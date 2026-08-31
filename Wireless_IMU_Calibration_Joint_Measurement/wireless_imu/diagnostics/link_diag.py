#!/usr/bin/env python3
"""
link_diag.py  -  find out WHERE packets are being lost.

THE QUESTION THIS ANSWERS
-------------------------
8-18% loss with 200-1800 ms gaps has two completely different possible causes,
and they need opposite fixes:

  AIR-SIDE      the ESP32's packets never reach the PC (weak signal, antenna,
                distance, channel congestion, brownout during TX bursts).
                Fix: hardware, placement, power.

  RECEIVER-SIDE the packets arrive but the kernel drops them because Python
                did not read the socket fast enough. imu_receiver.py only
                drains the socket when the consumer calls packetAvailable(),
                so any slow consumer stalls the drain and the 1 MB kernel
                buffer overflows. Fix: dedicated receive thread.
                (This is the flaw that would get WORSE under MuJoCo, since
                rendering a frame blocks the consumer.)

There is a THIRD pattern that neither of the above cover, and it matters
enough to be a first-class finding rather than something you notice by
accident:

  SENDER-PAUSE-OR-BUFFERED (ambiguous)   The sequence counter stays
                perfectly contiguous across a long silence -- no packets are
                dropped by the count -- but a lot of wall-clock time passed
                with nothing arriving. Sequence-based loss math is BLIND to
                this -- span and received count still match perfectly, so it
                reports 0% loss and "NO GAPS" even while, e.g., three
                separate 5-second silences went by. That much IS a solid
                finding on its own.

                What it does NOT tell you is WHY: packetSequence in
                ReBAIT_imu_firmware_v2.ino increments only AFTER a packet is
                built (past the early return on a failed mpu.update()), so
                an intact counter proves the SENDER PAUSED without building
                packets during the gap -- it says nothing about whether
                anything was buffered and delivered late. Sender-pause and
                buffered-and-late are NOT distinguishable from receiver-side
                data alone, and multi-second buffering isn't physically
                plausible on this hardware regardless (22s x 336 Hz x 92 B
                ~= 680 KB, versus the ESP32-C3's ~400 KB total SRAM). Do not
                default to "power-save" here. To find out which it is, check
                the ESP32's own serial log for "*** SENDER GAP <ms>" --
                printed with WiFi state, RSSI, and updFail/txFail counters
                at the moment of the gap.

HOW IT SEPARATES THEM
---------------------
This receiver is deliberately minimal: a dedicated thread that does nothing
but recvfrom() and read the 8-byte header. No float parsing, no analysis, no
disk. If loss persists even here, the consumer cannot be the cause, so it is
air-side.

Then --load adds an artificial consumer stall. If loss climbs with --load but
is near zero without it, the loss is receiver-side and a dedicated thread
fixes it. NOTE: --load only proves something if it actually pushes the
consumer's internal buffer toward its limit. If long time-domain gaps (see
below) give the consumer idle time to drain in between, --load can report a
clean pass without ever having stressed the failure mode it's meant to test --
check the reported buffer depth, not just the final loss %.

Independently of both, this script now measures TIME between consecutive
packets, not just their sequence numbers. Any inter-arrival gap over
--gap-threshold-ms (default 500 ms) is flagged AS ITS OWN FINDING, whether or
not the sequence counter shows any loss there. A time-domain gap with an
INTACT sequence counter across it is the sender-pause-or-buffered pattern
above, and is reported as such -- it is not folded into or hidden behind
"packet loss," and it is not labeled power-save without the ESP32 serial
log to back that up.

    python link_diag.py --seconds 60 --id 6              # baseline
    python link_diag.py --seconds 60 --id 6 --load 10    # simulate slow consumer

GAP PATTERN also tells you a lot:
    many single-packet gaps      -> ordinary radio interference
    few huge gaps (>50 packets)  -> association drops or brownout; check the
                                    ESP32 serial log's SENDER GAP line before
                                    assuming power save
    regularly spaced gaps        -> something periodic: WiFi scan, beacon,
                                    power management, or a CPU hog
"""

import argparse
import re
import socket
import struct
import subprocess
import sys
import threading
import time
from collections import deque

import numpy as np

# Only the header is unpacked: imu_id, flags, reserved, sequence.
# Deliberately NOT the 19 floats -- keeping this cheap is the whole point.
_HDR = "<BBHI"
_HDR_SIZE = struct.calcsize(_HDR)
_EXPECTED_PACKET = 92


class MinimalReceiver:
    """Dedicated receive thread. Does the least work possible per packet."""

    def __init__(self, port=5000, imu_id=None, bufsize=1 << 22):
        self.port = port
        self.imu_id = imu_id
        self.bufsize = bufsize
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.events = deque()          # (recv_time, imu_id, sequence)
        self.bad_size = 0
        self.actual_rcvbuf = None
        # Unconditional per-LOOP-ITERATION timestamp, taken every pass through
        # the while loop regardless of whether a packet arrived. socket
        # timeout is 0.2s, so under normal scheduling this should never show
        # a gap over ~200ms even during genuine silence on the wire -- the
        # loop keeps waking up on timeout and re-appending. If this log
        # itself shows a multi-second gap, the THREAD was not scheduled at
        # all for that stretch; that is categorically different from "no
        # packets arrived" and rules out anything on the network/air side,
        # since a starved thread can't read data that's sitting in the
        # kernel socket buffer either way. This is the direct, mechanical
        # test for GIL/scheduler starvation vs. genuine wire silence.
        self.iter_times = deque()

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2.0)

    def _run(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, self.bufsize)
        self.actual_rcvbuf = s.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
        s.bind(("0.0.0.0", self.port))
        s.settimeout(0.2)
        while not self._stop.is_set():
            self.iter_times.append(time.perf_counter())
            try:
                raw, _ = s.recvfrom(2048)
            except socket.timeout:
                continue
            except OSError:
                break
            if len(raw) != _EXPECTED_PACKET:
                self.bad_size += 1
                continue
            imu_id, _flags, _res, seq = struct.unpack(_HDR, raw[:_HDR_SIZE])
            if self.imu_id is not None and imu_id != self.imu_id:
                continue
            self.events.append((time.perf_counter(), imu_id, seq))
        s.close()


def _analyse_time_domain(t, seq, gap_threshold_ms):
    """
    FIRST-CLASS finding, independent of sequence-based loss math.

    Looks purely at the clock: any two consecutive received packets more
    than gap_threshold_ms apart get flagged, regardless of what their
    sequence numbers say. For each such gap, also reports whether the
    sequence counter was CONTIGUOUS across it (delta == 1) or not.

      contiguous  == True   -> sequence counter intact across the gap.
                                packetSequence only increments after a
                                packet is built, so this means EITHER the
                                sender paused without building packets OR
                                packets were buffered and delivered late --
                                these are not distinguishable from this
                                data. Sequence-based loss % reports this as
                                perfectly clean either way, which is why it
                                needs its own section.
      contiguous  == False  -> packets were BOTH delayed and some genuinely
                                went missing across the same stretch. Worth
                                knowing loss and delay coincided.

    Returns the list of (gap_ms, t_at_gap_end, contiguous) for the verdict
    logic to use.
    """
    dt_ms = np.diff(t) * 1e3
    dseq = np.diff(seq)
    idx = np.where(dt_ms > gap_threshold_ms)[0]

    print(f"\n{'-' * 68}")
    print(f"TIME-DOMAIN GAP CHECK   (threshold {gap_threshold_ms:.0f} ms)")
    print(f"{'-' * 68}")

    if idx.size == 0:
        print(f"  No inter-arrival gap exceeded {gap_threshold_ms:.0f} ms.")
        return []

    findings = []
    total_silent_ms = 0.0
    contiguous_count = 0
    for i in idx:
        gap_ms = dt_ms[i]
        contiguous = bool(dseq[i] == 1)
        findings.append((gap_ms, t[i + 1], contiguous))
        total_silent_ms += gap_ms
        if contiguous:
            contiguous_count += 1

    print(f"  {idx.size} gap(s) over threshold, "
          f"{total_silent_ms / 1000.0:.1f} s total silent time")
    print(f"  {contiguous_count}/{idx.size} of these had an INTACT sequence "
          f"counter across them")
    print(f"  {'  gap (ms)':>12}  {'ends at t=':>10}  {'seq intact?':>12}")
    for gap_ms, t_end, contiguous in sorted(findings, reverse=True)[:10]:
        print(f"  {gap_ms:12.0f}  {t_end:10.1f}s  "
              f"{'YES (sender paused OR buffered-late)' if contiguous else 'no (also lost data)'}")

    if contiguous_count and contiguous_count == idx.size:
        print(f"\n  *** ALL {idx.size} time-domain gap(s) had an intact sequence")
        print("  *** counter. A sequence-based loss metric alone reports this")
        print("  *** stretch as perfectly clean -- which is why this section")
        print("  *** exists as its own finding.")
        print("  *** This does NOT mean packets were buffered and delivered late.")
        print("  *** packetSequence only increments after a packet is built (past")
        print("  *** the early return on a failed mpu.update()), so an intact")
        print("  *** counter equally means the sender simply paused and never")
        print("  *** built any packets during the gap. Sender-pause and")
        print("  *** buffered-late are NOT distinguishable from this data.")
        print("  *** Check the ESP32's own serial log for '*** SENDER GAP <ms>' --")
        print("  *** it's printed with WiFi state, RSSI, and updFail/txFail")
        print("  *** counters at the moment of the gap, which is the only place")
        print("  *** this can actually be resolved.")
    elif contiguous_count:
        print(f"\n  Mixed: {contiguous_count} gap(s) had an intact sequence counter "
              f"(sender-pause-or-buffered, ambiguous -- see above), "
              f"{idx.size - contiguous_count} also lost packets.")
        print("  Both a sender-pause-or-buffered stretch and a genuine loss")
        print("  mechanism appear to be present -- don't let fixing one hide")
        print("  the other.")

    if idx.size >= 4:
        gap_ends = np.array([f[1] for f in findings])
        spacing = np.diff(gap_ends)
        if spacing.size:
            cv = spacing.std() / max(spacing.mean(), 1e-9)
            print(f"\n  gap spacing (time-domain)   mean {spacing.mean():.2f} s, "
                  f"CV {cv:.2f}")
            if cv < 0.35:
                print("  -> REGULARLY SPACED in wall-clock time. That points at")
                print("     a scheduled/periodic cause (adapter power-save timer,")
                print("     WiFi scan interval, beacon interval) rather than")
                print("     random interference.")

    return findings


def _analyse_scheduling(iter_times, time_gaps, wifi_call_log, gap_threshold_ms):
    """
    Decisive mechanism check: was the receive THREAD not scheduled during a
    gap, or was the thread running fine and simply saw no data at the socket?

    iter_times has one entry per pass through MinimalReceiver's while loop,
    logged BEFORE recvfrom() is called, regardless of whether that call
    times out or returns a packet. socket timeout is 0.2s, so under normal
    scheduling iter_times should never show a gap over ~200-300ms even
    during genuine silence on the wire -- the loop keeps waking up on
    timeout and re-logging. A gap in iter_times itself, not just in packet
    arrivals, means the thread was not given CPU time for that stretch --
    that rules out anything about the network, the AP, or the ESP32, since a
    starved thread can't drain the kernel socket buffer no matter what is
    sitting in it.

    Each such iter-time gap is also checked for direct overlap against
    wifi_call_log (netsh subprocess.run start/end times), so H1 is settled
    by an actual overlap test instead of eyeballing two timestamp columns.
    """
    print(f"\n{'-' * 68}")
    print("RECEIVE-THREAD SCHEDULING CHECK")
    print(f"{'-' * 68}")

    if len(iter_times) < 2:
        print("  Not enough loop iterations recorded to analyse.")
        return "insufficient"

    # MinimalReceiver's socket has a 0.2s recvfrom timeout, so an iteration
    # gap of ~200ms is NORMAL and expected on every idle cycle -- it is not
    # a finding. Floor this check's threshold comfortably above that so a
    # caller-supplied --gap-threshold-ms below ~250ms (meant for the
    # packet-arrival check) can't turn routine timeout cycles into false
    # "scheduling gap" positives here.
    gap_threshold_ms = max(gap_threshold_ms, 250.0)

    it = np.array(iter_times)
    it = it - it[0] if it[0] else it
    it_dt_ms = np.diff(it) * 1e3
    idx = np.where(it_dt_ms > gap_threshold_ms)[0]

    if idx.size == 0:
        print(f"  No gap over {gap_threshold_ms:.0f} ms in the receive loop's own")
        print("  iteration timestamps. The thread was scheduled normally the")
        print("  entire run -- if TIME-DOMAIN GAP CHECK above still found silent")
        print("  stretches, that silence happened at the SOCKET/network level,")
        print("  not because the thread was blocked from running.")
        return "not_starved" if time_gaps else "no_gaps"

    print(f"  {idx.size} gap(s) over {gap_threshold_ms:.0f} ms in the LOOP ITERATION")
    print("  log itself (not just packet arrivals). This means the receive")
    print("  thread was not scheduled to run during these stretches -- it is")
    print("  not a matter of no data being available, the thread simply did")
    print("  not get CPU time. This rules out the network, the AP, and the")
    print("  ESP32; the cause is inside this process.")
    for i in idx[:10]:
        gap_ms = it_dt_ms[i]
        t_end = it[i + 1]
        overlap = [c for c in wifi_call_log if c[0] <= t_end <= c[1] + 0.05
                   or (c[0] >= it[i] and c[0] <= t_end)]
        if overlap:
            c = overlap[0]
            print(f"    {gap_ms:9.0f} ms ending at t={t_end:6.1f}s  "
                  f"-> OVERLAPS a netsh call ({c[0]:.1f}s-{c[1]:.1f}s, "
                  f"{c[2]:.0f} ms)")
        else:
            print(f"    {gap_ms:9.0f} ms ending at t={t_end:6.1f}s  "
                  f"-> no netsh call was active during this gap")

    overlapping = sum(
        1 for i in idx
        if any(c[0] <= it[i + 1] <= c[1] + 0.05 or (c[0] >= it[i] and c[0] <= it[i + 1])
               for c in wifi_call_log)
    )
    print(f"\n  {overlapping}/{idx.size} scheduling gap(s) overlap an active netsh call.")
    if wifi_call_log:
        durs = np.array([c[2] for c in wifi_call_log])
        print(f"  netsh call durations seen this run: median {np.median(durs):.0f} ms, "
              f"max {durs.max():.0f} ms, n={len(durs)}")
    if overlapping == idx.size and idx.size > 0:
        print("\n  *** EVERY scheduling gap overlaps a netsh call. That is a direct,")
        print("  *** mechanical link (not correlation): disable WiFi state logging")
        print("  *** with --wifi-log-interval 0 and re-run to confirm the gaps")
        print("  *** vanish. If they do, the cause is subprocess.run()/CreateProcess")
        print("  *** blocking the interpreter during netsh.exe process creation --")
        print("  *** this is a Windows/CPython scheduling issue, NOT 802.11")
        print("  *** power-save, and disabling adapter power management will NOT")
        print("  *** fix it.")
    elif overlapping == 0 and idx.size > 0:
        print("\n  *** NONE of the scheduling gaps overlap a netsh call. WifiSampler")
        print("  *** is not the cause. Suspect something else in this process:")
        print("  *** console output (QuickEdit mode blocks writes to a clicked")
        print("  *** console until Esc is pressed -- redirect stdout to a file and")
        print("  *** never touch the window to test this), or another blocking")
        print("  *** call sharing the GIL.")

    if overlapping == idx.size and idx.size > 0:
        return "starved_netsh"
    return "starved_other"


class WifiSampler:
    """
    Periodically polls `netsh wlan show interfaces` in a background thread
    and records (time, signal_pct, channel, rx_mbps, tx_mbps, state) so gaps
    can be correlated against actual radio state instead of inferred after
    the fact. Windows only -- a no-op elsewhere.

    IMPORTANT CAVEAT, printed in the report: `netsh wlan show interfaces`
    reports the STATION interface -- the one the PC uses to join a network
    as a client. If the PC is running Windows Mobile Hotspot to host the AP
    the ESP32 connects to, the physical WiFi adapter is normally used as the
    internet UPLINK while a separate virtual "Microsoft Wi-Fi Direct Virtual
    Adapter" hosts the actual access point. In that setup this command may
    report the wrong adapter, or report "disconnected" even while the
    hotspot is actively serving the ESP32 -- that is a sign to also check
    `netsh wlan show hostednetwork`, not evidence the link is down.
    """

    _LINE_RE = re.compile(r"^\s*([A-Za-z0-9 ()/_.-]+?)\s*:\s*(.*?)\s*$")

    def __init__(self, interval_s=2.0):
        self.interval_s = interval_s
        self.samples = []              # (t, dict-of-parsed-fields)
        self.available = sys.platform.startswith("win")
        self.warned_wrong_adapter = False
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._t0 = None
        # Wall-clock start/end of every netsh subprocess.run() CALL (not just
        # the parsed sample), so a receive-thread stall can be checked for
        # actual time overlap with a netsh call in progress instead of by
        # eyeballing timestamps. subprocess.run() itself releases the GIL
        # while the child process runs, but process CREATION on Windows
        # (_winapi.CreateProcess) can be slow -- e.g. AV real-time scanning
        # of netsh.exe -- and that creation step does not release the GIL
        # for its duration, so a slow CreateProcess call CAN starve every
        # other Python thread in the process, including the receive thread.
        self.call_log = []             # (t_start, t_end, duration_ms)

    def start(self):
        if not self.available:
            return
        self._t0 = time.perf_counter()
        self._thread.start()

    def stop(self):
        if not self.available:
            return
        self._stop.set()
        self._thread.join(timeout=2.0)

    @staticmethod
    def _parse(output):
        fields = {}
        for line in output.splitlines():
            m = WifiSampler._LINE_RE.match(line)
            if not m:
                continue
            key, val = m.group(1).strip(), m.group(2).strip()
            fields[key] = val
        return fields

    def _run(self):
        while not self._stop.is_set():
            call_t0 = time.perf_counter() - self._t0
            try:
                proc = subprocess.run(
                    ["netsh", "wlan", "show", "interfaces"],
                    capture_output=True, text=True, timeout=2.0,
                )
                fields = self._parse(proc.stdout)
                t = time.perf_counter() - self._t0
                self.samples.append((t, fields))
                if fields.get("State", "").lower() == "disconnected" and not self.warned_wrong_adapter:
                    self.warned_wrong_adapter = True
            except Exception:
                pass
            call_t1 = time.perf_counter() - self._t0
            self.call_log.append((call_t0, call_t1, (call_t1 - call_t0) * 1000.0))
            self._stop.wait(self.interval_s)

    def report(self, findings_times):
        """
        Print the WiFi sample log and, for each timestamp of interest
        (large gaps, from either time-domain or sequence analysis), show
        the nearest sample so the two can be read side by side.
        """
        if not self.available:
            print(f"\n{'-' * 68}")
            print("WIFI STATE LOG")
            print(f"{'-' * 68}")
            print("  Skipped: `netsh` sampling is Windows-only.")
            return
        if not self.samples:
            print(f"\n{'-' * 68}")
            print("WIFI STATE LOG")
            print(f"{'-' * 68}")
            print("  No samples captured (netsh calls may have failed or")
            print("  the run was too short).")
            return

        print(f"\n{'-' * 68}")
        print(f"WIFI STATE LOG   ({len(self.samples)} samples, "
              f"every ~{self.interval_s:.0f}s)")
        print(f"{'-' * 68}")

        if self.warned_wrong_adapter:
            print("  *** NOTE: at least one sample showed State: disconnected.")
            print("  *** If you're using Windows Mobile Hotspot, the physical")
            print("  *** WiFi adapter queried here is normally the INTERNET")
            print("  *** UPLINK, not the virtual adapter hosting the AP the")
            print("  *** ESP32 connects to. A 'disconnected' reading here does")
            print("  *** NOT necessarily mean the hotspot link is down -- also")
            print("  *** check: netsh wlan show hostednetwork")
            print()

        signals, channels = [], []
        for t, f in self.samples:
            sig = f.get("Signal", "").rstrip("%")
            chan = f.get("Channel", "")
            if sig.isdigit():
                signals.append((t, int(sig)))
            if chan:
                channels.append((t, chan))

        if signals:
            vals = [s for _, s in signals]
            print(f"  signal %        min {min(vals)}   max {max(vals)}   "
                  f"mean {sum(vals)/len(vals):.0f}")
        if channels:
            uniq = sorted(set(c for _, c in channels))
            if len(uniq) > 1:
                print(f"  *** CHANNEL CHANGED during capture: {', '.join(uniq)}")
                print("  *** A channel switch (roaming, DFS radar event, driver")
                print("  *** reassociation) is a concrete, checkable cause of a")
                print("  *** sudden multi-second stall or burst of loss -- look")
                print("  *** for one lining up with a large gap below.")
            else:
                print(f"  channel        {uniq[0]} (constant)")

        if findings_times:
            print(f"\n  nearest WiFi sample to each notable gap:")
            for label, t_gap in findings_times:
                if not self.samples:
                    break
                nearest = min(self.samples, key=lambda s: abs(s[0] - t_gap))
                f = nearest[1]
                sig = f.get("Signal", "?").rstrip("%") or "?"
                chan = f.get("Channel", "?")
                state = f.get("State", "?")
                dt = nearest[0] - t_gap
                print(f"    {label:<28} t={t_gap:6.1f}s  -> sample at "
                      f"t={nearest[0]:6.1f}s ({dt:+.1f}s away): "
                      f"signal {sig}%  channel {chan}  state {state}")


def _analyse_segment(t, seq, label):
    span = int(seq[-1] - seq[0]) + 1
    received = len(seq)
    lost = span - received
    dseq = np.diff(seq)
    gaps = dseq[dseq > 1] - 1
    print(f"    {label:<12} duration {t[-1]:6.1f}s  received {received:6d}  "
          f"lost {lost:5d} ({100*lost/max(span,1):5.2f}%)  "
          f"gap events {int((dseq>1).sum())}")
    return span, received, gaps


def analyse(events, duration, load_ms, gap_threshold_ms=500.0, wifi_sampler=None,
            iter_times=None):
    notable_times = []   # (label, t) pairs to hand to wifi_sampler.report()

    if len(events) < 10:
        print("\n  Too few packets to analyse. Is the IMU streaming yet?")
        print("  (Remember: the firmware waits ~3.4 min before it prints 'Streaming.')")
        if wifi_sampler is not None:
            wifi_sampler.report(notable_times)
        return

    t = np.array([e[0] for e in events])
    seq = np.array([e[2] for e in events], dtype=np.int64)
    t = t - t[0]

    # SEQUENCE RESET DETECTION. If the ESP32 reboots (brownout, crash, manual
    # power cycle) mid-recording, packetSequence restarts at 0. That makes
    # seq non-monotonic, and computing one global span=seq[-1]-seq[0]+1
    # across a reset silently produces nonsense -- possibly a small or even
    # negative "loss" that reads as clean when whole seconds of data are
    # actually missing. So: find every reset FIRST, split into contiguous
    # segments at each one, and analyse each segment on its own. A reset is
    # itself the finding, not something to average away.
    dseq_raw = np.diff(seq)
    reset_idx = np.where(dseq_raw < 0)[0]
    if reset_idx.size:
        print(f"\n  *** {reset_idx.size} SEQUENCE RESET(S) DETECTED -- the sender")
        print("  *** restarted mid-recording (reboot / brownout / manual power")
        print("  *** cycle). Treating each stretch between resets separately;")
        print("  *** a single global loss % would be meaningless here.")
        for i in reset_idx:
            print(f"      reset at t={t[i+1]:.1f}s: seq {seq[i]} -> {seq[i+1]}")

    # Time-domain gap check runs on the FULL, unsegmented stream, before any
    # reset-based splitting -- a reset doesn't erase the wall-clock silence
    # that led up to it, and we want that silence reported either way.
    time_gaps = _analyse_time_domain(t, seq, gap_threshold_ms)
    for gap_ms, t_end, contiguous in time_gaps:
        tag = "delayed" if contiguous else "delayed+lost"
        notable_times.append((f"time-gap ({tag}, {gap_ms:.0f}ms)", t_end))
    for i in reset_idx:
        notable_times.append(("sequence reset", float(t[i + 1])))

    scheduling_verdict = None
    if iter_times is not None:
        wifi_calls = wifi_sampler.call_log if wifi_sampler is not None else []
        scheduling_verdict = _analyse_scheduling(iter_times, time_gaps, wifi_calls,
                                                  gap_threshold_ms)

    bounds = [0] + list(reset_idx + 1) + [len(seq)]
    segments = [(bounds[k], bounds[k + 1]) for k in range(len(bounds) - 1)
               if bounds[k + 1] - bounds[k] >= 2]

    if len(segments) > 1:
        print(f"\n  Analysing {len(segments)} segments between resets separately:")
        total_span = total_received = 0
        all_gaps = []
        for k, (a, b) in enumerate(segments):
            span, received, gaps = _analyse_segment(
                t[a:b] - t[a], seq[a:b], f"segment {k + 1}")
            total_span += span
            total_received += received
            all_gaps.append(gaps)
        agg_loss_pct = 100.0 * (total_span - total_received) / max(total_span, 1)
        agg_gaps = np.concatenate(all_gaps) if all_gaps else np.array([])
        print(f"\n  {'-' * 60}")
        print("  Any reset above is the dominant finding regardless of each")
        print("  segment's individual loss % -- a device that reboots under")
        print("  normal use has a power or firmware problem, not a WiFi one.")
        _print_verdict(agg_loss_pct, agg_gaps, time_gaps, load_ms, scheduling_verdict)
        if wifi_sampler is not None:
            wifi_sampler.report(notable_times)
        return  

    a, b = segments[0] if segments else (0, len(seq))
    t, seq = t[a:b], seq[a:b]

    span = int(seq[-1] - seq[0]) + 1
    received = len(seq)
    lost = span - received
    loss_pct = 100.0 * lost / max(span, 1)

    dseq = np.diff(seq)
    gaps = dseq[dseq > 1] - 1          # packets missing in each gap
    gap_times = t[1:][dseq > 1]

    dt_ms = np.diff(t) * 1e3
    contiguous = dt_ms[dseq == 1]      # inter-arrival where nothing was lost

    print(f"\n{'-' * 68}")
    print(f"LINK DIAGNOSTIC   ({duration:.0f} s"
          + (f", simulated consumer load {load_ms} ms/iter" if load_ms else ", no added load")
          + ")")
    print(f"{'-' * 68}")
    print(f"  received             {received}")
    print(f"  sequence span        {span}")
    print(f"  lost (by sequence)   {lost}  ({loss_pct:.2f}%)")
    if contiguous.size:
        print(f"  rate (contiguous)    {1000.0 / np.median(contiguous):.1f} Hz")
        print(f"  inter-arrival        p50 {np.percentile(contiguous, 50):.2f} ms   "
              f"p99 {np.percentile(contiguous, 99):.2f} ms   "
              f"max {contiguous.max():.1f} ms")

    if gaps.size == 0:
        print("\n  NO SEQUENCE GAPS. Every packet sent was received.")
        print("  (This does NOT mean delivery was well-timed -- see the")
        print("  TIME-DOMAIN GAP CHECK above, which checks the clock, not")
        print("  the sequence counter.)")
    else:
        print(f"\n  gap events           {gaps.size}")
        bins = [(1, 1, "single packet"), (2, 5, "2-5"), (6, 20, "6-20"),
                (21, 100, "21-100"), (101, 10 ** 9, ">100")]
        print(f"  {'gap size':<16}{'count':>8}{'packets lost':>15}")
        for lo, hi, lbl in bins:
            m = (gaps >= lo) & (gaps <= hi)
            if m.sum():
                print(f"  {lbl:<16}{m.sum():>8}{int(gaps[m].sum()):>15}")

        big = np.argsort(gaps)[-5:][::-1]
        print(f"\n  largest gaps:")
        for i in big:
            print(f"    {int(gaps[i]):5d} packets "
                  f"({gaps[i] * np.median(contiguous) if contiguous.size else 0:6.0f} ms) "
                  f"at t = {gap_times[i]:6.1f} s")
            notable_times.append((f"seq-gap ({int(gaps[i])} pkts)", float(gap_times[i])))

        # Periodicity: regularly spaced gaps point at something scheduled
        # (WiFi scan, power save, a CPU hog) rather than random interference.
        if gaps.size >= 4:
            spacing = np.diff(gap_times)
            cv = spacing.std() / max(spacing.mean(), 1e-9)
            print(f"\n  gap spacing          mean {spacing.mean():.2f} s, "
                  f"CV {cv:.2f}")
            if cv < 0.35:
                print("  -> REGULARLY SPACED. Something periodic is causing this,")
                print("     not random interference. Suspect WiFi power management,")
                print("     a background scan, or the hotspot's own housekeeping.")
            else:
                print("  -> irregularly spaced, consistent with random interference")
                print("     or intermittent signal strength.")

    _print_verdict(loss_pct, gaps, time_gaps, load_ms, scheduling_verdict)

    if wifi_sampler is not None:
        wifi_sampler.report(notable_times)


def _print_verdict(loss_pct, gaps, time_gaps, load_ms, scheduling_verdict=None):
    # ---- verdict -------------------------------------------------------
    # Time-domain findings take priority: an intact sequence counter across
    # a silent stretch is still worth flagging on its own, separate from any
    # sequence-based loss %. But do NOT claim it means "delayed, not lost" or
    # name 802.11 power-save as the cause -- neither is supported:
    #
    #   packetSequence (ReBAIT_imu_firmware_v2.ino) increments AFTER the
    #   early return on a failed mpu.update(), so it only advances when a
    #   packet is actually built. An intact counter across a gap proves the
    #   SENDER PAUSED without building packets -- it says nothing about
    #   whether anything was buffered and delivered late. From receiver-side
    #   data alone, sender-pause and buffered-and-late are NOT
    #   distinguishable; don't pick one.
    #
    #   The arithmetic also makes long buffering implausible regardless: 22s
    #   x 336 Hz x 92 bytes = ~680 KB, versus the ESP32-C3's ~400 KB of total
    #   SRAM. There is nowhere on that chip to hold a buffer that size.
    #
    # What CAN be said from receiver-side data alone is whether the RECEIVE
    # thread itself was scheduled (the SCHEDULING CHECK above) -- that's a
    # real, proven finding when it applies, kept below. What can't be said
    # from here is sender-pause vs. buffered-late; that requires the ESP32's
    # own serial log.
    print(f"\n  {'VERDICT':<20}")

    contiguous_time_gaps = [g for g in time_gaps if g[2]]
    lossy_time_gaps = [g for g in time_gaps if not g[2]]

    if contiguous_time_gaps and not lossy_time_gaps:
        print(f"  {len(contiguous_time_gaps)} time-domain gap(s), sequence counter intact")
        print("  across them. This means either (a) the sender paused without")
        print("  building packets, or (b) packets were buffered and delivered")
        print("  late. These are NOT distinguishable from receiver-side data")
        print("  alone -- packetSequence only increments after a packet is")
        print("  built, so an intact counter proves nothing about buffering.")

        if scheduling_verdict in ("starved_netsh", "starved_other"):
            print("\n  Separately, the RECEIVE-THREAD SCHEDULING CHECK above found")
            print("  that the receive thread itself was NOT scheduled during these")
            if scheduling_verdict == "starved_netsh":
                print("  gaps, and every such gap overlaps an active netsh call. That")
                print("  is a real, proven receiver-process-side stall")
                print("  (subprocess.run()/CreateProcess blocking the interpreter).")
                print("  Confirm with --wifi-log-interval 0 and re-run. Note this")
                print("  finding stands regardless of (a) vs (b) above -- it can")
                print("  compound with either.")
            else:
                print("  gaps, and it was NOT netsh. Something else in this process is")
                print("  blocking it -- see the SCHEDULING CHECK section above for what")
                print("  to check next (console QuickEdit mode is the next most")
                print("  likely). This finding stands regardless of (a) vs (b) above.")

        print("\n  To tell (a) sender-pause from (b) buffered-and-late apart, check")
        print("  the ESP32's own serial output for each gap: it prints")
        print("  '*** SENDER GAP <ms>' along with WiFi state, RSSI, and the")
        print("  updFail/txFail counters at that moment. That is the only place")
        print("  this can actually be resolved -- it cannot be settled from PC-side")
        print("  timing data, no matter how it's sliced.")
        print("  (For reference: buffering long enough to explain a multi-second")
        print("  gap is not physically plausible on this hardware anyway -- a 22s")
        print("  gap at 336 Hz x 92 bytes/packet is ~680 KB, more than the")
        print("  ESP32-C3's ~400 KB of total SRAM.)")
        return

    if loss_pct < 0.5 and not time_gaps:
        if load_ms:
            print("  Clean even WITH simulated consumer load, AND no time-domain")
            print("  gaps over threshold. The link is fine and a dedicated receive")
            print("  thread fully solves it -- your losses in imu_receiver.py were")
            print("  RECEIVER-SIDE (drain-on-demand stalling).")
        else:
            print("  Clean with a minimal dedicated-thread receiver, and no")
            print("  time-domain gaps over threshold.")
            print("  Now re-run with --load 10 to confirm the consumer was the")
            print("  cause -- and check the reported buffer depth in that run,")
            print("  not just the final loss %, since a run with no idle time")
            print("  for the consumer to catch up is a stronger test than one")
            print("  where it never got stressed.")
        return

    if loss_pct < 0.5 and time_gaps:
        # Some time-domain gaps exist but weren't all-contiguous, or were
        # already covered by the branch above -- this remaining case is
        # sequence-clean with at least one gap that ALSO lost packets.
        print("  Sequence-based loss is low, but at least one time-domain gap")
        print("  also lost packets across it (see TIME-DOMAIN GAP CHECK above).")
        print("  Treat that stretch as a genuine loss event, not noise -- a low")
        print("  overall percentage can hide one real incident.")
        return

    if gaps.size and float(gaps[gaps > 20].sum()) / max(gaps.sum(), 1) > 0.5:
        # Judge by the SHARE OF LOST PACKETS in big gaps, not the median gap
        # size. A handful of single drops alongside one huge burst gives a
        # median of 1 while the burst is plainly the real problem.
        share = 100.0 * gaps[gaps > 20].sum() / gaps.sum()
        print(f"  {share:.0f}% of lost packets came from gaps LARGER than 20,")
        print("  even with a minimal receiver. A slow consumer cannot cause")
        print("  bursts like that -- it is AIR-SIDE. Suspect, in order:")
        print("    1. Power. ESP32 WiFi TX draws current in bursts; a marginal USB")
        print("       cable or supply browns it out. Try a different cable/port.")
        print("    2. Antenna. Check the u.FL connector or antenna solder joint.")
        print("    3. Distance/obstruction, including your own body between the")
        print("       sensor and the PC once it is strapped to the thigh.")
    else:
        print("  Moderate loss spread across many small gaps, with a minimal")
        print("  receiver. Most consistent with radio interference or weak signal.")
        print("  Try a different WiFi channel on the hotspot, and move closer.")


def measure_sleep_resolution(requested_ms=0.5, n=500):
    """
    THE SPECIFIC, CHECKABLE HYPOTHESIS: imu_receiver.py's record() loop does
    `time.sleep(0.0005)` every time the buffer is momentarily empty. Windows'
    default timer resolution is commonly ~15.6 ms, so a REQUESTED 0.5 ms
    sleep can actually take ~15 ms -- 30x longer -- unless the process raises
    its timer resolution. That turns record()'s "briefly yield" into a real
    stall on every idle iteration, and under bursty WiFi delivery (packets
    do not arrive perfectly evenly) that is a concrete, mechanical way to
    lose packets even in an architecture that looks fine on paper.

    This measures ACTUAL sleep(0.0005) duration on whatever machine runs it.
    On Linux (this sandbox) it will typically be accurate to a fraction of a
    ms. Run this SAME function on the Windows machine that runs
    imu_receiver.py to see whether the hypothesis holds there.
    """
    req = requested_ms / 1000.0
    durs = []
    for _ in range(n):
        t0 = time.perf_counter()
        time.sleep(req)
        durs.append((time.perf_counter() - t0) * 1000.0)
    durs = np.array(durs)
    print(f"\n{'-' * 68}")
    print(f"SLEEP RESOLUTION PROBE  (requested {requested_ms:.2f} ms x {n})")
    print(f"{'-' * 68}")
    print(f"  actual        median {np.median(durs):.2f} ms   "
          f"p95 {np.percentile(durs, 95):.2f} ms   max {durs.max():.2f} ms")
    ratio = np.median(durs) / requested_ms
    print(f"  ratio to requested: {ratio:.1f}x")
    if ratio > 5:
        print(f"\n  *** COARSE TIMER CONFIRMED. Every idle iteration in")
        print(f"  *** record()'s loop stalls ~{np.median(durs):.0f} ms instead of")
        print(f"  *** the requested 0.5 ms. At ~333 Hz that is ~{np.median(durs)*0.333:.1f}")
        print(f"  *** packets' worth of buffer backlog created PER IDLE CYCLE.")
        print(f"  *** Fix: call time.sleep(0) instead of time.sleep(0.0005) --")
        print(f"  *** a zero-duration sleep just yields the GIL/scheduler slice")
        print(f"  *** without engaging the OS timer at all -- or wrap the")
        print(f"  *** recording loop with ctypes.windll.winmm.timeBeginPeriod(1)")
        print(f"  *** / timeEndPeriod(1) to request 1 ms system timer resolution.")
    else:
        print(f"\n  Timer resolution is fine here. If this is run on Linux and the")
        print(f"  real recordings happen on Windows, this does NOT rule the")
        print(f"  hypothesis out -- re-run this exact function on that machine.")
    return durs


def run_single_thread_test(seconds, port, imu_id, load_ms):
    """
    Mirrors imu_receiver.py's record() loop SHAPE EXACTLY:
        if packetAvailable(): pop one packet
        else: time.sleep(0.0005)
    rather than an invented per-iteration stall. This is what actually runs
    in the field, so this is what must be tested. load_ms, if given, ADDS an
    artificial per-popped-packet processing cost on top (e.g. numpy array
    construction, disk writes) to see whether normal per-packet work is
    itself enough to fall behind -- separate from the sleep-resolution
    question, which measure_sleep_resolution() tests directly.
    """
    from imu_receiver import UdpImuCallback

    cb = UdpImuCallback(port=port, expected_ids=(imu_id,) if imu_id else (6,))
    cb.enable()
    cb.attach()
    events = []
    t0 = time.perf_counter()
    last = 0
    while time.perf_counter() - t0 < seconds:
        if cb.packetAvailable():
            did, pkt = cb.getNextRaw()
            events.append((time.perf_counter(), pkt["imu_id"], pkt["sequence"]))
            if load_ms:                     # simulated PER-PACKET processing cost
                time.sleep(load_ms / 1000.0)
        else:
            time.sleep(0.0005)              # EXACT match to record()'s idle sleep
        el = time.perf_counter() - t0
        if int(el) // 5 > last:
            last = int(el) // 5
            print(f"  {el:5.1f} s   {len(events)} packets   "
                  f"(cb reports {sum(cb.received.values())} received, "
                  f"{sum(cb.dropped.values())} dropped, "
                  f"buf depth {len(cb._buf)}/{cb._buf.maxlen})")
    cb.detach()
    cb.close()
    return events


def main():
    ap = argparse.ArgumentParser(description="Separate air-side from receiver-side packet loss.")
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--id", type=int, default=None, help="only count this IMU_ID")
    ap.add_argument("--load", type=float, default=0.0,
                    help="ms of simulated PER-PACKET processing cost, added "
                         "on top of imu_receiver.py's OWN record()-shaped "
                         "loop (packetAvailable/getNextRaw/idle-sleep)")
    ap.add_argument("--gap-threshold-ms", type=float, default=500.0,
                    help="inter-arrival time (ms) above which a gap is "
                         "flagged as a TIME-DOMAIN finding, independent of "
                         "sequence-based loss (default: 500)")
    ap.add_argument("--wifi-log-interval", type=float, default=2.0,
                    help="seconds between `netsh wlan show interfaces` "
                         "polls during capture, Windows only (default: 2). "
                         "Use 0 to disable WiFi state logging entirely.")
    ap.add_argument("--sleep-probe", action="store_true",
                    help="measure actual time.sleep(0.0005) duration on THIS "
                         "machine -- run this on the Windows box that runs "
                         "imu_receiver.py, not just in a dev sandbox")
    a = ap.parse_args()

    if a.sleep_probe:
        measure_sleep_resolution()
        return

    wifi = WifiSampler(interval_s=a.wifi_log_interval) if a.wifi_log_interval > 0 else None
    if wifi is not None:
        if wifi.available:
            print(f"  WiFi state logging: netsh poll every {a.wifi_log_interval:.0f}s "
                  f"(Windows)")
        else:
            print(f"  WiFi state logging: skipped, not on Windows")
        wifi.start()

    if a.load:
        # SAME-THREAD test: reproduces imu_receiver.py's actual drain-on-
        # demand architecture, the only honest way to test this path.
        print(f"Listening on :{a.port} for {a.seconds:.0f} s"
              + (f" (IMU_ID {a.id})" if a.id is not None else " (all IDs)"))
        print(f"  SAME-THREAD test via imu_receiver.py's UdpImuCallback,")
        print(f"  simulating {a.load:.0f} ms of consumer work per iteration")
        print(f"  (a dedicated-thread receiver CANNOT show this failure mode")
        print(f"  by construction -- that separation is the fix, so --load")
        print(f"  deliberately bypasses it and drains on the same thread)")
        events = run_single_thread_test(a.seconds, a.port, a.id, a.load)
        if wifi is not None:
            wifi.stop()
        analyse(events, a.seconds, a.load, a.gap_threshold_ms, wifi_sampler=wifi)
    else:
        # baseline: dedicated-thread receiver, minimal per-packet work
        rx = MinimalReceiver(port=a.port, imu_id=a.id)
        rx.start()
        time.sleep(0.2)
        print(f"Listening on :{a.port} for {a.seconds:.0f} s"
              + (f" (IMU_ID {a.id})" if a.id is not None else " (all IDs)"))
        print(f"  SO_RCVBUF granted: {rx.actual_rcvbuf} bytes")
        print(f"  dedicated-thread receiver, no added load")

        t0 = time.perf_counter()
        last = 0
        while time.perf_counter() - t0 < a.seconds:
            time.sleep(0.05)
            el = time.perf_counter() - t0
            if int(el) // 5 > last:
                last = int(el) // 5
                print(f"  {el:5.1f} s   {len(rx.events)} packets")

        rx.stop()
        if wifi is not None:
            wifi.stop()
        if rx.bad_size:
            print(f"\n  NOTE: {rx.bad_size} packets had an unexpected size "
                  f"(expected {_EXPECTED_PACKET} bytes).")
            print("  Are you running MyoSuite_imu_firmware.ino (45-byte) instead of")
            print("  ReBAIT_imu_firmware_v2.ino (92-byte)?")

        analyse(list(rx.events), a.seconds, 0, a.gap_threshold_ms, wifi_sampler=wifi,
                iter_times=list(rx.iter_times))


if __name__ == "__main__":
    main()