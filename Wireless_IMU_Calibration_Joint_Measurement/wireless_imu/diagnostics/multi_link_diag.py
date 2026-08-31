#!/usr/bin/env python3
"""
multi_link_diag.py  -  record several IMU_IDs at once, on ONE socket, and
check whether any gaps are CORRELATED across devices.

WHY THIS IS A SEPARATE TOOL, NOT A PATCH TO link_diag.py
-----------------------------------------------------------
link_diag.py has been extended independently (netsh WiFi-state polling,
receive-thread scheduling check, --sleep-probe) since it was last shared
here. Patching it blind risks breaking working instrumentation. This is a
focused companion for one question link_diag.py cannot answer at all: does
a gap on one sensor line up in TIME with a gap on another sensor?

WHY THAT QUESTION MATTERS
---------------------------
Two independent ESP32 boards cannot coincidentally go quiet for the same
duration at the same wall-clock instant -- no per-board cause (WiFi
association, antenna, MPU stall) can produce that. If it happens, the
shared cause has to be upstream of both boards: the PC, the hotspot, or
something else common to the whole link. That correlation is exactly what
first suggested a receiver-side cause weeks ago, before per-board firmware
instrumentation (SENDER GAP, WIFI UP/DOWN) ruled the boards out one at a
time. This tool makes that correlation check a first-class, automatic
result instead of something spotted by eye across two separate terminal
outputs.

ONE SOCKET, MULTIPLE SENDERS
-------------------------------
Both ESP32s send to the same (IP, port). UDP is connectionless, so a single
bound socket receives from both -- no imu_id filter is applied at the
socket level, only when SORTING events into per-device buckets afterward.
This means the receive path (and therefore any receiver-side stall) is
IDENTICAL for every device in one run, which is what makes the correlation
check meaningful: if it stalls, it stalls for everyone at once, by
construction, since there's only one recvfrom() loop.

USAGE
    python multi_link_diag.py --seconds 60 --ids 5 6
    python multi_link_diag.py --seconds 60           # auto-discover IDs
"""

import argparse
import socket
import struct
import threading
import time
from collections import defaultdict, deque

import numpy as np

# Matches IMUPacket in ReBAIT_imu_firmware_v4.ino:
#   uint8_t imu_id; uint8_t flags; uint16_t reserved; uint32_t sequence; ...
_HDR = "<BBHI"
_HDR_SIZE = struct.calcsize(_HDR)
_EXPECTED_PACKET = 92


class MultiReceiver:
    """One socket, one thread, routes every packet by imu_id as it arrives."""

    def __init__(self, port=5000, ids=None, bufsize=1 << 22):
        self.port = port
        self.ids = set(ids) if ids else None   # None = accept any id seen
        self.bufsize = bufsize
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        # events[imu_id] = list of (recv_time, sequence)
        self.events = defaultdict(list)
        self.bad_size = 0
        self.actual_rcvbuf = None
        self.seen_ids = set()

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
            self.seen_ids.add(imu_id)
            if self.ids is not None and imu_id not in self.ids:
                continue
            self.events[imu_id].append((time.perf_counter(), seq))
        s.close()


def per_device_gaps(t, seq, time_gap_ms=500.0):
    """Sequence-reset-aware gap analysis for one device's event stream."""
    t = np.asarray(t, float)
    seq = np.asarray(seq, np.int64)

    dseq = np.diff(seq)
    reset_idx = np.where(dseq < 0)[0]
    bounds = [0] + list(reset_idx + 1) + [len(seq)]
    segs = [(bounds[k], bounds[k + 1]) for k in range(len(bounds) - 1)
           if bounds[k + 1] - bounds[k] >= 2]

    total_span = total_recv = total_lost = 0
    time_gaps = []   # (start_time, duration_ms)
    for a, b in segs:
        ts, sq = t[a:b], seq[a:b]
        span = int(sq[-1] - sq[0]) + 1
        total_span += span
        total_recv += len(sq)
        total_lost += span - len(sq)
        dt_ms = np.diff(ts) * 1000.0
        big = dt_ms > time_gap_ms
        for gt, dur in zip(ts[1:][big], dt_ms[big]):
            time_gaps.append((gt, dur))

    return {
        "n_resets": len(reset_idx),
        "received": len(seq),
        "span": total_span,
        "lost": total_lost,
        "loss_pct": 100.0 * total_lost / max(total_span, 1),
        "time_gaps": time_gaps,
        "rate_hz": len(seq) / max(t[-1] - t[0], 1e-9) if len(t) > 1 else 0.0,
    }


def find_correlated_gaps(per_id_gaps, overlap_tol_ms=1000.0):
    """
    Cross-reference every device's time-gaps against every other device's.
    Two gaps are CORRELATED if their time windows overlap within tolerance --
    this is the check that distinguishes a per-board problem from a shared
    upstream cause.
    """
    ids = list(per_id_gaps)
    correlated = []
    for i, id_a in enumerate(ids):
        for gt_a, dur_a in per_id_gaps[id_a]["time_gaps"]:
            a_start, a_end = gt_a - dur_a / 1000.0, gt_a
            hits = [(id_a, gt_a, dur_a)]
            for id_b in ids:
                if id_b == id_a:
                    continue
                for gt_b, dur_b in per_id_gaps[id_b]["time_gaps"]:
                    b_start, b_end = gt_b - dur_b / 1000.0, gt_b
                    if (a_start - overlap_tol_ms / 1000.0 <= b_end and
                            b_start - overlap_tol_ms / 1000.0 <= a_end):
                        hits.append((id_b, gt_b, dur_b))
            if len(hits) > 1:
                key = tuple(sorted(h[0] for h in hits))
                if not any(c[0] == key and abs(c[1] - gt_a) < 0.5 for c in correlated):
                    correlated.append((key, gt_a, hits))
    return correlated


def main():
    ap = argparse.ArgumentParser(
        description="Record multiple IMU_IDs at once; check for cross-device "
                    "gap correlation (a shared-cause signature no single-"
                    "sensor test can see).")
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--ids", type=int, nargs="+", default=None,
                    help="IMU_IDs to track (e.g. --ids 5 6). Omit to "
                         "auto-discover whatever IDs are seen.")
    ap.add_argument("--gap-threshold-ms", type=float, default=500.0)
    ap.add_argument("--overlap-tol-ms", type=float, default=1000.0,
                    help="how close in time two devices' gaps must be to "
                         "count as correlated")
    a = ap.parse_args()

    rx = MultiReceiver(port=a.port, ids=a.ids)
    rx.start()
    time.sleep(0.2)
    print(f"Listening on :{a.port} for {a.seconds:.0f} s"
          + (f"  (tracking IDs: {sorted(a.ids)})" if a.ids
             else "  (auto-discovering IDs)"))
    print(f"  SO_RCVBUF granted: {rx.actual_rcvbuf} bytes")
    print(f"  ONE socket, ONE receive thread, shared by every device in "
          f"this run\n")

    t0 = time.perf_counter()
    last = 0
    while time.perf_counter() - t0 < a.seconds:
        time.sleep(0.05)
        el = time.perf_counter() - t0
        if int(el) // 5 > last:
            last = int(el) // 5
            counts = "  ".join(f"id{i}={len(rx.events.get(i, []))}"
                               for i in sorted(rx.events))
            print(f"  {el:5.1f} s   {counts if counts else '(no packets yet)'}")

    rx.stop()

    if a.ids is None and rx.seen_ids:
        print(f"\n  Auto-discovered IDs: {sorted(rx.seen_ids)}")

    if not rx.events:
        print("\n  No packets received from any tracked ID. Is the "
              "firmware streaming yet? (waits ~3.4 min after power-up)")
        return

    print(f"\n{'=' * 68}")
    print("PER-DEVICE SUMMARY")
    print(f"{'=' * 68}")
    per_id = {}
    for imu_id in sorted(rx.events):
        t = [e[0] for e in rx.events[imu_id]]
        seq = [e[1] for e in rx.events[imu_id]]
        if len(t) < 10:
            print(f"  id {imu_id}: only {len(t)} packets, too few to analyse")
            continue
        g = per_device_gaps(t, seq, a.gap_threshold_ms)
        per_id[imu_id] = g
        print(f"\n  IMU_ID {imu_id}:")
        print(f"    received       {g['received']}")
        if g["n_resets"]:
            print(f"    *** {g['n_resets']} sequence reset(s) -- device "
                  f"rebooted mid-run")
        print(f"    loss (by seq)  {g['lost']} / {g['span']} "
              f"({g['loss_pct']:.2f}%)")
        print(f"    rate           {g['rate_hz']:.1f} Hz")
        if g["time_gaps"]:
            print(f"    time-domain gaps > {a.gap_threshold_ms:.0f} ms: "
                  f"{len(g['time_gaps'])}")
            for gt, dur in sorted(g["time_gaps"], key=lambda x: -x[1])[:5]:
                print(f"      {dur:8.0f} ms, ending at t={gt:.1f}s")
        else:
            print(f"    time-domain gaps: none")

    if len(per_id) < 2:
        print(f"\n  Only one device had enough data -- cross-device "
              f"correlation needs at least two.")
        return

    print(f"\n{'=' * 68}")
    print("CROSS-DEVICE CORRELATION CHECK")
    print(f"{'=' * 68}")
    print(f"  This is the check no single-sensor run can do. If a gap on")
    print(f"  one device lines up in time with a gap on another, no")
    print(f"  per-board explanation (that board's WiFi, antenna, MPU) can")
    print(f"  produce that -- the cause has to be shared: this PC, this")
    print(f"  socket, or the hotspot.\n")

    correlated = find_correlated_gaps(per_id, a.overlap_tol_ms)
    if correlated:
        print(f"  *** {len(correlated)} CORRELATED GAP EVENT(S) FOUND:")
        for ids_involved, gt, hits in correlated:
            print(f"\n    at t~{gt:.1f}s, devices {ids_involved}:")
            for imu_id, ht, dur in hits:
                print(f"      id {imu_id}: {dur:.0f} ms gap ending t={ht:.1f}s")
        print(f"\n  -> Cross-reference the exact millisecond against each")
        print(f"     board's own serial log (*** WIFI DOWN / *** SENDER GAP).")
        print(f"     If NEITHER board logged a problem at this instant, the")
        print(f"     cause is confirmed to be on the PC side of the socket,")
        print(f"     not on any sensor -- narrow it there, not in firmware.")
    else:
        any_gaps = any(per_id[i]["time_gaps"] for i in per_id)
        if any_gaps:
            print(f"  Gaps exist on at least one device, but none overlap in")
            print(f"  time with another device's gap (within "
                  f"{a.overlap_tol_ms:.0f} ms). That points at a cause")
            print(f"  specific to the affected device, not a shared one --")
            print(f"  check that board's own serial log for the same window.")
        else:
            print(f"  No time-domain gaps (silence stalls) on any device.")

    # SEPARATE from the time-domain/stall check above: total sequence loss.
    # A device can have zero large silence gaps -- the check above -- while
    # still losing a meaningful fraction of packets as many small, scattered
    # drops. Those two things are different findings and were previously
    # both folded into one "Clean multi-sensor run" verdict, which called a
    # run with a device at several percent sequence loss "clean" just
    # because none of that loss happened to cluster into a >500ms stall.
    lossy = {i: per_id[i]["loss_pct"] for i in per_id if per_id[i]["loss_pct"] > 0.5}
    if lossy:
        print(f"\n  Sequence loss by device (scattered, not stalls):")
        for imu_id, pct in sorted(lossy.items(), key=lambda x: -x[1]):
            flag = " ***" if pct > 2.0 else ""
            print(f"    id {imu_id}: {pct:.2f}%{flag}")
        print(f"  Not correlated with the stall check above -- this is many")
        print(f"  small drops (weak signal, marginal antenna connection, or")
        print(f"  distance), not the buffer-stall pattern chased earlier in")
        print(f"  this project. Re-run solo (link_diag.py --id N) on the")
        print(f"  worst device if this persists across sessions -- a single")
        print(f"  session's number alone doesn't distinguish a one-off blip")
        print(f"  from a real, reproducible characteristic of that link.")
    else:
        print(f"\n  Sequence loss: negligible on every device (<0.5%).")

    if rx.bad_size:
        print(f"\n  NOTE: {rx.bad_size} packets had an unexpected size "
              f"(expected {_EXPECTED_PACKET} bytes) and were discarded.")


if __name__ == "__main__":
    main()