#!/usr/bin/env python3
"""
imu_receiver.py  -  UDP receiver for the ESP32/MPU9250 node.

Provides three things:

  parse_packet()     decode one 92-byte packet from ReBait_imu_firemware.ino
  UdpImuCallback     a Callback class matching ReBAIT's expected interface,
                     so it can be dropped into DataCollector unchanged
  record()           capture a trial to .npz with full packet-loss accounting

UNITS
-----
The firmware already emits SI (rad/s, m/s^2) when OUTPUT_SI is 1, so this
module does NO unit conversion.  If you ever set OUTPUT_SI to 0, set
ASSUME_SI = False below and the conversions happen here instead.  Do not
convert in both places.

ReBAIT packet convention (what getNextPacket returns as `data`):
  data[0] time, seconds (float)
  data[1] free acceleration, m/s^2   (gravity REMOVED)  -> ZUPT magnitude check
  data[2] angular velocity, rad/s
  data[3] orientation quaternion, (w, x, y, z)
  data[4] calibrated acceleration, m/s^2 (gravity INCLUDED) -> gravity average
  data[5] Euler angles, degrees
"""

import argparse
import socket
import struct
import threading
import time
from collections import deque

import numpy as np

PACKET_FMT = "<BBHIQ19f"
PACKET_SIZE = struct.calcsize(PACKET_FMT)  # 92

ASSUME_SI = True
G_TO_MS2 = 9.80665
DEG_TO_RAD = np.pi / 180.0


def parse_packet(raw):
    """Decode one datagram into a dict. Returns None on a size mismatch."""
    if len(raw) != PACKET_SIZE:
        return None
    f = struct.unpack(PACKET_FMT, raw)

    gyr = np.array(f[5:8], dtype=float)
    acc = np.array(f[8:11], dtype=float)     # includes gravity
    lin = np.array(f[11:14], dtype=float)    # gravity removed
    mag = np.array(f[14:17], dtype=float)
    qut = np.array(f[17:21], dtype=float)    # w, x, y, z
    eul = np.array(f[21:24], dtype=float)

    if not ASSUME_SI:
        gyr = gyr * DEG_TO_RAD
        acc = acc * G_TO_MS2
        lin = lin * G_TO_MS2

    return {
        "imu_id": f[0],
        "flags": f[1],
        "sequence": f[3],
        "t": f[4] / 1e6,          # device micros -> seconds, already 64-bit
        "gyro": gyr,
        "cal_acc": acc,
        "free_acc": lin,
        "mag": mag,
        "quat": qut,
        "euler": eul,
    }


class UdpImuCallback:
    """
    Callback conforming to the ReBAIT interface.

    Fields   : devIds, samp_freq
    Functions: enable, attach, packetAvailable, getNextPacket, detach, close

    Device IDs are strings of the firmware's IMU_ID, so 'imu05' for IMU_ID 5.
    Pass expected_ids so devIds has a deterministic ORDER -- ReBAIT indexes
    devIds positionally, and discovery order over UDP is not repeatable.

    RECEIVE ARCHITECTURE. A dedicated background thread does nothing but
    recvfrom() + parse + enqueue, exactly like link_diag.py's MinimalReceiver.
    It runs continuously, independent of how often or how slowly the consumer
    calls packetAvailable()/getNextRaw(). The old version only drained the
    socket from inside packetAvailable(), so a slow consumer (e.g. one doing
    real per-packet work, or blocked rendering a MuJoCo frame) stalled the
    drain and the kernel's 1 MB socket buffer overflowed -- confirmed via
    link_diag.py --load 10, which showed 65% loss with 0 kernel-level drops,
    i.e. the loss was happening entirely on the application side of this class.

    BUFFER EVICTION IS NOW COUNTED AS LOSS. If the consumer genuinely can't
    keep up on average, the internal deque (bounded, maxlen=max_buffer) will
    still fill and start evicting the oldest unread packet per new arrival --
    no buffer size fixes a consumer that is structurally too slow, it only
    buys slack for transient stalls. Previously that eviction was tracked
    only in the internal _overflow counter, invisible in loss_report()'s
    per-device numbers. It's now folded into `dropped` too, so loss_report()
    can't read as clean while data is quietly being thrown away.
    """

    def __init__(self, port=5000, samp_freq=100, expected_ids=(5,),
                 max_buffer=4096, bind_addr="0.0.0.0"):
        self.port = port
        self.bind_addr = bind_addr
        self.samp_freq = samp_freq
        self.devIds = [f"imu{i:02d}" for i in expected_ids]
        self._buf = deque(maxlen=max_buffer)
        self._buf_lock = threading.Lock()
        self._sock = None
        self._attached = False
        self._overflow = 0
        self.last_seq = {}
        self.dropped = {d: 0 for d in self.devIds}
        self.received = {d: 0 for d in self.devIds}
        self._stop = threading.Event()
        self._thread = None

    # -- lifecycle -------------------------------------------------------
    def enable(self):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # 4 MB, matching MinimalReceiver -- generous headroom on the kernel
        # side even though the dedicated thread should keep this buffer
        # nearly empty at all times now.
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 22)
        self._sock.bind((self.bind_addr, self.port))
        self._sock.settimeout(0.2)   # so the thread can notice _stop promptly

    def attach(self):
        self._attached = True
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def detach(self):
        self._attached = False
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def close(self):
        if self._sock:
            self._sock.close()
            self._sock = None

    # -- receive thread ----------------------------------------------------
    def _run(self):
        """Runs continuously on its own thread: recv, parse, enqueue. Never
        waits on the consumer for anything."""
        while not self._stop.is_set():
            try:
                raw, _ = self._sock.recvfrom(2048)
            except socket.timeout:
                continue
            except OSError:
                break
            pkt = parse_packet(raw)
            if pkt is None:
                continue
            did = f"imu{pkt['imu_id']:02d}"
            with self._buf_lock:
                if did not in self.dropped:
                    self.dropped[did] = 0
                    self.received[did] = 0
                self.received[did] += 1
                prev = self.last_seq.get(did)
                if prev is not None and pkt["sequence"] > prev + 1:
                    self.dropped[did] += pkt["sequence"] - prev - 1
                self.last_seq[did] = pkt["sequence"]
                if len(self._buf) == self._buf.maxlen:
                    self._overflow += 1
                    self.dropped[did] += 1   # buffer eviction IS loss; count it
                self._buf.append((did, pkt))

    # -- packet flow (consumer side) ---------------------------------------
    def packetAvailable(self):
        with self._buf_lock:
            return len(self._buf) > 0

    def getNextPacket(self):
        """Return (devId, data) where data is ReBAIT's 6-element list."""
        with self._buf_lock:
            did, p = self._buf.popleft()
        data = [p["t"], p["free_acc"], p["gyro"], p["quat"], p["cal_acc"], p["euler"]]
        return did, data

    def getNextRaw(self):
        """Like getNextPacket but keeps mag, sequence and flags."""
        with self._buf_lock:
            return self._buf.popleft()

    # -- diagnostics -----------------------------------------------------
    def loss_report(self):
        lines = []
        for d in sorted(self.received):
            r, l = self.received[d], self.dropped[d]
            pct = 100.0 * l / max(r + l, 1)
            lines.append(f"  {d}: {r} received, {l} dropped ({pct:.2f}%)")
        if self._overflow:
            lines.append(f"  buffer overflow events: {self._overflow} "
                          f"(included in dropped counts above)")
        return "\n".join(lines)


def record(seconds, out_path, port=5000, expected_ids=(5,), quiet=False):
    """Capture a trial and save every field to .npz, one array set per device."""
    cb = UdpImuCallback(port=port, expected_ids=expected_ids)
    cb.enable()
    cb.attach()

    store = {}
    t_start = time.time()
    last_msg = t_start

    if not quiet:
        print(f"Recording {seconds:.0f} s on UDP :{port} ...")

    while time.time() - t_start < seconds:
        if cb.packetAvailable():
            did, p = cb.getNextRaw()
            s = store.setdefault(did, {k: [] for k in
                                       ("t", "seq", "gyro", "cal_acc",
                                        "free_acc", "mag", "quat", "euler")})
            s["t"].append(p["t"])
            s["seq"].append(p["sequence"])
            s["gyro"].append(p["gyro"])
            s["cal_acc"].append(p["cal_acc"])
            s["free_acc"].append(p["free_acc"])
            s["mag"].append(p["mag"])
            s["quat"].append(p["quat"])
            s["euler"].append(p["euler"])
        else:
            time.sleep(0.0005)

        if not quiet and time.time() - last_msg >= 5.0:
            last_msg = time.time()
            el = time.time() - t_start
            n = sum(len(v["t"]) for v in store.values())
            print(f"  {el:5.1f} s   {n} packets")

    cb.detach()
    cb.close()

    flat = {}
    for did, s in store.items():
        for k, v in s.items():
            flat[f"{did}__{k}"] = np.asarray(v)
    flat["devices"] = np.array(sorted(store.keys()))
    np.savez_compressed(out_path, **flat)

    if not quiet:
        print(f"\nSaved {out_path}")
        print(cb.loss_report())
        if not store:
            print("  WARNING: no valid packets were received; "
                  "start recording after the IMU reports 'Streaming.'")
        for did, s in store.items():
            t = np.asarray(s["t"])
            if len(t) > 10:
                dt = np.diff(t)
                print(f"  {did}: {len(t)} samples, "
                      f"mean {1/np.mean(dt):.1f} Hz, "
                      f"jitter SD {np.std(dt)*1000:.2f} ms, "
                      f"max gap {np.max(dt)*1000:.1f} ms")
    return out_path


def load(path, device=None):
    """Load a recording. Returns (device_id, dict_of_arrays)."""
    z = np.load(path, allow_pickle=False)
    devices = [str(d) for d in z["devices"]]
    if not devices:
        raise ValueError(
            f"{path} contains no IMU packets. Re-record after the ESP32 "
            "reports 'Streaming.'"
        )
    did = device or devices[0]
    keys = ("t", "seq", "gyro", "cal_acc", "free_acc", "mag", "quat", "euler")
    return did, {k: z[f"{did}__{k}"] for k in keys}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Record a trial from the IMU node.")
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--out", default="trial.npz")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--ids", type=int, nargs="+", default=[5])
    a = ap.parse_args()
    record(a.seconds, a.out, port=a.port, expected_ids=tuple(a.ids))