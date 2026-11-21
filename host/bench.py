"""Latency / throughput of the accelerator on the board (FPGA only).

    python host/bench.py --dir hw_export_512 --iters 20

Measures, per block, with the fixed vectors (output checked bit-exact every iteration):
  weights upload     2.33 MB over XDMA              -> MB/s
  x upload           8 KB
  compute            start -> done, from the host    -> ms; with the cycle counter gives the clock frequency
  readback           8 KB
and derives two per-image figures for the 4-block stage:
  "reload"   weights re-sent for every block (what run_image.py does now)
  "resident" weights already in BRAM, x in / y out only (upper bound without changing the hardware)
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hybrid import Xdma  # noqa: E402
from xdma_run import WEIGHT_FILES  # noqa: E402

MACS_PER_TOKEN = 14.82e6 / 16   # QKV, MS, attention, proj, expand, DW, PW per token (see README)


def med(v):
    return float(np.median(v))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="hw_export_512")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--blocks", default="0,1,2,3")
    ap.add_argument("--out", default=None, help="json file")
    args = ap.parse_args()

    x = Xdma()
    res = {}
    for blk in [int(b) for b in args.blocks.split(",")]:
        ld = lambda n: np.fromfile(os.path.join(args.dir, f"block_{blk}_{n}.bin"), dtype="<i2")
        w = {n: ld(n) for n, _ in WEIGHT_FILES if n != "x"}
        xin, gold = ld("x"), ld("y_golden")
        wbytes = sum(a.nbytes for a in w.values())
        t_w, t_x, t_c, t_r, cyc = [], [], [], [], []
        ok = True
        for it in range(args.iters):
            t0 = time.perf_counter(); x.write_weights(w); t1 = time.perf_counter()
            x.write_bram("BRAM_X", xin.tobytes()); t2 = time.perf_counter()
            c, _ = x.run(); t3 = time.perf_counter()
            y = np.frombuffer(x.read_bram("BRAM_Y", gold.nbytes), dtype="<i2"); t4 = time.perf_counter()
            ok &= bool(np.array_equal(y, gold))
            t_w.append(t1 - t0); t_x.append(t2 - t1); t_c.append(t3 - t2); t_r.append(t4 - t3); cyc.append(c)
        r = {"bit_exact_every_iteration": ok, "cycles": int(np.median(cyc)),
             "weights_MB": wbytes / 1e6, "t_weights_ms": 1e3 * med(t_w), "weights_MBps": wbytes / med(t_w) / 1e6,
             "t_x_ms": 1e3 * med(t_x), "t_compute_ms": 1e3 * med(t_c), "t_readback_ms": 1e3 * med(t_r)}
        r["clock_MHz_est"] = r["cycles"] / med(t_c) / 1e6          # upper bound: poll latency is inside t_compute
        tokens = xin.size // 256
        MACS_PER_BLOCK = MACS_PER_TOKEN * tokens
        r["tokens"] = tokens
        r["GMAC_per_s_compute_only"] = MACS_PER_BLOCK / med(t_c) / 1e9
        res[f"block_{blk}"] = r
        print(f"block {blk}: bit-exact {ok}, cycles {r['cycles']}, weights {r['t_weights_ms']:.1f} ms ({r['weights_MBps']:.0f} MB/s), "
              f"x {r['t_x_ms']:.2f} ms, compute {r['t_compute_ms']:.2f} ms (~{r['clock_MHz_est']:.0f} MHz), readback {r['t_readback_ms']:.2f} ms")

    b = list(res.values())
    per_img_reload = sum(v["t_weights_ms"] + v["t_x_ms"] + v["t_compute_ms"] + v["t_readback_ms"] for v in b)
    per_img_resident = sum(v["t_x_ms"] + v["t_compute_ms"] + v["t_readback_ms"] for v in b)
    compute_only = sum(v["t_compute_ms"] for v in b)
    res["per_image"] = {"blocks": len(b), "reload_ms": per_img_reload, "reload_fps": 1e3 / per_img_reload,
                        "resident_ms": per_img_resident, "resident_fps": 1e3 / per_img_resident,
                        "compute_only_ms": compute_only, "compute_only_fps": 1e3 / compute_only,
                        "GMAC_per_s_compute_only": sum(v["tokens"] for v in b) * MACS_PER_TOKEN / compute_only / 1e6}
    p = res["per_image"]
    print(f"\nper image ({len(b)} blocks):")
    print(f"  weights reloaded every block : {p['reload_ms']:.1f} ms  -> {p['reload_fps']:.1f} img/s   (current host flow)")
    print(f"  weights resident, x/y only   : {p['resident_ms']:.1f} ms  -> {p['resident_fps']:.1f} img/s   (needs 4 blocks of weights on chip)")
    print(f"  compute only                 : {p['compute_only_ms']:.1f} ms  -> {p['compute_only_fps']:.1f} img/s, {p['GMAC_per_s_compute_only']:.2f} GMAC/s")
    x.close()
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
