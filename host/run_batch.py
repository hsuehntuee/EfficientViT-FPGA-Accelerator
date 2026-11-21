"""HARDWARE workstation (numpy only): run a batch of per-image vectors through the FPGA.

    python host/run_batch.py --batch batch_demo --weights hw_export_512 --out batch_demo_results

<batch>/<stem>/block_N_x.bin + block_N_y_golden.bin come from host/export_batch.py (software side).
Block-major order: the weights of block N are written once, then every image's x is streamed
through, so the per-image time is the "weights resident" figure. Output:
    <out>/results.json    per image x block: bit_exact, cycles, t_x_ms, t_compute_ms, t_readback_ms
    <out>/<stem>/block_N_y_hw.bin   hardware output (only needed when something is not bit-exact)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", required=True)
    ap.add_argument("--weights", default="hw_export_512")
    ap.add_argument("--out", required=True)
    ap.add_argument("--save-y", action="store_true", help="save every hardware output, not only mismatches")
    ap.add_argument("--timeout", type=float, default=5.0)
    args = ap.parse_args()

    stems = sorted(d for d in os.listdir(args.batch) if os.path.exists(os.path.join(args.batch, d, "block_0_x.bin")))
    if not stems:
        sys.exit(f"no <stem>/block_0_x.bin under {args.batch}")
    os.makedirs(args.out, exist_ok=True)
    x = Xdma()
    res = {s: {} for s in stems}
    t_batch = time.perf_counter()
    for blk in range(4):
        w = {n: np.fromfile(os.path.join(args.weights, f"block_{blk}_{n}.bin"), dtype="<i2") for n, _ in WEIGHT_FILES if n != "x"}
        t0 = time.perf_counter()
        x.write_weights(w)
        t_w = time.perf_counter() - t0
        n_bad = 0
        for s in stems:
            d = os.path.join(args.batch, s)
            xin = np.fromfile(os.path.join(d, f"block_{blk}_x.bin"), dtype="<i2")
            gold = np.fromfile(os.path.join(d, f"block_{blk}_y_golden.bin"), dtype="<i2")
            t0 = time.perf_counter()
            x.write_bram("BRAM_X", xin.tobytes())
            t1 = time.perf_counter()
            cycles, _ = x.run(args.timeout)
            t2 = time.perf_counter()
            y = np.frombuffer(x.read_bram("BRAM_Y", gold.nbytes), dtype="<i2")
            t3 = time.perf_counter()
            dmax = int(np.abs(y.astype(np.int64) - gold.astype(np.int64)).max())
            ok = dmax == 0
            n_bad += (not ok)
            res[s][f"block_{blk}"] = {"bit_exact": ok, "max_abs_diff": dmax, "cycles": cycles,
                                      "t_x_ms": 1e3 * (t1 - t0), "t_compute_ms": 1e3 * (t2 - t1), "t_readback_ms": 1e3 * (t3 - t2)}
            if args.save_y or not ok:
                os.makedirs(os.path.join(args.out, s), exist_ok=True)
                y.tofile(os.path.join(args.out, s, f"block_{blk}_y_hw.bin"))
        print(f"block {blk}: weights {1e3 * t_w:.1f} ms, {len(stems)} images, {n_bad} not bit-exact", flush=True)
    x.close()

    per_img = [sum(res[s][f"block_{b}"]["t_x_ms"] + res[s][f"block_{b}"]["t_compute_ms"] + res[s][f"block_{b}"]["t_readback_ms"]
                   for b in range(4)) for s in stems]
    summary = {"n_images": len(stems), "all_bit_exact": all(v["bit_exact"] for s in stems for v in res[s].values()),
               "cycles_mean": float(np.mean([v["cycles"] for s in stems for v in res[s].values()])),
               "per_image_resident_ms_median": float(np.median(per_img)),
               "resident_fps": 1e3 / float(np.median(per_img)),
               "batch_wall_s": time.perf_counter() - t_batch,
               "t_x_ms_median": float(np.median([v["t_x_ms"] for s in stems for v in res[s].values()])),
               "t_compute_ms_median": float(np.median([v["t_compute_ms"] for s in stems for v in res[s].values()])),
               "t_readback_ms_median": float(np.median([v["t_readback_ms"] for s in stems for v in res[s].values()]))}
    with open(os.path.join(args.out, "results.json"), "w") as fh:
        json.dump({"summary": summary, "images": res}, fh, indent=1)
    print(f"\n{summary['n_images']} images x 4 blocks: all bit-exact = {summary['all_bit_exact']}, "
          f"cycles {summary['cycles_mean']:.0f}, per image (weights resident) {summary['per_image_resident_ms_median']:.1f} ms "
          f"= {summary['resident_fps']:.1f} img/s")
    print(f"written: {args.out}/results.json  -> send back to the software workstation (host/make_figures.py)")


if __name__ == "__main__":
    main()
