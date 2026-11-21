"""Figures for the report (matplotlib only). Each figure is produced when its input exists.

    python host/make_figures.py --out figs \
        [--bench bench.json] [--batch-manifest batch/manifest.json] [--batch-results batch_results/results.json] \
        [--eval summary.json] [--power "total=6.1,dynamic=4.9,static=1.2"] [--clock-mhz 62.5]

Always produced (constants from the project):
    fig_error_budget.png     block-0 error vs. golden through the fixes
    fig_mac_breakdown.png    MACs per layer and the 1.4 % DSP utilisation
    fig_resources.png        May vs. October LUT/FF/BRAM/DSP
From inputs:
    fig_latency.png          upload / compute / readback per block, reload vs. resident per image   (bench.json)
    fig_cosine.png           cosine per block over the batch, mask agreement histogram               (manifest.json)
    fig_per_class_iou.png    per-class IoU FP32 vs. hybrid                                           (summary.json / manifest.json)
    fig_board_cycles.png     cycles per (image, block) on the board, bit-exact count                  (results.json)
"""
import argparse
import json
import os

import numpy as np

CLASSES = ["road", "sidewalk", "building", "wall", "fence", "pole", "tr. light", "tr. sign", "vegetation", "terrain",
           "sky", "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"]
GREEN, DARK, LIGHT, GREY = "#2E7D32", "#143C1E", "#A5D6A7", "#9E9E9E"


def setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})
    return plt


def fig_error_budget(plt, out):
    steps = ["May RTL\nMay export", "+ #1 address\ntranspose", "+ #2 #3 #8\nhandshake, MS", "+ #6a expand\nhardswish", "+ #6b DW hswish\ninteger golden"]
    lsb = [92.3, 28.3, 26.7, 1.52, 0.0]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    bars = ax.bar(steps, lsb, color=[GREY, GREY, GREY, LIGHT, GREEN])
    for b, v in zip(bars, lsb):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:g}", ha="center", fontsize=10)
    ax.set_ylabel("mean |HW − golden| (LSB, Q10.5)")
    ax.set_title("Block 0 error budget (golden mean magnitude ≈ 1018 LSB)")
    ax.set_ylim(0, 105)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_error_budget.png"))
    plt.close(fig)


def fig_mac_breakdown(plt, out):
    """Left: MACs per layer of one block at 512x512 (256 tokens). Right: cycles per block for every RTL version."""
    layers = ["QKV 1×1", "MS DW+PW", "attention", "proj 1×1", "expand 1×1", "DW 5×5", "point 1×1"]
    macs = [50.33, 8.06, 4.46, 33.55, 67.11, 6.55, 67.11]          # M MAC, 256 tokens
    total = sum(macs)
    vers = ["v1 (May→10-04)\n128², 16 tok", "v2 10-10\n512², 256 tok", "v3 cluster K=8", "v4 overlap"]
    cyc = [271019, 1055995, 170788, 126999]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={"width_ratios": [3, 2]})
    a1.barh(layers, macs, color=GREEN)
    for i, v in enumerate(macs):
        a1.text(v + 1, i, f"{v:.1f} M", va="center", fontsize=9)
    a1.set_xlabel("MACs per block (M)")
    a1.set_xlim(0, 80)
    a1.set_title(f"{total:.0f} M MAC per block (512×512)\nv4: {total * 1e6 / cyc[-1]:,.0f} MAC/cycle = {100 * total * 1e6 / cyc[-1] / 4755:.0f}% of 4,755 DSPs", fontsize=10)
    a1.invert_yaxis()
    bars = a2.bar(range(4), [c / 1e3 for c in cyc], color=[GREY, GREY, LIGHT, GREEN])
    for b, c in zip(bars, cyc):
        a2.text(b.get_x() + b.get_width() / 2, c / 1e3 + 15, f"{c:,}", ha="center", fontsize=8)
    a2.set_xticks(range(4), vers, fontsize=7)
    a2.set_ylabel("cycles per block (k)")
    a2.set_ylim(0, 1200)
    a2.set_title("cycles per block: same datapath, 8.3× from v2", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_mac_breakdown.png"))
    plt.close(fig)


def fig_resources(plt, out):
    names = ["LUT (k)", "FF (k)", "BRAM36", "DSP"]
    may, v1, v4 = [87, 125, 804, 2513], [99.7, 156, 1038, 3963], [264, 274, 1712, 4755]
    avail = [1304, 2607, 2016, 9024]
    fig, axs = plt.subplots(1, 4, figsize=(10, 3))
    for ax, n, m, o, q, a in zip(axs, names, may, v1, v4, avail):
        ax.bar(["May", "v1", "v4"], [m, o, q], color=[GREY, LIGHT, GREEN])
        ax.set_title(f"{n}\n(v4: {100 * q / a:.0f}% of device)")
        for i, v in enumerate([m, o, q]):
            ax.text(i, v, f"{v:g}", ha="center", va="bottom", fontsize=9)
    fig.suptitle("xcvu37p utilisation: May / Oct v1 (128²) / Oct v4 (512², K=8 cluster);  v4: WNS +0.43 ns @ 62.5 MHz, 11.1 W", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_resources.png"))
    plt.close(fig)


def fig_cycle_breakdown(plt, out):
    """v4 cycles per block by stage (from the schedule: 416 cluster steps x 256 tokens, 2 x 17 x 260 attention, 4096 in / out)
    and the measured per-image time on the board vs the CPU."""
    stages = ["x in", "QKV 1×1", "attention q", "attention ms", "proj 1×1", "expand 1×1", "point 1×1", "y out", "bias / preload / waits"]
    cyc = [4096, 24576, 4420, 4420, 16384, 32768, 32768, 4096, 126999 - (4096 + 24576 + 4420 * 2 + 16384 + 32768 * 2 + 4096)]
    cols = [LIGHT, GREEN, DARK, DARK, GREEN, GREEN, GREEN, LIGHT, GREY]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.4), gridspec_kw={"width_ratios": [3, 2]})
    left = 0
    for n, c, k in zip(stages, cyc, cols):
        a1.barh([0], [c], left=left, color=k, edgecolor="white")
        if c > 6000:
            a1.text(left + c / 2, 0, f"{n}\n{100 * c / 126999:.0f}%", ha="center", va="center", fontsize=8, color="white" if k != LIGHT else "black")
        left += c
    a1.set_xlim(0, 126999)
    a1.set_yticks([])
    a1.set_xlabel("cycles per block (126,999)")
    a1.set_title("Cycles per block; multi-scale and depthwise passes overlap QKV / expand", fontsize=10)
    names = ["CPU FP32\n(i9-13900H)", "FPGA now\n(weights re-sent)", "FPGA, weights\nresident", "FPGA\ncompute only"]
    ms = [120.5, 95.1, 16.4, 8.2]
    bars = a2.bar(names, ms, color=[GREY, LIGHT, GREEN, DARK])
    for b, v in zip(bars, ms):
        a2.text(b.get_x() + b.get_width() / 2, v + 2, f"{v:.1f} ms", ha="center", fontsize=9)
    a2.set_ylabel("Stage 4 per image (ms)")
    a2.set_ylim(0, 140)
    a2.set_title("Stage 4, 512×512: CPU vs. board", fontsize=10)
    a2.tick_params(axis="x", labelsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_cycle_breakdown.png"))
    plt.close(fig)


def fig_latency(plt, out, bench, clock_mhz):
    blocks = [k for k in bench if k.startswith("block_")]
    w = np.array([bench[b]["t_weights_ms"] for b in blocks])
    xx = np.array([bench[b]["t_x_ms"] for b in blocks])
    c = np.array([bench[b]["t_compute_ms"] for b in blocks])
    r = np.array([bench[b]["t_readback_ms"] for b in blocks])
    p = bench["per_image"]
    fig, a1 = plt.subplots(figsize=(6.2, 3.8))
    idx = np.arange(len(blocks))
    a1.bar(idx, w, color=GREY, label=f"weights upload ({bench[blocks[0]]['weights_MB']:.2f} MB, {bench[blocks[0]]['weights_MBps']:.0f} MB/s)")
    a1.bar(idx, xx, bottom=w, color=LIGHT, label="x upload (128 KB)")
    a1.bar(idx, c, bottom=w + xx, color=GREEN, label="compute (start→done)")
    a1.bar(idx, r, bottom=w + xx + c, color=DARK, label="readback (128 KB)")
    for i in idx:
        a1.text(i, w[i] + xx[i] + c[i] + r[i] + 0.4, f"{w[i] + xx[i] + c[i] + r[i]:.1f} ms", ha="center", fontsize=9)
        a1.text(i, w[i] / 2, f"{w[i]:.1f}", ha="center", va="center", fontsize=9, color="white")
        a1.text(i, w[i] + xx[i] + c[i] / 2, f"{c[i]:.2f}", ha="center", va="center", fontsize=8, color="white")
    a1.set_xticks(idx, [b.replace("_", " ") for b in blocks])
    a1.set_ylabel("ms per block (median of 20 runs)")
    a1.set_ylim(0, max(w + xx + c + r) * 1.18)
    clk = bench[blocks[0]].get("clock_MHz_est")
    a1.set_title(f"Per block on the board; cycle counter → {clk:.1f} MHz\nper image: {p['reload_ms']:.1f} ms with weights re-sent, {p['resident_ms']:.1f} ms resident, {p['compute_only_ms']:.1f} ms compute", fontsize=10)
    a1.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_latency.png"))
    plt.close(fig)


def fig_cosine(plt, out, manifest):
    rows = manifest["images"]
    cos = np.array([r["cosine"] for r in rows])
    agr = np.array([r["agreement"] for r in rows]) * 100
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.4))
    a1.boxplot([cos[:, i] for i in range(4)], tick_labels=[f"block {i}" for i in range(4)])
    a1.set_ylabel("cosine similarity vs FP32")
    a1.set_title(f"Per-block output similarity, {len(rows)} images")
    a2.hist(agr, bins=20, color=GREEN)
    a2.axvline(agr.mean(), color=DARK, ls="--")
    a2.set_xlabel("mask agreement hybrid vs FP32 (%)")
    a2.set_ylabel("images")
    a2.set_title(f"mean {agr.mean():.2f}%, min {agr.min():.2f}%")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_cosine.png"))
    plt.close(fig)


def fig_per_class(plt, out, fp32, hw, miou_fp32, miou_hw, n):
    fp32 = np.array([np.nan if v is None else v for v in fp32]) * 100
    hw = np.array([np.nan if v is None else v for v in hw]) * 100
    idx = np.arange(len(CLASSES))
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.bar(idx - 0.2, fp32, 0.4, color=GREY, label=f"FP32 (mIoU {100 * miou_fp32:.1f}%)")
    ax.bar(idx + 0.2, hw, 0.4, color=GREEN, label=f"FPGA stage 4 (mIoU {100 * miou_hw:.1f}%)")
    ax.set_xticks(idx, CLASSES, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("IoU (%)")
    ax.set_title(f"Per-class IoU, {n} labelled images, 512×512 input")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_per_class_iou.png"))
    plt.close(fig)


def fig_board(plt, out, results):
    imgs = results["images"]
    cyc = np.array([[imgs[s][f"block_{b}"]["cycles"] for b in range(4)] for s in imgs])
    ok = sum(imgs[s][f"block_{b}"]["bit_exact"] for s in imgs for b in range(4))
    tot = cyc.size
    t = np.array([[imgs[s][f"block_{b}"]["t_x_ms"] + imgs[s][f"block_{b}"]["t_compute_ms"] + imgs[s][f"block_{b}"]["t_readback_ms"]
                   for b in range(4)] for s in imgs])
    fig, a2 = plt.subplots(figsize=(6.2, 3.8))
    a2.hist(t.sum(1), bins=20, color=GREEN)
    a2.set_xlabel("ms per image, 4 blocks, weights resident")
    a2.set_ylabel("images")
    lo, hi = int(cyc.min()), int(cyc.max())
    cyc_txt = f"{lo:,} cycles every run" if lo == hi else f"cycles {lo:,}–{hi:,}"
    a2.set_title(f"{len(imgs)} KITTI images on the board: {ok}/{tot} block runs bit-exact, {cyc_txt}\nmedian {np.median(t.sum(1)):.1f} ms = {1e3 / np.median(t.sum(1)):.1f} img/s", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_board_cycles.png"))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figs")
    ap.add_argument("--bench", default=None)
    ap.add_argument("--batch-manifest", default=None)
    ap.add_argument("--batch-results", default=None)
    ap.add_argument("--eval", default=None, help="summary.json from eval_dataset.py")
    ap.add_argument("--clock-mhz", type=float, default=None)
    args = ap.parse_args()
    plt = setup()
    os.makedirs(args.out, exist_ok=True)
    fig_error_budget(plt, args.out)
    fig_mac_breakdown(plt, args.out)
    fig_resources(plt, args.out)
    fig_cycle_breakdown(plt, args.out)
    made = ["fig_error_budget", "fig_mac_breakdown", "fig_resources"]
    if args.bench:
        fig_latency(plt, args.out, json.load(open(args.bench)), args.clock_mhz)
        made.append("fig_latency")
    if args.batch_manifest:
        m = json.load(open(args.batch_manifest))
        fig_cosine(plt, args.out, m)
        made.append("fig_cosine")
        s = m.get("summary", {})
        if "per_class_iou_hw" in s and not args.eval:
            fig_per_class(plt, args.out, s["per_class_iou_fp32"], s["per_class_iou_hw"], s["miou_fp32"], s["miou_hw"], s["n_images"])
            made.append("fig_per_class_iou")
    if args.eval:
        s = json.load(open(args.eval))
        if "per_class_iou" in s:
            pc = s["per_class_iou"]
            fig_per_class(plt, args.out, [pc["fp32"][c] for c in pc["fp32"]], [pc["hw"][c] for c in pc["hw"]],
                          s["miou_fp32"], s["miou_hw"], s["n_labelled"])
            made.append("fig_per_class_iou")
    if args.batch_results:
        fig_board(plt, args.out, json.load(open(args.batch_results)))
        made.append("fig_board_cycles")
    print("written to", args.out + ":", ", ".join(f + ".png" for f in made))


if __name__ == "__main__":
    main()
