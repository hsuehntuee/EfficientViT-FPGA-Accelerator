"""Qualitative figure: a few images as input | ground truth | FP32 mask | FPGA-path mask | where they differ.

The FPGA path is the hybrid model (Stage 4 by the backend, rest FP32). With --backend model the Stage 4
blocks run in the integer model, which is what the FPGA produces bit for bit, so the figure can be made
on the software workstation before the board run; --backend file replays block_N_y_hw.bin recorded on
the board for one image (run_image.py --save / run_batch.py) and gives the same picture.

    # software workstation, pick images from an eval_dataset.py run (best / median / worst mIoU of the FPGA path)
    python host/make_mask_examples.py --per-image results/kitti512/per_image.json --cityscapes --out figs
    # or explicit images
    python host/make_mask_examples.py --images "data/kitti/leftImg8bit/val/*/*_leftImg8bit.png" --cityscapes --n 4 --out figs
    # no labels (e.g. the test image): the GT column is dropped
    python host/make_mask_examples.py --images model/evit_hw_golden/assets/test.png --out figs --backend model

Writes <out>/fig_mask_examples.png (and fig_mask_examples_<stem>.png per image with --per-image-files).
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hybrid import (CITYSCAPES_CLASSES, DATASETS, FileBackend, HybridModel, ModelBackend, confusion,  # noqa: E402
                    default_ckpt, iou_from_confusion, label_for, load_golden_module, load_label, palette)


def pick_images(rows, n, how):
    """rows: eval_dataset per_image.json. Returns rows sorted by the FPGA-path mIoU, spread / best / worst."""
    scored = [r for r in rows if "miou_hw" in r]
    if not scored:
        return rows[:n]
    scored.sort(key=lambda r: r["miou_hw"])
    if how == "worst":
        return scored[:n]
    if how == "best":
        return scored[::-1][:n]
    idx = np.linspace(len(scored) - 1, 0, n).round().astype(int)   # best -> worst, evenly spread
    return [scored[i] for i in idx]


def colorize(mask, n_cls):
    pal = palette(n_cls)
    m = np.asarray(mask)
    out = pal[np.clip(m, 0, n_cls - 1)]
    out[m == 255] = 0                                              # ignore -> black
    return out.astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default=None, help="glob of images")
    ap.add_argument("--per-image", default=None, help="per_image.json from eval_dataset.py: pick images by FPGA-path mIoU")
    ap.add_argument("--pick", choices=["spread", "best", "worst"], default="spread", help="with --per-image")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--out", default="figs")
    ap.add_argument("--cityscapes", action="store_true", help="label from the Cityscapes layout (KITTI export uses it too)")
    ap.add_argument("--label-dir", default=None)
    ap.add_argument("--label-suffix", default=".png")
    ap.add_argument("--dataset", choices=list(DATASETS), default="cityscapes")
    ap.add_argument("--backend", choices=["model", "file"], default="model")
    ap.add_argument("--y-dir", default=None, help="backend file: directory with block_N_y_hw.bin (single image)")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--golden-repo", default=None)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--no-diff", action="store_true", help="drop the FP32-vs-FPGA difference column")
    ap.add_argument("--per-image-files", action="store_true", help="also one png per image")
    ap.add_argument("--dpi", type=int, default=150)
    args = ap.parse_args()

    if args.per_image:
        rows = json.load(open(args.per_image))
        chosen = pick_images(rows, args.n, args.pick)
        files = [r["image"] for r in chosen]
    elif args.images:
        files = sorted(glob.glob(args.images))
        if args.n and len(files) > args.n:
            files = [files[i] for i in np.linspace(0, len(files) - 1, args.n).round().astype(int)]
    else:
        sys.exit("give --images or --per-image")
    if not files:
        sys.exit("no images")
    if args.backend == "file" and len(files) != 1:
        sys.exit("--backend file replays one image's block_N_y_hw.bin: use exactly one image")

    g, gdir = load_golden_module(args.golden_repo)
    import torch
    from PIL import Image
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    n_cls = DATASETS[args.dataset]["n_classes"]
    backend = FileBackend(g, args.y_dir) if args.backend == "file" else ModelBackend(g)
    hm = HybridModel(g, args.ckpt or default_ckpt(gdir, args.dataset), backend, args.size, args.dataset)

    def label_path(f):
        if args.cityscapes:
            return label_for(f, args.dataset)
        if args.label_dir:
            return os.path.join(args.label_dir, os.path.splitext(os.path.basename(f))[0] + args.label_suffix)
        return None

    panels = []                                                    # per image: dict of arrays + numbers
    for f in files:
        img = Image.open(f)
        x = hm.preprocess(img)
        lp = label_path(f)
        lab = load_label(lp, args.dataset) if lp and os.path.exists(lp) else None
        out_size = lab.shape[:2] if lab is not None else (img.height, img.width)
        m_fp32 = torch.argmax(hm.logits(x, "fp32", out_size), 1).squeeze().numpy()
        m_hw = torch.argmax(hm.logits(x, "hw", out_size), 1).squeeze().numpy()
        p = {"file": f, "img": np.array(img.convert("RGB").resize(out_size[::-1])), "lab": lab, "fp32": m_fp32, "hw": m_hw,
             "agreement": float((m_fp32 == m_hw).mean()), "bit_exact": all(hm.per_block[i]["bit_exact"] for i in range(4))}
        if lab is not None:
            p["miou_fp32"] = iou_from_confusion(confusion(m_fp32, lab, n_cls))[1]
            p["miou_hw"] = iou_from_confusion(confusion(m_hw, lab, n_cls))[1]
        panels.append(p)
        msg = f"{os.path.basename(f)}: agreement {100 * p['agreement']:.2f}%  bit-exact {p['bit_exact']}"
        if lab is not None:
            msg += f"  mIoU fp32 {100 * p['miou_fp32']:.2f}%  fpga-path {100 * p['miou_hw']:.2f}%"
        print(msg, flush=True)
    backend.close()

    have_gt = any(p["lab"] is not None for p in panels)
    cols = ["input"] + (["ground truth"] if have_gt else []) + ["FP32 (CPU)", f"Stage 4 on FPGA ({backend.name})"]
    if not args.no_diff:
        cols.append("FP32 vs FPGA differ")
    hw_ratio = np.mean([p["img"].shape[0] / p["img"].shape[1] for p in panels])
    w = 3.2 if hw_ratio > 0.6 else 4.0                              # wide (KITTI) images get wider panels
    fig, axes = plt.subplots(len(panels), len(cols), figsize=(w * len(cols), w * hw_ratio * len(panels) + 0.9), squeeze=False)
    for r, p in enumerate(panels):
        ims = [p["img"]]
        if have_gt:
            ims.append(colorize(p["lab"], n_cls) if p["lab"] is not None else np.zeros_like(p["img"]))
        ims += [colorize(p["fp32"], n_cls), colorize(p["hw"], n_cls)]
        if not args.no_diff:
            d = p["img"].copy() // 3
            d[p["fp32"] != p["hw"]] = (255, 0, 0)
            ims.append(d)
        for c, (a, im) in enumerate(zip(axes[r], ims)):
            a.imshow(im)
            a.set_xticks([]); a.set_yticks([])
            if r == 0:
                a.set_title(cols[c], fontsize=10)
        stats = f"agreement {100 * p['agreement']:.1f}%"
        if "miou_hw" in p:
            stats = f"mIoU FP32 {100 * p['miou_fp32']:.1f}%  FPGA {100 * p['miou_hw']:.1f}%  |  " + stats
        axes[r][0].set_ylabel(os.path.basename(p["file"]).replace("_leftImg8bit.png", ""), fontsize=8)
        axes[r][-1].text(1.0, -0.03, stats, transform=axes[r][-1].transAxes, ha="right", va="top", fontsize=8)
    if n_cls == 19:
        pal = palette(n_cls)
        handles = [Patch(color=pal[i] / 255.0, label=CITYSCAPES_CLASSES[i]) for i in range(n_cls)]
        fig.legend(handles=handles, loc="lower center", ncol=10, fontsize=7, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.subplots_adjust(left=0.03, right=0.995, top=0.93, bottom=0.14 if n_cls == 19 else 0.05, wspace=0.04, hspace=0.3)
    os.makedirs(args.out, exist_ok=True)
    out = os.path.join(args.out, "fig_mask_examples.png")
    fig.savefig(out, dpi=args.dpi)
    plt.close(fig)
    print(f"written {out}")

    if args.per_image_files:
        for p in panels:
            stem = os.path.splitext(os.path.basename(p["file"]))[0]
            ims = [("input", p["img"])] + ([("ground truth", colorize(p["lab"], n_cls))] if p["lab"] is not None else [])
            ims += [("FP32", colorize(p["fp32"], n_cls)), ("FPGA path", colorize(p["hw"], n_cls))]
            fig, ax = plt.subplots(1, len(ims), figsize=(w * len(ims), w * p["img"].shape[0] / p["img"].shape[1] + 0.4), squeeze=False)
            for a, (t, im) in zip(ax[0], ims):
                a.imshow(im); a.set_title(t, fontsize=10); a.axis("off")
            fig.tight_layout()
            fig.savefig(os.path.join(args.out, f"fig_mask_examples_{stem}.png"), dpi=args.dpi)
            plt.close(fig)

    with open(os.path.join(args.out, "fig_mask_examples.json"), "w") as fh:
        json.dump([{k: v for k, v in p.items() if k not in ("img", "lab", "fp32", "hw")} for p in panels], fh, indent=2)


if __name__ == "__main__":
    main()
