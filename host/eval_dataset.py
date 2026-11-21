"""Accuracy of the hybrid model (Stage 4 on FPGA / integer model, rest FP32) over a labelled image set.

The FPGA is bit-exact with the integer model, so --backend model gives the hardware's accuracy
without a board; --backend fpga runs the same thing through the board (slow, use --limit).

    # Cityscapes val layout: <root>/leftImg8bit/val/<city>/*_leftImg8bit.png,
    #                        <root>/gtFine/val/<city>/*_gtFine_labelTrainIds.png
    python host/eval_dataset.py --images "cityscapes/leftImg8bit/val/*/*_leftImg8bit.png" --cityscapes --out results/val
    # any folder: label file = --label-dir/<image stem><label-suffix>
    python host/eval_dataset.py --images "imgs/*.png" --label-dir labels --label-suffix _trainId.png --out results/x
    # no labels: only FP32-vs-hybrid agreement
    python host/eval_dataset.py --images "imgs/*.png" --out results/x

Labels are Cityscapes trainIds (0..18, 255 = ignore). Predictions are upsampled to the label size
before scoring, as in the efficientvit eval; the network input is --size (default 512: Stage 4 = 16x16
= 256 tokens, the RTL's N_TOKENS).
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hybrid import (CITYSCAPES_CLASSES, DATASETS, FpgaBackend, HybridModel, ModelBackend, confusion,  # noqa: E402
                    default_ckpt, iou_from_confusion, label_for, load_golden_module, load_label)


def label_path(img_path, args):
    if args.cityscapes or args.ade20k:
        return label_for(img_path, args.dataset)
    if args.label_dir:
        stem = os.path.splitext(os.path.basename(img_path))[0]
        return os.path.join(args.label_dir, stem + args.label_suffix)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="glob")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cityscapes", action="store_true", help="derive label paths from the Cityscapes layout")
    ap.add_argument("--ade20k", action="store_true", help="ADE20K layout (images/validation/*.jpg, annotations/validation/*.png); implies --dataset ade20k")
    ap.add_argument("--dataset", choices=list(DATASETS), default=None, help="model head / label space (default: cityscapes, or ade20k with --ade20k)")
    ap.add_argument("--label-dir", default=None)
    ap.add_argument("--label-suffix", default=".png")
    ap.add_argument("--backend", choices=["fpga", "model"], default="model")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--golden-repo", default=None)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--timeout", type=float, default=5.0)
    ap.add_argument("--fp32-only", action="store_true",
                    help="only the FP32 model (any --size, e.g. 256/512/1024): shows what resolution the task needs")
    args = ap.parse_args()
    args.dataset = args.dataset or ("ade20k" if args.ade20k else "cityscapes")
    n_cls = DATASETS[args.dataset]["n_classes"]

    g, gdir = load_golden_module(args.golden_repo)
    import torch
    from PIL import Image

    files = sorted(glob.glob(args.images))
    if args.limit:
        files = files[:args.limit]
    if not files:
        sys.exit(f"no images match {args.images}")
    os.makedirs(args.out, exist_ok=True)
    backend = FpgaBackend(timeout=args.timeout) if args.backend == "fpga" else ModelBackend(g)
    hm = HybridModel(g, args.ckpt or default_ckpt(gdir, args.dataset), backend, args.size, args.dataset)

    conf = {"fp32": np.zeros((n_cls, n_cls), np.int64), "hw": np.zeros((n_cls, n_cls), np.int64)}
    rows, n_lab, bit_exact_all, cos_sum, cyc = [], 0, True, np.zeros(4), []
    t_start = time.perf_counter()
    for k, f in enumerate(files):
        img = Image.open(f)
        x = hm.preprocess(img)
        lp = label_path(f, args)
        lab = load_label(lp, args.dataset) if lp and os.path.exists(lp) else None
        out_size = lab.shape[:2] if lab is not None else None
        m = {"fp32": torch.argmax(hm.logits(x, "fp32", out_size), 1).squeeze().numpy()}
        if args.fp32_only:
            m["hw"] = m["fp32"]
            row = {"image": f, "agreement": 1.0, "bit_exact": True, "cosine": [1.0] * 4, "cycles": [None] * 4}
        else:
            m["hw"] = torch.argmax(hm.logits(x, "hw", out_size), 1).squeeze().numpy()
            sim = hm.block_similarity()
            row = {"image": f, "agreement": float((m["fp32"] == m["hw"]).mean()),
                   "bit_exact": all(hm.per_block[i]["bit_exact"] for i in range(4)),
                   "cosine": [sim[i]["cosine_vs_fp32"] for i in range(4)],
                   "cycles": [hm.per_block[i]["cycles"] for i in range(4)]}
        bit_exact_all &= row["bit_exact"]
        cos_sum += np.array(row["cosine"])
        if row["cycles"][0] is not None:
            cyc += row["cycles"]
        if lab is not None:
            n_lab += 1
            for key in conf:
                conf[key] += confusion(m[key], lab, n_cls)
            row["miou_fp32"] = iou_from_confusion(confusion(m["fp32"], lab, n_cls))[1]
            row["miou_hw"] = iou_from_confusion(confusion(m["hw"], lab, n_cls))[1]
        rows.append(row)
        if (k + 1) % 10 == 0 or k + 1 == len(files):
            el = time.perf_counter() - t_start
            print(f"[{k + 1}/{len(files)}] agreement so far {100 * np.mean([r['agreement'] for r in rows]):.2f}%  "
                  f"bit-exact {'all' if bit_exact_all else 'NOT all'}  {el / (k + 1):.2f} s/img", flush=True)
    backend.close()

    n = len(rows)
    summary = {"backend": backend.name, "dataset": args.dataset, "size": args.size, "n_images": n, "n_labelled": n_lab, "all_bit_exact": bit_exact_all,
               "mask_agreement_mean": float(np.mean([r["agreement"] for r in rows])),
               "cosine_mean_per_block": (cos_sum / n).tolist(),
               "cycles_mean": float(np.mean(cyc)) if cyc else None}
    print(f"\n{n} images, backend {'fp32 only' if args.fp32_only else backend.name}, input {args.size}x{args.size}")
    print(f"all blocks bit-exact vs integer model: {bit_exact_all}")
    print(f"mask agreement hybrid vs FP32: {100 * summary['mask_agreement_mean']:.2f}% (mean over images)")
    print("cosine per block (mean): " + "  ".join(f"{c:.4f}" for c in summary["cosine_mean_per_block"]))
    if n_lab:
        print(f"\n{'class':14s} {'IoU fp32':>9s} {'IoU hw':>9s} {'diff':>7s}")
        pc = {}
        for key in conf:
            pc[key], summary[f"miou_{key}"], summary[f"pixel_acc_{key}"] = iou_from_confusion(conf[key])
        names = CITYSCAPES_CLASSES if n_cls == 19 else [f"class {i}" for i in range(n_cls)]
        shown = range(n_cls) if n_cls == 19 else np.argsort(-np.nan_to_num(pc["fp32"]))[:15]   # ADE: 15 most present classes
        for i in shown:
            a, b = pc["fp32"][i], pc["hw"][i]
            print(f"{names[i]:14s} {100 * a:8.2f}% {100 * b:8.2f}% {100 * (b - a):+6.2f}")
        print(f"{'mIoU':14s} {100 * summary['miou_fp32']:8.2f}% {100 * summary['miou_hw']:8.2f}% "
              f"{100 * (summary['miou_hw'] - summary['miou_fp32']):+6.2f}   ({n_lab} labelled images, dataset-level)")
        print(f"{'pixel acc':14s} {100 * summary['pixel_acc_fp32']:8.2f}% {100 * summary['pixel_acc_hw']:8.2f}%")
        summary["per_class_iou"] = {k: {c: (None if np.isnan(v) else float(v)) for c, v in zip(names, pc[k])} for k in pc}
    with open(os.path.join(args.out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    with open(os.path.join(args.out, "per_image.json"), "w") as fh:
        json.dump(rows, fh, indent=1)
    print(f"\nwritten: {args.out}/summary.json, per_image.json")


if __name__ == "__main__":
    main()
