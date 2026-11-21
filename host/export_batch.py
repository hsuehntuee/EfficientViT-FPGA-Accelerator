"""SOFTWARE workstation (needs torch): turn a set of images into per-image FPGA test vectors.

    python host/export_batch.py --images "imgs/*.png" --out batch_demo [--limit 50] [--cityscapes | --label-dir D --label-suffix S]

For every image the FP32 front-end runs on CPU and the 4 Stage 4 blocks run in the integer model
(bit-exact with the FPGA). Written per image to <out>/<stem>/:
    block_N_x.bin, block_N_y_golden.bin   (what the FPGA must read / produce)
    mask_fp32.npy, mask_hw.npy             (128x128 trainIds, FP32 model vs. hybrid)
plus <out>/manifest.json with per-image cosine / mask agreement (and mIoU when labels exist).
The weights are the same for every image and are NOT repeated here: the hardware side uses
hw_export_512/block_N_*.bin. Zip <out>/ and send it to the FPGA workstation
(host/run_batch.py); it returns results.json which host/make_figures.py turns into plots.
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hybrid import (DATASETS, HybridModel, ModelBackend, confusion, default_ckpt, iou_from_confusion,  # noqa: E402
                    label_for, load_golden_module, load_label)


def label_path(img_path, args):
    if args.cityscapes or args.ade20k:
        return label_for(img_path, args.dataset)
    if args.label_dir:
        return os.path.join(args.label_dir, os.path.splitext(os.path.basename(img_path))[0] + args.label_suffix)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="glob")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--cityscapes", action="store_true")
    ap.add_argument("--ade20k", action="store_true", help="ADE20K layout; implies --dataset ade20k")
    ap.add_argument("--dataset", choices=list(DATASETS), default=None)
    ap.add_argument("--label-dir", default=None)
    ap.add_argument("--label-suffix", default=".png")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--golden-repo", default=None)
    ap.add_argument("--size", type=int, default=512)
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
    hm = HybridModel(g, args.ckpt or default_ckpt(gdir, args.dataset), ModelBackend(g), args.size, args.dataset)
    hm.keep_io = True

    weights_dir = "hw_export_512" if args.dataset == "cityscapes" else "hw_export_ade512"
    manifest = {"size": args.size, "dataset": args.dataset, "weights": weights_dir, "images": []}
    conf = {"fp32": np.zeros((n_cls, n_cls), np.int64), "hw": np.zeros((n_cls, n_cls), np.int64)}
    t0 = time.perf_counter()
    for k, f in enumerate(files):
        stem = os.path.splitext(os.path.basename(f))[0]
        d = os.path.join(args.out, stem)
        os.makedirs(d, exist_ok=True)
        x = hm.preprocess(Image.open(f))
        m_fp32 = torch.argmax(hm.logits(x, "fp32"), 1).squeeze().numpy().astype(np.uint8)
        m_hw = torch.argmax(hm.logits(x, "hw"), 1).squeeze().numpy().astype(np.uint8)
        for i in range(4):
            hm.io[i][0].tofile(os.path.join(d, f"block_{i}_x.bin"))
            hm.io[i][1].tofile(os.path.join(d, f"block_{i}_y_golden.bin"))
        np.save(os.path.join(d, "mask_fp32.npy"), m_fp32)
        np.save(os.path.join(d, "mask_hw.npy"), m_hw)
        sim = hm.block_similarity()
        row = {"stem": stem, "image": os.path.abspath(f), "agreement": float((m_fp32 == m_hw).mean()),
               "cosine": [sim[i]["cosine_vs_fp32"] for i in range(4)], "mse": [sim[i]["mse_vs_fp32"] for i in range(4)]}
        lp = label_path(f, args)
        if lp and os.path.exists(lp):
            lab = load_label(lp, args.dataset)
            lab = np.array(Image.fromarray(lab.astype(np.uint8)).resize((args.size, args.size), Image.NEAREST)).astype(np.int64)
            for key, m in (("fp32", m_fp32), ("hw", m_hw)):
                c = confusion(m, lab, n_cls)
                conf[key] += c
                row[f"miou_{key}"] = iou_from_confusion(c)[1]
        manifest["images"].append(row)
        if (k + 1) % 10 == 0 or k + 1 == len(files):
            print(f"[{k + 1}/{len(files)}] {(time.perf_counter() - t0) / (k + 1):.2f} s/img", flush=True)

    n = len(manifest["images"])
    manifest["summary"] = {"n_images": n,
                           "agreement_mean": float(np.mean([r["agreement"] for r in manifest["images"]])),
                           "cosine_mean": np.mean([r["cosine"] for r in manifest["images"]], 0).tolist()}
    if conf["fp32"].sum():
        for key in conf:
            pc, miou, acc = iou_from_confusion(conf[key])
            manifest["summary"][f"miou_{key}"] = miou
            manifest["summary"][f"pixel_acc_{key}"] = acc
            manifest["summary"][f"per_class_iou_{key}"] = [None if np.isnan(v) else float(v) for v in pc]
    with open(os.path.join(args.out, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=1)
    s = manifest["summary"]
    print(f"\n{n} images -> {args.out}/   agreement {100 * s['agreement_mean']:.2f}%   cosine " +
          " ".join(f"{c:.4f}" for c in s["cosine_mean"]) +
          (f"   mIoU fp32 {100 * s['miou_fp32']:.2f}% / hw {100 * s['miou_hw']:.2f}%" if "miou_hw" in s else ""))
    print(f"next: zip this folder, run host/run_batch.py --weights {weights_dir} on the FPGA workstation")


if __name__ == "__main__":
    main()
