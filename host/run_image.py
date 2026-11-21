"""Image -> FPGA (4 Stage 4 blocks) -> CPU head -> semantic mask + numbers.

    python host/run_image.py --image test.png --out results/test                   # on the board host
    python host/run_image.py --image test.png --out results/test --backend model   # dry run, no FPGA

Per block: the FP32 model runs up to the block, its input is quantised to Q.5 and sent to the
accelerator together with that block's weights; the accelerator's output replaces the block's
output and the FP32 model continues. The integer reference model (evit_hw_golden.run_block_int)
runs alongside, so every block has a bit-exact pass/fail, not just a similarity number.

Needs the evit_hw_golden repo (import path from --golden-repo / $EVIT_HW_GOLDEN / sibling dir)
and the efficientvit package ($EFFICIENTVIT_ROOT or installed). torch runs on CPU.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hybrid import (DATASETS, FileBackend, FpgaBackend, HybridModel, ModelBackend, confusion, default_ckpt,  # noqa: E402
                    iou_from_confusion, load_golden_module, load_label, overlay)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--ckpt", default=None, help="default: <golden repo>/assets/<checkpoint of --dataset>")
    ap.add_argument("--dataset", choices=list(DATASETS), default="cityscapes")
    ap.add_argument("--backend", choices=["fpga", "model", "file"], default="fpga")
    ap.add_argument("--y-dir", default=None, help="backend file: directory with block_N_y_hw.bin recorded on the board")
    ap.add_argument("--golden-repo", default=None)
    ap.add_argument("--size", type=int, default=512, help="network input; Stage 4 tokens = (size/32)^2, RTL N_TOKENS must match")
    ap.add_argument("--label", default=None, help="Cityscapes trainId label png (0..18, 255 = ignore) for mIoU")
    ap.add_argument("--timeout", type=float, default=5.0)
    args = ap.parse_args()

    g, gdir = load_golden_module(args.golden_repo)
    import torch
    from PIL import Image

    ckpt = args.ckpt or default_ckpt(gdir, args.dataset)
    n_cls = DATASETS[args.dataset]["n_classes"]
    os.makedirs(args.out, exist_ok=True)
    backend = FpgaBackend(timeout=args.timeout) if args.backend == "fpga" else FileBackend(g, args.y_dir) if args.backend == "file" else ModelBackend(g)
    hm = HybridModel(g, ckpt, backend, args.size, args.dataset)
    hm.keep_io = True

    img = Image.open(args.image)
    x = hm.preprocess(img)
    t0 = time.perf_counter()
    mask_fp32 = torch.argmax(hm.logits(x, "fp32"), 1).squeeze().numpy()
    t_fp32 = time.perf_counter() - t0
    t0 = time.perf_counter()
    mask_hw = torch.argmax(hm.logits(x, "hw"), 1).squeeze().numpy()
    t_hw = time.perf_counter() - t0
    backend.close()

    sim = hm.block_similarity()
    report = {"image": os.path.basename(args.image), "image_sha256": g.sha256_file(args.image), "ckpt_sha256": g.sha256_file(ckpt),
              "size": args.size, "dataset": args.dataset, "backend": backend.name, "blocks": {}, "t_fp32_ms": 1e3 * t_fp32, "t_hw_pipeline_ms": 1e3 * t_hw}
    print(f"\n{'block':5s} {'bit-exact':9s} {'cosine':>8s} {'MSE':>9s} {'cycles':>8s} {'upload':>9s} {'compute':>9s} {'readback':>9s}")
    for i in range(len(hm.blocks)):
        info = dict(hm.per_block[i], **sim[i])
        report["blocks"][f"block_{i}"] = info
        cyc = "-" if info["cycles"] is None else str(info["cycles"])
        print(f"{i:5d} {'yes' if info['bit_exact'] else 'NO':9s} {info['cosine_vs_fp32']:8.4f} {info['mse_vs_fp32']:9.4f} {cyc:>8s} "
              f"{info['t_upload_ms']:7.1f}ms {info['t_compute_ms']:7.1f}ms {info['t_readback_ms']:7.1f}ms")
        if not info["bit_exact"]:
            print(f"      {info['mismatches']} values differ from the integer model, max |d| = {info['max_abs_diff']} LSB"
                  + (f"  [{info['source']}]" if "source" in info else ""))
        hm.io[i][0].tofile(os.path.join(args.out, f"block_{i}_x.bin"))
        hm.io[i][1].tofile(os.path.join(args.out, f"block_{i}_y_hw.bin"))
    report["mask_agreement_vs_fp32"] = float((mask_fp32 == mask_hw).mean())
    print(f"\nmask agreement {backend.name} vs FP32: {100 * report['mask_agreement_vs_fp32']:.2f}%   (mask {mask_hw.shape[1]}x{mask_hw.shape[0]})")
    print(f"CPU FP32 forward {1e3 * t_fp32:.0f} ms; hybrid forward incl. transfers {1e3 * t_hw:.0f} ms")
    if args.label:
        lab = load_label(args.label, args.dataset)
        lab = np.array(Image.fromarray(lab.astype(np.uint8)).resize(mask_hw.shape[::-1], Image.NEAREST)).astype(np.int64)
        for k, m in (("fp32", mask_fp32), (backend.name, mask_hw)):
            _, miou, acc = iou_from_confusion(confusion(m, lab, n_cls))
            report[f"miou_{k}"], report[f"pixel_acc_{k}"] = miou, acc
            print(f"{k:5s} vs label: mIoU {100 * miou:.2f}%  pixel acc {100 * acc:.2f}%")

    # ---- pictures
    np.save(os.path.join(args.out, "mask_fp32.npy"), mask_fp32)
    np.save(os.path.join(args.out, "mask_hw.npy"), mask_hw)
    img_v = np.array(img.convert("RGB").resize((args.size, args.size)))
    Image.fromarray(overlay(img_v, mask_hw, n_cls)).save(os.path.join(args.out, "mask_hw.png"))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        panels = (img_v, overlay(img_v, mask_fp32, n_cls), overlay(img_v, mask_hw, n_cls))
        for a, im, t in zip(ax, panels, ("input", "FP32 (CPU)", f"{backend.name} Stage 4 + CPU head  (agreement {100 * report['mask_agreement_vs_fp32']:.1f}%)")):
            a.imshow(im)
            a.set_title(t)
            a.axis("off")
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, "comparison.png"), dpi=120)
        plt.close(fig)
    except ImportError:
        pass
    with open(os.path.join(args.out, "result.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nwritten to {args.out}/: result.json, comparison.png, mask_hw.png, block_N_x.bin, block_N_y_hw.bin")


if __name__ == "__main__":
    main()
