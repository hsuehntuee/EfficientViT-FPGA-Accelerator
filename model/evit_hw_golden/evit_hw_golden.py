#!/usr/bin/env python3
"""EfficientViT-Seg-B1 (Cityscapes) Stage 4 hardware golden generator.

One file, three subcommands:

  gen      image -> FP32 run + integer-only (W=Q.11, A=Q.5) run that mirrors the
           RTL bit for bit -> per-block weights / input / every intermediate
           stage / golden output as .bin, .hex, .npy, .npz, plus an
           FP32-vs-quant report and overlay figure.
  compare  diff a hardware dump (.bin / .hex / .mem / .npy) against a golden file.
  verify   regenerate everything from the bundled assets and check the SHA-256
           of every exported file against reference/SHA256SUMS.

BRAM layouts come from semantic_final_final.py, the 256-bit .hex text format
from gen_hex_4.py, and the arithmetic rules from the hardware team's answers.
"""
import argparse
import hashlib
import json
import os
import sys
import tempfile

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_IMAGE = os.path.join(HERE, "assets", "test.png")
DEFAULT_CKPT = os.path.join(HERE, "assets", "b1-cityscapes.pt")
DEFAULT_MANIFEST = os.path.join(HERE, "reference", "SHA256SUMS")
LANES = 16  # int16 lanes per 256-bit BRAM word
# Files whose bytes depend on library versions rather than on the golden data.
MANIFEST_SKIP_EXT = (".png", ".json", ".npz")

CITYSCAPES_PALETTE = np.array([
    [128, 64, 128],  [244, 35, 232],  [70, 70, 70],    [102, 102, 156],
    [190, 153, 153], [153, 153, 153], [250, 170, 30],  [220, 220, 0],
    [107, 142, 35],  [152, 251, 152], [70, 130, 180],  [220, 20, 60],
    [255, 0, 0],     [0, 0, 142],     [0, 0, 70],      [0, 60, 100],
    [0, 80, 100],    [0, 0, 230],     [119, 11, 32]
], dtype=np.uint8)


# ==========================================
# 0. efficientvit import / file helpers
# ==========================================
def import_efficientvit():
    """Import efficientvit from the environment, $EFFICIENTVIT_ROOT, or a parent directory."""
    candidates = [None, os.environ.get("EFFICIENTVIT_ROOT")]
    d = HERE
    for _ in range(3):
        d = os.path.dirname(d)
        candidates.append(d)
    last_err = None
    for root in candidates:
        if root is not None:
            if not os.path.isdir(os.path.join(root, "efficientvit")):
                continue
            sys.path.insert(0, root)
        try:
            from efficientvit.models.efficientvit.seg import efficientvit_seg_b1
            from efficientvit.models.nn.ops import EfficientViTBlock
            return efficientvit_seg_b1, EfficientViTBlock
        except ImportError as e:
            last_err = e
            if root is not None:
                sys.path.remove(root)
    raise RuntimeError(
        f"[FATAL] Cannot import efficientvit ({last_err}). Clone "
        "https://github.com/mit-han-lab/efficientvit and set EFFICIENTVIT_ROOT to it."
    )


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_hex(arr, path):
    """One 256-bit word per line: 16 x int16, element 0 in the LSBs (rightmost), for $readmemh."""
    flat = np.asarray(arr).astype(np.int64).flatten()
    if flat.size % LANES:
        flat = np.concatenate([flat, np.zeros(LANES - flat.size % LANES, dtype=np.int64)])
    with open(path, "w") as f:
        for row in flat.reshape(-1, LANES):
            f.write("".join(format(int(v) & 0xFFFF, "04x") for v in reversed(row)) + "\n")


def read_hex(path):
    vals = []
    with open(path) as f:
        for line in f:
            line = line.split("//")[0].strip().replace("_", "")
            if not line or line.startswith("@"):
                continue
            if len(line) % 4:
                raise ValueError(f"{path}: line length {len(line)} is not a multiple of 4 hex digits")
            words = [int(line[i:i + 4], 16) for i in range(0, len(line), 4)]
            vals.extend(reversed(words))
    return np.array(vals, dtype=np.uint16).astype(np.int16)


def load_int16(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        return np.load(path).astype(np.int16).flatten()
    if ext in (".hex", ".mem", ".txt"):
        return read_hex(path)
    return np.fromfile(path, dtype="<i2")


def overlay_semantic_mask(img_vis, mask_np):
    h, w = img_vis.shape[:2]
    mask = Image.fromarray(mask_np.astype(np.uint8)).resize((w, h), Image.NEAREST)
    color = CITYSCAPES_PALETTE[np.clip(np.array(mask), 0, 18)]
    return (0.4 * img_vis + 0.6 * color).astype(np.uint8)


# ==========================================
# 1. Integer-only block model (mirrors the RTL)
# ==========================================
I16_MIN, I16_MAX = -32768, 32767
# Intermediate dump points, in data-flow order. Last one is the block output.
STAGES = ("s1_qkv", "s2_agg", "s3_att", "s4_proj", "s5_mid", "s6_exp", "s7_dw", "s8_pw", "y_golden")


def sat16(a):
    return np.clip(a, I16_MIN, I16_MAX)


def round_shift(acc, shift):
    """Add half then arithmetic right shift (round-half-up)."""
    return (acc + (1 << (shift - 1))) >> shift if shift > 0 else acc


def quantize(t, frac):
    return torch.clamp(torch.round(t * float(1 << frac)), I16_MIN, I16_MAX).cpu().numpy().astype(np.int64)


def fuse_conv_bn(conv, bn):
    w = conv.weight.data
    b = conv.bias.data if conv.bias is not None else torch.zeros(w.size(0), device=w.device)
    if bn is not None:
        multiplier = bn.weight.data / torch.sqrt(bn.running_var + bn.eps)
        return w * multiplier.view(-1, 1, 1, 1), (b - bn.running_mean) * multiplier + bn.bias.data
    return w, b


def extract_block_params(block, w_frac, a_frac):
    """Fuse BN and quantize every conv of one EfficientViTBlock to int16 (held in int64 arrays)."""
    mbconv = getattr(block.local_module, 'main', block.local_module)
    litemla = getattr(block.context_module, 'main', block.context_module)
    if len(litemla.aggreg) != 1:
        raise ValueError("Only a single aggregation scale is supported")
    layers = {
        'qkv': (litemla.qkv.conv, litemla.qkv.norm),
        'agg_dw': (litemla.aggreg[0][0], None),
        'agg_pw': (litemla.aggreg[0][1], None),
        'proj': (litemla.proj.conv, litemla.proj.norm),
        'exp': (mbconv.inverted_conv.conv, mbconv.inverted_conv.norm),
        'dw': (mbconv.depth_conv.conv, mbconv.depth_conv.norm),
        'pw': (mbconv.point_conv.conv, mbconv.point_conv.norm),
    }
    p = {'dim': litemla.dim}
    for name, (conv, bn) in layers.items():
        w, b = fuse_conv_bn(conv, bn)
        p[f'{name}_w'] = quantize(w, w_frac).squeeze()
        p[f'{name}_b'] = quantize(b, a_frac)
    return p


def conv1x1_int(x, w, b, w_frac):
    C, H, W = x.shape
    acc = w @ x.reshape(C, H * W)
    return sat16(round_shift(acc, w_frac) + b[:, None]).reshape(-1, H, W)


def dwconv_int(x, w, b, w_frac):
    C, H, W = x.shape
    k = w.shape[-1]
    xp = np.pad(x, ((0, 0), (k // 2, k // 2), (k // 2, k // 2)))
    acc = np.zeros_like(x)
    for ky in range(k):
        for kx in range(k):
            acc += xp[:, ky:ky + H, kx:kx + W] * w[:, ky, kx, None, None]
    return sat16(round_shift(acc, w_frac) + b[:, None, None])


def group_conv1x1_int(x, w, b, w_frac):
    """w: [out_ch, in_per_group]; group g owns channels g*n .. g*n+n-1 on both sides."""
    C, H, W = x.shape
    n = w.shape[1]
    acc = np.einsum('goi,git->got', w.reshape(C // n, n, n), x.reshape(C // n, n, H * W))
    return sat16(round_shift(acc.reshape(C, H * W), w_frac) + b[:, None]).reshape(C, H, W)


def hswish_int(x, a_frac):
    """y = (x * clamp(x + 3, 0, 6) * floor(2^24 / 6) + 2^23) >> 24, pass-through once x + 3 >= 6."""
    three, six = 3 << a_frac, 6 << a_frac
    t = np.clip(x + three, 0, six)
    y = (x * t * ((1 << 24) // six) + (1 << 23)) >> 24
    return np.where(x + three >= six, x, y)


def linear_att_int(qkv, dim, a_frac):
    """ReLU linear attention. qkv: [heads * 3 * dim, H, W] with each head laid out as q | k | v."""
    C, H, W = qkv.shape
    T = H * W
    heads = C // (3 * dim)
    qkv = qkv.reshape(heads, 3 * dim, T)
    q = np.maximum(qkv[:, :dim], 0)
    k = np.maximum(qkv[:, dim:2 * dim], 0)
    vpad = np.concatenate([qkv[:, 2 * dim:], np.full((heads, 1, T), 1 << a_frac, dtype=np.int64)], axis=1)
    ctx = np.einsum('hrt,hct->hrc', k, vpad)   # [heads, dim, dim + 1], no shift
    out = np.einsum('hrt,hrc->htc', q, ctx)    # [heads, T, dim + 1], no shift
    # The pre-shifted numerator exceeds 64 bits, so finish in Python integers.
    num = out[:, :, :dim].astype(object) * (1 << 16)
    den = out[:, :, dim:].astype(object)
    den = np.where(den <= 0, 1, den)
    quot = num // den                          # floor division -> Q.16
    y = (quot + (1 << (15 - a_frac))) >> (16 - a_frac)
    y = sat16(y).astype(np.int64)              # [heads, T, dim]
    return y.transpose(0, 2, 1).reshape(heads * dim, H, W)


def run_block_int(p, x, w_frac, a_frac, dw_act=False):
    """x: int64 [C, H, W] in Q.a_frac. Returns every stage output as int64 [C', H, W]."""
    s = {}
    s['s1_qkv'] = conv1x1_int(x, p['qkv_w'], p['qkv_b'], w_frac)
    agg = dwconv_int(s['s1_qkv'], p['agg_dw_w'], p['agg_dw_b'], w_frac)
    s['s2_agg'] = group_conv1x1_int(agg, p['agg_pw_w'], p['agg_pw_b'], w_frac)
    s['s3_att'] = linear_att_int(np.concatenate([s['s1_qkv'], s['s2_agg']]), p['dim'], a_frac)
    s['s4_proj'] = conv1x1_int(s['s3_att'], p['proj_w'], p['proj_b'], w_frac)
    s['s5_mid'] = sat16(x + s['s4_proj'])
    s['s6_exp'] = hswish_int(conv1x1_int(s['s5_mid'], p['exp_w'], p['exp_b'], w_frac), a_frac)
    dw = dwconv_int(s['s6_exp'], p['dw_w'], p['dw_b'], w_frac)
    s['s7_dw'] = hswish_int(dw, a_frac) if dw_act else dw
    s['s8_pw'] = conv1x1_int(s['s7_dw'], p['pw_w'], p['pw_b'], w_frac)
    s['y_golden'] = sat16(s['s5_mid'] + s['s8_pw'])
    return s


# ==========================================
# 2. BRAM layouts
# ==========================================
def format_act(a):
    """int [C, H, W] -> int16 stream ordered [token][chunk][lane], token = row * W + col."""
    C, H, W = a.shape
    C_padded = ((C + 15) // 16) * 16
    a = np.concatenate([a, np.zeros((C_padded - C, H, W), dtype=a.dtype)]) if C_padded != C else a
    return a.reshape(C_padded // 16, 16, H * W).transpose(2, 0, 1).flatten().astype(np.int16)


def format_weights(p):
    def fmt_1x1(w, order):
        out_ch, in_ch = w.shape
        return w.reshape(out_ch // 16, 16, in_ch // 16, 16).transpose(order).flatten()

    def fmt_dw(w):
        """Kernel zero-padded to a centred 5x5, [chunk][tap = ky * 5 + kx][lane]."""
        ch, k = w.shape[0], w.shape[-1]
        padded = np.zeros((ch, 5, 5), dtype=np.int64)
        o = (5 - k) // 2
        padded[:, o:o + k, o:o + k] = w
        return padded.reshape(ch // 16, 16, 25).transpose(0, 2, 1)

    dim = p['dim']
    groups = p['agg_pw_w'].shape[0] // 16
    heads = groups // 3
    # Hardware chunk = {q,k,v} * heads + head  <->  PyTorch group = head * 3 + {q,k,v}
    hw2torch = np.array([(c % heads) * 3 + c // heads for c in range(groups)])
    # hw QKV channel s * heads * dim + h * dim + d  <-  PyTorch channel h * 3 * dim + s * dim + d
    qkv_hw_ch = np.array([h * 3 * dim + s * dim + d for s in range(3) for h in range(heads) for d in range(dim)])
    if dim != 16:
        raise ValueError("BRAM layout assumes 16 channels per q/k/v group")

    # dw_w: MBConv DW at word 0, aggregation DW at word 16384 + chunk * 25 + tap
    AGG_DW_W_BASE, AGG_DW_B_BASE = 16384, 1024
    mb_dw = fmt_dw(p['dw_w']).reshape(-1, 16)
    agg_dw = fmt_dw(p['agg_dw_w'])[hw2torch].reshape(-1, 16)
    dw_w = np.zeros((AGG_DW_W_BASE + agg_dw.shape[0], 16), dtype=np.int64)
    dw_w[:mb_dw.shape[0]] = mb_dw
    dw_w[AGG_DW_W_BASE:] = agg_dw
    # dw_b: MBConv DW bias at word 0, aggregation DW bias at word 1024 + chunk
    mb_dw_b = p['dw_b'].reshape(-1, 16)
    dw_b = np.zeros((AGG_DW_B_BASE + groups, 16), dtype=np.int64)
    dw_b[:mb_dw_b.shape[0]] = mb_dw_b
    dw_b[AGG_DW_B_BASE:] = p['agg_dw_b'].reshape(groups, 16)[hw2torch]
    # ms_pw_w: word chunk * 16 + r, lane c = weight from group input r to group output c
    ms_pw_w = p['agg_pw_w'].reshape(groups, 16, 16)[hw2torch].transpose(0, 2, 1)

    out = {
        'w': fmt_1x1(p['exp_w'], (0, 2, 3, 1)),      # [out_chunk][in_chunk][in_lane][out_lane]
        'dw_w': dw_w,
        'pw_w': fmt_1x1(p['pw_w'], (2, 0, 3, 1)),    # [in_chunk][out_chunk][in_lane][out_lane]
        'b': p['exp_b'],
        'dw_b': dw_b,
        'pw_b': p['pw_b'],
        # QKV output channels in hardware order: all heads' Q | all K | all V (chunk = {q,k,v} * heads + head).
        # The RTL writes M1 output chunk c to sram_q (c < 16), sram_k (c < 32) or sram_v, and head h reads chunk h of each.
        'mla_w': np.concatenate([fmt_1x1(p['qkv_w'][qkv_hw_ch], (0, 2, 3, 1)), fmt_1x1(p['proj_w'], (0, 2, 3, 1))]),
        'mla_b': np.concatenate([p['qkv_b'][qkv_hw_ch], p['proj_b']]),
        'ms_pw_w': ms_pw_w,
        'ms_pw_b': p['agg_pw_b'].reshape(groups, 16)[hw2torch],
    }
    return {k: v.flatten().astype(np.int16) for k, v in out.items()}


# ==========================================
# 3. gen
# ==========================================
def export_array(export_dir, stem, arr, npy_aliases=()):
    arr.tofile(os.path.join(export_dir, f"{stem}.bin"))
    write_hex(arr, os.path.join(export_dir, f"{stem}.hex"))
    for alias in npy_aliases:
        np.save(os.path.join(export_dir, f"{alias}.npy"), arr)


def write_manifest(export_dir):
    names = sorted(n for n in os.listdir(export_dir)
                   if not n.endswith(MANIFEST_SKIP_EXT) and n != "SHA256SUMS")
    lines = [f"{sha256_file(os.path.join(export_dir, n))}  {n}\n" for n in names]
    with open(os.path.join(export_dir, "SHA256SUMS"), "w") as f:
        f.writelines(lines)
    return lines


def run_gen(image, ckpt, export_dir, size=128, w_frac=11, a_frac=5, dw_act=True, plot=True, dataset="cityscapes"):
    efficientvit_seg_b1, EfficientViTBlock = import_efficientvit()
    os.makedirs(export_dir, exist_ok=True)
    device = torch.device("cpu")
    a_scale = float(1 << a_frac)
    print(f"[*] HW Export Directory: {export_dir}")
    print(f"[*] Quantization: W=Q.{w_frac}, A=Q.{a_frac} | hardswish after MBConv DW: {'on' if dw_act else 'off'}")
    print(f"[*] Image: {image}\n[*] Checkpoint: {ckpt}")
    if not dw_act:
        print("[WARN] No hardswish after the MBConv depthwise conv (RTL with DW_ACT=0). The trained model "
              "has one,\n       so this golden is NOT a valid segmentation result; it only matches the old RTL.")

    for path in (image, ckpt):
        if not os.path.exists(path):
            raise FileNotFoundError(f"[FATAL] Missing input file: {path}")

    model = efficientvit_seg_b1(dataset=dataset).to(device)
    chk = torch.load(ckpt, map_location=device)
    model.load_state_dict(chk['state_dict'] if 'state_dict' in chk else chk, strict=True)
    model.eval()

    img = Image.open(image).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((size, size)), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    x = transform(img).unsqueeze(0).to(device)

    target_blocks = [m for name, m in model.named_modules() if isinstance(m, EfficientViTBlock) and "stages.3" in name]
    config = {'sim_mode': 'fp32'}
    feature_cache = {'fp32': {}, 'quant': {}}
    saturation = {}

    def to_mask(out):
        return torch.argmax(out['output'] if isinstance(out, dict) else out, dim=1).squeeze().numpy()

    def make_block_hook(b_idx, params):
        def hook(mod, inp, out):
            if config['sim_mode'] == 'quant':
                x_int = quantize(inp[0][0], a_frac)
                stages = run_block_int(params, x_int, w_frac, a_frac, dw_act)
                export_array(export_dir, f"block_{b_idx}_x", format_act(x_int),
                             npy_aliases=(f"block_{b_idx}_x", f"block_{b_idx}_input"))
                for name in STAGES:
                    aliases = (f"block_{b_idx}_y_golden", f"block_{b_idx}_golden_out") if name == "y_golden" else ()
                    export_array(export_dir, f"block_{b_idx}_{name}", format_act(stages[name]), npy_aliases=aliases)
                saturation[f"Block_{b_idx:02d}"] = {
                    name: int(((stages[name] == I16_MIN) | (stages[name] == I16_MAX)).sum()) for name in STAGES}
                out = torch.from_numpy(stages['y_golden'] / a_scale).to(out.dtype).unsqueeze(0)
            feature_cache[config['sim_mode']][f"Block_{b_idx:02d}"] = out.detach().clone()
            return out
        return hook

    hooks = []
    for i, block in enumerate(target_blocks):
        params = extract_block_params(block, w_frac, a_frac)
        weights = format_weights(params)
        np.savez(os.path.join(export_dir, f"block_{i}_weights.npz"), **weights)
        for wn, wd in weights.items():
            export_array(export_dir, f"block_{i}_{wn}", wd)
        hooks.append(block.register_forward_hook(make_block_hook(i, params)))

    with torch.no_grad():
        mask_fp32 = to_mask(model(x))
        config['sim_mode'] = 'quant'
        mask_quant = to_mask(model(x))
    for h in hooks: h.remove()

    report = {
        'image': os.path.basename(image), 'image_sha256': sha256_file(image),
        'ckpt_sha256': sha256_file(ckpt), 'size': size, 'w_frac': w_frac, 'a_frac': a_frac,
        'dw_act': dw_act, 'torch': torch.__version__, 'numpy': np.__version__, 'blocks': {},
    }
    print("\n" + "="*86)
    print(f"| {'Layer Name':<12} | {'Cosine Similarity':<18} | {'MSE Error':<12} | {'Saturated values (stage: count)':<31} |")
    print("="*86)
    for name in sorted(feature_cache['fp32'].keys()):
        f, q = feature_cache['fp32'][name].flatten(), feature_cache['quant'][name].flatten()
        cs = F.cosine_similarity(f.unsqueeze(0), q.unsqueeze(0)).item() if torch.sum(torch.abs(f)) != 0 else 0.0
        mse = F.mse_loss(f, q).item()
        sat = {k: v for k, v in saturation[name].items() if v}
        report['blocks'][name] = {'cosine_similarity': cs, 'mse': mse, 'saturated': saturation[name]}
        print(f"| {name:<12} | {cs:<18.6f} | {mse:<12.6f} | {(str(sat) if sat else 'none'):<31} |")
    print("="*86)
    report['mask_agreement'] = float((mask_fp32 == mask_quant).mean())
    print(f"[*] FP32 vs Quant segmentation mask agreement: {report['mask_agreement'] * 100:.2f}%")

    with open(os.path.join(export_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2)

    if plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        img_v = np.array(img.resize((size, size)))
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1); plt.imshow(overlay_semantic_mask(img_v, mask_fp32)); plt.title("Software FP32"); plt.axis('off')
        plt.subplot(1, 2, 2); plt.imshow(overlay_semantic_mask(img_v, mask_quant)); plt.title(f"Bit-exact W{w_frac}A{a_frac}"); plt.axis('off')
        plt.savefig(os.path.join(export_dir, "pipeline_evaluation.png"))
        plt.close()

    lines = write_manifest(export_dir)
    print(f"\n[SUCCESS] Exported {len(lines)} golden files (+ report/figure) to {export_dir}")
    return report


# ==========================================
# 4. compare / verify
# ==========================================
def golden_part_channels(part, heads=16, dim=16):
    """(total golden channels, golden channel indices in hardware order) for one RTL dump point."""
    if part in ("q", "k", "v"):            # hw_m1_{q,k,v}: all heads of one of q/k/v, chunk = head
        s = "qkv".index(part)
        return 3 * heads * dim, [h * 3 * dim + s * dim + d for h in range(heads) for d in range(dim)]
    if part == "ms_hw":                    # hw_ms_pw_mid: chunk = {q,k,v} * heads + head
        return 3 * heads * dim, [(h * 3 + s) * dim + d for s in range(3) for h in range(heads) for d in range(dim)]
    if part == "att_h1":                   # hw_m2_h1: heads 0..15 (from QKV)
        return 2 * heads * dim, list(range(0, heads * dim))
    if part == "att_h2":                   # hw_m2_h2: heads 16..31 (from aggregation)
        return 2 * heads * dim, list(range(heads * dim, 2 * heads * dim))
    raise ValueError(part)


def run_compare(golden_path, hw_path, tol=0, max_show=10, hw_chunk_major_tokens=0, golden_part=None):
    golden, hw = load_int16(golden_path), load_int16(hw_path)
    if golden_part:
        # golden stage files are [token][channel] in PyTorch order; keep the channels the RTL dumps, in its order
        total, ch = golden_part_channels(golden_part)
        golden = golden.reshape(-1, total)[:, ch].flatten()
    if hw_chunk_major_tokens:
        # hardware stage dumps are [chunk][token][lane]; golden files are [token][chunk][lane]
        hw = hw.reshape(-1, hw_chunk_major_tokens, LANES).transpose(1, 0, 2).flatten()
    print(f"[*] golden: {golden_path} ({golden.size} x int16)")
    print(f"[*] hw    : {hw_path} ({hw.size} x int16)")
    ok = True
    if golden.size != hw.size:
        print(f"[FAIL] Length mismatch: golden={golden.size}, hw={hw.size} (comparing the common prefix)")
        ok = False
    n = min(golden.size, hw.size)
    if n == 0:
        print("[FAIL] Nothing to compare")
        return False
    diff = hw[:n].astype(np.int32) - golden[:n].astype(np.int32)
    bad = np.flatnonzero(np.abs(diff) > tol)
    print(f"[*] compared {n} values | exact match {int((diff == 0).sum())} "
          f"({(diff == 0).mean() * 100:.4f}%) | max |diff| = {int(np.abs(diff).max())} LSB | "
          f"mean |diff| = {np.abs(diff).mean():.4f} LSB")
    if bad.size:
        ok = False
        print(f"[FAIL] {bad.size} values differ by more than {tol} LSB. First {min(max_show, bad.size)}:")
        for idx in bad[:max_show]:
            print(f"    idx {idx:>8} (word {idx // LANES}, lane {idx % LANES}): "
                  f"golden={int(golden[idx]):>6} hw={int(hw[idx]):>6} diff={int(diff[idx]):+d}")
    if ok:
        print("[PASS] Hardware output matches golden" + (f" within {tol} LSB" if tol else " bit-exactly"))
    return ok


def run_verify(manifest, update=False):
    with tempfile.TemporaryDirectory(prefix="evit_hw_verify_") as tmp:
        run_gen(DEFAULT_IMAGE, DEFAULT_CKPT, tmp, plot=False)
        with open(os.path.join(tmp, "SHA256SUMS")) as f:
            got = dict(reversed(line.split()) for line in f)
        if update:
            os.makedirs(os.path.dirname(manifest), exist_ok=True)
            with open(os.path.join(tmp, "SHA256SUMS")) as src, open(manifest, "w") as dst:
                dst.write(src.read())
            print(f"[*] Wrote reference manifest ({len(got)} files) to {manifest}")
            return True
    with open(manifest) as f:
        want = dict(reversed(line.split()) for line in f)
    bad = sorted(n for n in want.keys() | got.keys() if want.get(n) != got.get(n))
    for n in bad:
        state = "missing" if n not in got else "unexpected" if n not in want else "hash differs"
        print(f"    [DIFF] {n}: {state}")
    if bad:
        print(f"[FAIL] {len(bad)} of {len(want)} files do not match {manifest}")
        return False
    print(f"[PASS] All {len(want)} regenerated files match {manifest}")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gen", help="generate weights / inputs / golden outputs from an image")
    g.add_argument("--image", default=DEFAULT_IMAGE)
    g.add_argument("--ckpt", default=DEFAULT_CKPT)
    g.add_argument("--out", default="hw_export")
    g.add_argument("--size", type=int, default=128, help="network input resolution (square)")
    g.add_argument("--w-frac", type=int, default=11, help="weight fractional bits")
    g.add_argument("--a-frac", type=int, default=5, help="activation / bias fractional bits")
    g.add_argument("--no-dw-act", action="store_true",
                   help="drop the hardswish after the MBConv depthwise conv, matching the RTL built with DW_ACT=0")
    g.add_argument("--no-plot", action="store_true")
    g.add_argument("--dataset", choices=["cityscapes", "ade20k"], default="cityscapes", help="model head (backbone is identical)")

    c = sub.add_parser("compare", help="compare a hardware dump against a golden file")
    c.add_argument("golden", help=".bin / .hex / .mem / .txt / .npy")
    c.add_argument("hw", help=".bin / .hex / .mem / .txt / .npy")
    c.add_argument("--hw-chunk-major", type=int, default=0, metavar="TOKENS",
                   help="hw file is ordered [chunk][token] with this many tokens (RTL stage dumps); reorder before comparing")
    c.add_argument("--golden-part", choices=["q", "k", "v", "ms_hw", "att_h1", "att_h2"],
                   help="golden is a stage file; keep only the channels of this RTL dump, in RTL order "
                        "(q/k/v: hw_m1_*, ms_hw: hw_ms_pw_mid, att_h1/h2: hw_m2_h1/h2)")
    c.add_argument("--tol", type=int, default=0, help="allowed |diff| in LSB")
    c.add_argument("--max-show", type=int, default=10)

    v = sub.add_parser("verify", help="regenerate from bundled assets and check against the reference manifest")
    v.add_argument("--manifest", default=DEFAULT_MANIFEST)
    v.add_argument("--update", action="store_true", help="overwrite the reference manifest instead of checking")

    args = ap.parse_args()
    if args.cmd == "gen":
        run_gen(args.image, args.ckpt, args.out, args.size, args.w_frac, args.a_frac, not args.no_dw_act, not args.no_plot, args.dataset)
        ok = True
    elif args.cmd == "compare":
        ok = run_compare(args.golden, args.hw, args.tol, args.max_show, args.hw_chunk_major, args.golden_part)
    else:
        ok = run_verify(args.manifest, args.update)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
