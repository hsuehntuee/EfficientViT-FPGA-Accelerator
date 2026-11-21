"""Bit-accurate numpy model of rtl/accelerator_top.v
(EfficientViT-B1 stage-4 block = backbone.stages[3]: 256 ch, IMG x IMG tokens (256 for 512x512 input), W frac 11 / A frac 5).

Written from the RTL. The three bug flags (run() defaults) reproduce the May 2026 RTL; with the flags
off (FIXED) it is the RTL from 2026-10-04 on. The vectors it reads are ../hw_export_512
(block 0-3, every stage golden); the 128x128 vectors and the May xsim dumps were removed on 2026-10-17.

  python rtl_model.py --block N                        # model vs hw_export_512 stage files (expect 100%)
  python rtl_model.py --block 0 --dumps <dir>          # model vs xsim dumps (hw_*.txt) made with hw_export_512
"""
import argparse
import os

import numpy as np

WF, AF = 11, 5                              # weight frac bits, activation frac bits
T, IMG = 256, 16                            # tokens and map side; set_tokens() follows the x file (16 -> 128x128 input, 256 -> 512x512)


def set_tokens(t):
    global T, IMG, P1_BUG, Q_BUG
    T, IMG = t, int(round(t ** 0.5))
    assert IMG * IMG == T, f"tokens {t} is not a square map"
    P1_BUG = [0] + list(range(T - 1))
    Q_BUG = [T - 1] + list(range(T - 1))
HERE = os.path.dirname(os.path.abspath(__file__))
NAMES = ["x", "w", "b", "dw_w", "dw_b", "pw_w", "pw_b", "mla_w", "mla_b", "ms_pw_w", "ms_pw_b", "y_golden"]

# RTL handshake bug (accelerator_top <-> litemla_core): in_valid is not gated by in_ready.
P1_BUG = [0] + list(range(T - 1))           # context accumulates token 0 twice, the last token never
Q_BUG = [T - 1] + list(range(T - 1))        # output slot t is computed from q of token t-1
FIXED = dict(h2_zero=False, mid_transposed=False, mla_bug=False, act="hswish", dw_act=True)   # accelerator_top.v now


def sat(x):
    return np.clip(x, -32768, 32767)


def rs(x, n=WF):
    """Round-half-up arithmetic right shift, as in psum_buffer / dw_conv_engine."""
    return (x + (1 << (n - 1))) >> n


def load(d, blk=0, legacy=False):
    """hw_export/block_{blk}_{name}.bin -> {name: int64 [words, 16]} (one row = one 256-bit BRAM word).
    legacy=True takes the MS weights from legacy_ms_order/ (what the 2026-05-26 simulation used)."""
    out = {}
    for n in NAMES:
        p = os.path.join(d, f"block_{blk}_{n}.bin")
        old = os.path.join(d, "legacy_ms_order", f"block_{blk}_{n}.bin")
        if legacy and os.path.exists(old):
            p = old
        if os.path.exists(p):
            out[n] = np.fromfile(p, dtype=np.int16).astype(np.int64).reshape(-1, 16)
    if "x" in out:
        set_tokens(len(out["x"]) // 16)
    return out


def conv1x1(act, W, nin, nout, base, tile):
    """Systolic 1x1 conv. act[token, nin*16]; W word (base + tile(oc, ic)*16 + r)[c] = weight(in r, out c)."""
    out = np.zeros((T, nout * 16), dtype=np.int64)
    for oc in range(nout):
        for ic in range(nin):
            a = base + tile(oc, ic) * 16
            out[:, oc * 16:(oc + 1) * 16] += act[:, ic * 16:(ic + 1) * 16] @ W[a:a + 16]
    return out


def dw5(x, k, b):
    """5x5 depthwise on an IMG x IMG map, zero padded. x[token,16], k[25,16] (row-major taps), b[16]."""
    img = np.zeros((IMG + 4, IMG + 4, 16), dtype=np.int64)
    img[2:IMG + 2, 2:IMG + 2] = x.reshape(IMG, IMG, 16)
    out = np.zeros((IMG, IMG, 16), dtype=np.int64)
    for ky in range(5):
        for kx in range(5):
            out += img[ky:ky + IMG, kx:kx + IMG] * k[ky * 5 + kx]
    return sat(rs(out) + b).reshape(T, 16)


def mla(q, k, v, bug):
    """One LiteMLA head: relu(q) . (relu(k)^T [v, 1]) / norm, 64-bit floor divide then round to A frac."""
    q, k = np.maximum(q, 0), np.maximum(k, 0)
    vp = np.concatenate([v, np.full((T, 1), 1 << AF, dtype=np.int64)], 1)
    if bug:
        k, vp, q = k[P1_BUG], vp[P1_BUG], q[Q_BUG]
    qc = q @ (k.T @ vp)                        # 48-bit context, 64-bit dot products in the RTL
    den = np.where(qc[:, 16:] <= 0, 1, qc[:, 16:])
    quo = [[(int(n) << 16) // int(d[0]) for n in row[:16]] for row, d in zip(qc, den)]
    return sat(np.array([[(z + (1 << (16 - AF - 1))) >> (16 - AF) for z in r] for r in quo], dtype=object)).astype(np.int64)


def hswish_float(p):
    """What the PyTorch reference does after the expand conv (float hswish on dequantised ints)."""
    f = p / (1 << AF)
    return np.round(f * np.clip(f + 3, 0, 6) / 6 * (1 << AF)).astype(np.int64)


def hswish_rtl(x):
    """hswish_array in accelerator_top.v: x * clamp(x + 3, 0, 6) / 6 with a 24-bit reciprocal."""
    six = 6 << AF
    t = np.clip(x + (3 << AF), 0, six)
    return np.where(t == six, x, (x * t * ((1 << 24) // six) + (1 << 23)) >> 24)


def run(D, h2_zero=True, mid_transposed=True, mla_bug=True, act="relu6", dw_act=False):
    """Returns every stage as [token, channels]. Defaults = RTL as of 2026-05-26."""
    S = {}
    x = D["x"][:T * 16].reshape(T, 256)

    # M1: QKV 1x1 (256 -> 768). Channel order is all-Q | all-K | all-V, 16 heads each.
    qkv = sat(rs(conv1x1(x, D["mla_w"], 16, 48, 0, lambda oc, ic: oc * 16 + ic)) + D["mla_b"][:48].reshape(-1))
    S["q"], S["k"], S["v"] = qkv[:, :256], qkv[:, 256:512], qkv[:, 512:]

    # MS: 5x5 DW + group 1x1 per 16-channel chunk (weights at dw_w word 16384+, dw_b word 1024+).
    ms = np.zeros_like(qkv)
    for c in range(48):
        d = dw5(qkv[:, c * 16:(c + 1) * 16], D["dw_w"][16384 + c * 25:16384 + c * 25 + 25], D["dw_b"][1024 + c])
        ms[:, c * 16:(c + 1) * 16] = sat(rs(d @ D["ms_pw_w"][c * 16:(c + 1) * 16]) + D["ms_pw_b"][c])
    S["ms_pw"] = ms

    # M2: attention, 16 heads on qkv then 16 heads on ms.
    heads = lambda z: np.concatenate(
        [mla(z[:, h * 16:h * 16 + 16], z[:, 256 + h * 16:272 + h * 16], z[:, 512 + h * 16:528 + h * 16], mla_bug) for h in range(16)], 1)
    S["h1"], S["h2"] = heads(qkv), heads(ms)
    h = np.concatenate([S["h1"], np.zeros_like(S["h2"]) if h2_zero else S["h2"]], 1)   # RTL forces h2 = 0

    # M3: proj 1x1 (512 -> 256) + residual.
    proj = sat(rs(conv1x1(h, D["mla_w"], 32, 16, 12288, lambda oc, ic: oc * 32 + ic)) + D["mla_b"][48:64].reshape(-1))
    mid = sat(x + proj)
    S["proj"], S["mid"] = proj, mid

    # P2: expand 1x1 (256 -> 1024) + act (EXPAND_ACT: 0 = relu6, 1 = hswish), then DW (3x3 embedded
    # in 5x5), then hswish if DW_ACT = 1.
    # RTL bug: sram_mid is written at token*16+chunk but read at chunk*N_TOKENS+token.
    mid_rd = mid.reshape(T, 16, 16).transpose(1, 0, 2).reshape(T, 256) if mid_transposed else mid
    pre = sat(rs(conv1x1(mid_rd, D["w"], 16, 64, 0, lambda oc, ic: oc * 16 + ic)) + D["b"][:64].reshape(-1))
    inv = {"relu6": lambda p: np.clip(p, 0, 6 << AF), "hswish": hswish_rtl, "hswish_float": hswish_float}[act](pre)
    dw = np.concatenate([dw5(inv[:, c * 16:(c + 1) * 16], D["dw_w"][c * 25:c * 25 + 25], D["dw_b"][c]) for c in range(64)], 1)
    if dw_act:                                  # DW_ACT = 1
        dw = hswish_rtl(dw)
    S["inv"], S["dw"] = inv, dw

    # P3: point 1x1 (1024 -> 256); P4: residual.
    pw = sat(rs(conv1x1(dw, D["pw_w"], 64, 16, 0, lambda oc, ic: ic * 16 + oc)) + D["pw_b"][:16].reshape(-1))
    S["pw"], S["y"] = pw, sat(mid + pw)
    return S


def load_dump(path):
    """xsim dump -> [lines, 16], channel 0 first. Lines are chunk-major: [chunk][token]."""
    rows = [l.strip() for l in open(path) if l.strip()]
    a = np.array([[int(l[i * 4:i * 4 + 4], 16) for i in range(15, -1, -1)] for l in rows], dtype=np.int64)
    return np.where(a >= 32768, a - 65536, a)


def chunk_major(a):
    """[token, chunks*16] -> [chunks*token, 16] in dump order."""
    return a.reshape(T, -1, 16).transpose(1, 0, 2).reshape(-1, 16)


def dump_tokens(path, layout="chunk", k=8):
    """xsim dump -> [token, channels]. layout 'chunk': lines are [chunk][token] (MS, attention, DW dumps);
    'cluster': lines are [group][token][k] with chunk = group*k + k (conv_cluster outputs, v3 RTL)."""
    a = load_dump(path)
    if layout == "chunk":
        return a.reshape(-1, T, 16).transpose(1, 0, 2).reshape(T, -1)
    return a.reshape(-1, T, k, 16).transpose(1, 0, 2, 3).reshape(T, -1)


# evit_hw_golden stage files ([token][channel], PyTorch channel order) -> this model's stages (hardware order).
def torch_qkv_to_hw(a):
    return a.reshape(T, 16, 3, 16).transpose(0, 2, 1, 3).reshape(T, 768)


def compare_sw_golden(D, S, d, blk):
    """Model vs evit_hw_golden block_{blk}_s*_*.bin in directory d (if present). Returns True if compared."""
    f = lambda n: os.path.join(d, f"block_{blk}_{n}.bin")
    if not os.path.exists(f("s1_qkv")):
        return False
    ld = lambda n: np.fromfile(f(n), dtype=np.int16).astype(np.int64).reshape(T, -1)
    x = D["x"][:T * 16].reshape(T, 256)
    s3 = ld("s3_att")
    pairs = [("q|k|v", np.concatenate([S["q"], S["k"], S["v"]], 1), torch_qkv_to_hw(ld("s1_qkv")), "s1_qkv"),
             ("ms_pw", S["ms_pw"], torch_qkv_to_hw(ld("s2_agg")), "s2_agg"),
             ("h1", S["h1"], s3[:, :256], "s3_att[:256]"), ("h2", S["h2"], s3[:, 256:], "s3_att[256:]"),
             ("proj", S["mid"] - x, ld("s4_proj"), "s4_proj (valid while mid does not saturate)"),
             ("mid", S["mid"], ld("s5_mid"), "s5_mid"), ("inv", S["inv"], ld("s6_exp"), "s6_exp"),
             ("dw", S["dw"], ld("s7_dw"), "s7_dw"), ("pw", S["pw"], ld("s8_pw"), "s8_pw"), ("y", S["y"], ld("y_golden"), "y_golden")]
    print(f"\nBlock {blk} ({T} tokens): this model (RTL now) vs evit_hw_golden stage files in {os.path.relpath(d)}:")
    for name, a, b, src in pairs:
        print(f"  {name:6s} exact = {100 * np.mean(a == b):6.2f}%   max|d| = {np.abs(a - b).max():4d}   ({src})")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", default=os.path.join(HERE, "..", "hw_export_512"),
                    help="vector directory (default ../hw_export_512: block 0-3 x / weights / per-stage golden)")
    ap.add_argument("--dumps", default=None, help="xsim hw_*.txt directory to compare against (optional)")
    ap.add_argument("--block", type=int, default=0)
    ap.add_argument("--rtl", choices=["v2", "v3"], default="v3", help="v2: every dump [chunk][token]; v3: cluster outputs [group][token][k]")
    ap.add_argument("--par", type=int, default=8, help="v3 CONV_PAR")
    args = ap.parse_args()
    D = load(args.export, args.block)
    D_old = load(args.export, args.block, legacy=True)

    if args.block == 0 and args.dumps and os.path.isdir(args.dumps):
        # The dumps must come from a simulation that used the same hw_export/ as --export.
        # sim_0526 was made with the old RTL; a re-run after the 2026-10-04 fixes should match "fixed".
        old, new = run(D_old), run(D, **FIXED)
        print(f"Model vs xsim dumps in {os.path.relpath(args.dumps)} (% of values bit-exact, RTL {args.rtl} dump layout):")
        print(f"  {'stage':6s} {'RTL 0526':>9s} {'RTL fixed':>10s}     (0526 column uses hw_export/legacy_ms_order)")
        cl = "cluster" if args.rtl == "v3" else "chunk"
        for key, f, lay in [("q", "hw_m1_q", cl), ("k", "hw_m1_k", cl), ("v", "hw_m1_v", cl), ("ms_pw", "hw_ms_pw_mid", "chunk"),
                            ("h1", "hw_m2_h1", "chunk"), ("h2", "hw_m2_h2", "chunk"), ("proj", "hw_m3_proj", cl),
                            ("mid", "hw_m3_mid", "chunk"), ("dw", "hw_p2_dw", "chunk"), ("pw", "hw_p3_pw", cl)]:
            p = os.path.join(args.dumps, f + ".txt")
            if os.path.exists(p):
                hw = dump_tokens(p, lay, args.par)
                a, b = (100 * np.mean(S[key] == hw) for S in (old, new))
                print(f"  {key:6s} {a:8.2f}% {b:9.2f}%")
        print("  (the 0526 h2 dump is the forced-zero value, so it matches neither column)")

    if compare_sw_golden(D, run(D, **FIXED), args.export, args.block):
        return

    if "y_golden" in D:
        g = D["y_golden"].reshape(T, 256)
        print(f"\nBlock {args.block} ({IMG}x{IMG} tokens) output vs y_golden (mean |golden| = {np.abs(g).mean():.1f} LSB):")
        cases = [("RTL 0526, 0524 export", D_old, {}),
                 ("+ fix sram_mid address", D_old, dict(mid_transposed=False)),
                 ("+ MLA handshake, h2, MS reorder", D, dict(FIXED, act="relu6", dw_act=False)),
                 ("+ expand hswish (DW_ACT=0)", D, dict(FIXED, dw_act=False)),
                 ("RTL now (DW_ACT=1) *", D, FIXED)]
        for name, data, kw in cases:
            d = np.abs(run(data, **kw)["y"] - g)
            print(f"  {name:36s} mean|d| = {d.mean():8.2f}   max = {d.max():6d}   rel = {100 * d.mean() / np.abs(g).mean():6.2f}%")
        print("  * only meaningful against a y_golden generated with the DW hswish (the 2026-05-24 one has none)")


if __name__ == "__main__":
    main()
