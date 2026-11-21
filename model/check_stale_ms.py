"""Model of the 2026-10-10 board bug (fixed in accelerator_top.v v4.1): in run N+1 the MS branch started
chunk 0 one cycle early (stale cl_groups_done) and computed q2 chunk 0 from the PREVIOUS block's q chunk 0.
Prints the predicted y error for every (previous block, current block) pair; it reproduces the board
(1 after 0 -> 1.0 LSB, 3 after 1 -> 3.4 LSB, blocks 0/2 and same-block repeats -> 0).

  python model/check_stale_ms.py [hw_export dir]
"""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rtl_model as M

exp = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "hw_export_512")
D = {b: M.load(exp, b) for b in range(4)}
S = {b: M.run(D[b], **M.FIXED) for b in range(4)}
for b in range(4):
    assert np.array_equal(S[b]["y"], D[b]["y_golden"].reshape(M.T, 256)), f"model != golden for block {b}"

def run_stale(cur, prev, chunks=(0,)):
    """Model of block `cur` where MS chunks in `chunks` are fed q/k/v of block `prev`."""
    Dc, Sc, Sp = D[cur], S[cur], S[prev]
    qkv_c = np.concatenate([Sc["q"], Sc["k"], Sc["v"]], 1)
    qkv_p = np.concatenate([Sp["q"], Sp["k"], Sp["v"]], 1)
    ms = Sc["ms_pw"].copy()
    for c in chunks:
        src = qkv_p[:, c*16:(c+1)*16]
        d = M.dw5(src, Dc["dw_w"][16384 + c*25:16384 + c*25 + 25], Dc["dw_b"][1024 + c])
        ms[:, c*16:(c+1)*16] = M.sat(M.rs(d @ Dc["ms_pw_w"][c*16:(c+1)*16]) + Dc["ms_pw_b"][c])
    heads = lambda z: np.concatenate(
        [M.mla(z[:, h*16:h*16+16], z[:, 256+h*16:272+h*16], z[:, 512+h*16:528+h*16], False) for h in range(16)], 1)
    h2 = heads(ms)
    x = Dc["x"][:M.T*16].reshape(M.T, 256)
    h = np.concatenate([Sc["h1"], h2], 1)
    proj = M.sat(M.rs(M.conv1x1(h, Dc["mla_w"], 32, 16, 12288, lambda oc, ic: oc*32 + ic)) + Dc["mla_b"][48:64].reshape(-1))
    mid = M.sat(x + proj)
    pre = M.sat(M.rs(M.conv1x1(mid, Dc["w"], 16, 64, 0, lambda oc, ic: oc*16 + ic)) + Dc["b"][:64].reshape(-1))
    inv = M.hswish_rtl(pre)
    dw = np.concatenate([M.dw5(inv[:, c*16:(c+1)*16], Dc["dw_w"][c*25:c*25+25], Dc["dw_b"][c]) for c in range(64)], 1)
    dw = M.hswish_rtl(dw)
    pw = M.sat(M.rs(M.conv1x1(dw, Dc["pw_w"], 64, 16, 0, lambda oc, ic: ic*16 + oc)) + Dc["pw_b"][:16].reshape(-1))
    y = M.sat(mid + pw)
    return y, ms, h2

g = lambda b: D[b]["y_golden"].reshape(M.T, 256)
print("y error of block CUR when MS chunk 0 is computed from block PREV's q chunk 0")
print("board: 1 after 0 -> 1.0 LSB, 3 after 1 -> 3.4 LSB, 0 after 1 -> 0, 2 after 3 -> 0, same block -> 0")
print(f"{'prev/cur':>8s}" + "".join(f"{c:>18d}" for c in range(4)))
for p in range(4):
    row = f"{p:>8d}"
    for c in range(4):
        y, ms, h2 = run_stale(c, p)
        d = np.abs(y - g(c))
        words = np.any((y != g(c)).reshape(M.T*16, 16), 1).sum()
        row += f"  mean {d.mean():5.2f} max {d.max():4d} w{words:4d}"
    print(row)

print("\nwhy: relu(q2 chunk 0) stats per block (own data) and h2 chunk 0 difference when stale")
for c in range(4):
    q2 = S[c]["ms_pw"][:, 0:16]
    print(f"  block {c}: relu(q2[:,0:16]) nonzero = {np.mean(q2 > 0)*100:5.1f}%  max = {q2.max()}  mean = {q2.mean():.1f}")
    for p in range(4):
        if p == c: continue
        y, ms, h2 = run_stale(c, p)
        print(f"     prev {p}: ms chunk0 diff mean {np.abs(ms[:,0:16]-S[c]['ms_pw'][:,0:16]).mean():6.2f}  "
              f"h2 chunk0 diff mean {np.abs(h2[:,0:16]-S[c]['h2'][:,0:16]).mean():6.2f} max {np.abs(h2[:,0:16]-S[c]['h2'][:,0:16]).max()}")
