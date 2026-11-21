# evit_hw_golden

Integer reference model and test-vector generator for the four Stage 4 blocks of EfficientViT-Seg-B1 (Cityscapes), written from the PyTorch network. It loads the checkpoint, runs the FP32 network up to Stage 4, rounds the block 0 input to Q.5 and then runs the four blocks in pure integer arithmetic with the same shifts, rounding, saturation and division as the RTL. Everything is in `evit_hw_golden.py`.

The arithmetic rules are listed in [`docs/DESIGN.md`](../../docs/DESIGN.md#fixed-point-arithmetic). The second integer model, `model/rtl_model.py`, was written from the RTL; both agree at every stage of every block.

## Files

| Path | Contents |
|---|---|
| `evit_hw_golden.py` | `gen`, `compare` and `verify` commands |
| `assets/test.png` | Test image (1024x1024, resized to the network input) |
| `assets/b1-cityscapes.pt` | Checkpoint, not in git: `efficientvit_seg_b1_cityscapes.pt` from the mit-han-lab/efficientvit release, renamed |
| `reference/SHA256SUMS` | Hashes of the 176 files that `gen` writes at the default 128x128 size; `verify` checks against them |
| `requirements.txt` | Python packages |

The efficientvit package is found in this order: installed package, `$EFFICIENTVIT_ROOT`, parent directories. `scripts/run_evit_gen.py` sets `EFFICIENTVIT_ROOT` to `third_party/efficientvit` and stubs optional packages that the generator never uses.

## Usage

```bash
# 512x512 vectors as used by the RTL (N_TOKENS = 256)
python scripts/run_evit_gen.py gen --size 512 --out hw_export_512 --no-plot

# Compare a hardware or simulation dump with a golden file (.bin / .hex / .mem / .txt / .npy)
python scripts/run_evit_gen.py compare hw_export_512/block_0_y_golden.bin y_hw.bin
python scripts/run_evit_gen.py compare hw_export_512/block_0_s1_qkv.hex hw_m1_k.txt --hw-chunk-major 16 --golden-part k

# Regenerate the 128x128 set and check it against reference/SHA256SUMS
python scripts/run_evit_gen.py verify
```

`compare` and `verify` exit with code 1 on a mismatch.

## Output files

For each block `i` (0 to 3):

- Weights, one 256-bit word = 16 int16 values: `block_i_w` (MBConv expand 1x1), `block_i_dw_w` (MBConv depthwise, then the multi-scale 5x5 depthwise from word 16384), `block_i_pw_w` (MBConv point 1x1), `block_i_mla_w` (QKV 1x1 then projection), `block_i_ms_pw_w` (multi-scale grouped 1x1), and the matching `_b` bias files. Conv and BatchNorm are fused before quantisation.
- Activations, Q.5 int16 in `[token][chunk][lane]` order: `block_i_x` (input), `s1_qkv`, `s2_agg`, `s3_att`, `s4_proj`, `s5_mid`, `s6_exp`, `s7_dw`, `s8_pw` and `block_i_y_golden` (output).
- `.bin` is little-endian int16, `.hex` is one 256-bit word per line for `$readmemh` (element 0 in the least significant bits).
- `report.json` holds per-block cosine similarity and MSE against FP32 and saturation counts.

RTL stage dumps are `[chunk][token]`; `--hw-chunk-major 16` reorders them, and `--golden-part` selects the channels a given dump holds (`q`, `k`, `v`, `ms_hw`, `att_h1`, `att_h2`).

## Reproducibility

The four blocks are exact integer arithmetic. The block 0 input comes from the FP32 front end, so a different torch version or CPU can move a few values by 1 LSB after rounding, and the difference then propagates. Compare hardware against one fixed vector set rather than regenerating it on each machine. On the machine used for this repository (torch 2.6, CPU), `host/run_image.py --backend model` reproduced the stored block inputs and outputs byte for byte.
