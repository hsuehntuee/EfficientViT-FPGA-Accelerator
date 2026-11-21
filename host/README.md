# host/

Python scripts that drive the board and run the hybrid network. Run them from the repository root; they find the other folders by relative path.

| Script | What it does | Needs |
|---|---|---|
| `hybrid.py` | Shared code: XDMA access, loading the integer model, `HybridModel` (the full network on the CPU with the four Stage 4 blocks sent to a backend: `fpga`, `model` or `file`), IoU helpers | numpy; torch for every backend except `fpga` |
| `xdma_run.py` | One block on the board: write weights and `x`, start, poll `done`, read `y`, compare word by word with `y_golden` | numpy, XDMA driver |
| `bench.py` | Times weight upload, `x` upload, compute and readback separately (median of `--iters` runs) and derives the clock from the cycle counter | numpy, XDMA driver |
| `run_batch.py` | Runs an image batch made by `export_batch.py` block by block (weights loaded once per block) and checks every image and block for bit-exactness | numpy, XDMA driver |
| `run_image.py` | One image end to end: image, four Stage 4 blocks on the chosen backend, CPU head, mask, cosine similarity per block, timings | torch |
| `eval_dataset.py` | mIoU of FP32 and of the hardware path over a labelled image set | torch |
| `export_batch.py` | Writes the per-image block inputs (`block_N_x.bin`) and a manifest with per-image mIoU for the board | torch |
| `get_dataset.py` | Downloads KITTI semantic (no account) or Cityscapes (account needed) into the Cityscapes layout | numpy, pillow |
| `make_figures.py` | Figures for the design notes: error budget, MAC and cycle breakdown, latency, cosine, per-class IoU, board cycles | matplotlib |
| `make_mask_examples.py` | Best, median and worst mask examples by mIoU | torch |

Install: `pip install numpy torch torchvision timm pillow matplotlib`. The integer model is loaded from `model/evit_hw_golden/` (override with `--golden-repo` or `$EVIT_HW_GOLDEN`) and the network code from `third_party/efficientvit/`.

## Without a board

```bash
# One image, Stage 4 on the integer model (about 20 s on a laptop CPU)
python host/run_image.py --image model/evit_hw_golden/assets/test.png --out results/test_model --backend model

# Accuracy over the 200 labelled KITTI images (hardware path = integer model, since the board is bit-exact with it)
python host/get_dataset.py kitti --out data/kitti
python host/eval_dataset.py --images "data/kitti/leftImg8bit/val/*/*_leftImg8bit.png" --cityscapes --out results/kitti512 --backend model

# Batch for the board
python host/export_batch.py --images "data/kitti/leftImg8bit/val/*/*_leftImg8bit.png" --cityscapes --out batch_kitti512
```

`eval_dataset.py` upsamples the logits to the label resolution before scoring (43.26% FP32, 44.52% hardware path on KITTI). The manifest written by `export_batch.py` scores at 512x512 against nearest-downsampled labels (43.25% and 44.50%); the two differ by less than 0.02 points. `results/kitti512_manifest.json` is the manifest of the 200-image run.

## On the board host (VCU128 with the XDMA driver loaded)

```bash
ls /dev/xdma0_user /dev/xdma0_h2c_0 /dev/xdma0_c2h_0           # BRAM addresses are in BRAM_MAP in xdma_run.py
for b in 0 1 1 3 3 2 0 2 1; do python3 host/xdma_run.py --dir hw_export_512 --block $b; done   # mixed order: expect 0 mismatches, 126,999 cycles
python3 host/bench.py --dir hw_export_512 --iters 20 --out bench.json
python3 host/run_batch.py --batch batch_kitti512 --weights hw_export_512 --out batch_kitti512_results
sudo lspci -d 10ee: -vv | grep -E "LnkSta:|LnkCap:"             # link width and speed
```

If `done` never rises and the cycle count is 0, the start write did not arrive over AXI-Lite; a non-zero count that stops changing means the FSM is stuck. If a block is not bit-exact, save its output with `--save` and compare it stage by stage with `model/rtl_model.py`.

## How the numbers were measured

- Correctness: every run compares all 4,096 output words (256 bits each) with the integer model; bit-exact means zero words differ.
- Time: `time.perf_counter()` on the host around each step. Compute time runs from the start write until `done` is seen, so it includes polling and is an upper bound. The cycle counter clears on start and is read after `done`.
- CPU baseline: PyTorch FP32, same image, median of 10 runs (i9-13900H, torch 2.6, CPU).
- Power: Vivado post-route estimate, not a measurement.
