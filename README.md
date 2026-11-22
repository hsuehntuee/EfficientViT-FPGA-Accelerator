<h1 align="center">Bit-Exact FPGA Accelerator for EfficientViT-B1 Stage 4</h1>
<p align="center">A SystemVerilog accelerator for the four linear-attention blocks of EfficientViT-B1 Stage 4 on an AMD Xilinx VCU128, driven from a PyTorch segmentation network over PCIe: 126,999 cycles per block, and 800 of 800 on-board block runs bit-exact against the integer reference.</p>
<p align="center"><img src="docs/figures/architecture.png" width="720" alt="Architecture"></p>

## Overview

EfficientViT-Seg-B1 is a semantic segmentation network whose last backbone stage (Stage 4, `backbone.stages[3]` in the code) holds four identical blocks. Each block is a multi-scale ReLU linear attention module (LiteMLA) followed by an MBConv. This project moves those four blocks onto an FPGA and leaves the rest of the network in PyTorch on the host CPU. A 512x512 input gives 256 tokens of 256 channels at Stage 4, and one RTL design runs all four blocks by loading different weights.

The datapath uses int16 fixed point (activations Q10.5, weights Q4.11). The RTL was written to match two integer models written separately: one follows the RTL (`model/rtl_model.py`) and one follows the PyTorch network (`model/evit_hw_golden/evit_hw_golden.py`). The board output matches them word for word. Between design versions v2 and v4, cycles per block fell from 1,055,995 to 126,999 at the same 512x512 workload, an 8.3x reduction. Most of the gain came from a cluster of eight 16x16 systolic arrays with shadow weights; the rest came from overlapping the attention, multi-scale and depthwise passes with the 1x1 convolutions. [`docs/DESIGN.md`](docs/DESIGN.md) explains the design decisions, the version history and the debugging.

## Results

All numbers are for one Stage 4 block at 512x512 input unless stated otherwise.

| Metric | Value | Source |
|---|---|---|
| Cycles per block | 126,999 (identical on every run, all four blocks) | `results/board_v41/bench.json`, `results/board_v41/results.json` (on-board cycle counter) |
| Cycle reduction, v2 to v4, same workload | 1,055,995 to 126,999 cycles, 8.3x | version table in `docs/DESIGN.md` (simulation cycle counts) |
| Conv cluster utilisation | 84% (106,496 of 126,999 cycles) | cycle breakdown in `docs/DESIGN.md`; `docs/figures/cycle_breakdown.png` |
| Correctness on the board | 800 of 800 block runs bit-exact (200 KITTI images x 4 blocks); mixed block order `0 1 1 3 3 2 0 2 1` also 0 mismatches | `results/board_v41/results.json`; mixed-order runs described in `docs/DESIGN.md` |
| Compute time per block | 2.05 ms (median of 20, start to done, includes polling) | `results/board_v41/bench.json` |
| Throughput, compute only | 115.6 GMAC/s = 231 GOPS | `results/board_v41/bench.json` |
| Stage 4 per image, weights resident in BRAM | 16.9 ms median (59.2 images/s) over the 200-image batch | `results/board_v41/results.json` |
| Weight upload per block | 19.7 ms for 2.33 MB (118 MB/s; the XDMA IP was still at its default Gen1 x1 setting) | `results/board_v41/bench.json` |
| CPU baseline, four blocks, FP32 | 120.5 ms on an i9-13900H (PyTorch 2.6, median of 10) vs. 8.2 ms board compute (14.7x) | measured on a laptop on 2026-10-10 and recorded in `docs/DESIGN.md`; no log file in the repository |
| Clock and timing | 62.5 MHz, WNS +0.532 ns, no failing endpoints | `results/implementation_summary.md` |
| Resources (full design, XCVU37P) | LUT 264,391 (20.3%), FF 273,825, RAMB36 1,703 (84.5%, 770 of them in the accelerator), DSP 4,755 (52.7%), URAM 0 | `results/implementation_summary.md` |
| Power | 11.13 W total on-chip (Vivado post-route estimate, not measured) | `results/implementation_summary.md` |
| Segmentation accuracy, 200 KITTI images | mIoU 44.52% with the integer path vs. 43.26% FP32; mean per-image difference -0.22 points, uncertainty +/-0.4 points | `docs/DESIGN.md`; manifest version (43.25% / 44.50%) in `results/kitti512_manifest.json` |

The mIoU was computed with the integer model (`--backend model`). Because the board output is bit-exact with that model, it is also the accuracy of the hardware path. The 8.3x figure compares two RTL versions on the same workload. An earlier 128x128 version (v1, 271,019 cycles) ran a different workload and is not used for comparison.

<p align="center"><img src="docs/figures/cycle_breakdown.png" width="720" alt="Cycle breakdown and per-image time"></p>

## Architecture

The figure at the top shows the system as built. Module names come from `rtl/`.

- Host: `host/hybrid.py` runs the full EfficientViT-Seg-B1 network in PyTorch and swaps the four Stage 4 blocks for a backend: `fpga` (the board), `model` (the integer model) or `file` (recorded board outputs). It writes the block input and weights over PCIe, starts the accelerator, polls `done` and reads the result back.
- Board: an XDMA PCIe endpoint feeds AXI BRAM controllers that hold the input `x`, ten weight and bias arrays and the output `y`. An AXI-Lite slave exposes `start`, `done` and a cycle counter.
- `conv_cluster`: eight weight-stationary 16x16 systolic arrays compute eight output-channel chunks of a 1x1 convolution at once. Each PE has a shadow weight register, so the next tile's weights load while the current tile computes and tiles run back to back without a flush. The cluster runs all four 1x1 convolutions of a block (QKV, projection, expand, point), which account for 92% of the block's multiply-accumulates.
- `litemla_core2`: ReLU linear attention for 16-dimensional heads with a 64-bit divider. Two context banks let head h+1 accumulate its k/v context while head h streams its queries.
- Depthwise engines: two `dw_conv_engine` instances with line buffers. One handles the 5x5 multi-scale branch (followed by a grouped 1x1 in `conv_engine_top`), the other the MBConv 5x5 depthwise with hardswish. Both trail the cluster group by group, so their work hides under the QKV and expand passes.
- `accelerator_top`: its FSM sequences LOAD_X, M1 (QKV), ATT_Q, ATT_MS, M3 (projection), P2 (expand), P3 (point) and P4 (residual adds into `y`).

## Repository layout

| Path | What it holds |
|---|---|
| `rtl/` | The RTL (v4.1): `accelerator_top.v`, `conv_cluster.v`, `litemla_core2.v`, `dw_conv_engine.v`, `dw_line_buffer.v`, `conv_engine_top.v`, `psum_buffer.v`, `systolic_core.v`, `bram_wrapper.v`, the block-design wrapper `Efficient_vit_v2_v1_0.v` and its AXI-Lite slave. `litemla_core.v` is the earlier single-bank attention core, kept for reference and no longer instantiated |
| `tb/` | `tb_fused.sv`: self-checking testbench that runs one or more blocks back to back from the test vectors, dumps every stage and reports cycles and the error against the golden output |
| `model/` | `rtl_model.py` (NumPy model written from the RTL), `check_stale_ms.py` (model of the cross-run bug fixed in v4.1), and `evit_hw_golden/` (integer model and test-vector generator written from PyTorch; see its README) |
| `host/` | Board scripts, the hybrid network, dataset evaluation and figure scripts; see `host/README.md` |
| `scripts/` | `run_evit_gen.py` (runs the vector generator with the bundled efficientvit code) and `vivado/`: `sim_block0.tcl` (batch simulation), `update_bd.tcl` (block-design changes and address map), `build_bitstream.tcl` (headless build) |
| `constraints/` | VCU128 PCIe clock and reset pins |
| `results/` | Board results (`board_v41/`), the 200-image KITTI manifest, a single-image run of the integer model (`test_model/`), and `implementation_summary.md` |
| `docs/` | `DESIGN.md` and `figures/` |
| `third_party/efficientvit/` | The seven files of mit-han-lab/efficientvit needed to build EfficientViT-Seg-B1 (Apache-2.0; see `NOTICE`) |

Not in the repository: the test vectors `hw_export_512/` (60 MB, regenerated with the command below), the checkpoint, the KITTI images, and the Vivado project with its block design and packaged IP. The Vivado reports are summarised in `results/implementation_summary.md`.

## Getting started

Python 3.10 or later with `pip install numpy torch torchvision timm pillow matplotlib`.

Two inputs have to be fetched first:

- the EfficientViT-Seg-B1 Cityscapes checkpoint (`efficientvit_seg_b1_cityscapes.pt` from the mit-han-lab/efficientvit release), saved as `model/evit_hw_golden/assets/b1-cityscapes.pt`;
- the test vectors (blocks 0 to 3: inputs, weights and the golden output of every stage): `python scripts/run_evit_gen.py gen --size 512 --out hw_export_512 --no-plot`. Block 0's input comes from the FP32 front end, so another torch version can shift a few values by 1 LSB; generate the set once and keep it for all comparisons.

These commands were run for this README on a Windows laptop (Python 3.12, torch 2.6, CPU only), using the vector set and checkpoint from the original project:

```bash
# NumPy RTL model vs. the PyTorch-derived golden files, every stage of every block (100.00%, max |d| = 0 for blocks 0 to 3)
python model/rtl_model.py --block 0      # repeat for 1, 2, 3

# Reproduce the cross-run stale-state bug in the model (predicts 0.99 LSB for block 1 after 0, 3.46 LSB for block 3 after 1)
python model/check_stale_ms.py

# One image end to end with the Stage 4 blocks on the integer model (about 20 s); all four blocks bit-exact, 99.16% mask agreement with FP32
python host/run_image.py --image model/evit_hw_golden/assets/test.png --out results/test_model --backend model
```

The block inputs and outputs written by the last command were byte-identical to the stored vector set.

These need tools or hardware and were not re-run here:

- RTL simulation (Vivado 2023.2, xsim): `vivado -mode batch -source scripts/vivado/sim_block0.tcl -tclargs hw_export_512 0` builds a throw-away project from `rtl/` and `tb/` and should report 126,999 cycles and 0 LSB error. A second argument such as `13` runs blocks 1 and 3 back to back without reset.
- Bitstream build: `update_bd.tcl` and `build_bitstream.tcl` run against the Vivado project (block design with XDMA, SmartConnect, AXI BRAM controllers and the packaged accelerator IP), which is not part of this repository.
- Board runs (VCU128 with the XDMA driver): `host/xdma_run.py`, `host/bench.py`, `host/run_batch.py`.
- Dataset accuracy (KITTI semantic labels): `host/get_dataset.py`, `host/eval_dataset.py`, `host/export_batch.py`.

`host/README.md` lists the full command sequence.

## Team and my role

I (Yen-Hung Hsueh, `hsuehntuee`) designed and verified the RTL, wrote both integer models, the host software and the board scripts, and ran the board and dataset measurements. The work was advised by Prof. Chia-Hsiang Yang in the Digital Circuits and Systems Lab at National Taiwan University.

## Status

The version in this repository, v4.1, runs all four blocks on the board inside the full network and matches the integer models bit for bit, at 62.5 MHz with WNS +0.532 ns.

A follow-up version, v5, is in progress (November 2026) and is not in this repository yet. It removes the first three limits of v4.1:

- The slow PCIe link came from a setting left at its default: the XDMA IP was generated for one lane at 2.5 GT/s. v5 configures it for Gen3 x8.
- The weights of all four blocks stay resident in UltraRAM, and one start command runs one to four blocks.
- The accelerator has its own clock, and the datapath was re-pipelined for it.

v5 is bit-exact in RTL simulation against the same test vectors at 123,193 cycles per block, and its implementation meets timing at 200 MHz on the same device with no margin (WNS 0.000 ns). It has not run on the board yet, so this README states no v5 speed. The v5 sources will be added after the board test.

Still not done: INT8 arithmetic, the 16-array cluster, the classification stem and Stages 1 and 2, and token batching. The power number is a Vivado estimate. Some RTL comments are in Chinese.

## Acknowledgements / License

`third_party/efficientvit/` holds seven files from [mit-han-lab/efficientvit](https://github.com/mit-han-lab/efficientvit), licensed under Apache-2.0 (`third_party/efficientvit/LICENSE`); `third_party/efficientvit/NOTICE` lists what was changed. The model and checkpoint are from Cai et al., "EfficientViT: Lightweight Multi-Scale Attention for High-Resolution Dense Prediction", ICCV 2023. Accuracy was measured on images from the KITTI semantic segmentation benchmark.

No license has been chosen for the rest of the code in this repository.
