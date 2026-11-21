# Implementation summary (RTL v4.1, final build)

Vivado 2023.2, device `xcvu37p-fsvh2892-2L-e` (VCU128), post-route. The design is the full block design: XDMA PCIe endpoint, SmartConnect, AXI BRAM controllers with their BRAMs, and the accelerator IP. The raw Vivado reports are not included; the numbers below are copied from them.

## Clock and timing

| Item | Value |
|---|---|
| Clock | 62.5 MHz (`axi_aclk` of the XDMA IP, 62,500,000 Hz), shared by the accelerator |
| Setup WNS | +0.532 ns, TNS 0.000 ns, 0 failing of 1,017,601 endpoints |
| Hold WHS | +0.009 ns, THS 0.000 ns, 0 failing endpoints |
| Pulse width WPWS | +0.094 ns, 0 failing endpoints |
| Worst setup path, whole design | XDMA user reset register to a host-side weight BRAM (reset recovery) |
| Worst setup path through the accelerator | +2.901 ns, also from the XDMA user reset (reset recovery into the conv cluster) |

The worst paths are reset-recovery paths; the worst register-to-register path inside the compute datapath was not reported on its own.

## Resources

| Resource | Full design | Used by the accelerator IP | Device total |
|---|---|---|---|
| LUT | 264,391 (20.28%) | 229,824 (17.63%) | about 1,304 k |
| FF | 273,825 (10.50%) | 219,290 (8.41%) | about 2,607 k |
| RAMB36 | 1,703 (84.47%) | 770 (38.19%) | 2,016 |
| RAMB18 | 18 (0.45%) | 14 (0.35%) | 4,032 |
| URAM | 0 | 0 | 960 |
| DSP | 4,755 (52.69%) | 4,755 (52.69%) | 9,024 |

The 933 RAMB36 outside the accelerator are the host-side BRAMs that hold the input, the ten weight and bias arrays and the output.

## Power (Vivado estimate, not measured)

| Item | Value |
|---|---|
| Total on-chip | 11.134 W |
| Dynamic | 7.747 W |
| Device static | 3.387 W |
| Junction temperature | 29.8 C |

At 231 GOPS of compute-only throughput (`results/board_v41/bench.json`), the estimate gives about 21 GOPS/W.
