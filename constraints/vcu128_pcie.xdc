# =========================================================
# VCU128: PCIe reference clock and PERST# pins (the constraints used for the build)
# =========================================================
set_property PACKAGE_PIN AR15 [get_ports pcie_refclk_clk_p]
set_property PACKAGE_PIN AR14 [get_ports pcie_refclk_clk_n]
create_clock -name pcie_refclk -period 10.000 [get_ports pcie_refclk_clk_p]

set_property PACKAGE_PIN BF41 [get_ports pcie_rst_n]
set_property IOSTANDARD LVCMOS12 [get_ports pcie_rst_n]
set_property PULLUP true [get_ports pcie_rst_n]
