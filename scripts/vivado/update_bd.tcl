# Update design_1.bd for the MS branch. Run with VIT.xpr open, after re-packaging the accelerator IP:
#     source <repo>/scripts/vivado/update_bd.tcl
#
# What it does (each step is skipped if already done, so it is safe to re-run):
#   1. upgrade the Efficient_vit_v2 cell to the re-packaged IP (adds the bram_ms_pw_* pins)
#   2. add ctrl_ms_pw_w/b (axi_bram_ctrl) + bram_ms_pw_w/b (blk_mem_gen) + const_we_ms_pw_w/b,
#      configured like the existing ctrl_pw_b / bram_pw_b / const_we_pw_b
#   3. smc_main NUM_MI 10 -> 12, M10 -> ctrl_ms_pw_w, M11 -> ctrl_ms_pw_b
#   4. addresses: ms_pw_b 0xD4000000/8K, ms_pw_w 0xD6000000/32K,
#      ctrl_dw_w 64K -> 1M (MS depthwise weights sit at byte 524288..562687)
#   5. validate + save
# host/xdma_run.py BRAM_MAP already uses these addresses.

set IP   Efficient_vit_v2_v1_0_0
set XDMA xdma_host
set SMC  smc_main

update_ip_catalog -rebuild
open_bd_design [get_files design_1.bd]

# ---- 1. upgrade the accelerator IP
if {[llength [get_bd_pins -quiet $IP/bram_ms_pw_w_addr]] == 0} {
    set ips [get_ips -filter "NAME =~ design_1_${IP}*"]
    upgrade_ip $ips
}
if {[llength [get_bd_pins -quiet $IP/bram_ms_pw_w_addr]] == 0} {
    error "$IP still has no bram_ms_pw_* pins: re-package the accelerator IP first (and check Settings -> IP -> Repository)"
}

# ---- 2. new BRAMs
proc set_cfg {cell args} {
    # one property at a time: in BRAM_Controller mode some of these are derived and read-only
    foreach {k v} $args {
        if {[catch {set_property CONFIG.$k $v $cell} msg]} { puts "  (skip $k on $cell: derived)" }
    }
}

proc add_bram {name} {
    global IP XDMA
    if {[llength [get_bd_cells -quiet bram_$name]]} { puts "bram_$name already present"; return }

    set ctrl [create_bd_cell -type ip -vlnv xilinx.com:ip:axi_bram_ctrl:4.1 ctrl_$name]
    set_cfg $ctrl DATA_WIDTH 256 SINGLE_PORT_BRAM 1

    set bram [create_bd_cell -type ip -vlnv xilinx.com:ip:blk_mem_gen:8.4 bram_$name]
    set_cfg $bram Memory_Type True_Dual_Port_RAM Enable_B Use_ENB_Pin EN_SAFETY_CKT false \
        Operating_Mode_A WRITE_FIRST Operating_Mode_B WRITE_FIRST \
        Read_Width_A 256 Write_Width_A 256 Read_Width_B 256 Write_Width_B 256 \
        Register_PortA_Output_of_Memory_Primitives false \
        Register_PortB_Output_of_Memory_Primitives false \
        Register_PortB_Output_of_Memory_Core false

    set const [create_bd_cell -type ip -vlnv xilinx.com:ip:xlconstant:1.1 const_we_$name]
    set_cfg $const CONST_WIDTH 32 CONST_VAL 0

    # host side (port A)
    connect_bd_intf_net [get_bd_intf_pins ctrl_$name/BRAM_PORTA] [get_bd_intf_pins bram_$name/BRAM_PORTA]
    connect_bd_net [get_bd_pins $XDMA/axi_aclk]    [get_bd_pins ctrl_$name/s_axi_aclk]
    connect_bd_net [get_bd_pins $XDMA/axi_aresetn] [get_bd_pins ctrl_$name/s_axi_aresetn]

    # accelerator side (port B, read only)
    connect_bd_net [get_bd_pins $IP/bram_${name}_clk]  [get_bd_pins bram_$name/clkb]
    connect_bd_net [get_bd_pins $IP/bram_${name}_en]   [get_bd_pins bram_$name/enb]
    connect_bd_net [get_bd_pins $IP/bram_${name}_addr] [get_bd_pins bram_$name/addrb]
    connect_bd_net [get_bd_pins bram_$name/doutb]      [get_bd_pins $IP/bram_${name}_rdata]
    connect_bd_net [get_bd_pins const_we_$name/dout]   [get_bd_pins bram_$name/web]
}

add_bram ms_pw_w
add_bram ms_pw_b

# ---- 3. SmartConnect
if {[get_property CONFIG.NUM_MI [get_bd_cells $SMC]] < 12} {
    set_property CONFIG.NUM_MI 12 [get_bd_cells $SMC]
}
foreach {mi name} {M10_AXI ms_pw_w M11_AXI ms_pw_b} {
    if {[llength [get_bd_intf_nets -quiet -of_objects [get_bd_intf_pins ctrl_$name/S_AXI]]] == 0} {
        connect_bd_intf_net [get_bd_intf_pins $SMC/$mi] [get_bd_intf_pins ctrl_$name/S_AXI]
    }
}

# ---- 4. address map
set space [get_bd_addr_spaces $XDMA/M_AXI]
assign_bd_address -force -offset 0xD4000000 -range 8K  -target_address_space $space [get_bd_addr_segs ctrl_ms_pw_b/S_AXI/Mem0]
assign_bd_address -force -offset 0xD6000000 -range 32K -target_address_space $space [get_bd_addr_segs ctrl_ms_pw_w/S_AXI/Mem0]
set_property range 1M [get_bd_addr_segs $XDMA/M_AXI/SEG_ctrl_dw_w]

# ---- 5. validate, save
validate_bd_design
save_bd_design

puts "================================================================"
puts " xdma_host/M_AXI address map (must match host/xdma_run.py BRAM_MAP):"
foreach seg [lsort [get_bd_addr_segs -of_objects $space]] {
    puts [format "   %-34s %s  %s" [get_property NAME $seg] [get_property OFFSET $seg] [get_property RANGE $seg]]
}
puts " Next: Generate Bitstream (synth_1 / impl_1 will be out of date)."
puts "================================================================"
