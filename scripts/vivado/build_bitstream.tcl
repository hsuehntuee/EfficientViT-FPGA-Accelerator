# Headless build of the Vivado project: IP upgrade -> synth -> impl -> bitstream -> reports. Safe to re-run.
# The project (block design_1 with XDMA, AXI BRAM controllers and the packaged accelerator IP) is not part of
# this repository; set VIT_XPR to its .xpr file.
#
#   vivado -mode batch -source scripts/vivado/build_bitstream.tcl -tclargs status   # only print run state
#   vivado -mode batch -source scripts/vivado/build_bitstream.tcl -tclargs resume   # continue what is done
#   vivado -mode batch -source scripts/vivado/build_bitstream.tcl -tclargs fresh    # reset and rebuild
#   vivado -mode batch -source scripts/vivado/build_bitstream.tcl -tclargs reports  # impl already finished: only the reports
#
# Run it inside tmux; the log goes to build_<mode>.log next to the project, the summary to build_report.txt and the full
# timing / hierarchical utilisation / power reports to build_timing.rpt, build_util.rpt, build_power.rpt (same directory).

set here [file dirname [file normalize [info script]]]
if {![info exists ::env(VIT_XPR)]} { error "set VIT_XPR to the Vivado project (.xpr)" }
set proj [file normalize $::env(VIT_XPR)]
set mode [expr {[llength $argv] ? [lindex $argv 0] : "status"}]
set jobs 16
set_param general.maxThreads 16

open_project $proj
update_ip_catalog -rebuild

proc run_state {r} {
    set run [get_runs $r]
    return [format "%-8s status=%-40s progress=%-5s needs_refresh=%s" $r [get_property STATUS $run] [get_property PROGRESS $run] [get_property NEEDS_REFRESH $run]]
}
puts "================================================================"
puts [run_state synth_1]
puts [run_state impl_1]
puts "================================================================"

if {$mode eq "status"} { close_project; exit }

if {$mode eq "reports"} {
    if {![string match "*write_bitstream Complete*" [get_property STATUS [get_runs impl_1]]]} {
        puts "impl_1 is not finished ([get_property STATUS [get_runs impl_1]]); use resume"; close_project; exit 1
    }
    open_bd_design [get_files design_1.bd]
} else {

# ---- make sure the BD uses the re-packaged IP (no-op if already current)
open_bd_design [get_files design_1.bd]
set ip [get_ips -filter {NAME =~ design_1_Efficient_vit_v2_v1_0_0*}]
puts "IP revision in BD before upgrade: [get_property IP_REVISION $ip]"
catch {upgrade_ip $ip}
puts "IP revision in BD after upgrade : [get_property IP_REVISION $ip]"
validate_bd_design
save_bd_design

# ---- decide what to run
set synth_done [expr {[string match "*Complete*" [get_property STATUS [get_runs synth_1]]] && ![get_property NEEDS_REFRESH [get_runs synth_1]]}]
if {$mode eq "fresh" || !$synth_done} {
    puts ">> synthesis needed (mode=$mode, synth complete=$synth_done): reset_run synth_1"
    reset_run synth_1
} else {
    puts ">> synth_1 is up to date, continuing with implementation"
    reset_run impl_1
}
launch_runs impl_1 -to_step write_bitstream -jobs $jobs
wait_on_run impl_1
puts [run_state synth_1]
puts [run_state impl_1]
}

# ---- reports
set bit [glob -nocomplain [file join [get_property DIRECTORY [get_runs impl_1]] *.bit]]
open_run impl_1
set rpt [file join [file dirname $proj] build_report.txt]
set fh [open $rpt w]
puts $fh "bitstream: $bit"
puts $fh "axi_aclk FREQ_HZ: [get_property CONFIG.FREQ_HZ [get_bd_pins xdma_host/axi_aclk]]"
set rdir [file dirname $proj]
report_timing_summary -max_paths 2 -file [file join $rdir build_timing.rpt]
report_utilization -hierarchical -hierarchical_depth 7 -hierarchical_percentages -file [file join $rdir build_util.rpt]
report_power -file [file join $rdir build_power.rpt]
# timing numbers (the old grep lost the value rows)
foreach {lbl opt} {setup -setup hold -hold} {
    set tp [lindex [get_timing_paths -max_paths 1 {*}$opt] 0]
    puts $fh "worst $lbl (design): slack [get_property SLACK $tp]  [get_property STARTPOINT_PIN $tp] -> [get_property ENDPOINT_PIN $tp]"
    set tp [lindex [get_timing_paths -max_paths 1 {*}$opt -through [get_cells -hier -filter {NAME =~ *Efficient_vit_v2_v1_0_0}]] 0]
    puts $fh "worst $lbl (accelerator): slack [get_property SLACK $tp]  [get_property STARTPOINT_PIN $tp] -> [get_property ENDPOINT_PIN $tp]"
}
foreach {f pat} [list [file join $rdir build_timing.rpt] {WNS|TNS|Design Timing Summary|^\s+-?[0-9]+\.[0-9]+\s+-?[0-9]+\.[0-9]+\s+[0-9]+\s+[0-9]+} \
                      [file join $rdir build_util.rpt] {Instance |design_1_wrapper|Efficient_vit|u_accelerator_top|u_cluster|arr\[0\]|u_litemla|u_ms_|u_dw_engine|u_line_buf|inst_sram|u_add} \
                      [file join $rdir build_power.rpt] {Total On-Chip Power|Dynamic \(W\)|Device Static|Junction Temperature}] {
    set g [open $f r]
    while {[gets $g line] >= 0} { if {[regexp $pat $line]} { puts $fh [regsub -all {\s+} $line " "] } }
    close $g
}
close $fh
puts "================================================================"
puts [exec cat $rpt]
puts "================================================================"
close_project
exit
