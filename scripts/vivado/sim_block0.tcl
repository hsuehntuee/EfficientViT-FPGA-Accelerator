# Behavioural simulation of the accelerator (tb_fused.sv, top = tb_accelerator_top) in a throw-away
# project built from rtl/*.v and tb/tb_fused.sv directly (no IP packager, no IP catalog).
# Copies the per-stage dumps (hw_*.txt) and simulate.log to sim_out/sim_<yyyymmdd>_seq<SEQ>/.
#
#   vivado -mode batch -source scripts/vivado/sim_block0.tcl
#   vivado -mode batch -source scripts/vivado/sim_block0.tcl -tclargs <hex dir> [<block sequence, e.g. 01>]
#   The sequence runs those blocks back to back without reset (board behaviour); dumps are from the last block.
#
# Default hex directory: hw_export_512 (512x512 input, N_TOKENS=256).

set here    [file dirname [file normalize [info script]]]
set repo    [file normalize [file join $here .. ..]]
set tmp     [file normalize [file join $repo .sim_tmp]]
set hex_dir [file normalize [file join $repo hw_export_512]]
if {[llength $argv] > 0} { set hex_dir [file normalize [lindex $argv 0]] }
# NOTE: no [expr] here -- expr would turn "01" into the integer 1 and the sequence would collapse to a single run.
set seq "0"
if {[llength $argv] > 1} { set seq [string trim [lindex $argv 1]] }
if {![regexp {^[0-3]+$} $seq]} { error "block sequence must be digits 0-3, e.g. 01 (got '$seq')" }
set out_dir [file normalize [file join $repo sim_out sim_[clock format [clock seconds] -format %Y%m%d]_seq$seq]]
set part xcvu37p-fsvh2892-2L-e

set hex {}
foreach n {x w b dw_w dw_b pw_w pw_b ms_pw_w ms_pw_b mla_w mla_b y_golden} {
    foreach f [glob -nocomplain [file join $hex_dir block_*_$n.hex]] { lappend hex $f }
}
if {[llength $hex] == 0} { error "no block_*_*.hex in $hex_dir" }
set rtl [glob [file join $repo rtl *.v]]
set tb  [file join $repo tb tb_fused.sv]
puts "RTL: [llength $rtl] files, hex: [llength $hex] files, sequence: $seq"

file delete -force $tmp
create_project sim_tmp $tmp -part $part -force
add_files -norecurse -fileset sources_1 $rtl
add_files -norecurse -fileset sim_1 [concat $tb $hex]
set_property FILE_TYPE {Memory File} [get_files -of_objects [get_filesets sim_1] -filter {NAME =~ *.hex}]
set_property top tb_accelerator_top [get_filesets sim_1]
set_property top_lib xil_defaultlib [get_filesets sim_1]
set_property -name {xsim.simulate.runtime} -value {-all} -objects [get_filesets sim_1]
set_property -name {xsim.simulate.xsim.more_options} -value "-testplusarg SEQ=$seq" -objects [get_filesets sim_1]
update_compile_order -fileset sources_1
update_compile_order -fileset sim_1

launch_simulation
set sim_dir [file join [get_property DIRECTORY [current_project]] [current_project].sim sim_1 behav xsim]
close_sim

file mkdir $out_dir
foreach f [concat [glob -nocomplain [file join $sim_dir hw_*.txt]] [glob -nocomplain [file join $sim_dir simulate.log]]] {
    file copy -force $f $out_dir
}
close_project

puts "================================================================"
puts " dumps copied to $out_dir"
puts " check: python model/rtl_model.py --rtl v3 --par 8 --export $hex_dir --block [string index $seq end] --dumps $out_dir"
puts "================================================================"
