set starttime [clock seconds]
echo "INFORM: Start job at: " [clock format $starttime -gmt false]
set is_si_enabled false

set top_design mc_top

set link_library "* ../Timing_Lib/scc14nsfp_90sdb_9tc16_rvt_ssg_v0p63_125c_ccs.db"


set netlist "../Icc2Output/${top_design}_route.v"
set sdc "../Icc2Output/${top_design}_route.sdc"
set spef "../Icc2Output/${top_design}.rcworst_125_1.08_1.08_1_1.spef"

source -e -v ../pt_variable.tcl

set NET_FILE $netlist 
set SDC_FILE $sdc
set SPEF_FILE $spef


set_app_var read_parasitics_load_locations true
set_app_var eco_allow_filler_cells_as_open_sites true
###################################################################
read_verilog  $NET_FILE
#current_design $top_design
link


read_parasitics -keep_capacitive_coupling  -format SPEF  $SPEF_FILE
#read_parasitics -keep_capacitive_coupling -format gpd -eco /home/jiajiexu/mylib/StarRC_Lab/SMIC14/eco_step_1.gpd



source -e -v $SDC_FILE

#create_clock -period 0 -name clk1 [get_port clk]

set_propagated_clock [all_clocks]
#set_propagated_clock [get_generated_clock *]

set timing_remove_clock_reconvergence_pessimism true

#set_clock_uncertainty 0.1 [all_clocks] -setup
#set_clock_uncertainty 0.2  [all_clocks] -hold
     
set timing_disable_clock_gating_checks true  
set timing_report_unconstrained_paths true


update_timing -full

set_eco_options -physical_icc2_lib ../Icc2Ndm/${top_design}_nlib -physical_icc2_blocks ${top_design}.design
set_app_var eco_allow_filler_cells_as_open_sites false
fix_eco_timing -physical_mode open_site -type setup -methods size_cell

write_changes -format icctcl -output ../ECO_ChangeList/${top_design}_eco_physical.tcl

set endtime   [clock seconds]
echo "INFORM: End job at: " [clock format $endtime -gmt false]
set pwd [pwd]
set runtime "[format %02d [expr ($endtime - $starttime)/3600]]:[format %02d [expr (($endtime - $starttime)%3600)/60]]:[format %02d [expr ((($endtime - $starttime))%3600)%60]]"
echo [format "%-15s %-2s %-70s" " | runtime" "|" "$runtime"]
exit
