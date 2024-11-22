set_svf -off
set_host_options -max_cores 8
set bench aes_cipher_top
open_lib ../Icc2Ndm/${bench}_nlib2
copy_block -from_block ${bench} -to_block ${bench}_test
open_block ${bench}_test
link_block

set fill_cell_list "*/F_FILL*"
source ../ECO_ChangeList/aes_cipher_top_eco_occupied.tcl
place_eco_cells -eco_changed_cells -legalize_only -legalize_mode minimum_physical_impact -remove_filler_references $fill_cell_list
set_app_options -name route.global.timing_driven    -value false
set_app_options -name route.track.timing_driven     -value false
set_app_options -name route.detail.timing_driven    -value false 
set_app_options -name route.global.crosstalk_driven -value false 
set_app_options -name route.track.crosstalk_driven  -value false
route_eco -utilize_dangling_wires true -reroute modified_nets_first_then_others -open_net_driven true
create_stdcell_fillers -lib_cells $fill_cell_list
connect_pg_net
report_global_timing
route_detail -incremental true
save_block
check_design -checks routes
#source ../icc2_eco_rpt.tcl
#exit
