from typing import List, Optional
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch as th
import dgl
from sklearn.preprocessing import MinMaxScaler

def find_module_dir(module_name, search_paths):
    """
    find the target module in the given paths
    :param module_name: target module name
    :param search_paths: list of search paths
    :return: the path which contains the target module or None
    """
    for path in search_paths:
        module_path = os.path.join(path, module_name + ".py")
        if os.path.exists(module_path):
            return path
    return None


import os
import sys
main_script_path = os.path.abspath(sys.argv[0])  # absolute path to the python script being run
main_script_dir = os.path.dirname(main_script_path)  # main script directory
parent_dir = os.path.dirname(main_script_dir) # root directory
search_paths = [
    main_script_dir,
    parent_dir,
    '../'
]
module_dir = find_module_dir('Global_var', search_paths)
if module_dir:
    sys.path.append(module_dir)
else:
    raise ImportError("Project Main dir not found by the 'rl_cpd' environment file. Please check the working directory!")
import Global_var
import DataBuilder
import TimingGraphTrans
import Interaction
from RL_CTD_Space import GraphSpace
import time


def report_time(elapsed_time, stage):
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = elapsed_time % 60
    print(f'Runtime for {stage}: {hours} hours, {minutes} minutes, {seconds:.2f} seconds')

class RL_CTD(gym.Env):
    metadata = {"render_modes": ["human"]}
    
    def __init__(self, render_mode: Optional[str] = None, current_design: Optional[str] = 'mc_top', ReBuildRpt: Optional[bool] = True):
        start_time = time.time()
        super(RL_CTD, self).__init__()
        print(f'Initializing RL_CTD environment for {current_design}')
        self.render_mode = render_mode  # online monitor
        self.current_design = current_design  # the design to be optimized
        
        if ReBuildRpt:
            Interaction.ReBuildAllRpt(self.current_design)
        
        # Layout Partition
        self.row, self.column = 8, 8
        print(f'Layout Partioning to {self.row} rows, {self.column} columns')
        self.Sub_CellLayout, self.CellLocation = DataBuilder.BuildCellData(self.current_design, self.row, self.column, verbose=True)
        # The count of cells in one region
        self.RegionCellCount = np.zeros((self.row, self.column), dtype=np.int32)  # 9x9 grid for counting cells
        # Initialize RegionID array with region numbers starting from left-top to right-bottom
        self.RegionID = np.arange(self.row * self.column).reshape(self.row, self.column)
        
        # Pin Layout Partition
        self.Sub_PinLayout = DataBuilder.BuildPinData(self.current_design, self.row, self.column, verbose=True)
        
        # H, V congestion Partition
        self.Sub_H_congestion, self.Sub_V_congestion = DataBuilder.BuildCongestionData(self.current_design, 128, self.row, self.column, verbose=True)
        
        # The RegionID array holds the region numbers from 0 to (row*column - 1) from left-top to right-bottom order
        ''' self.cell_region_mapping self.region_cell_mapping Initialed here'''
        self._Cell_Cluster()
        
        # Get the eco command for three optimization strategy
        # Data type: {cellname:command_line}
        # self.strategy_eco, self.strategy_occ, self.strategy_phy
        self.strategy = DataBuilder.GetSizedCellCommand(self.current_design)
        
        scaler = MinMaxScaler()
        # Caculate total negative slack of each region
        ''' self.graph self.nodes Initialed here'''
        self.Region_TNS = self._Get_Region_TNS(self.current_design)
        
        RegionTNS_flattened = self.Region_TNS.flatten().reshape(-1, 1)
        RegionTNS_normalized = scaler.fit_transform(RegionTNS_flattened)
        RegionTNS_normalized = RegionTNS_normalized.reshape(self.row, self.column)
        # used as obs in this version
        self.RegionTNS_normalized = RegionTNS_normalized
        # print(f'Initialized DT_ECO environment with observation space: \n<{self.observation_space}>\nand action space: \n<{self.action_space}>')
        
        # obs list generation
        self.obs_list = self._get_obs_list()
        
        # Get the DRC count of each region
        self.Region_DRC = self._Get_Region_DRC(self.current_design)
        RegionDRC_flattened = self.Region_DRC.flatten().reshape(-1, 1)
        RegionDRC_normalized = scaler.fit_transform(RegionDRC_flattened)
        RegionDRC_normalized = RegionDRC_normalized.reshape(self.row, self.column)
        # used as obs in this version
        self.RegionDRC_normalized = RegionDRC_normalized
        
        # Initilizing obs space, action space and reward space
        
        self.observation_space = spaces.Dict({
            'timing_graph': GraphSpace(
                self.graph
            ),
            'physical_image':spaces.Box(low=0, high=1, shape=(4, 64, 64), dtype=np.float32)
        })
        # self.action_space_list = [0, 0.33, 0.66, 1] # 0: no optimization,  1: eco, 2: occ, 3: phy
        self.action_space = spaces.Discrete(3)
        self.reward_space = spaces.Box(
            low=np.array([0, 0]),
            high=np.array([2, 2]),
            dtype=np.float32,
        )
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        report_time(elapsed_time, 'Initialization')
        
    def render(self): # needed
        pass
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        # self.current_step = 0
        # self.Region_Action = np.zeros((self.row, self.column), dtype=np.float32)
        return self.obs_list
    
    def _Get_Region_DRC(self, design):
        return DataBuilder.BuildRegionalDrc(design, self.row, self.column)
    
    def _Get_Region_TNS(self, design):
        # Get gate wise timing graph
        self.graph = TimingGraphTrans.LoadTimingGraph(design, rebuild=True)
        self.nodes, _ = TimingGraphTrans.LoadNodeDict(design)
        RegionTNS = np.zeros((self.row, self.column), dtype=np.float32)
        for cell in self.cell_region_mapping:
            if cell in self.nodes.keys():
                node_num = self.nodes[cell]
                row, column = self._Get_Row_Column(self.cell_region_mapping[cell])
                RegionTNS[row, column] += self.graph.ndata['bidirection_feature'][node_num, 1]
        print("Region TNS value (Left-Top to Right-Bottom):")
        print(RegionTNS)
        return RegionTNS
                
    def _Get_Row_Column(self, region):
        row = region // self.column
        column = region % self.column
        return row, column
    
    def _Cell_Cluster(self):
        # Initialize a dictionary to store cell region assignments
        self.cell_region_mapping = {}
        self.region_cell_mapping = {}

        # Iterate over each cell and its location to determine its region
        for cell, location in self.CellLocation.items():
            min_x = min(point[0] for point in location)
            max_x = max(point[0] for point in location)
            min_y = min(point[1] for point in location)
            max_y = max(point[1] for point in location)

            # Calculate grid indices for the bounding box
            row_start = max(0, min(int(min_y * self.row), self.row - 1))
            row_end = max(0, min(int(max_y * self.row), self.row - 1))
            col_start = max(0, min(int(min_x * self.column), self.column - 1))
            col_end = max(0, min(int(max_x * self.column), self.column - 1))

            # Find the region with the most overlap
            max_overlap_region = None
            max_overlap_area = 0

            for r in range(row_start, row_end + 1):
                for c in range(col_start, col_end + 1):
                    # Calculate the bounding box for this region
                    region_min_x = c / self.column
                    region_max_x = (c + 1) / self.column
                    region_min_y = r / self.row  # Adjust to match bottom-up indexing
                    region_max_y = (r + 1) / self.row        # Adjust to match bottom-up indexing

                    # Calculate overlap area between the cell and the region
                    overlap_width = max(0, min(max_x, region_max_x) - max(min_x, region_min_x))
                    overlap_height = max(0, min(max_y, region_max_y) - max(min_y, region_min_y))
                    overlap_area = overlap_width * overlap_height

                    # Check if this region has the most overlap
                    if overlap_area > max_overlap_area:
                        max_overlap_area = overlap_area
                        max_overlap_region = self.RegionID[r, c]

            # Assign the cell to the region with the most overlap
            self.cell_region_mapping[cell] = max_overlap_region
            # Assign the region with the cell
            if max_overlap_region not in self.region_cell_mapping:
                self.region_cell_mapping[max_overlap_region] = []
            self.region_cell_mapping[max_overlap_region].append(cell)
            # Increment the region counter in the RegionCellCount grid
            self.RegionCellCount[max_overlap_region // self.column, max_overlap_region % self.column] += 1

        # Print the RegionID array
        print("Region ID Grid (Left-Top to Right-Bottom):")
        print(self.RegionID)

        # Print the RegionCellCount grid with region numbers
        print("Region Cell Count Grid:")
        print(self.RegionCellCount)
        return
    
    def _get_tns_drc(self, ECO=True):
        if ECO:
            wns, tns = DataBuilder.BuildGlobalTimingData(self.current_design+'_eco')
            drc = DataBuilder.BuildDrcNumber(self.current_design+'_eco')
        else:
            wns, tns = DataBuilder.BuildGlobalTimingData(self.current_design)
            drc = DataBuilder.BuildDrcNumber(self.current_design)
        return [wns, tns, drc]

    def _get_obs_list(self):
        obs_list = []
        # get the subgraph and Image data of each region, normalize the graph data first
        subgraph_list = []
        timing_graph = self.graph.clone()
        scaler = MinMaxScaler()
        timing_graph.ndata['bidirection_feature'] = th.from_numpy(scaler.fit_transform(timing_graph.ndata['bidirection_feature'].numpy()))
        timing_graph.ndata['forward_feature'] = th.from_numpy(scaler.fit_transform(timing_graph.ndata['forward_feature'].numpy()))
        timing_graph.ndata['backward_feature'] = th.from_numpy(scaler.fit_transform(timing_graph.ndata['backward_feature'].numpy()))
        for region in range(self.row * self.column):
            cells = self.region_cell_mapping[region]
            nodes = [self.nodes[cell] for cell in cells if cell in self.nodes.keys()]
            subgraph = dgl.node_subgraph(timing_graph, nodes)
            subgraph_list.append(subgraph)
        
        subimage_list = []
        for cell_layout, pin_layout, H_congestion, V_congestion in zip(self.Sub_CellLayout, self.Sub_PinLayout, self.Sub_H_congestion, self.Sub_V_congestion):
            combined_layout = np.stack([cell_layout, pin_layout, H_congestion, V_congestion], axis=0)
            subimage_list.append(th.from_numpy(combined_layout))
        
        for graph, image in zip(subgraph_list, subimage_list):
            obs_list.append({'timing_graph':graph, 'physical_image':image})
            
        return obs_list
    
    def step(self, action_list):
        ECO_Commands = []
        for i in range(len(action_list)):
            action = action_list[i]
            for cell in self.region_cell_mapping[i]:
                if cell in self.strategy[int(action)].keys():
                    for command in self.strategy[int(action)][cell]:
                        ECO_Commands.append(command)
                            
        Interaction.Write_Icc2_ECO_Command(self.current_design, ECO_Commands)
        # run icc2 and pt
        Interaction.RLCTD_Iteration(self.current_design)
        wns, tns, drc = self._get_tns_drc(ECO=False)
        wns_eco, tns_eco, drc_eco = self._get_tns_drc(ECO=True)
        rewards = np.array([1 - drc_eco/max(drc, drc_eco), 1 - tns_eco/min(tns, tns_eco)])

        return rewards, (wns_eco, tns_eco, drc_eco)

if __name__ == "__main__": 
    start_time = time.time()
    env = RL_CTD()
    obs_list = env.reset()
    for i in range(0):
        env.render()
        action_list = []
        for obs in obs_list:
            action_list.append(env.action_space.sample())
        rewards, state_vector = env.step(action_list)
        # np.set_printoptions(edgeitems=10, threshold=20, precision=4, suppress=False, linewidth=np.inf)
        print(f'step {i} state: {state_vector}')
    env.close()
    end_time = time.time()
    elapsed_time = end_time - start_time
    report_time(elapsed_time, 'One Eposide')
    # time = elapsed_time/10
    # print(f'average time per step:{time}')
