import DataBuilder
# from sklearn.preprocessing import MinMaxScaler
import dgl
from dgl.data.utils import save_graphs
from dgl.data.utils import load_graphs
import torch
import numpy as np
import pickle
import sys
import os
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)
import Global_var

def TimingGraphTrans(design, rebuilt=False, verbose=False):
    print(f'Building {design} Timing Graph.')
    
    # Load or rebuild timing data
    if rebuilt:
        CellArcs, _ = DataBuilder.BuildTimingArc(design)
        PtCells = DataBuilder.BuildPtCells(design)
        PtNets = DataBuilder.BuildPtNets(design)
        Critical_Paths = DataBuilder.BuildPtRpt(design)
    else:
        CellArcs, _ = DataBuilder.LoadTimingArc(design)
        PtCells = DataBuilder.LoadPtCells(design)
        PtNets = DataBuilder.LoadPtNets(design)
        Critical_Paths = DataBuilder.LoadPtRpt(design)

    # Initialize node mappings and edges
    nodes = {cell: idx for idx, cell in enumerate(PtCells.keys())}
    nodes_rev = {idx: cell for cell, idx in nodes.items()}
    U_forward, V_forward = [], []

    # Build the Pt Netlist graph, nodes are the cells
    for net in PtNets.values():
        inpins = [inpin.split('/')[0] for inpin in net.inpins if inpin.split('/')[0] in PtCells]
        outpins = [outpin.split('/')[0] for outpin in net.outpins if outpin.split('/')[0] in PtCells]
        
        for inpin in inpins:
            for outpin in outpins:
                U_forward.append(nodes[inpin])
                V_forward.append(nodes[outpin])
                
    # Initialize node features
    num_nodes = len(nodes)
    # outslew, tns, wns, delay
    nodes_feature_bidirectional = np.zeros((num_nodes, 4), dtype=np.float32)
    # inslew
    nodes_feature_forward = np.zeros((num_nodes, 1), dtype=np.float32)
    # cap, res
    nodes_feature_backward = np.zeros((num_nodes, 2), dtype=np.float32)


    # Populate node features from CellArcs
    for arc in CellArcs.values():
        if arc:
            cell = arc.from_pin.split('/')[0]
            if cell in nodes.keys():
                node_number = nodes[cell]

                # Update bidirectional, forward, and backward features
                outslew = max(arc.outslew)
                inslew = max(arc.inslew)
                delay = max(arc.Delay)

                nodes_feature_bidirectional[node_number] = [
                    max(nodes_feature_bidirectional[node_number][0], outslew),
                    0, 0,
                    max(nodes_feature_bidirectional[node_number][3], delay)
                ]
                nodes_feature_forward[node_number][0] = max(nodes_feature_forward[node_number][0], inslew)
                nodes_feature_backward[node_number] = [arc.loadCap, arc.loadRes]

    # Collect Critical Path data
    Endpoint = {nodes[path.Cellarcs[-1].name.split('->')[0].split('/')[0]]: path.slack for path in Critical_Paths if path.Cellarcs}
    FlipFlops = [nodes[path.Endpoint.replace('', '')] for path in Critical_Paths if path.Endpoint.replace('', '') in nodes]

    # Build graph
    G = dgl.graph((U_forward, V_forward), num_nodes=num_nodes)
    G.ndata['bidirection_feature'] = torch.from_numpy(nodes_feature_bidirectional)
    G.ndata['forward_feature'] = torch.from_numpy(nodes_feature_forward)
    G.ndata['backward_feature'] = torch.from_numpy(nodes_feature_backward)
    # G.ndata['Pt_slack'] = torch.from_numpy(np.zeros((num_nodes, 2), dtype=np.float32)) # tns ,wns for pt optimization

    # # DFS function to calculate TNS and WNS
    # def dfs(node, visited, current_slack, start_endpoint):
    #     if node in visited or node in FlipFlops or (node in Endpoint and node != start_endpoint):
    #         return

    #     visited.add(node)
    #     tns = current_slack
    #     wns = current_slack

    #     for pred in G.predecessors(node):
    #         dfs(pred.item(), visited, current_slack, start_endpoint)
            
    #     G.ndata['bidirection_feature'][node, 1] += tns  # Accumulate TNS
    #     G.ndata['bidirection_feature'][node, 2] = min(G.ndata['bidirection_feature'][node, 2], wns)  # Update WNS

    # # Perform DFS for each endpoint
    # for endpoint, slack in Endpoint.items():
    #     visited = set()
    #     G.ndata['bidirection_feature'][endpoint, 1] = slack  # TNS for endpoint is its slack
    #     G.ndata['bidirection_feature'][endpoint, 2] = slack  # WNS for endpoint is its slack
    #     dfs(endpoint, visited, slack, endpoint)

    def dfs_iterative(start_node, initial_slack, start_endpoint, G, FlipFlops, Endpoint):
        # Initialize stack with start node, initial slack, and visited set
        stack = [(start_node, initial_slack, initial_slack)]
        visited = set()

        while stack:
            node, tns, wns = stack.pop()

            # Skip if already visited, or if it is a FlipFlop or different endpoint
            if node in visited or node in FlipFlops or (node in Endpoint and node != start_endpoint):
                continue

            # Mark node as visited
            visited.add(node)

            # Accumulate TNS and update WNS for this node
            G.ndata['bidirection_feature'][node, 1] += tns
            G.ndata['bidirection_feature'][node, 2] = min(G.ndata['bidirection_feature'][node, 2], wns)

            # Process predecessors with updated slack
            for pred in G.predecessors(node):
                # Pass along the accumulated TNS and update WNS as the min of current path slack
                stack.append((pred.item(), initial_slack, initial_slack))

    # Perform DFS iteratively for each endpoint
    for endpoint, slack in Endpoint.items():
        # Start the iterative DFS
        dfs_iterative(endpoint, slack, endpoint, G, FlipFlops, Endpoint)
    
    
    # Save the graph and node dictionary
    save_dir = os.path.join(Global_var.Trans_Data_Path, 'TimingGraph')
    os.makedirs(save_dir, exist_ok=True)
    save_graphs(os.path.join(save_dir, f'{design}_TimingGraph.bin'), G)
    with open(os.path.join(save_dir, f'{design}_nodeDict.sav'), 'wb') as f:
        pickle.dump([nodes, nodes_rev], f)

    if verbose:
        print(f'{design} Timing Graph complete!')
    
def LoadTimingGraph(design, rebuild=False, verbose = False):
    # Load Timing Graph
    if verbose:
        print(f'Loading {design} Timing Graph.')
    Save_Dir = Global_var.Trans_Data_Path + 'TimingGraph' 
    save_path = os.path.join(Save_Dir, design + '_TimingGraph.bin')
    if not os.path.exists(save_path) or rebuild:
        TimingGraphTrans(design, True)
    G, _ = load_graphs(save_path)
    if verbose:
        print(f'{design} Timing Graph loaded!')
    return G[0]

def LoadNodeDict(design, rebuild = False, verbose = False):
    # node number and number node
    if verbose:
        print(f'Loading {design} Node Dict.')
    Save_Dir = Global_var.Trans_Data_Path + 'TimingGraph'
    save_path = os.path.join(Save_Dir, design + '_nodeDict.sav')
    if not os.path.exists(save_path) or rebuild:
        TimingGraphTrans(design, True)
    with open(save_path, 'rb') as f:
        nodes, nodes_rev = pickle.load(f)
    if verbose:
        print(f'{design} Node Dict loaded!')
    return nodes, nodes_rev