import sys
import os
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)
import Global_var
import Verilog_Parser
import TimingLib_Parser
import Drc_Parser
import Def_Parser
import NetRpt_Parser
import CellRpt_Parser
import Density_Parser
import Congestion_Parser
import PtRpt_Parser
import PtCellRpt_Parser
import PtNetRpt_Parser
import PtDelayRpt_Parser
import PtGlobalRpt_Parser
import SDF_Parser
import EndPoint_Parser
import PortLocation_Parser
import PinLocation_Parser
import ECOChangeList_Parser
import ECOCommand_Parser
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

Save_Path = Global_var.Saved_Data_Path
# the scaler factor of linewidth
scaler_base = 20

# cell layout builder
def BuildCellLayout(design, verbose=False):
    if verbose:
        print(f'Building {design} cell layout.')
    Rpt = Global_var.Icc2Rpt_Path + design + '_cell.rpt'
    Def = Global_var.Def_Path + design + '_route.def'
    coreArea, _, _, _ = Def_Parser.Read_def(Def)
    cellList = CellRpt_Parser.Read_CellRpt(Rpt)
    cell_location_dict = {cell.name: (cell.pin1, cell.pin2, cell.pin3, cell.pin4) for cell in cellList}
    fig, ax = plt.subplots()
    core_polygon = patches.Polygon(coreArea, closed=True, fill=True, color='black')
    ax.add_patch(core_polygon)
    for cell in cellList:
        cell_polygon = patches.Polygon([cell.pin1, cell.pin2, cell.pin3, cell.pin4], 
                                closed=True, fill=True, edgecolor='black', facecolor='white', linewidth=1/(coreArea[1][1]/scaler_base))
        ax.add_patch(cell_polygon)
    ax.axis('off')

    # plan plot area
    ax.set_xlim(coreArea[0][0], coreArea[2][0])
    ax.set_ylim(coreArea[0][1], coreArea[2][1])
    ax.set_aspect('equal') # x=y
    save_dir = Save_Path + 'CellLayout/' + design
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # save_path = os.path.join(save_dir, design + '_fig.pkl')
    # with open(save_path, 'wb') as f:
    #     pickle.dump(fig, f)

    save_path = os.path.join(save_dir, f'CellLayout.png')
    plt.savefig(save_path, dpi=300, format='png', bbox_inches='tight', pad_inches=0)
    # plt.show()
    plt.close(fig)

    # convert image to 512*512
    img = Image.open(save_path).convert('L')
    img = img.resize((512, 512), Image.Resampling.LANCZOS)
    # img.show()
    save_path = os.path.join(save_dir, f'CellLayout.npy')
    img_array = np.array(img, dtype=np.float32)

    # nomalization
    scaler = MinMaxScaler()
    img_array = scaler.fit_transform(img_array.reshape(-1, 1)).reshape(512, 512)
    # np.set_printoptions(threshold=np.inf)
    # print(img_array)
    np.save(save_path, img_array)

    save_path = os.path.join(save_dir, 'CellDict.sav')
    normalized_cell_location_dict = {}
    for cell_name, pins in cell_location_dict.items():
        normalized_pins = [
            (pin[0] / coreArea[2][0], pin[1] / coreArea[2][1]) for pin in pins
        ]
        normalized_cell_location_dict[cell_name] = normalized_pins
    with open(save_path, 'wb') as f:
        pickle.dump(normalized_cell_location_dict, f)
    if verbose:
        print(f'{design} cell layout complete!')

def BuildCellData(design, row, column, verbose=False):
    if verbose:
        print(f'Building {design} cell layout with {row} rows and {column} columns.')

    Rpt = Global_var.Icc2Rpt_Path + design + '_cell.rpt'
    Def = Global_var.Def_Path + design + '_route.def'
    coreArea, _, _, _ = Def_Parser.Read_def(Def)
    cellList = CellRpt_Parser.Read_CellRpt(Rpt)
    cell_location_dict = {cell.name: (cell.pin1, cell.pin2, cell.pin3, cell.pin4) for cell in cellList}

    # Initialize an empty matrix for the target area, filled with 0 for the background
    target_height, target_width = row * 64, column * 64
    img_array = np.zeros((target_height, target_width), dtype=np.float32)  # Fill with 0 as background

    # Map coordinates to the image matrix based on coreArea size and cell coordinates
    x_scale = target_width / (coreArea[2][0] - coreArea[0][0])
    y_scale = target_height / (coreArea[2][1] - coreArea[0][1])

    # Draw each cell in the image matrix
    for cell in cellList:
        cell_coords = [
            (int(pin[0] * x_scale), int(pin[1] * y_scale)) for pin in cell_location_dict[cell.name]
        ]
        xs, ys = zip(*cell_coords)
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        img_array[y_min:y_max, x_min:x_max] = 1.0  # Grayscale level for cells

    # Normalize the image
    scaler = MinMaxScaler()
    img_array = scaler.fit_transform(img_array.reshape(-1, 1)).reshape(target_height, target_width)
    
    # Split the image into row*column sub-images of 64x64 pixels
    sub_images = []
    sub_h, sub_w = 64, 64  # Fixed sub-image size of 64x64

    for r in range(row):
        for c in range(column):
            sub_img = img_array[r * sub_h:(r + 1) * sub_h, c * sub_w:(c + 1) * sub_w]
            sub_images.append(sub_img)

    # Normalize cell locations for saving
    normalized_cell_location_dict = {}
    for cell_name, pins in cell_location_dict.items():
        normalized_pins = [
            (pin[0] / coreArea[2][0], pin[1] / coreArea[2][1]) for pin in pins
        ]
        normalized_cell_location_dict[cell_name] = normalized_pins

    if verbose:
        print(f'{design} cell layout complete! Created {len(sub_images)} sub-images of 64x64 pixels.')
    
    return sub_images, normalized_cell_location_dict

def LoadCellLayout(design, verbose=False):
    if verbose:
        print(f'Loading {design} cell layout.')
    save_dir = Save_Path + 'CellLayout/' + design
    save_path = os.path.join(save_dir, 'CellLayout.npy')
    if not os.path.exists(save_path):
        BuildCellLayout(design)
    img_array = np.load(save_path)
    save_path = os.path.join(save_dir, 'CellDict.sav')
    with open(save_path, 'rb') as f:
        cell_location_dict = pickle.load(f)
    if verbose:
        print(f'{design} cell layout loaded!')
    return img_array, cell_location_dict

# def LoadAndShowFigure(design):
#     save_dir = Save_Path + 'CellLayout'
#     save_path = os.path.join(save_dir, design + '_fig.pkl')
#     with open(save_path, 'rb') as f:
#         fig = pickle.load(f)
#     plt.show()

# metal layer builder
def BuildMetalLayer(design, verbose=False):
    if verbose:
        print(f'Building {design} metal layer.')
    Rpt = Global_var.Icc2Rpt_Path + design + '_net.rpt'
    Def = Global_var.Def_Path + design + '_route.def'
    rpt_nets = NetRpt_Parser.Read_NetRpt(Rpt)
    coreArea, _, _, layers = Def_Parser.Read_def(Def)
    layer_seg = {}
    for layer in layers:
        layer_seg[layer] = []
    for net in rpt_nets:
        for seg in net.segs:
            layer_seg[seg.metal].append((seg.width, (seg.sx, seg.sy), (seg.ex, seg.ey)))
    
    MetalLayer = {}
    save_dir = Save_Path + 'MetalLayers/' + design
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for layer, segments in layer_seg.items():
        fig, ax = plt.subplots()
        ax.set_xlim(coreArea[0][0], coreArea[2][0])
        ax.set_ylim(coreArea[0][1], coreArea[2][1])
        ax.set_aspect('equal')

        for (width, (sx, sy), (ex, ey)) in segments:
            dx = ex - sx
            dy = ey - sy
            length = (dx**2 + dy**2)**0.5

            nx = -dy / length
            ny = dx / length

            # end point
            x1 = sx + width/2 * nx
            y1 = sy + width/2 * ny
            x2 = sx - width/2 * nx
            y2 = sy - width/2 * ny
            x3 = ex - width/2 * nx
            y3 = ey - width/2 * ny
            x4 = ex + width/2 * nx
            y4 = ey + width/2 * ny

            # plot
            segment_polygon = patches.Polygon(((x1, y1), (x2, y2), (x3, y3), (x4, y4)), 
                                                closed=True, edgecolor='black', facecolor='white', linewidth=1/(coreArea[1][1]/scaler_base))
            ax.add_patch(segment_polygon)

        # no axis
        ax.axis('off')
        
        save_path = os.path.join(save_dir, f'{layer}.png')
        plt.savefig(save_path, dpi=300, format='png', bbox_inches='tight', pad_inches=0)
        # plt.show()

        # pickle_save_path = os.path.join(save_dir, f"{layer}_fig.pkl")
        # with open(pickle_save_path, 'wb') as f:
        #     pickle.dump(fig, f)
        plt.close(fig)

        img = Image.open(save_path).convert('L')
        img = img.resize((512, 512), Image.Resampling.LANCZOS)
        
        img_array = np.array(img, dtype=np.float32) 

        scaler = MinMaxScaler()
        img_array = scaler.fit_transform(img_array.reshape(-1, 1)).reshape(512, 512)
        # np.set_printoptions(threshold=np.inf)
        # print(img_array)

        MetalLayer[layer] = img_array

    npy_save_path = os.path.join(save_dir, f'MetalLayer.npz')
    np.savez(npy_save_path, **MetalLayer)
    if verbose:
        print(f'{design} metal layer complete!')
        

def LoadMetalLayer(design, verbose = False):
    if verbose:
        print(f'Loading {design} metal layer.')
    save_dir = Save_Path + 'MetalLayers/' + design
    npy_save_path = os.path.join(save_dir, f'MetalLayer.npz')
    if not os.path.exists(npy_save_path):
        BuildMetalLayer(design)
    loaded_data = np.load(npy_save_path)
    MetalLayer = {key: loaded_data[key] for key in loaded_data}
    if verbose:
        print(f'{design} metal layer loaded!')
    return MetalLayer

# def LoadAndShowFigure(pickle_file):
#     with open(pickle_file, 'rb') as f:
#         fig = pickle.load(f)
#     plt.show()

# def BuildRouteCongestion(design, scale, verbose=False):
#     if verbose:
#         print(f'Building {design} route congestion of scale {scale}.')

#     Rpt = Global_var.Icc2Rpt_Path + design + '_congestion_' + str(scale) + '.rpt'
#     H_congestion, V_congestion, Layer_congestion = Congestion_Parser.ReadRouteCongestion(Rpt, scale)
#     scaler = MinMaxScaler()

#     # normalization
#     H_congestion_reshaped = H_congestion.reshape(-1, 1)
#     H_congestion_normalized = scaler.fit_transform(H_congestion_reshaped).reshape(H_congestion.shape)
#     V_congestion_reshaped = V_congestion.reshape(-1, 1)
#     V_congestion_normalized = scaler.fit_transform(V_congestion_reshaped).reshape(V_congestion.shape)            
#     Layer_congestion_normalized = {}
#     for key, value in Layer_congestion.items():
#         value_reshaped = value.reshape(-1, 1)
#         Layer_congestion_normalized[key] = scaler.fit_transform(value_reshaped).reshape(value.shape)

#     plt.imshow(H_congestion_normalized, cmap='Reds', interpolation='nearest', origin='lower')
#     # add color bar
#     plt.colorbar()
#     # turn off axis
#     # plt.axis('off')
#     save_dir = Save_Path + 'RouteCongestion/'+ design
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)
#     save_path = os.path.join(save_dir, f'H_congestion_{scale}.png')
#     plt.savefig(save_path, dpi=300, format='png', bbox_inches='tight')
#     # plt.show()
#     plt.close()

#     plt.imshow(V_congestion_normalized, cmap='Blues', interpolation='nearest', origin='lower')
#     # add color bar
#     plt.colorbar()
#     # turn off axis
#     # plt.axis('off')
#     save_path = os.path.join(save_dir, f'V_congestion_{scale}.png')
#     plt.savefig(save_path, dpi=300, format='png', bbox_inches='tight')
#     # plt.show()
#     plt.close()

#     save_path = os.path.join(save_dir, f'Route_congestion_{scale}.npz')
#     np.savez(save_path,
#         H_congestion=H_congestion_normalized,
#         V_congestion=V_congestion_normalized,
#         **Layer_congestion_normalized)
#     if verbose:
#         print(f'{design} route congestion of sclae {scale} complete!')

def BuildRouteCongestion(design, scale, verbose=False):
    if verbose:
        print(f'Building {design} route congestion of scale {scale} with {row} rows and {column} columns.')

    Rpt = Global_var.Icc2Rpt_Path + design + '_congestion_' + str(scale) + '.rpt'
    H_congestion, V_congestion = Congestion_Parser.ReadRouteCongestion(Rpt, scale)
    scaler = MinMaxScaler()

    # Normalization
    H_congestion_normalized = scaler.fit_transform(H_congestion.reshape(-1, 1)).reshape(H_congestion.shape)
    V_congestion_normalized = scaler.fit_transform(V_congestion.reshape(-1, 1)).reshape(V_congestion.shape)

    # Plot and save H_congestion
    save_dir = Save_Path + 'RouteCongestion/' + design
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    plt.imshow(H_congestion_normalized, cmap='Reds', interpolation='nearest', origin='lower')
    plt.colorbar()
    save_path = os.path.join(save_dir, f'H_congestion_{scale}.png')
    plt.savefig(save_path, dpi=300, format='png', bbox_inches='tight')
    plt.close()

    # Plot and save V_congestion
    plt.imshow(V_congestion_normalized, cmap='Blues', interpolation='nearest', origin='lower')
    plt.colorbar()
    save_path = os.path.join(save_dir, f'V_congestion_{scale}.png')
    plt.savefig(save_path, dpi=300, format='png', bbox_inches='tight')
    plt.close()

    if verbose:
        print(f'{design} route congestion of scale {scale} complete with {row * column} sub-images!')
    

def BuildCongestionData(design, scale, row, column, verbose=False):
    if verbose:
        print(f'Building {design} route congestion of scale {scale} with {row} rows and {column} columns.')

    # Read the raw congestion data
    Rpt = Global_var.Icc2Rpt_Path + design + '_congestion_' + str(scale) + '.rpt'
    H_congestion, V_congestion = Congestion_Parser.ReadRouteCongestion(Rpt, scale)
    scaler = MinMaxScaler()

    # Normalize congestion data
    H_congestion_normalized = scaler.fit_transform(H_congestion.reshape(-1, 1)).reshape(H_congestion.shape)
    V_congestion_normalized = scaler.fit_transform(V_congestion.reshape(-1, 1)).reshape(V_congestion.shape)

    # Resize the congestion data to match the target dimensions (row * 64, column * 64)
    target_height, target_width = row * 64, column * 64
    H_congestion_resized = np.array(
        Image.fromarray(H_congestion_normalized).resize((target_width, target_height), Image.Resampling.LANCZOS)
    )
    V_congestion_resized = np.array(
        Image.fromarray(V_congestion_normalized).resize((target_width, target_height), Image.Resampling.LANCZOS)
    )

    # Split the resized congestion data into row*column sub-images of 64x64 pixels
    sub_h, sub_w = 64, 64
    H_sub_images = []
    V_sub_images = []

    for r in range(row):
        for c in range(column):
            H_sub_img = H_congestion_resized[r * sub_h:(r + 1) * sub_h, c * sub_w:(c + 1) * sub_w]
            V_sub_img = V_congestion_resized[r * sub_h:(r + 1) * sub_h, c * sub_w:(c + 1) * sub_w]
            H_sub_images.append(H_sub_img)
            V_sub_images.append(V_sub_img)

    # save_path = "img_array_output.txt"  # print the image array
    # np.set_printoptions(threshold=np.inf)
    # with open(save_path, 'w') as f:
    #     f.write(np.array2string(H_congestion_resized, separator=',', max_line_width=np.inf))
    
    if verbose:
        print(f'{design} route congestion of scale {scale} complete with {row * column} sub-images!')

    return H_sub_images, V_sub_images

# def LoadRouteCongestion(design, scale, verbose=False):
#     if verbose:
#         print(f'Loading {design} route congestion of scale {scale}.')
#     save_dir = Save_Path + 'RouteCongestion/' + design
#     save_path = os.path.join(save_dir, f'Route_congestion_{scale}.npz')
#     if not os.path.exists(save_path):
#         BuildRouteCongestion(design, scale)

#     loaded_data = np.load(save_path)
#     Hcongestion = loaded_data['H_congestion']
#     Vcongestion = loaded_data['V_congestion']
#     Layercongestion = {key: loaded_data[key] for key in loaded_data if key not in ['H_congestion', 'V_congestion']}
#     if verbose:
#         print(f'{design} route congestion of scale {scale} loaded!')
#     return Hcongestion, Vcongestion, Layercongestion

def BuildCellDensity(design, scale, verbose=False):
    if verbose:
        print(f'Building {design} cell density of scale {scale}.')

    Rpt = Global_var.Icc2Rpt_Path + design + '_density_' + str(scale) + '.rpt'
    Cell_density = Density_Parser.ReadCellDensity(Rpt, scale)

    scaler = MinMaxScaler()
    # normalization
    Cell_density_reshaped = Cell_density.reshape(-1, 1)
    Cell_density_normalized = scaler.fit_transform(Cell_density_reshaped).reshape(Cell_density.shape)
    plt.imshow(Cell_density_normalized, cmap='Greens', interpolation='nearest', origin='lower')
    # add color bar
    plt.colorbar()
    # turn off axis
    # plt.axis('off')

    save_dir = Save_Path + 'CellDensity/'+ design
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    save_path = os.path.join(save_dir, f'Cell_Density_{scale}.png')
    plt.savefig(save_path, dpi=300, format='png', bbox_inches='tight')
    # plt.show()
    plt.close()

    save_path = os.path.join(save_dir, f'Cell_Density_{scale}.npy')
    np.save(save_path, Cell_density_normalized)
    if verbose:
        print(f'{design} cell density of scale {scale} complete!')

def LoadCellDensity(design, scale, verbose=False):
    if verbose:
        print(f'Loading {design} cell density.')
    scale_CellDensity = {}
    save_dir = Save_Path + 'CellDensity/'+ design
    save_path = os.path.join(save_dir, f'Cell_Density_128.npy')
    if not os.path.exists(save_path):
        BuildCellDensity(design, scale)

    save_path = os.path.join(save_dir, f'Cell_Density_{scale}.npy')
    scale_CellDensity[scale] = np.load(save_path)
    if verbose:
        print(f'{design} cell density loaded!')
    return scale_CellDensity

def BuildDrcNumber(design, verbose=False):
    if verbose:
        print(f'Building {design} drc number.')
    Rpt = Global_var.Icc2Rpt_Path + design + '_drc.rpt'
    DrcList = Drc_Parser.Read_Drc(Rpt)
    DrcNumber = len(DrcList)
    if verbose:
        print(f'{design} drc number complete!')
    return float(DrcNumber)
    
def BuildDrcMap(design, verbose=False):
    if verbose:
        print(f'Building {design} drc map.')
    Rpt = Global_var.Icc2Rpt_Path + design + '_drc.rpt'
    Def = Global_var.Def_Path + design + '_route.def'
    coreArea, _, _, _ = Def_Parser.Read_def(Def)
    DrcList = Drc_Parser.Read_Drc(Rpt)
    fig, ax = plt.subplots()
    core_polygon = patches.Polygon(coreArea, closed=True, fill=True, color='black')
    ax.add_patch(core_polygon)

    for drc in DrcList:
        if drc.pins:
            drc_polygon = patches.Polygon(drc.pins, closed=True, fill=True, facecolor='white')
            ax.add_patch(drc_polygon)
    
    # Set limits and aspect ratio
    ax.set_xlim(coreArea[0][0], coreArea[2][0])
    ax.set_ylim(coreArea[0][1], coreArea[2][1])
    ax.set_aspect('equal')
    save_dir = Save_Path + 'DrcMap/' + design
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)       
    save_path = os.path.join(save_dir, f'DrcMap.png')
    plt.savefig(save_path, dpi=300, format='png', bbox_inches='tight', pad_inches=0)
    # plt.show()
    plt.close(fig)

    img = Image.open(save_path).convert('L')
    img = img.resize((512, 512), Image.Resampling.LANCZOS)
    save_path = os.path.join(save_dir, f'DrcMap.npy')
    img_array = np.array(img, dtype=np.float32)
    scaler = MinMaxScaler()
    img_array = scaler.fit_transform(img_array.reshape(-1, 1)).reshape(512, 512)
    # np.set_printoptions(threshold=np.inf)
    # print(img_array)
    np.save(save_path, img_array)
    if verbose:
        print(f'{design} drc map complete!')

def LoadDrcMap(design, verbose=False):
    if verbose:
        print(f'Loading {design} cell density.')
    save_dir = Save_Path + 'DrcMap/'+ design
    save_path = os.path.join(save_dir, f'DrcMap.npy')
    if not os.path.exists(save_path):
        BuildDrcMap(design)
    DrcMap = np.load(save_path)
    if verbose:
        print(f'{design} drc map loaded!')
    return DrcMap

def BuildTimingLib():
    print(f'Building cell timing lib.')
    cells = TimingLib_Parser.Read_TimingLib()
    save_dir = Save_Path + 'TimingLib'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, 'LibData.sav')
    with open(save_path, 'wb') as f:
        pickle.dump(cells, f)
    
    footprint = {}
    for cellname, cell in cells.items():
        if(cell.footprint not in footprint.keys()):
            footprint[cell.footprint] = []
        footprint[cell.footprint].append(cellname)
    save_path = os.path.join(save_dir, 'FootprintDict.sav')
    with open(save_path, 'wb') as f:
        pickle.dump(footprint, f)

    scaler = MinMaxScaler()
    all_index1 = []
    all_index2 = []
    all_delay = []
    all_trans = []

    for cellname, cell in cells.items():
        for (arc, rf), delay_obj in cell.delay.items():
            if delay_obj.index1 and delay_obj.index2:
                all_index1.extend(delay_obj.index1)
                all_index2.extend(delay_obj.index2)
                all_delay.extend(np.array(delay_obj.delay).flatten())

        for (arc, rf), trans_obj in cell.trans.items():
            if trans_obj.index1 and trans_obj.index2 and not trans_obj.isscalar:
                all_index1.extend(trans_obj.index1)
                all_index2.extend(trans_obj.index2)
                all_trans.extend(np.array(trans_obj.trans).flatten())

    all_index1 = np.array(all_index1).reshape(-1, 1)
    all_index2 = np.array(all_index2).reshape(-1, 1)
    all_delay = np.array(all_delay).reshape(-1, 1)
    all_trans = np.array(all_trans).reshape(-1, 1)

    scaler_index1 = MinMaxScaler()
    scaler_index2 = MinMaxScaler()
    scaler_delay = MinMaxScaler()
    scaler_trans = MinMaxScaler()

    all_index1_normalized = scaler_index1.fit_transform(all_index1).flatten()
    all_index2_normalized = scaler_index2.fit_transform(all_index2).flatten()
    all_delay_normalized = scaler_delay.fit_transform(all_delay).flatten()
    all_trans_normalized = scaler_trans.fit_transform(all_trans).flatten()

    index1_count = 0
    index2_count = 0
    delay_count = 0
    trans_count = 0

    for cellname, cell in cells.items():
        for (arc, rf), delay_obj in cell.delay.items():
            if delay_obj.index1 and delay_obj.index2:
                count = len(delay_obj.index1)
                delay_obj.index1 = all_index1_normalized[index1_count:index1_count + count].tolist()
                index1_count += count

                count = len(delay_obj.index2)
                delay_obj.index2 = all_index2_normalized[index2_count:index2_count + count].tolist()
                index2_count += count

                count = len(np.array(delay_obj.delay).flatten())
                reshaped_delay = all_delay_normalized[delay_count:delay_count + count].reshape(len(delay_obj.index2), len(delay_obj.index1))
                delay_obj.delay = reshaped_delay.tolist()
                delay_count += count

        for (arc, rf), trans_obj in cell.trans.items():
            if trans_obj.index1 and trans_obj.index2 and not trans_obj.isscalar:
                count = len(trans_obj.index1)
                trans_obj.index1 = all_index1_normalized[index1_count:index1_count + count].tolist()
                index1_count += count

                count = len(trans_obj.index2)
                trans_obj.index2 = all_index2_normalized[index2_count:index2_count + count].tolist()
                index2_count += count

                count = len(np.array(trans_obj.trans).flatten())
                reshaped_trans = all_trans_normalized[trans_count:trans_count + count].reshape(len(trans_obj.index2), len(trans_obj.index1))
                trans_obj.trans = reshaped_trans.tolist()
                trans_count += count
    save_path = os.path.join(save_dir, 'LibData_normalized.sav')
    with open(save_path, 'wb') as f:
        pickle.dump(cells, f)
    print(f'Cell timing lib complete!')

def LoadTimingLib(verbose=False):
    if verbose:
        print(f'Loading cell timing lib.')
    save_dir = Save_Path + 'TimingLib'
    save_path = os.path.join(save_dir, 'LibData.sav')
    if not os.path.exists(save_path):
        BuildTimingLib()
    with open(save_path, 'rb') as f:
        cells = pickle.load(f)
    save_path = os.path.join(save_dir, 'FootprintDict.sav')
    if not os.path.exists(save_path):
        BuildTimingLib()
    with open(save_path, 'rb') as f:
        footprint = pickle.load(f)
    if verbose:
        print(f'Timing lib loaded!')
    return cells, footprint

def LoadNormalizedTimingLib(verbose=False):
    if verbose:
        print(f'Loading normalized cell timing lib.')
    save_dir = Save_Path + 'TimingLib'
    save_path = os.path.join(save_dir, 'LibData_normalized.sav')
    if not os.path.exists(save_path):
        BuildTimingLib()
    with open(save_path, 'rb') as f:
        cells = pickle.load(f)
    save_path = os.path.join(save_dir, 'FootprintDict.sav')
    if not os.path.exists(save_path):
        BuildTimingLib()
    with open(save_path, 'rb') as f:
        footprint = pickle.load(f)
    if verbose:
        print(f'Normalized timing lib loaded!')
    return cells, footprint

def BuildVerilog(design, verbose=False):
    if verbose:
        print(f'Building {design} verilog data.')
    inVerilog = Global_var.Verilog_Path + design + '_route.v'
    Verilog = Verilog_Parser.Read_Verilog(inVerilog)
    Nets = {} # Net -> (Cell, pins)
    for input in Verilog.inputs:
        if input not in Nets.keys():
            Nets[input] = []
            Nets[input].append((None, input))
    for output in Verilog.outputs:
        if output not in Nets.keys():
            Nets[output] = []
            Nets[output].append((None, output))
    for wire in Verilog.wires:
        if wire not in Nets.keys():
            Nets[wire] = []
    
    Cells = {} # CellName -> cell
    for cell in Verilog.cells:
        Cells[cell.name] = cell
        for pin, net in cell.pins.items():
            if net not in Nets.keys():
                Nets[net] = []
            Nets[net].append((cell.name, pin))
    
    save_dir = Save_Path + 'Verilog/' + design
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, 'Verilog.sav')
    with open(save_path, 'wb') as f:
        pickle.dump((Verilog, Nets, Cells), f)
    if verbose:
        print(f'\n{design} verilog complete!')

def LoadVerilog(design, verbose=False):
    if verbose:
        print(f'Loading {design} verilog.')
    save_dir = Save_Path + 'Verilog/' + design
    save_path = os.path.join(save_dir, 'Verilog.sav')
    if not os.path.exists(save_path):
        BuildVerilog(design)
    with open(save_path, 'rb') as f:
        Verilog, Nets, Cells = pickle.load(f)
    if verbose:
        print(f'{design} verilog loaded!')
    return Verilog, Nets, Cells

def BuildPtRpt(design, verbose=False):
    if verbose:
        print(f'Building {design} PrimeTime Rpt data.')
    inPtRpt = Global_var.PtRpt_Path + design + '.rpt'
    paths = PtRpt_Parser.Read_PtRpt(inPtRpt)
    save_dir = Save_Path + 'PtRpt/' + design
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, 'PtRpt.sav')
    with open(save_path, 'wb') as f:
        pickle.dump(paths, f)
    if verbose:
        print(f'{design} PrimeTime Rpt complete!')
    return paths

def LoadPtRpt(design, verbose=False):
    if verbose:
        print(f'Loading {design} PrimeTime Rpt.')
    save_dir = Save_Path + 'PtRpt/' + design
    save_path = os.path.join(save_dir, 'PtRpt.sav')
    if not os.path.exists(save_path):
        BuildPtRpt(design)
    with open(save_path, 'rb') as f:
        paths = pickle.load(f)
    if verbose:
        print(f'{design} PrimeTime Rpt loaded!')
    return paths

class Net_Arc:
    def __init__(self):
        self.totalCap = 0
        self.resistance = 0
        self.Delay = [] # Rise Fall
        self.from_pin = ''
        self.inpin_caps = [0 ,0] # Min Max
        self.to_pin = ''
        self.outpin_caps = [0, 0]
        self.isinPinPIPO = 0
        self.isoutPinPIPO = 0
    def __repr__(self):
        delay_repr = ', '.join([f'{delay:.8f}' for delay in self.Delay])
        return f"NetArc(from_pin='{self.from_pin}', to_pin='{self.to_pin}', from_pin_cap='{self.inpin_cap:.8f}', to_pin_cap='{self.outpin_cap:.8f}', total_cap='{self.totalCap:.8f}', \nDelay={{ {delay_repr} }}\n)"

def BuildPtCells(design, verbose=False):
    if verbose:
        print(f'Building {design} PrimeTime Cells.')
    inPtCellRpt = Global_var.PtRpt_Path + design + '_cell.rpt'
    PtCells = PtCellRpt_Parser.Read_PtCellRpt(inPtCellRpt)
    save_dir = Save_Path + 'PtCell/' + design
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, 'PtCell.sav')
    with open(save_path, 'wb') as f:
        pickle.dump(PtCells, f)
    if verbose:
        print(f'{design} PrimeTime Cells complete!')
    return PtCells
        
def LoadPtCells(design, verbose=False):
    if verbose:
        print(f'Loading {design} PrimeTime Cells.')
    save_dir = Save_Path + 'PtCell/' + design
    save_path = os.path.join(save_dir, 'PtCell.sav')
    if not os.path.exists(save_path):
        BuildPtCells(design)
    with open(save_path, 'rb') as f:
        PtCells = pickle.load(f)
    if verbose:
        print(f'{design} PrimeTime Cells loaded!')
    return PtCells

def BuildPtNets(design, verbose=False):
    if verbose:
        print(f'Building {design} PrimeTime Nets.')
    inPtNetRpt = Global_var.PtRpt_Path + design + '_net.rpt'
    PtNets = PtNetRpt_Parser.Read_PtNetRpt(inPtNetRpt)
    save_dir = Save_Path + 'PtNet/' + design
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, 'PtNet.sav')
    with open(save_path, 'wb') as f:
        pickle.dump(PtNets, f)
    if verbose:
        print(f'{design} PrimeTime Nets complete!')
    return PtNets

def LoadPtNets(design, verbose=False):
    if verbose:
        print(f'Loading {design} PrimeTime Nets.')
    save_dir = Save_Path + 'PtNet/' + design
    save_path = os.path.join(save_dir, 'PtNet.sav')
    if not os.path.exists(save_path):
        BuildPtNets(design)
    with open(save_path, 'rb') as f:
        PtNets = pickle.load(f)
    if verbose:
        print(f'{design} PrimeTime Nets loaded!')
    return PtNets

def BuildCellArc(design, verbose=False):
    if verbose:
        print(f'Building {design} Timing Arc.')
    inPtCellRpt = Global_var.PtRpt_Path + design + '_cell.rpt'
    inPtDelayRpt = Global_var.PtRpt_Path + design + '_Delay.rpt'
    PtCells = PtCellRpt_Parser.Read_PtCellRpt(inPtCellRpt)
    PtCellArcs = PtDelayRpt_Parser.Read_PtDelayRpt(inPtDelayRpt)
    CellArcs = {}
    for cellname, cell in PtCells.items():
        for inpin in cell.inpins:
            for outpin in cell.outpins:
                CellArcs[(cellname + '/' + inpin, cellname + '/' + outpin)] = None

    for _, arc in PtCellArcs.items():
        CellArcs[(arc.from_pin, arc.to_pin)] = arc

    if verbose:
        print(f'{design} Timing Arc complete!')
    return CellArcs, PtCells

def BuildTimingArc(design, verbose=False):
    if verbose:
        print(f'Building {design} Timing Arc.')
    inPtCellRpt = Global_var.PtRpt_Path + design + '_cell.rpt'
    inPtNetRpt = Global_var.PtRpt_Path + design + '_net.rpt'
    inPtDelayRpt = Global_var.PtRpt_Path + design + '_Delay.rpt'
    PtCells = PtCellRpt_Parser.Read_PtCellRpt(inPtCellRpt)
    PtNets = PtNetRpt_Parser.Read_PtNetRpt(inPtNetRpt)
    PtCellArcs = PtDelayRpt_Parser.Read_PtDelayRpt(inPtDelayRpt)
    SdfNetArcs = SDF_Parser.Read_SDF(Global_var.PtRpt_Path + design + '.sdf')
    CellArcs = {}; NetArcs = {}
    for cellname, cell in PtCells.items():
        for inpin in cell.inpins:
            for outpin in cell.outpins:
                CellArcs[(cellname + '/' + inpin, cellname + '/' + outpin)] = None
    for _, net in PtNets.items():
        i = 0; j = 0
        for inpin in net.inpins:
            for outpin in net.outpins:
                newNetArc = Net_Arc()
                newNetArc.from_pin = inpin
                newNetArc.to_pin = outpin
                newNetArc.inpin_caps = net.inpin_caps[i]
                newNetArc.outpin_caps = net.outpin_caps[j]
                newNetArc.isinPinPIPO = net.isinPinPIPO[i]
                newNetArc.isoutPinPIPO = net.isoutPinPIPO[j]
                newNetArc.totalCap = net.totalCap
                newNetArc.resistance = net.resistance
                newNetArc.Delay = [0, 0]
                NetArcs[(inpin, outpin)] = newNetArc
                j += 1
            i += 1
    for _, arc in PtCellArcs.items():
        CellArcs[(arc.from_pin, arc.to_pin)] = arc
    for _, arc in SdfNetArcs.items():
        if (arc.from_pin, arc.to_pin) in NetArcs.keys():
            NetArcs[(arc.from_pin, arc.to_pin)].Delay = arc.Delay
        elif (arc.to_pin, arc.from_pin) in NetArcs.keys():
            NetArcs[(arc.to_pin, arc.from_pin)].Delay = arc.Delay
        else:
            pass
    Save_Dir = Save_Path + 'TimingArc/' + design
    if not os.path.exists(Save_Dir):
        os.makedirs(Save_Dir)
    save_path = os.path.join(Save_Dir, 'CellArc.sav')
    with open(save_path, 'wb') as f:
        pickle.dump((CellArcs, NetArcs), f)
    if verbose:
        print(f'{design} Timing Arc complete!')
    return CellArcs, NetArcs

def LoadTimingArc(design, verbose=False):
    if verbose:
        print(f'Loading {design} Timing Arc.')
    save_path = Save_Path + 'TimingArc/' + design + '/CellArc.sav'
    if not os.path.exists(save_path):
        BuildTimingArc(design)
    with open(save_path, 'rb') as f:
        CellArcs, NetArcs = pickle.load(f)
    if verbose:
        print(f'{design} Timing Arc loaded!')
    return CellArcs, NetArcs

def BuildEndPoint(design, verbose=False):
    if verbose:
        print(f'Building {design} End Point Slack.')
    inEndPointMetRpt = Global_var.PtRpt_Path + design + '_met_endpoint.rpt'
    inEndPointViolatedRpt = Global_var.PtRpt_Path + design + '_violated_endpoint.rpt'
    inEndPointUntestedRpt = Global_var.PtRpt_Path + design + '_untested_endpoint.rpt'
    EndPointMet = EndPoint_Parser.Read_EndPoint(inEndPointMetRpt, 'met')
    EndPointViolated = EndPoint_Parser.Read_EndPoint(inEndPointViolatedRpt, 'violated')
    EndPointUntested = EndPoint_Parser.Read_EndPoint(inEndPointUntestedRpt, 'untested')
    EndPoints = {**EndPointMet, **EndPointViolated, **EndPointUntested}
    save_dir = Save_Path + 'EndPoint/' + design
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, 'EndPoint.sav')
    with open(save_path, 'wb') as f:
        pickle.dump(EndPoints, f)
    if verbose:
        print(f'{design} End Point Slack complete!')
    return EndPoints

def LoadEndPoint(design, verbose=False):
    if verbose:
        print(f'Loading {design} End Point Slack.')
    save_path = Save_Path + 'EndPoint/' + design + '/EndPoint.sav'
    if not os.path.exists(save_path):
        BuildEndPoint(design)
    with open(save_path, 'rb') as f:
        EndPoints = pickle.load(f)
    if verbose:
        print(f'{design} End Point Slack loaded!')
    return EndPoints

def BuildGlobalTimingData(design, verbose=False):
    if verbose:
        print(f'Building {design} Global Timing Data.')
    inGlobalRpt = Global_var.PtRpt_Path + design + '_global.rpt'
    wns, tns = PtGlobalRpt_Parser.Read_GlobalRpt(inGlobalRpt)
    if verbose:
        print(f'{design} Global Timing Data complete!')
    return wns, tns

def BuildPortData(design, verbose=False):
    if verbose:
        print(f'Building {design} Port Data.')
    inPort = Global_var.Icc2Rpt_Path + design + '_port.rpt'
    Def = Global_var.Def_Path + design + '_route.def'
    coreArea, _, _, _ = Def_Parser.Read_def(Def)
    PortList = PortLocation_Parser.Read_PortLocation(inPort)

    Normalized_PortList = {}
    for key, value in PortList.items():
        for i in range(len(value)):
            value[i] = (value[i][0] / coreArea[2][0], value[i][1] / coreArea[2][1])
        Normalized_PortList[key] = value

    save_dir = Save_Path + 'PortData/' + design
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, 'PortList.sav')
    with open(save_path, 'wb') as f:
        pickle.dump(Normalized_PortList, f)

    if verbose:
        print(f'{design} Port Data complete!')
    
def LoadPortData(design, verbose=False):
    if verbose:
        print(f'Loading {design} Port Data.')
    save_path = Save_Path + 'PortData/' + design + '/PortList.sav'
    if not os.path.exists(save_path):
        BuildPortData(design)
    with open(save_path, 'rb') as f:
        PortList = pickle.load(f)
    if verbose:
        print(f'{design} Port Data loaded!')
    return PortList

def BuildPinLayout(design, verbose=False):
    if verbose:
        print(f'Building {design} Pin Data.')
    inPin = Global_var.Icc2Rpt_Path + design + '_pin.rpt'
    Def = Global_var.Def_Path + design + '_route.def'
    PinList = PinLocation_Parser.Read_PinLocation(inPin)
    coreArea, _, _, _ = Def_Parser.Read_def(Def)
    fig, ax = plt.subplots()
    core_polygon = patches.Polygon(coreArea, closed=True, fill=True, color='black')
    ax.add_patch(core_polygon)
    
    for pin_name, pin_coords in PinList.items():
        bottom_left = pin_coords[0]
        top_right = pin_coords[1]

        width = top_right[0] - bottom_left[0]
        height = top_right[1] - bottom_left[1]

        pin_rect = patches.Rectangle(bottom_left, width, height, edgecolor='black', facecolor='white', linewidth=1/(coreArea[1][1]/scaler_base))
        ax.add_patch(pin_rect)
    ax.axis('off')
        
    ax.set_xlim(coreArea[0][0], coreArea[2][0])
    ax.set_ylim(coreArea[0][1], coreArea[2][1])
    ax.set_aspect('equal')
    
    save_dir = Save_Path + 'PinLayout/' + design
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    save_path = os.path.join(save_dir, f'PinLayout.png')
    plt.savefig(save_path, dpi=300, format='png', bbox_inches='tight', pad_inches=0)
    # plt.show()
    plt.close(fig)
    
    # convert image to 512*512
    img = Image.open(save_path).convert('L')
    img = img.resize((512, 512), Image.Resampling.LANCZOS)
    # img.show()
    save_path = os.path.join(save_dir, f'PinLayout.npy')
    img_array = np.array(img, dtype=np.float32)
    
    # nomalization
    scaler = MinMaxScaler()
    img_array = scaler.fit_transform(img_array.reshape(-1, 1)).reshape(512, 512)
    # np.set_printoptions(threshold=np.inf)
    # print(img_array)
    np.save(save_path, img_array)

    if verbose:
        print(f'\n{design} pin layout complete!')

def BuildPinData(design, row, column, verbose=False):
    if verbose:
        print(f'Building {design} Pin Data with {row} rows and {column} columns.')

    inPin = Global_var.Icc2Rpt_Path + design + '_pin.rpt'
    Def = Global_var.Def_Path + design + '_route.def'
    PinList = PinLocation_Parser.Read_PinLocation(inPin)
    coreArea, _, _, _ = Def_Parser.Read_def(Def)

    # Initialize an empty matrix for the target area, filled with 0 for the background
    target_height, target_width = row * 64, column * 64
    img_array = np.zeros((target_height, target_width), dtype=np.float32)  # Fill with 0 as background

    # Map coordinates to the image matrix based on coreArea size and pin coordinates
    x_scale = target_width / (coreArea[2][0] - coreArea[0][0])
    y_scale = target_height / (coreArea[2][1] - coreArea[0][1])

    # Draw each pin in the image matrix
    for pin_name, pin_coords in PinList.items():
        bottom_left = (int(pin_coords[0][0] * x_scale), int(pin_coords[0][1] * y_scale))
        top_right = (int(pin_coords[1][0] * x_scale), int(pin_coords[1][1] * y_scale))
        img_array[bottom_left[1]:top_right[1], bottom_left[0]:top_right[0]] = 1.0  # Grayscale level for pins

    # Normalize the image
    scaler = MinMaxScaler()
    img_array = scaler.fit_transform(img_array.reshape(-1, 1)).reshape(target_height, target_width)
        
    # Split the image into row*column sub-images of 64x64 pixels
    sub_images = []
    sub_h, sub_w = 64, 64  # Fixed sub-image size of 64x64

    for r in range(row):
        for c in range(column):
            sub_img = img_array[r * sub_h:(r + 1) * sub_h, c * sub_w:(c + 1) * sub_w]
            sub_images.append(sub_img)

    if verbose:
        print(f'\n{design} pin layout complete! Created {len(sub_images)} sub-images of 64x64 pixels.')
    
    return sub_images

def BuildPinData(design, row, column, verbose=False):
    if verbose:
        print(f'Building {design} Pin Data with {row} rows and {column} columns.')

    inPin = Global_var.Icc2Rpt_Path + design + '_pin.rpt'
    Def = Global_var.Def_Path + design + '_route.def'
    PinList = PinLocation_Parser.Read_PinLocation(inPin)
    coreArea, _, _, _ = Def_Parser.Read_def(Def)

    # Initialize an empty matrix for the entire area (assumes a fixed size based on target dimensions)
    target_height, target_width = row * 64, column * 64
    img_array = np.zeros((target_height, target_width), dtype=np.float32)

    # Map coordinates to the image matrix based on coreArea size and pin coordinates
    x_scale = target_width / (coreArea[2][0] - coreArea[0][0])
    y_scale = target_height / (coreArea[2][1] - coreArea[0][1])

    # Fill core area
    core_x_min = int(coreArea[0][0] * x_scale)
    core_x_max = int(coreArea[2][0] * x_scale)
    core_y_min = int(coreArea[0][1] * y_scale)
    core_y_max = int(coreArea[2][1] * y_scale)
    img_array[core_y_min:core_y_max, core_x_min:core_x_max] = 0.5  # Set a grayscale level for core

    # Draw each pin as a rectangle in the image matrix
    for pin_name, pin_coords in PinList.items():
        bottom_left = (min(int(pin_coords[0][0] * x_scale), target_width - 1), 
                       min(int(pin_coords[0][1] * y_scale), target_height - 1))
        top_right = (min(int(pin_coords[1][0] * x_scale), target_width), 
                     min(int(pin_coords[1][1] * y_scale), target_height))
        img_array[bottom_left[1]:top_right[1], bottom_left[0]:top_right[0]] = 1.0  # Set a grayscale level for pins

    # Normalize the image
    scaler = MinMaxScaler()
    img_array = scaler.fit_transform(img_array.reshape(-1, 1)).reshape(target_height, target_width)

    # Split the image into row*column sub-images of 64x64 pixels
    sub_images = []
    sub_h, sub_w = 64, 64  # Fixed sub-image size of 64x64

    for r in range(row):
        for c in range(column):
            sub_img = img_array[r * sub_h:(r + 1) * sub_h, c * sub_w:(c + 1) * sub_w]
            sub_images.append(sub_img)

    if verbose:
        print(f'\n{design} pin layout complete! Created {len(sub_images)} sub-images of 64x64 pixels.')
    
    return sub_images

# def LoadPinLayout(design, verbose=False):
#     if verbose:
#         print(f'Loading {design} Pin Data.')
#     save_path = Save_Path + 'PinLayout/' + design + '/PinLayout.npy'
#     if not os.path.exists(save_path):
#         BuildPinLayout(design)
#     img_array = np.load(save_path)
#     if verbose:
#         print(f'{design} Pin Data loaded!')
#     return img_array

def GetSizedCellList(design):
    ChangeListPath = os.path.join(Global_var.work_dir, 'ECO_ChangeList/' + design + '_eco.tcl')
    sized_cell_list = ECOChangeList_Parser.Read_ECOChangeList(ChangeListPath)
    return sized_cell_list

def GetSizedCellCommand(design):
    ChangeListPath = os.path.join(Global_var.work_dir, 'ECO_ChangeList/' + design + '_eco.tcl')
    command_eco = ECOCommand_Parser.Read_ECOCommand(ChangeListPath)
    ChangeListPath = os.path.join(Global_var.work_dir, 'ECO_ChangeList/' + design + '_eco_occupied.tcl')
    command_occupied = ECOCommand_Parser.Read_ECOCommand(ChangeListPath)
    ChangeListPath = os.path.join(Global_var.work_dir, 'ECO_ChangeList/' + design + '_eco_physical.tcl')
    command_physical = ECOCommand_Parser.Read_ECOCommand(ChangeListPath)
    return [command_eco, command_occupied, command_physical]

def BuildRegionalDrc(design, row, column, verbose=False):
    print(f'Building {design} regional DRC.')
    Rpt = Global_var.Icc2Rpt_Path + design + '_drc.rpt'
    Def = Global_var.Def_Path + design + '_route.def'
    coreArea, _, _, _ = Def_Parser.Read_def(Def)
    DrcList = Drc_Parser.Read_Drc(Rpt)

    core_width = coreArea[2][0] - coreArea[0][0]
    core_height = coreArea[2][1] - coreArea[0][1]
    cell_width = core_width / column
    cell_height = core_height / row
    
    drc_count_matrix = np.zeros((row, column), dtype=np.float32)
    
    for drc in DrcList:
        if drc.pins:
            # Determine the bounding box of the drc polygon
            min_x = min(point[0] for point in drc.pins)
            max_x = max(point[0] for point in drc.pins)
            min_y = min(point[1] for point in drc.pins)
            max_y = max(point[1] for point in drc.pins)

            # Determine grid indices for the bounding box
            col_start = max(0, min(column - 1, int((min_x - coreArea[0][0]) / cell_width)))
            col_end = max(0, min(column - 1, int((max_x - coreArea[0][0]) / cell_width)))
            row_start = max(0, min(row - 1, int((min_y - coreArea[0][1]) / cell_height)))
            row_end = max(0, min(row - 1, int((max_y - coreArea[0][1]) / cell_height)))

            # Determine the region with the maximum overlap area
            max_overlap_region = None
            max_overlap_area = 0

            for r in range(row_start, row_end + 1):
                for c in range(col_start, col_end + 1):
                    # Calculate bounding box for the grid cell
                    cell_min_x = coreArea[0][0] + c * cell_width
                    cell_max_x = coreArea[0][0] + (c + 1) * cell_width
                    cell_min_y = coreArea[0][1] + r * cell_height
                    cell_max_y = coreArea[0][1] + (r + 1) * cell_height

                    # Calculate overlap area between the drc and the cell
                    overlap_width = max(0, min(max_x, cell_max_x) - max(min_x, cell_min_x))
                    overlap_height = max(0, min(max_y, cell_max_y) - max(min_y, cell_min_y))
                    overlap_area = overlap_width * overlap_height

                    # Check if this cell has the most overlap area
                    if overlap_area > max_overlap_area:
                        max_overlap_area = overlap_area
                        max_overlap_region = (r, c)

            # Increment the count for the region with maximum overlap
            if max_overlap_region:
                drc_count_matrix[max_overlap_region] += 1

    # Adjust the matrix to match the desired numbering from bottom-left to top-right
    drc_count_matrix = np.flipud(drc_count_matrix)

    print("DRC counts per region:\n", drc_count_matrix)

    return drc_count_matrix

if __name__ == "__main__": 
    # DataBuilder.BuildCellLayout()
    # DataBuilder.LoadCellLayout('aes_cipher_top')
    # DataBuilder.BuildMetalLayer()
    # DataBuilder.LoadMetalLayer('aes_cipher_top')
    # DataBuilder.BuildRouteCongestion()
    # DataBuilder.LoadRouteCongestion('aes_cipher_top')
    # DataBuilder.BuildCellDensity()
    # DataBuilder.LoadCellDensity('aes_cipher_top')
    # DataBuilder.BuildDrcMap()
    # DataBuilder.LoadDrcMap('sasc_top')
    # DataBuilder.BuildTimingLib()
    # DataBuilder.LoadTimingLib()
    # DataBuilder.LoadNormalizedTimingLib()
    # DataBuilder.BuildVerilog('nova')
    # DataBuilder.LoadVerilog('sasc_top')
    # DataBuilder.BuildPtRpt('aes_cipher_top')
    # DataBuilder.LoadPtRpt('aes_cipher_top')
    # DataBuilder.BuildTimingArc('aes_cipher_top')
    # DataBuilder.BuildEndPoint('aes_cipher_top')
    # DataBuilder.LoadEndPoint('aes_cipher_top')
    TimingGraphTrans.TimingGraphTrans('usbf_top', rebuilt=True)
    # DataBuilder.BuildPortData('aes_cipher_top')
    # PhysicalDataTrans.PhysicalDataTrans('vga_enh_top', 512)
    # DataBuilder.BuildGlobalTimingData('vga_enh_top')
    # DataBuilder.BuildPinLayout('aes_cipher_top')
    # MergeMultiPath.MergeMultiPath('aes_cipher_top')
    # TimingGraphTrans.LoadTimingGraph('aes_cipher_top_eco', True)
    # print(DataBuilder.GetSizedCellList('aes_cipher_top'))
    # command_eco, command_occupied, command_physical = DataBuilder.GetSizedCellCommand('aes_cipher_top')
    # for key, value in command_physical.items():
    #     print(key)
    #     print(value)