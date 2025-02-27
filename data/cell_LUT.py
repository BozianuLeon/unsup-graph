import torch
import torch_geometric

import h5py
import numpy as np
import os
import json

EPS=1e-8


def make_edge_LUT(path_to_h5_file, output_dir='./'):
    '''
    Function to create and store the LUT for connecting 
    cells in neighbouring eta-phi buckets.
    Uses a single event from a single file, since cellId 
    and positions remain unchanged. Saves to json file
    named cell_neighbours.json
    Args:
        path_to_h5_file (str): Direct path to a single h5 file - cells
    Outputs:
        None
    '''
    cell_neighbours_dict = {}

    print("\t",path_to_h5_file)
    f1 = h5py.File(path_to_h5_file,"r")

    cells = f1["caloCells"]["2d"][0] # event 0 cells

    cell_etas = cells['cell_eta']
    cell_phis = cells['cell_phi']
    cell_ids  = cells['cell_IdCells']
    bins_x = np.linspace(min(cell_etas), max(cell_etas)+EPS, int((max(cell_etas) - min(cell_etas)) / 0.1 + 1))
    bins_y = np.linspace(min(cell_phis), max(cell_phis)+EPS, int((max(cell_phis) - min(cell_phis)) / ((2*np.pi)/64) + 1))
    x_indices = np.digitize(cell_etas, bins_x,right=False) # gives right hand bin edges
    y_indices = np.digitize(cell_phis, bins_y,right=False) # gives right hand bin edges

    for i in range(len(cell_ids)):
        print(i,f'/ {len(cell_ids)}')
        idx = int(cell_ids[i])

        neighbour_bucket_mask = (x_indices==x_indices[i]-1) & (y_indices==y_indices[i]-1) \
                              | (x_indices==x_indices[i]-1) & (y_indices==y_indices[i]  ) \
                              | (x_indices==x_indices[i]-1) & (y_indices==y_indices[i]+1) \
                              | (x_indices==x_indices[i]  ) & (y_indices==y_indices[i]-1) \
                              | (x_indices==x_indices[i]  ) & (y_indices==y_indices[i]  ) \
                              | (x_indices==x_indices[i]  ) & (y_indices==y_indices[i]+1) \
                              | (x_indices==x_indices[i]+1) & (y_indices==y_indices[i]-1) \
                              | (x_indices==x_indices[i]+1) & (y_indices==y_indices[i]  ) \
                              | (x_indices==x_indices[i]+1) & (y_indices==y_indices[i]+1) 

        cell_ids_in_same_bin = cell_ids[neighbour_bucket_mask]
        cell_neighbours_dict[idx] = cell_ids_in_same_bin.tolist() # list of all cells in neighbouring buckets

    f1.close()

    print('Saving cell json neighbours dict json file...')
    with open(output_dir+"cell_neighbours.json",'w') as json_file:
        json.dump(cell_neighbours_dict,json_file)


def make_edge_npy_LUT(path_to_h5_file, output_dir='./'):
    
    print("\t",path_to_h5_file)
    f1 = h5py.File(path_to_h5_file,"r")

    cells = f1["caloCells"]["2d"][0] # event 0 cells

    cell_etas = cells['cell_eta']
    cell_phis = cells['cell_phi']
    cell_ids  = cells['cell_IdCells']

    bins_x = np.linspace(min(cell_etas), max(cell_etas)+EPS, int((max(cell_etas) - min(cell_etas)) / 0.1 + 1))
    bins_y = np.linspace(min(cell_phis), max(cell_phis)+EPS, int((max(cell_phis) - min(cell_phis)) / ((2*np.pi)/64) + 1))

    x_indices = np.digitize(cell_etas, bins_x,right=False) 
    y_indices = np.digitize(cell_phis, bins_y,right=False) 
    
    big_storage_array = np.empty((len(cell_ids),750),dtype=int)
    source_node_array = np.empty((len(cell_ids),750),dtype=int)
    max_number_neighbours = []
    for i in range(len(cell_ids)):
        print(i,f'/ {len(cell_ids)}')
        idx = int(cell_ids[i])
        print('cell ID',idx,' index', i)
        print('cell eta,phi',cell_etas[i],cell_phis[i])
        print('cell bin x ID',x_indices[i],'[',bins_x[x_indices[i]-1],',',bins_x[x_indices[i]],']')
        print('cell bin y ID',y_indices[i],'[',bins_y[y_indices[i]-1],',',bins_y[y_indices[i]],']')

        neighbour_bucket_mask = (x_indices==x_indices[i]-1) & (y_indices==y_indices[i]-1) \
                              | (x_indices==x_indices[i]-1) & (y_indices==y_indices[i]  ) \
                              | (x_indices==x_indices[i]-1) & (y_indices==y_indices[i]+1) \
                              | (x_indices==x_indices[i]  ) & (y_indices==y_indices[i]-1) \
                              | (x_indices==x_indices[i]  ) & (y_indices==y_indices[i]  ) \
                              | (x_indices==x_indices[i]  ) & (y_indices==y_indices[i]+1) \
                              | (x_indices==x_indices[i]+1) & (y_indices==y_indices[i]-1) \
                              | (x_indices==x_indices[i]+1) & (y_indices==y_indices[i]  ) \
                              | (x_indices==x_indices[i]+1) & (y_indices==y_indices[i]+1) 

        cell_ids_in_same_bin = cell_ids[neighbour_bucket_mask]
        cell_ids_in_same_bin = np.pad(cell_ids_in_same_bin, (0,750-len(cell_ids_in_same_bin)), 'constant',constant_values=(999))
        source_cell_ids = idx*np.ones_like(cell_ids_in_same_bin)


        big_storage_array[i,:] = cell_ids_in_same_bin
        source_node_array[i,:] = source_cell_ids
        max_number_neighbours.append(len(cell_ids_in_same_bin))
    
    f1.close()

    print('Saving numpy array to .npy file')
    np.save(output_dir+'cell_neighbours.npy', big_storage_array)    # .npy extension is added if not given
    np.save(output_dir+'src_cell_neighbours.npy', source_node_array)   

    return




if __name__=="__main__":
    print("In the process of making teh graph adjacency matrix according to:")
    print("1. All cells with |significance| > 4 are fully connected. All to all.")
    print("2. All cells with |significance| > 2 are connected to cells in surrounding eta-phi buckets.")


    # make_edge_LUT("/Users/leonbozianu/work/phd/graph/dmon/user.cantel.34126190._000001.calocellD3PD_mc16_JZ4W.r10788.h5",output_dir="./pyg/")
    make_edge_npy_LUT("/Users/leonbozianu/work/phd/graph/dmon/user.cantel.34126190._000001.calocellD3PD_mc16_JZ4W.r10788.h5",output_dir="./pyg/")
