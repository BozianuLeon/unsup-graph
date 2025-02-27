import torch
import torch_geometric

import h5py
import numpy as np
import numpy.lib.recfunctions as rf
import os
import json
import time


# 
# 




def make_one_A_matrix(path_to_h5_file, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    '''
    Function to create the (sparse) adjacency matrix for the cell
    point cloud. 
    Fully connects the >4sigma cells.
    Uses the LUT saved in nearby json file to find close neighbours
    to each >2sigma cell. 
    This function is called once in each inference. Therefore must 
    be efficient.
    See also:
    https://pytorch.org/docs/stable/sparse.html
    https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/utils/sparse.html 
    '''
    print("\t",path_to_h5_file)
    f1 = h5py.File(path_to_h5_file,"r")
    print("\t","./pyg/cell_neighbours.json")
    cell_neighbours_json = open("./pyg/cell_neighbours.json", 'r')
    cell_neighbours = json.load(cell_neighbours_json)

    kk = 2
    cells = f1["caloCells"]["2d"][kk] # event 0 cells
    mask_2sigma = abs(cells['cell_E'] / cells['cell_Sigma']) >= 2
    mask_4sigma = abs(cells['cell_E'] / cells['cell_Sigma']) >= 4
    cells2sig = cells[mask_2sigma]
    cells4sig = cells[mask_4sigma]

    # get cell IDs
    cell_ids_2 = torch.tensor(cells2sig['cell_IdCells'].astype(int))
    cell_ids_4 = cells4sig['cell_IdCells'].astype(int)
    num_nodes = cell_ids_2.shape[0]


    # get edges for 2 sigma cells first:
    source_node_indices = []
    dest_node_indices  = []
    for i in range(len(cell_ids_2)):
        cell_id = cell_ids_2[i].item()
        # print('ID ',cell_id)

        # find cells in neighbouring buckets
        neighb = torch.tensor(cell_neighbours[str(cell_id)])
        # print(neighb)
        # && with 2 sigma mask
        neighb_2sig = neighb[torch.isin(neighb,cell_ids_2)]
        # print(neighb_2sig)
        # get indices in regular cells array
        neighb_2sig_indices = torch.searchsorted(cell_ids_2, neighb_2sig)
        # print(neighb_2sig_indices)
        # print(cell_ids_2[neighb_2sig_indices]) # check neighb_2sig
        # match this with cell i in sparse adj mat
        # source_node_idxs = np.repeat(i,len(neighb_2sig_indices))
        # source_node_idxs = torch.tensor([i])
        # source_node_idxs = source_node_idxs.repeat_interleave(repeats=len(neighb_2sig_indices)).to(dtype=torch.int32, device=device)
        source_node_idxs = torch.tensor([i])*torch.ones_like(neighb_2sig_indices)
        #append to overall arrays
        source_node_indices.append(source_node_idxs)
        dest_node_indices.append(neighb_2sig_indices)


    row_indices = torch.cat(source_node_indices,dim=0)
    col_indices = torch.cat(dest_node_indices,dim=0)

    adj_matrix = torch.sparse_coo_tensor(
        indices=torch.stack([row_indices, col_indices]), 
        values=torch.ones(row_indices.shape[0]), 
        size=(num_nodes, num_nodes)
    )

    edge_index = torch.stack((row_indices,col_indices),dim=0)
    print('HERE should be [2,ei]',edge_index.shape)
    print("FOR LOOP IMP")
    print(row_indices)
    print(col_indices)
    print(row_indices.shape,col_indices.shape)
    print(sum(row_indices),sum(col_indices))
    return adj_matrix

def make_A_matrix_faster(path_to_h5_file,device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    print("\nI'm now in the faster function!")
    print("\t",path_to_h5_file)
    f1 = h5py.File(path_to_h5_file,"r")
    print("\t","./pyg/cell_neighbours.json")
    cell_neighbours_json = open("./pyg/cell_neighbours.json", 'r')
    cell_neighbours = json.load(cell_neighbours_json)
    values = np.array(list(cell_neighbours.values()))
    print(values[0])
    # Can use numba here, but slow at appending/extending. 
    # will need to know the size in advance. This is possible with smart padding
    # add a default placeholder value (-1) that we can filter out of 

    return

def make_A_faster(path_to_h5_file, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    print("\t",path_to_h5_file)
    f1 = h5py.File(path_to_h5_file,"r")
    print("\t","./pyg/cell_neighbours.npy")
    cell_neighbours = np.load('./pyg/cell_neighbours.npy')
    src_cell_neighbours = np.load('./pyg/src_cell_neighbours.npy')


    kk = 2
    cells = f1["caloCells"]["2d"][kk] # event 0 cells
    mask_2sigma = abs(cells['cell_E'] / cells['cell_Sigma']) >= 2
    mask_4sigma = abs(cells['cell_E'] / cells['cell_Sigma']) >= 4
    cells2sig = cells[mask_2sigma]
    cells4sig = cells[mask_4sigma]

    # get cell IDs
    cell_ids_2 = np.array(cells2sig['cell_IdCells'].astype(int)) # THIS IS THE GRAND LIST OF CELLS WE CAN USE IN THIS EVENT
    cell_ids_4 = np.array(cells4sig['cell_IdCells'].astype(int))

    # get the neighbour arrays for the 2 sigma cells
    cell_neighb_2 = cell_neighbours[mask_2sigma]
    src_cell_neighb_2 = src_cell_neighbours[mask_2sigma]

    # filter cell neighbours, only >2sigma and remove padded -999 values
    actual_cell_neighb_2 = np.where(np.isin(cell_neighb_2,cell_ids_2), cell_neighb_2, np.nan) # actual cells we can use from cell_neighbours
    actual_src_cell_neighb_2 = np.where(np.isin(cell_neighb_2,cell_ids_2), src_cell_neighb_2, np.nan) 

    # find the cellID indices from cell_ids_2, what index are they in this event?
    neighb_2sig_indices = np.searchsorted(cell_ids_2,actual_cell_neighb_2)
    neighb_src_2sig_indices = np.searchsorted(cell_ids_2,actual_src_cell_neighb_2)
    print(neighb_2sig_indices.shape,neighb_src_2sig_indices.shape)

    # use the nan array to again extract just the valid node indices we want
    dst_node_indices = neighb_2sig_indices[~np.isnan(actual_cell_neighb_2)]
    src_node_indices = neighb_src_2sig_indices[~np.isnan(actual_src_cell_neighb_2)]
    print(dst_node_indices)
    print(src_node_indices)
    print(dst_node_indices.shape,src_node_indices.shape)
    print(sum(dst_node_indices),sum(src_node_indices))

    edge_index = np.stack((dst_node_indices,src_node_indices),axis=0)

    return edge_index


if __name__=="__main__":
    print("In the process of making the graph adjacency matrix according to:")
    print("1. All cells with |significance| > 4 are fully connected. All to all.")
    print("2. All cells with |significance| > 2 are connected to cells in surrounding eta-phi buckets.")


    print("Numpy array functions")
    a = time.perf_counter()
    A = make_A_faster("/Users/leonbozianu/work/phd/graph/dmon/user.cantel.34126190._000001.calocellD3PD_mc16_JZ4W.r10788.h5")
    b = time.perf_counter()
    print(f"Time taken (nump): {b-a}")
    print()
    print()
    print(A.shape)
    print(A.shape[0]**2)

    a = time.perf_counter()
    A = make_one_A_matrix("/Users/leonbozianu/work/phd/graph/dmon/user.cantel.34126190._000001.calocellD3PD_mc16_JZ4W.r10788.h5")
    b = time.perf_counter()
    print(f"Time taken: {b-a}")
    print(A.shape)
    print(A.shape[0]**2)
    quit()
    # print("FOR LOOP IMP")
    # print(row_indices)
    # print(col_indices)
    # print(row_indices.shape,col_indices.shape)
    # print(sum(row_indices),sum(col_indices))
    # quit()
    # FOR LOOP IMP
    # tensor([   0,    0,    0,  ..., 9247, 9247, 9247], dtype=torch.int32)
    # tensor([   0,    1,    5,  ..., 6801, 6822, 9247])
    # torch.Size([312720]) torch.Size([312720])
    # tensor(1406910199, dtype=torch.int32) tensor(1406910199)


    row_indices = torch.tensor([0,0,0,1,2])
    col_indices = torch.tensor([1,2,3,0,3])
    adj_matrix = torch.sparse_coo_tensor(
        indices=torch.stack([row_indices, col_indices]), 
        values=torch.ones(row_indices.shape[0]), 
        size=(4, 4)
    )
    print(row_indices.shape,col_indices.shape)
    print(adj_matrix.shape)
    print(adj_matrix.to_dense())

    print()
    print()
    print()
    A_prime = make_A_matrix_faster("/Users/leonbozianu/work/phd/graph/dmon/user.cantel.34126190._000001.calocellD3PD_mc16_JZ4W.r10788.h5")
    