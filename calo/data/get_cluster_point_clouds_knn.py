import numpy as np
import scipy
import h5py
import matplotlib.pyplot as plt
import os
import argparse

import torch
import torch_geometric

import utils
MIN_CELLS_PHI,MAX_CELLS_PHI = -3.1334076, 3.134037
MIN_CELLS_ETA,MAX_CELLS_ETA = -4.823496, 4.823496
image_format = "png"

parser = argparse.ArgumentParser()
parser.add_argument('-idx','--idx', required=False)
args = vars(parser.parse_args())

truth_box_number = int(args['idx']) if args['idx'] is not None else 0

with open("../../../struc_array.npy", "rb") as file:
    inference = np.load(file)


for i in range(len(inference)):
    print(i)
    extent_i = inference[i]['extent']
    preds = inference[i]['p_boxes']
    trues = inference[i]['t_boxes']
    scores = inference[i]['p_scores']
    pees = preds[np.where(preds[:,0] != 0)]
    tees = trues[np.where(trues[:,0] != 0)]

    pees = torch.tensor(pees)
    tees = torch.tensor(tees)

    #make boxes cover extent
    tees[:,(0,2)] = (tees[:,(0,2)]*(extent_i[1]-extent_i[0]))+extent_i[0]
    tees[:,(1,3)] = (tees[:,(1,3)]*(extent_i[3]-extent_i[2]))+extent_i[2]

    pees[:,(0,2)] = (pees[:,(0,2)]*(extent_i[1]-extent_i[0]))+extent_i[0]
    pees[:,(1,3)] = (pees[:,(1,3)]*(extent_i[3]-extent_i[2]))+extent_i[2]

    #wrap check
    # pees = wrap_check_NMS(pees,scores,MIN_CELLS_PHI,MAX_CELLS_PHI,threshold=0.2)
    tees = utils.wrap_check_truth(tees,MIN_CELLS_PHI,MAX_CELLS_PHI)

    #get the cells
    h5f = inference[i]['h5file']
    event_no = inference[i]['event_no']
    if h5f.decode('utf-8')=="01":
        cells_file = "../../../user.cantel.34126190._0000{}.calocellD3PD_mc16_JZ4W.r10788.h5".format(h5f.decode('utf-8'))
        with h5py.File(cells_file,"r") as f:
            h5group = f["caloCells"]
            cells = h5group["2d"][event_no]

        clusters_file = "../../../user.cantel.34126190._0000{}.topoclusterD3PD_mc16_JZ4W.r10788.h5".format(h5f.decode('utf-8'))
        with h5py.File(clusters_file,"r") as f:
            cl_data = f["caloCells"] 
            event_data = cl_data["1d"][event_no]
            cluster_data = cl_data["2d"][event_no]
            cluster_cell_data = cl_data["3d"][event_no]

        list_truth_cells, list_cl_cells = utils.RetrieveClusterCellsFromBox(cluster_data,cluster_cell_data,cells,tees)

        truth_box_cells_i = list_truth_cells[truth_box_number]
        mask = abs(truth_box_cells_i['cell_E'] / truth_box_cells_i['cell_Sigma']) > 2
        truth_box_cells_2sig_i = truth_box_cells_i[mask]
        cluster_cells_i = list_cl_cells[truth_box_number]

        feature_matrix = truth_box_cells_i[['cell_xCells','cell_yCells','cell_zCells','cell_E','cell_eta','cell_phi','cell_Sigma','cell_IdCells','cell_TimeCells']].view(np.float32).reshape(truth_box_cells_i.shape + (-1,))
        edge_index = torch_geometric.nn.knn_graph(torch.tensor(feature_matrix[:, :3]),k=3)


        if os.path.exists(f"../plots/{truth_box_number}/") is False: os.makedirs(f"../plots/{truth_box_number}/")
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(truth_box_cells_i['cell_xCells'], truth_box_cells_i['cell_zCells'], truth_box_cells_i['cell_yCells'], c='b', marker='o')
        for src, dst in edge_index.t().tolist():
            x_src, y_src, z_src = truth_box_cells_i[src][['cell_xCells','cell_yCells','cell_zCells']]
            x_dst, y_dst, z_dst = truth_box_cells_i[dst][['cell_xCells','cell_yCells','cell_zCells']]
            
            ax.plot([x_src, x_dst], [z_src, z_dst], [y_src, y_dst], c='r')

        ax.set(xlabel='X',ylabel='Z',zlabel='Y',title=f'Graph with KNN 3 Edges')
        plt.show()
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

        break


