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

        l_topo_cells = utils.RetrieveCellIdsFromCluster(cells,cluster_cell_data)
        l_true_cells = utils.RetrieveCellIdsFromBox(cells,tees)

        list_truth_cells, list_cl_cells = utils.RetrieveClusterCellsFromBox(cluster_data,cluster_cell_data,cells,tees)



        # truth_box_number = 1
        truth_box_cells_i = list_truth_cells[truth_box_number]
        mask = abs(truth_box_cells_i['cell_E'] / truth_box_cells_i['cell_Sigma']) > 2
        truth_box_cells_2sig_i = truth_box_cells_i[mask]
        cluster_cells_i = list_cl_cells[truth_box_number]

        if os.path.exists(f"../plots/{truth_box_number}/") is False: os.makedirs(f"../plots/{truth_box_number}/")

        figure = plt.figure(figsize=(14, 8))
        ax1 = figure.add_subplot(131, projection='3d')
        ax2 = figure.add_subplot(132, projection='3d')
        ax3 = figure.add_subplot(133, projection='3d')
        ax1.scatter(truth_box_cells_i['cell_xCells'], truth_box_cells_i['cell_zCells'], truth_box_cells_i['cell_yCells'], marker='o',s=4,color='green')
        ax3.scatter(truth_box_cells_2sig_i['cell_xCells'], truth_box_cells_2sig_i['cell_zCells'], truth_box_cells_2sig_i['cell_yCells'], marker='o',s=4,color='green')
        for cl_idx in range(len(cluster_cells_i)):
            cluster_inside_box = cluster_cells_i[cl_idx]
            ax2.scatter(cluster_inside_box['cell_xCells'], cluster_inside_box['cell_zCells'], cluster_inside_box['cell_yCells'], marker='^',s=4)

        ax2.set(xlim=ax1.get_xlim(),ylim=ax1.get_ylim(),zlim=ax1.get_zlim())
        ax3.set(xlim=ax1.get_xlim(),ylim=ax1.get_ylim(),zlim=ax1.get_zlim())
        ax1.set(xlabel='X',ylabel='Z',zlabel='Y',title=f'({len(truth_box_cells_i)}) Truth Box Cells')
        ax2.set(xlabel='X',ylabel='Z',zlabel='Y',title=f'({len(np.concatenate(cluster_cells_i))}) Total Cluster Cells, {len(cluster_cells_i)} Clusters')
        ax3.set(xlabel='X',ylabel='Z',zlabel='Y',title=f'({len(truth_box_cells_2sig_i)}) Truth Box Cells |> 2sig|')
        figure.savefig(f'../plots/{truth_box_number}/cells-graph.{image_format}',dpi=400,format=image_format,bbox_inches="tight")


        figure = plt.figure(figsize=(14, 8))
        ax1 = figure.add_subplot(131, projection='3d')
        ax2 = figure.add_subplot(132, projection='3d')
        ax3 = figure.add_subplot(133, projection='3d')

        ax1.scatter(truth_box_cells_i['cell_phi'], truth_box_cells_i['cell_zCells'], np.sqrt(truth_box_cells_i['cell_yCells']**2+truth_box_cells_i['cell_xCells']**2), marker='o',s=4,color='green')
        ax3.scatter(truth_box_cells_2sig_i['cell_phi'], truth_box_cells_2sig_i['cell_zCells'], np.sqrt(truth_box_cells_2sig_i['cell_yCells']**2+truth_box_cells_2sig_i['cell_xCells']**2), marker='o',s=4,color='green')
        for cl_idx in range(len(cluster_cells_i)):
            cluster_inside_box = cluster_cells_i[cl_idx]
            ax2.scatter(cluster_inside_box['cell_phi'], cluster_inside_box['cell_zCells'], np.sqrt(cluster_inside_box['cell_yCells']**2+cluster_inside_box['cell_xCells']**2), marker='^',s=4)

        ax2.set(xlim=ax1.get_xlim(),ylim=ax1.get_ylim(),zlim=ax1.get_zlim())
        ax3.set(xlim=ax1.get_xlim(),ylim=ax1.get_ylim(),zlim=ax1.get_zlim())
        ax1.set(xlabel='phi',ylabel='Z',zlabel='R',title=f'({len(truth_box_cells_i)}) Truth Box Cells')
        ax2.set(xlabel='phi',ylabel='Z',zlabel='R',title=f'({len(np.concatenate(cluster_cells_i))}) Total Cluster Cells')
        ax3.set(xlabel='phi',ylabel='Z',zlabel='R',title=f'({len(truth_box_cells_2sig_i)}) Truth Box Cells |> 2sig|')
        figure.savefig(f'../plots/{truth_box_number}/cells-graph-zphir.{image_format}',dpi=400,format=image_format,bbox_inches="tight")



        break




#now we have the boxes, the clusters and the cells we can begin making


#file structure is
#plots/index/total_event/
#plots/index/cluster_X/





