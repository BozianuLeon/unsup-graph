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

        cells_r = np.sqrt(cells['cell_yCells']**2+cells['cell_xCells']**2)
        plt.figure(figsize=(14,8))
        n,bins,_=plt.hist(cells_r,bins=1000,histtype='step')
        plt.xlabel('r')
        plt.title(f'All cells radius ({len(cells_r)})')
        plt.savefig(f'../plots/cells/logradius_hist.png')
        print(np.unique(cells['cell_DetCells']))
        print(len(np.unique(cells['cell_DetCells'])))
        

        # split into subdetectors
        EM_layers = [65,81,97,113,  #EM barrel
                    257,273,289,305, #EM Endcap
                    145,161, # IW EM (inner wheel)
                    2052] #EM FCAL

        HAD_layers = [2,514,1026,1538, #HEC layers
                     4100,6148, #FCAL HAD
                     65544,73736,81928, #Tile barrel
                     131080,139272,147464, #Tile endcap
                     811016,278536,270344] #Tile gap
        
        EM_indices = np.isin(cells['cell_DetCells'],EM_layers)
        HAD_indices = np.isin(cells['cell_DetCells'],HAD_layers)
        EMBar_indices = np.isin(cells['cell_DetCells'],[65,81,97,113])
        EMEC_indices = np.isin(cells['cell_DetCells'],[257,273,289,305])
        EMIW_indices = np.isin(cells['cell_DetCells'],[145,161])
        EMFCAL_indices = np.isin(cells['cell_DetCells'],[2052])

        HEC_indices = np.isin(cells['cell_DetCells'],[2,514,1026,1538])
        HFCAL_indices = np.isin(cells['cell_DetCells'],[4100,6148])
        TileBar_indices = np.isin(cells['cell_DetCells'],[65544,73736,81928])
        TileEC_indices = np.isin(cells['cell_DetCells'],[131080,139272,147464])
        TileGap_indices = np.isin(cells['cell_DetCells'],[811016,278536,270344])

        plt.figure(figsize=(14,8))
        plt.hist(cells_r[EM_indices],bins=bins,histtype='step',label='EM')
        plt.hist(cells_r[HAD_indices],bins=bins,histtype='step',label='HAD')
        plt.xlabel('r')
        plt.legend()
        # plt.yscale('log')
        plt.title(f'All cells radius ({len(cells_r[EM_indices]),len(cells_r[HAD_indices])})')
        plt.savefig(f'../plots/cells/radius_EMHAD_hist.png')

        plt.figure(figsize=(14,8))
        plt.hist(cells_r[EMBar_indices],bins=bins,histtype='step',label='EMBar')
        plt.hist(cells_r[EMEC_indices],bins=bins,histtype='step',label='EMEC')
        plt.hist(cells_r[EMIW_indices],bins=bins,histtype='step',label='EMIW')
        plt.hist(cells_r[EMFCAL_indices],bins=bins,histtype='step',label='EMFCAL')
        plt.hist(cells_r[HEC_indices],bins=bins,histtype='step',label='HEC')
        plt.hist(cells_r[HFCAL_indices],bins=bins,histtype='step',label='HFCAL')
        plt.hist(cells_r[TileBar_indices],bins=bins,histtype='step',label='TileBar')
        plt.hist(cells_r[TileEC_indices],bins=bins,histtype='step',label='TileEC')
        plt.hist(cells_r[TileGap_indices],bins=bins,histtype='step',label='TileGap')
        # plt.axvline(x=250,color='red',label='Buckets')
        # plt.axvline(x=325,color='red')
        # plt.axvline(x=615,color='red')
        # plt.axvline(x=1450,color='red')
        # plt.axvline(x=1650,color='red')
        # plt.axvline(x=1775,color='red')
        # plt.axvline(x=2150,color='red')
        # plt.axvline(x=2500,color='red')
        # plt.axvline(x=3000,color='red')
        # plt.axvline(x=3250,color='red')
        # plt.axvline(x=3500,color='red')
        plt.xlabel('r')
        plt.legend()
        plt.yscale('log')
        plt.title(f'All cells radius ({len(cells_r[EM_indices]),len(cells_r[HAD_indices])})')
        plt.savefig(f'../plots/cells/radius_EMHADLayers_hist_log.png')



        plt.figure(figsize=(14,8))
        f,ax = plt.subplots(8,1,figsize=(14,15))
        #EM Cells
        ax[0].hist(cells_r[cells['cell_DetCells']==65],bins=bins,histtype='step',label='PreSamplerB (65)')
        ax[0].hist(cells_r[cells['cell_DetCells']==81],bins=bins,histtype='step',label='EMB1 (81)')
        ax[0].hist(cells_r[cells['cell_DetCells']==97],bins=bins,histtype='step',label='EMB2 (97)')
        ax[0].hist(cells_r[cells['cell_DetCells']==113],bins=bins,histtype='step',label='EMB3 (113)')

        ax[4].hist(cells_r[cells['cell_DetCells']==257],bins=bins,histtype='step',label='PresamplerE (257)')
        ax[4].hist(cells_r[cells['cell_DetCells']==273],bins=bins,histtype='step',label='EME1 (273)')

        ax[5].hist(cells_r[cells['cell_DetCells']==145],bins=bins,histtype='step',label='EME2 (IW) (145)')
        ax[5].hist(cells_r[cells['cell_DetCells']==161],bins=bins,histtype='step',label='EME3 (IW) (161)')
        ax[5].hist(cells_r[cells['cell_DetCells']==289],bins=bins,histtype='step',label='EME2 (289)')
        ax[5].hist(cells_r[cells['cell_DetCells']==305],bins=bins,histtype='step',label='EME3 (305)')

        #Hadronic cells
        ax[1].hist(cells_r[cells['cell_DetCells']==65544],bins=bins,histtype='step',label='TileBar0 (65544)')
        ax[1].hist(cells_r[cells['cell_DetCells']==73736],bins=bins,histtype='step',label='TileBar1 (73736)')
        ax[1].hist(cells_r[cells['cell_DetCells']==81928],bins=bins,histtype='step',label='TileBar2 (81928)')

        ax[2].hist(cells_r[cells['cell_DetCells']==131080],bins=bins,histtype='step',label='TileExt0 (131080)')
        ax[2].hist(cells_r[cells['cell_DetCells']==139272],bins=bins,histtype='step',label='TileExt1 (139272)')
        ax[2].hist(cells_r[cells['cell_DetCells']==147464],bins=bins,histtype='step',label='TileExt2 (147464)')

        ax[3].hist(cells_r[cells['cell_DetCells']==270344],bins=bins,histtype='step',label='TileGap1 (270344)')
        ax[3].hist(cells_r[cells['cell_DetCells']==278536],bins=bins,histtype='step',label='TileGap2 (278536)')
        ax[3].hist(cells_r[cells['cell_DetCells']==811016],bins=bins,histtype='step',label='TileGap3 (811016)')

        ax[6].hist(cells_r[cells['cell_DetCells']==2],bins=bins,histtype='step',label='HEC0 (2)')
        ax[6].hist(cells_r[cells['cell_DetCells']==514],bins=bins,histtype='step',label='HEC1 (514)')
        ax[6].hist(cells_r[cells['cell_DetCells']==1026],bins=bins,histtype='step',label='HEC2 (1026)')
        ax[6].hist(cells_r[cells['cell_DetCells']==1538],bins=bins,histtype='step',label='HEC3 (1538)')

        ax[7].hist(cells_r[cells['cell_DetCells']==2052],bins=bins,histtype='step',label='FCAL0 (EM) (2052)')
        ax[7].hist(cells_r[cells['cell_DetCells']==4100],bins=bins,histtype='step',label='FCAL1 (HAD) (4100)')
        ax[7].hist(cells_r[cells['cell_DetCells']==6148],bins=bins,histtype='step',label='FCAL2 (HAD) (6148)')
        

        for a in range(len(ax)):
            ax[a].legend(fontsize='x-small')
            # ax[a].set(ylabel='Freq.',yscale='log')
        ax[7].set(xlabel='cell_r',)
        # plt.yscale('log')
        plt.xlabel('cell_r')
        ax[0].set_title(f'Barrel cells')
        ax[4].set_title(f'Endcap cells')
        f.subplots_adjust(hspace=0.7)
        plt.savefig(f'../plots/cells/logradius_individual_hist.png',bbox_inches="tight")
        quit()

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





