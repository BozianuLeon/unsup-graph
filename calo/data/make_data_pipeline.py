import numpy as np
import numpy.lib.recfunctions as rf
import scipy
import h5py
import matplotlib.pyplot as plt
import pickle
import os

import torch
import torch_geometric

import utils
MIN_CELLS_PHI,MAX_CELLS_PHI = -3.1334076, 3.134037
MIN_CELLS_ETA,MAX_CELLS_ETA = -4.823496, 4.823496

with open("../../../struc_array.npy", "rb") as file:
    inference = np.load(file)


data_list = []
clusters_list = []
for i in range(len(inference)):
    
    h5f = inference[i]['h5file']
    event_no = inference[i]['event_no']
    if h5f.decode('utf-8')=="01":
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

        print(i)
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

        for truth_box_number in range(len(list_truth_cells)):
            truth_box_cells_i = list_truth_cells[truth_box_number]
            cluster_cells_i = list_cl_cells[truth_box_number]

            if np.abs(np.mean(truth_box_cells_i['cell_eta']))<1.5:
                mask = abs(truth_box_cells_i['cell_E'] / truth_box_cells_i['cell_Sigma']) >= 2
                truth_box_cells_2sig_i = truth_box_cells_i[mask]
                truth_box_cells_i = truth_box_cells_2sig_i
                # calculate edges
                struc_array = truth_box_cells_i[['cell_xCells','cell_yCells','cell_zCells','cell_E','cell_eta','cell_phi','cell_Sigma','cell_IdCells','cell_TimeCells']].copy()
                feature_matrix =  rf.structured_to_unstructured(struc_array,dtype=np.float32)
                cell_radius = torch.tensor(np.sqrt(truth_box_cells_i['cell_yCells']**2+truth_box_cells_i['cell_xCells']**2))
                vectorized_mapping = np.vectorize(utils.encode_layers)
                cell_layer = torch.tensor(vectorized_mapping(truth_box_cells_i['cell_DetCells']))
                cell_significance = torch.tensor(abs(truth_box_cells_i['cell_E'] / truth_box_cells_i['cell_Sigma']))
                feature_tensor = torch.column_stack((torch.tensor(feature_matrix),cell_radius,cell_layer,cell_significance))
                
                #normalise features
                def normalize(x):
                    x_normed = x / x.max(0, keepdim=True)[0]
                    return x_normed
                feature_tensor_norm = normalize(feature_tensor)

                ten_perc = max(int(len(cell_significance)/10),24) #make sure we have at least 12 neighbours
                one_perc = max(int(len(cell_significance)/100),8)
                edge_index = utils.var_knn_graph(feature_tensor_norm[:, [0,1,2]],k=[1,2,3,one_perc,ten_perc],quantiles=[0.25,0.5,0.9,0.99],x_ranking=cell_significance)
                # edge_index = utils.var_knn_graph(feature_tensor_norm[:, [0,1,2]],k=[1,2,3,25],quantiles=[0.25,0.5,0.98],x_ranking=cell_significance)
                # edge_index = torch_geometric.nn.knn_graph(feature_tensor_norm[:, [0,1,2]],k=3,loop=False)

                # put in graph Data object
                # truth_box_graph_data = torch_geometric.data.Data(x=feature_tensor[:, [0,1,2,4,5,-3,-2,-1]],edge_index=edge_index) 
                
                truth_box_graph_data = torch_geometric.data.Data(x=feature_tensor_norm[:, [0,1,2]],edge_index=edge_index) 

                # append to list
                data_list.append(truth_box_graph_data)
                clusters_list.append(cluster_cells_i)


                # figure = plt.figure(figsize=(14, 8))
                # # 1. input graph
                # ax1 = figure.add_subplot(121, projection='3d')
                # ax1.scatter(truth_box_graph_data.x[:, 0], truth_box_graph_data.x[:, 2], truth_box_graph_data.x[:, 1], c="b", marker='.')
                # # ax1.scatter(truth_box_cells_i['cell_xCells'], truth_box_cells_i['cell_yCells'], truth_box_cells_i['cell_zCells'], c="b", marker='o',s=4*cell_significance)
                # for src, dst in truth_box_graph_data.edge_index.t().tolist():
                #     x_src, y_src, z_src = truth_box_graph_data.x[src][:3]
                #     x_dst, y_dst, z_dst = truth_box_graph_data.x[dst][:3]
                #     ax1.plot([x_src, x_dst], [z_src, z_dst], [y_src, y_dst], c='r',alpha=0.5)
                
                # # 2. Model output
                # # ax2 = figure.add_subplot(132, projection='3d')
                # # scatter = ax2.scatter(truth_box_graph_data.x[:, 0], truth_box_graph_data.x[:, 2], truth_box_graph_data.x[:, 1], c='green', marker='o',s=3)
                # # labels = [f"{value}: ({count})" for value,count in zip(unique_values, counts)]
                # # ax2.legend(handles=scatter.legend_elements(num=None)[0],labels=labels,title=f"Classes {len(unique_values)}/{num_clusters}",bbox_to_anchor=(1.07, 0.25),loc='lower left')

                # # 3. True topoclusters
                # ax3 = figure.add_subplot(122, projection='3d')
                # ax3.set(xlim=ax1.get_xlim(),ylim=ax1.get_ylim(),zlim=ax1.get_zlim())
                # for cl_idx in range(len(cluster_cells_i)):
                #     cluster_inside_box = cluster_cells_i[cl_idx]
                #     ax3.scatter(cluster_inside_box['cell_xCells'], cluster_inside_box['cell_zCells'], cluster_inside_box['cell_yCells'], marker='^',s=4*abs(cluster_inside_box['cell_E']/cluster_inside_box['cell_Sigma']))
                # ax1.set(xlabel='X',ylabel='Z',zlabel='Y',title=f'Input Graph (2sig cut {len(truth_box_graph_data.x)} cells) varKNN ({len(truth_box_graph_data.edge_index.t())} edges)')
                # # ax2.set(xlabel='X',ylabel='Z',zlabel='Y',title=f'Model Output Cluster Assignments')
                # ax3.set(xlabel='X',ylabel='Z',zlabel='Y',title=f'{len(cluster_cells_i)} Topocluster(s), ({len(np.concatenate(cluster_cells_i))} tot. cells)')
                # # figure.savefig(f'../plots/0/graphs_and_clusters/{truth_box_number}dmon_clus_norm_zphiRsignif.png', bbox_inches="tight")
                # plt.show()
                # plt.close()
                # # quit()
                # if truth_box_number==len(list_truth_cells)-1:
                #     quit()



with open('lists/truth_box_graphs_2sig_knn123onetenperc_norm_xyz.pkl', 'wb') as f1:
    pickle.dump(data_list, f1)

with open('lists/clusters_list.pkl', 'wb') as f2:
    pickle.dump(clusters_list, f2)

