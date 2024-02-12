import torch
import torchvision
import torch_geometric
import h5py
import os
import numpy as np
import numpy.lib.recfunctions as rf
import pickle

MIN_CELLS_PHI,MAX_CELLS_PHI = -3.1334076, 3.134037
MIN_CELLS_ETA,MAX_CELLS_ETA = -4.823496, 4.823496


from utils import wrap_check_truth, remove_nan, perpendicular_dists, RetrieveCellIdsFromCluster, RetrieveClusterCellsFromBox



def make_calo_data_list(
        inference_array,
        box_eta_cut=1.5,
        cell_significance_cut=2,
        k=3,    
    ):
    '''
    Turn truth (green) boxes into point clouds in pytorch geometric, save these for later training
    '''

    data_list = list()
    cluster_list = list()
    for i in range(len(inference_array)):
        h5f = inference_array[i]['h5file']
        event_no = inference_array[i]['event_no']
        if h5f.decode('utf-8')=="01":
            extent_i = inference_array[i]['extent']
            preds = inference_array[i]['p_boxes']
            trues = inference_array[i]['t_boxes']
            scores = inference_array[i]['p_scores']
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
            tees = wrap_check_truth(tees,MIN_CELLS_PHI,MAX_CELLS_PHI)

            print(i)
            cells_file = "../../user.cantel.34126190._0000{}.calocellD3PD_mc16_JZ4W.r10788.h5".format(h5f.decode('utf-8'))
            with h5py.File(cells_file,"r") as f:
                h5group = f["caloCells"]
                cells = h5group["2d"][event_no]

            clusters_file = "../../user.cantel.34126190._0000{}.topoclusterD3PD_mc16_JZ4W.r10788.h5".format(h5f.decode('utf-8'))
            with h5py.File(clusters_file,"r") as f:
                cl_data = f["caloCells"] 
                event_data = cl_data["1d"][event_no]
                cluster_data = cl_data["2d"][event_no]
                cluster_cell_data = cl_data["3d"][event_no]    

            list_truth_cells, list_cl_cells = RetrieveClusterCellsFromBox(cluster_data,cluster_cell_data,cells,tees)
               
            for truth_box_number in range(len(list_truth_cells)):
                truth_box_cells_i = list_truth_cells[truth_box_number]
                cluster_cells_i = list_cl_cells[truth_box_number]

                if (np.abs(np.mean(truth_box_cells_i['cell_eta']))<box_eta_cut):
                    # only take cells above 2 sigma
                    mask = abs(truth_box_cells_i['cell_E'] / truth_box_cells_i['cell_Sigma']) >= cell_significance_cut
                    truth_box_cells_2sig_i = truth_box_cells_i[mask]
                    if (len(truth_box_cells_2sig_i['cell_eta'])>50):
                        # calculate edges
                        struc_array = truth_box_cells_2sig_i[['cell_xCells','cell_yCells','cell_zCells','cell_E','cell_eta','cell_phi','cell_Sigma','cell_IdCells','cell_TimeCells']].copy()
                        feature_matrix =  rf.structured_to_unstructured(struc_array,dtype=np.float32)
                        # calculate features
                        coordinates = torch.tensor(feature_matrix[:,[0,1,2]])
                        cell_radius = torch.tensor(np.sqrt(truth_box_cells_2sig_i['cell_yCells']**2+truth_box_cells_2sig_i['cell_xCells']**2))
                        cell_significance = torch.tensor(abs(truth_box_cells_2sig_i['cell_E'] / truth_box_cells_2sig_i['cell_Sigma']))
                        cell_ids = torch.tensor(truth_box_cells_2sig_i['cell_IdCells'].astype(np.int64))

                        weighted_mean = torch.sum(coordinates*cell_significance.view(-1,1), dim=0) / torch.sum(cell_significance)
                        perp_vectors = perpendicular_dists(coordinates,weighted_mean)
                        cell_delta_R = torch.linalg.norm(perp_vectors,dim=1)
                        # feature_tensor = torch.column_stack((torch.tensor(feature_matrix),cell_radius, cell_delta_R,cell_significance))

                        edge_indices = torch_geometric.nn.knn_graph(coordinates,k=k,loop=False)
                        # edge_indicesxy = torch_geometric.nn.knn_graph(feature_tensor[:, [0,1]],k=k,loop=False)
                        event_graph  = torch_geometric.data.Data(x=torch.column_stack([coordinates,cell_delta_R]),edge_index=edge_indices,y=cell_ids) 
                        data_list.append(event_graph)
                        cluster_list.append(cluster_cells_i)
    
    return data_list, cluster_list




if __name__=="__main__":

    box_eta_cut=2.1
    cell_significance_cut=2
    norm=False
    k=4

    with open("../../struc_array.npy", "rb") as file:
        inference = np.load(file)

    data_list, topocl_list = make_calo_data_list(inference,
                                                 box_eta_cut,
                                                 cell_significance_cut,
                                                 k)

    with open(f'datasets/data_xyzdeltaR_{len(data_list)}_{box_eta_cut}_{cell_significance_cut}_{k}.pkl', 'wb') as f1:
        pickle.dump(data_list, f1)

    with open(f'datasets/topocl_data_{len(topocl_list)}.pkl', 'wb') as f2:
        pickle.dump(topocl_list, f2)






