import torch
import torchvision
import torch_geometric
import numpy as np
import numpy.lib.recfunctions as rf
import matplotlib.pyplot as plt
import matplotlib
import h5py
import os
import time

from utils import wrap_check_truth, remove_nan, perpendicular_dists, RetrieveCellIdsFromCluster, RetrieveClusterCellsFromBox
from metrics import get_physics_dictionary

MIN_CELLS_PHI,MAX_CELLS_PHI = -3.1334076, 3.134037
MIN_CELLS_ETA,MAX_CELLS_ETA = -4.823496, 4.823496
markers = [".",",","o","v","^","<",">","1","2","3","4","8","s","p","P","*","h","H","+","x","X","D","d","|","_"]*4


class Net(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=128):
        super().__init__()
        self.n_clusters = out_channels

        self.norm = torch_geometric.nn.GraphNorm(in_channels)
        self.conv1 = torch_geometric.nn.GCNConv(in_channels, hidden_channels)
        self.relu = torch.nn.ReLU()
        self.selu = torch.nn.SELU()
        self.pool1 = torch_geometric.nn.DMoNPooling(hidden_channels,out_channels)

    def forward(self, x, edge_index, batch):

        x = self.norm(x)
        x = self.conv1(x, edge_index)
        # x = self.relu(x)
        x = self.selu(x)
        # return F.log_softmax(x,dim=-1), 0

        x, mask = torch_geometric.utils.to_dense_batch(x, batch)
        adj = torch_geometric.utils.to_dense_adj(edge_index, batch)
        s, x, adj, sp1, o1, c1 = self.pool1(x, adj, mask)

        return torch.nn.functional.log_softmax(x, dim=-1), sp1+o1+c1, s






if __name__=='__main__':

    # # Get data
    with open("../../struc_array.npy", "rb") as file:
        inference_array = np.load(file)

    #Load in models!
    n_graphs = 604
    box_eta_cut=2.1
    cell_significance_cut=2
    k=4
    name = "calo"
    dataset_name = f"xyzdeltaR_{n_graphs}_{box_eta_cut}_{cell_significance_cut}_{k}"
    num_clusters = "..."
    hidden_channels = 256
    num_epochs = 30

    # Initialise 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model3 = Net(in_channels=4,hidden_channels=hidden_channels, out_channels=3).to(device)
    model5 = Net(in_channels=4,hidden_channels=hidden_channels, out_channels=5).to(device)
    model10 = Net(in_channels=4,hidden_channels=hidden_channels, out_channels=10).to(device)
    model15 = Net(in_channels=4,hidden_channels=hidden_channels, out_channels=15).to(device)
    model20 = Net(in_channels=4,hidden_channels=hidden_channels, out_channels=20).to(device)


    # Load
    model3.load_state_dict(torch.load("models/" + f"calo_dmon_{dataset_name}_data_{hidden_channels}nn_{model3.n_clusters}c_{num_epochs}e" + ".pt"))
    model5.load_state_dict(torch.load("models/" + f"calo_dmon_{dataset_name}_data_{hidden_channels}nn_{model5.n_clusters}c_{num_epochs}e" + ".pt"))
    model10.load_state_dict(torch.load("models/" + f"calo_dmon_{dataset_name}_data_{hidden_channels}nn_{model10.n_clusters}c_{num_epochs}e" + ".pt"))
    model15.load_state_dict(torch.load("models/" + f"calo_dmon_{dataset_name}_data_{hidden_channels}nn_{model15.n_clusters}c_{num_epochs}e" + ".pt"))
    model20.load_state_dict(torch.load("models/" + f"calo_dmon_{dataset_name}_data_{hidden_channels}nn_{model20.n_clusters}c_{num_epochs}e" + ".pt"))
    model3.eval()
    model5.eval()
    model10.eval()
    model15.eval()
    model20.eval()


    # Save here:
    save_loc = "plots/" + name +"/one_event/" + time.strftime("%Y%m%d-%H") + "/"
    os.makedirs(save_loc) if not os.path.exists(save_loc) else None


    # Make graphs for one event

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
                raw_E_mask = (cluster_data['cl_E_em']+cluster_data['cl_E_had']) > 5000 #5GeV cut
                cluster_data = cluster_data[raw_E_mask]
                cluster_cell_data = cl_data["3d"][event_no]
                cluster_cell_data = cluster_cell_data[raw_E_mask]
        
            jets_file = "../../user.cantel.34126190._0000{}.jetD3PD_mc16_JZ4W.r10788.h5".format(h5f.decode('utf-8'))
            with h5py.File(jets_file,"r") as f:
                j_data = f["caloCells"]
                jet_data = j_data["2d"][event_no]
                jet_data = remove_nan(jet_data)

            ##----------------------------------------------------------------------------------------
            # Take truth boxes, return GNN clusters:
            list_truth_cells, list_cl_cells = RetrieveClusterCellsFromBox(cluster_data, cluster_cell_data, cells, tees)
            list_topo_cells = RetrieveCellIdsFromCluster(cells,cluster_cell_data)
            list_gnn_cells = list()   
            for truth_box_number in range(len(list_truth_cells)):
                print(f'\tBox number {truth_box_number}')
                truth_box_cells_i = list_truth_cells[truth_box_number]
                cluster_cells_i = list_cl_cells[truth_box_number]

                if (np.abs(np.mean(truth_box_cells_i['cell_eta']))<box_eta_cut):
                    # only take cells above 2 sigma
                    mask = abs(truth_box_cells_i['cell_E'] / truth_box_cells_i['cell_Sigma']) >= cell_significance_cut
                    truth_box_cells_2sig_i = truth_box_cells_i[mask]
                    if (len(truth_box_cells_2sig_i['cell_eta'])>50):
                        # make edges
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

                        edge_indices = torch_geometric.nn.knn_graph(coordinates,k=k,loop=False)
                        event_graph  = torch_geometric.data.Data(x=torch.column_stack([coordinates,cell_delta_R]),edge_index=edge_indices,y=cell_ids) 

                        ##----------------------------------------------------------------------------------------
                        # Run inference
                        ##----------------------------------------------------------------------------------------
                        
                        # how many clusters should the GNN look for? Naive cut:
                        if (len(truth_box_cells_2sig_i['cell_eta'])<100):
                            pred,tot_loss,clus_ass = model3(event_graph.x,event_graph.edge_index,batch=None) #no batch?
                        elif (len(truth_box_cells_2sig_i['cell_eta'])<175):
                            pred,tot_loss,clus_ass = model5(event_graph.x,event_graph.edge_index,batch=None) 
                        elif (len(truth_box_cells_2sig_i['cell_eta'])<300):
                            pred,tot_loss,clus_ass = model10(event_graph.x,event_graph.edge_index,batch=None) 
                        elif (len(truth_box_cells_2sig_i['cell_eta'])<425):
                            pred,tot_loss,clus_ass = model15(event_graph.x,event_graph.edge_index,batch=None) 
                        else:
                            pred,tot_loss,clus_ass = model20(event_graph.x,event_graph.edge_index,batch=None) 

                        # retrieve clusters from GNN
                        predicted_classes = clus_ass.squeeze().argmax(dim=1).numpy()
                        unique_values, counts = np.unique(predicted_classes, return_counts=True)

                        gnn_cluster_cells_i = list()
                        for cluster_no in unique_values:
                            cluster_id = predicted_classes==cluster_no
                            gnn_cluster_ids = event_graph.y[cluster_id]
                            cell_mask = np.isin(cells['cell_IdCells'],gnn_cluster_ids.detach().numpy())
                            gnn_desired_cells = cells[cell_mask][['cell_E','cell_eta','cell_phi','cell_Sigma','cell_IdCells','cell_DetCells','cell_xCells','cell_yCells','cell_zCells','cell_TimeCells']]
                            gnn_cluster_cells_i.append(gnn_desired_cells)
                            list_gnn_cells.append(gnn_desired_cells)
                        print('\t',len(gnn_cluster_cells_i), 'GNN clusters there were', len(cluster_cells_i), 'TCs', len(coordinates), 'cells')
                    
                    # Get boxes that are too small (<50 2sig cells)
                    else:
                        print('\tToo few cells box')
                        small_point_clouds = truth_box_cells_2sig_i[['cell_E','cell_eta','cell_phi','cell_Sigma','cell_IdCells','cell_DetCells','cell_xCells','cell_yCells','cell_zCells','cell_TimeCells']]
                        list_gnn_cells.append(small_point_clouds)
                # Get boxes that are in the forward region, where we do not use GNN
                else:
                    print('\tToo forward box')
                    mask = abs(truth_box_cells_i['cell_E'] / truth_box_cells_i['cell_Sigma']) >= cell_significance_cut
                    truth_box_cells_2sig_i = truth_box_cells_i[mask]
                    point_clouds_in_forward = truth_box_cells_2sig_i[['cell_E','cell_eta','cell_phi','cell_Sigma','cell_IdCells','cell_DetCells','cell_xCells','cell_yCells','cell_zCells','cell_TimeCells']]
                    list_gnn_cells.append(point_clouds_in_forward)
            
            print('Number of clusters?',len(list_topo_cells))
            print('Number of truth boxes',len(list_truth_cells),tees.shape)
            print('Number of GNN clusters',len(list_gnn_cells))
            print('\n\n\n')
            ##----------------------------------------------------------------------------------------

            tc_phys_dict = get_physics_dictionary(list_topo_cells)
            tb_phys_dict = get_physics_dictionary(list_truth_cells)
            gcl_phys_dict = get_physics_dictionary(list_gnn_cells)


            # project in eta-phi
            f,ax = plt.subplots(1,2,figsize=(10,5))
            for t in tees:
                ax[0].add_patch(matplotlib.patches.Rectangle((t[0],t[1]),t[2]-t[0],t[3]-t[1],lw=1.25,ec='green',fc='none'))

            ax[0].scatter(tc_phys_dict['eta'],tc_phys_dict['phi'],marker='*',color='dodgerblue',s=12,label='Topocl>5GeV')
            ax[1].scatter(gcl_phys_dict['eta'],gcl_phys_dict['phi'],marker='*',color='firebrick',s=12,label='GNN Clus.')

            ax[0].grid()
            ax[1].grid()
            ax[0].set(xlim=(MIN_CELLS_ETA,MAX_CELLS_ETA),ylim=(MIN_CELLS_PHI,MAX_CELLS_PHI),title=f'Event {i}, {len(tc_phys_dict["eta"])},{len(list_topo_cells)} Topocluster(s)')
            ax[1].set(xlim=(MIN_CELLS_ETA,MAX_CELLS_ETA),ylim=(MIN_CELLS_PHI,MAX_CELLS_PHI),title=f'Event {i}, {len(gcl_phys_dict["eta"])},{len(list_gnn_cells)} GNN Clusters')
            ax[0].legend(bbox_to_anchor=(1.01, 0.25),loc='lower left',fontsize='x-small')
            ax[1].legend(bbox_to_anchor=(1.01, 0.25),loc='lower left',fontsize='x-small')
            f.tight_layout()
            plt.show()
            plt.close()

            f,ax = plt.subplots(1,3,figsize=(15,5))
            for t in tees:
                ax[0].add_patch(matplotlib.patches.Rectangle((t[0],t[1]),t[2]-t[0],t[3]-t[1],lw=1.25,ec='green',fc='none'))

            ax[0].scatter(tc_phys_dict['eta'],tc_phys_dict['phi'],marker='*',color='dodgerblue',s=np.abs(tc_phys_dict['significance']),label='Topocl>5GeV')
            ax[1].scatter(gcl_phys_dict['eta'],gcl_phys_dict['phi'],marker='*',color='firebrick',s=np.abs(gcl_phys_dict['significance']),label='GNN Clus.')
            ax[2].scatter(tb_phys_dict['eta'],tb_phys_dict['phi'],marker='*',color='green',s=np.abs(tb_phys_dict['significance']),label='TBoxes')
            for ax_i in ax:
                ax_i.grid()
                ax_i.set(xlim=(MIN_CELLS_ETA,MAX_CELLS_ETA),ylim=(MIN_CELLS_PHI,MAX_CELLS_PHI))
                ax_i.legend(bbox_to_anchor=(1.01, 0.25),loc='lower left',fontsize='x-small')
            f.tight_layout()
            plt.show()
            plt.close()




            quit()


        
        
        
        
        
        


