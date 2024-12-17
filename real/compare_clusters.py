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
import random

from utils import wrap_check_truth, perpendicular_dists, RetrieveCellIdsFromCluster, RetrieveClusterCellsFromBox

MIN_CELLS_PHI,MAX_CELLS_PHI = -3.1334076, 3.134037
MIN_CELLS_ETA,MAX_CELLS_ETA = -4.823496, 4.823496
markers = [".",",","o","v","^","<",">","1","2","3","4","8","s","p","P","*","h","H","+","x","X","D","d","|","_"]*4


class Net(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=128):
        super().__init__()

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

    # # get data
    with open("../../struc_array.npy", "rb") as file:
        inference_array = np.load(file)

    # dataset_name = "xyzdeltaR_689_1.5_2_3"
    # with open(f'datasets/data_{dataset_name}.pkl', 'rb') as f:
    #    data_list = pickle.load(f)

    # with open(f'datasets/topocl_data_689.pkl', 'rb') as f:
    #    topocl_list = pickle.load(f)
    # train_size = 0.8
    # train_data_list = data_list[:int(train_size*len(data_list))]
    # test_data_list = data_list[int(train_size*len(data_list)):]
    # test_topocl_list = topocl_list[int(train_size*len(data_list)):]

    n_graphs = 604
    box_eta_cut=2.1
    cell_significance_cut=2
    k=4
    name = "calo"
    dataset_name = f"xyzdeltaR_{n_graphs}_{box_eta_cut}_{cell_significance_cut}_{k}"

    #initialise model
    num_clusters = 15
    hidden_channels = 256
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net(in_channels=4,
                hidden_channels=hidden_channels, 
                out_channels=num_clusters).to(device)

    num_epochs = 30
    #load model
    model_name = f"calo_dmon_{dataset_name}_data_{hidden_channels}nn_{num_clusters}c_{num_epochs}e"
    model.load_state_dict(torch.load("models/"+model_name + ".pt"))
    model.eval()

    save_loc = "plots/" + name +"/" + model_name + "/" + time.strftime("%Y%m%d-%H") + "/"
    os.makedirs(save_loc) if not os.path.exists(save_loc) else None


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

            list_truth_cells, list_cl_cells = RetrieveClusterCellsFromBox(cluster_data, cluster_cell_data, cells, tees)
               
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


                        edge_indices = torch_geometric.nn.knn_graph(coordinates,k=k,loop=False)
                        event_graph  = torch_geometric.data.Data(x=torch.column_stack([coordinates,cell_delta_R]),edge_index=edge_indices,y=cell_ids) 

                        ##----------------------------------------------------------------------------------------
                        #immediately run inference:
                        pred,tot_loss,clus_ass = model(event_graph.x,event_graph.edge_index,None) #no batch?

                        predicted_classes = clus_ass.squeeze().argmax(dim=1).numpy()
                        unique_values, counts = np.unique(predicted_classes, return_counts=True)

                        gnn_cluster_cells_i = list()
                        for cluster_no in unique_values:
                            cluster_id = predicted_classes==cluster_no
                            gnn_cluster_ids = event_graph.y[cluster_id]
                            print(gnn_cluster_ids)
                            cell_mask = np.isin(cells['cell_IdCells'],gnn_cluster_ids.detach().numpy())
                            gnn_desired_cells = cells[cell_mask][['cell_E','cell_eta','cell_phi','cell_Sigma','cell_IdCells','cell_DetCells','cell_xCells','cell_yCells','cell_zCells','cell_TimeCells']]
                            print(gnn_desired_cells.shape)
                            # gnn_cluster_cells_i.append(gnn_cluster_ids.detach().numpy())
                            gnn_cluster_cells_i.append(gnn_desired_cells)


                        ##----------------------------------------------------------------------------------------
                        #make plots
                        fig = plt.figure(figsize=(12, 8))
                        # 1. Model Input
                        ax1 = fig.add_subplot(223, projection='3d')
                        ax1.scatter(event_graph.x[:, 0], event_graph.x[:, 2], event_graph.x[:, 1], c="b", marker='o',s=30)
                        for src, dst in event_graph.edge_index.t().tolist():
                            x_src, y_src, z_src = event_graph.x[src][:3]
                            x_dst, y_dst, z_dst = event_graph.x[dst][:3]
                            ax1.plot([x_src, x_dst], [z_src, z_dst], [y_src, y_dst], c='r',alpha=0.5)


                        # 2. Model Output
                        ax2 = fig.add_subplot(224, projection='3d')
                        ax2.set(xlim=ax1.get_xlim(),ylim=ax1.get_ylim(),zlim=ax1.get_zlim())
                        xs = event_graph.x[:, 0]
                        ys = event_graph.x[:, 1]
                        zs = event_graph.x[:, 2]
                        random.shuffle(markers)
                        colors = matplotlib.cm.gnuplot(np.linspace(0, 1, num_clusters))
                        for i in unique_values:
                            mask = predicted_classes==i
                            ax2.scatter(xs[mask],zs[mask],ys[mask],color=colors[i],marker=markers[i],s=64,label=f'{i}: {counts[np.where(unique_values==i)[0]][0]}')
                        # ax2.legend(bbox_to_anchor=(1.05, 0.25),loc='lower left',fontsize='x-small')

                        # 3. TopoClusters
                        ax3 = fig.add_subplot(221, projection='3d')
                        ax3.set(xlim=ax1.get_xlim(),ylim=ax1.get_ylim(),zlim=ax1.get_zlim())
                        for cl_idx in range(len(cluster_cells_i)):
                            cluster_inside_box = cluster_cells_i[cl_idx]
                            ax3.scatter(cluster_inside_box['cell_xCells'], cluster_inside_box['cell_zCells'], cluster_inside_box['cell_yCells'], marker=markers[cl_idx],s=48,label=f'TC {cl_idx}: {len(cluster_inside_box)}')
                        # ax3.legend(bbox_to_anchor=(1.07, 0.25),loc='lower left',fontsize='x-small')


                        # 4. TopoClusters 2 sigma cells
                        ax4 = fig.add_subplot(222, projection='3d')
                        ax4.set(xlim=ax1.get_xlim(),ylim=ax1.get_ylim(),zlim=ax1.get_zlim())
                        twosigmacl_cells = 0
                        for cl_idx in range(len(cluster_cells_i)):
                            cluster_inside_box = cluster_cells_i[cl_idx]
                            mask = abs(cluster_inside_box['cell_E'] / cluster_inside_box['cell_Sigma']) >= cell_significance_cut
                            cluster_cells_2sig_i = cluster_inside_box[mask]
                            twosigmacl_cells += len(cluster_cells_2sig_i)
                            ax4.scatter(cluster_cells_2sig_i['cell_xCells'], cluster_cells_2sig_i['cell_zCells'], cluster_cells_2sig_i['cell_yCells'], marker=markers[cl_idx],s=48,label=f'TC {cl_idx}: {len(cluster_cells_2sig_i)}')
                        # ax4.legend(bbox_to_anchor=(1.07, 0.25),loc='lower left',fontsize='x-small')

                        ax1.set(xlabel='X',ylabel='Z',zlabel='Y')
                        ax1.set_title(f'Model Input: {len(event_graph.x)} >2 signif. Cells, {len(event_graph.edge_index.t())} Edges',fontsize=10,y=0.98)
                        ax2.set(xlabel='X',ylabel='Z',zlabel='Y')
                        ax2.set_title(f'Model Output: {len(unique_values)} clusters',fontsize=10,y=0.98)
                        ax3.set(xlabel='X',ylabel='Z',zlabel='Y')
                        ax3.set_title(f'{len(cluster_cells_i)} Topocluster(s): {len(np.concatenate(cluster_cells_i))} cells',fontsize=10,y=0.98)
                        ax4.set(xlabel='X',ylabel='Z',zlabel='Y')
                        ax4.set_title(f'{len(cluster_cells_i)} Topocluster(s): {twosigmacl_cells} >2 signif. cells',fontsize=10,y=0.98)
                        fig.tight_layout()
                        plt.show()
                        plt.close()

                        if input("Continue to physics plots for this event (y/n)?") != 'y':
                            print('Exiting gracefully')
                            quit()

                        ##calculate physical properties
                        cl_etas, cl_phis, cl_Es, cl_xs, cl_ys, cl_zs = list(),list(),list(),list(),list(),list()
                        cl_etas_2sig, cl_phis_2sig, cl_Es_2sig, cl_xs_2sig, cl_ys_2sig, cl_zs_2sig, cl_ncells_2sig = list(),list(),list(),list(),list(),list(),list()
                        for cl_idx in range(len(cluster_cells_i)):
                            cluster_inside_box = cluster_cells_i[cl_idx]
                            cluster_eta = np.dot(cluster_inside_box['cell_eta'],np.abs(cluster_inside_box['cell_E'])) / sum(np.abs(cluster_inside_box['cell_E']))
                            cluster_phi = np.arctan2(np.sum(cluster_inside_box['cell_E'] * np.sin(cluster_inside_box['cell_phi'])), np.sum(cluster_inside_box['cell_E'] * np.cos(cluster_inside_box['cell_phi'])))
                            cl_etas.append(cluster_eta)
                            cl_phis.append(cluster_phi)
                            cl_Es.append(np.sum(cluster_inside_box['cell_E']))
                            maskInTime = cluster_inside_box['cell_E']>0
                            cl_xs.append(np.dot(cluster_inside_box[maskInTime]['cell_xCells'],cluster_inside_box[maskInTime]['cell_E']) / sum(cluster_inside_box[maskInTime]['cell_E']))
                            cl_ys.append(np.dot(cluster_inside_box[maskInTime]['cell_yCells'],cluster_inside_box[maskInTime]['cell_E']) / sum(cluster_inside_box[maskInTime]['cell_E']))
                            cl_zs.append(np.dot(cluster_inside_box[maskInTime]['cell_zCells'],cluster_inside_box[maskInTime]['cell_E']) / sum(cluster_inside_box[maskInTime]['cell_E']))
                            mask2sig = abs(cluster_inside_box['cell_E'] / cluster_inside_box['cell_Sigma']) >= cell_significance_cut
                            mask2sigInTime = (cluster_inside_box['cell_E']>0) & (abs(cluster_inside_box['cell_E'] / cluster_inside_box['cell_Sigma']) >= cell_significance_cut)
                            cluster_cells_2sig_i = cluster_inside_box[mask2sig]
                            cluster_eta_2sig = np.dot(cluster_cells_2sig_i['cell_eta'],np.abs(cluster_cells_2sig_i['cell_E'])) / sum(np.abs(cluster_cells_2sig_i['cell_E']))
                            cluster_phi_2sig = np.arctan2(np.sum(cluster_cells_2sig_i['cell_E'] * np.sin(cluster_cells_2sig_i['cell_phi'])), np.sum(cluster_cells_2sig_i['cell_E'] * np.cos(cluster_cells_2sig_i['cell_phi'])))
                            cl_etas_2sig.append(cluster_eta_2sig)
                            cl_phis_2sig.append(cluster_phi_2sig)
                            cl_Es_2sig.append(np.sum(cluster_cells_2sig_i['cell_E']))
                            cl_ncells_2sig.append(len(cluster_cells_2sig_i))
                            cl_xs_2sig.append(np.dot(cluster_inside_box[mask2sigInTime]['cell_xCells'],cluster_inside_box[mask2sigInTime]['cell_E']) / sum(cluster_inside_box[mask2sigInTime]['cell_E']))
                            cl_ys_2sig.append(np.dot(cluster_inside_box[mask2sigInTime]['cell_yCells'],cluster_inside_box[mask2sigInTime]['cell_E']) / sum(cluster_inside_box[mask2sigInTime]['cell_E']))
                            cl_zs_2sig.append(np.dot(cluster_inside_box[mask2sigInTime]['cell_zCells'],cluster_inside_box[mask2sigInTime]['cell_E']) / sum(cluster_inside_box[mask2sigInTime]['cell_E']))

                        gcl_etas, gcl_phis, gcl_Es, gcl_xs, gcl_ys, gcl_zs = list(),list(),list(), list(),list(),list()
                        for gcl_idx in range(len(gnn_cluster_cells_i)):
                            gcl_inside_box = gnn_cluster_cells_i[gcl_idx]
                            gcl_eta = np.dot(gcl_inside_box['cell_eta'],np.abs(gcl_inside_box['cell_E'])) / sum(np.abs(gcl_inside_box['cell_E']))
                            gcl_phi = np.arctan2(np.sum(gcl_inside_box['cell_E'] * np.sin(gcl_inside_box['cell_phi'])), np.sum(gcl_inside_box['cell_E'] * np.cos(gcl_inside_box['cell_phi'])))
                            gcl_etas.append(gcl_eta)
                            gcl_phis.append(gcl_phi)
                            gcl_Es.append(np.sum(gcl_inside_box['cell_E']))
                            maskInTime = gcl_inside_box['cell_E']>0
                            gcl_xs.append(np.dot(gcl_inside_box[maskInTime]['cell_xCells'],gcl_inside_box[maskInTime]['cell_E']) / sum(gcl_inside_box[maskInTime]['cell_E']))
                            gcl_ys.append(np.dot(gcl_inside_box[maskInTime]['cell_yCells'],gcl_inside_box[maskInTime]['cell_E']) / sum(gcl_inside_box[maskInTime]['cell_E']))
                            gcl_zs.append(np.dot(gcl_inside_box[maskInTime]['cell_zCells'],gcl_inside_box[maskInTime]['cell_E']) / sum(gcl_inside_box[maskInTime]['cell_E']))


                        # project in eta-phi
                        f,ax = plt.subplots(1,1,figsize=(5,6))
                        truth_box_i = tees[truth_box_number]
                        ax.add_patch(matplotlib.patches.Rectangle((truth_box_i[0],truth_box_i[1]),truth_box_i[2]-truth_box_i[0],truth_box_i[3]-truth_box_i[1],lw=1.25,ec='green',fc='none'))    
                        ax.scatter(cl_etas,cl_phis,marker='*',color='dodgerblue',s=12)

                        ax.grid()
                        ax.axhline(MIN_CELLS_PHI,ls='--',lw=2.0, color='purple')
                        ax.axhline(MAX_CELLS_PHI,ls='--',lw=2.0, color='purple')
                        ax.set(xlim=(MIN_CELLS_ETA,MAX_CELLS_ETA),ylim=(MIN_CELLS_PHI,MAX_CELLS_PHI),title=f'Box {truth_box_number}, ({truth_box_i[0]:.1f},{truth_box_i[1]:.1f}): {len(cluster_cells_i)} Topocluster(s)')
                        f.tight_layout()
                        plt.show()
                        plt.close()


                        f,ax = plt.subplots(1,2,figsize=(10,5))
                        truth_box_i = tees[truth_box_number]
                        ax[0].add_patch(matplotlib.patches.Rectangle((truth_box_i[0],truth_box_i[1]),truth_box_i[2]-truth_box_i[0],truth_box_i[3]-truth_box_i[1],lw=1.25,ec='green',fc='none'))
                        ax[1].add_patch(matplotlib.patches.Rectangle((truth_box_i[0],truth_box_i[1]),truth_box_i[2]-truth_box_i[0],truth_box_i[3]-truth_box_i[1],lw=2.5,ec='green',fc='none'))
                        ax[0].scatter(cl_etas,cl_phis,marker='*',color='dodgerblue',s=12)
                        for cl_idx in range(len(cluster_cells_i)):
                            ax[1].scatter(cl_etas[cl_idx],cl_phis[cl_idx],marker='*',color='dodgerblue',s=400,ec='black',label=f'TC {cl_idx}: ({len(cluster_cells_i[cl_idx])})')
                            ax[1].scatter(cl_etas_2sig[cl_idx],cl_phis_2sig[cl_idx],marker='*',color='cyan',s=400,ec='black',label=f'({cl_ncells_2sig[cl_idx]} > 2sig)')
                            ax[1].arrow(cl_etas[cl_idx],cl_phis[cl_idx],(cl_etas_2sig[cl_idx]-cl_etas[cl_idx]),(cl_phis_2sig[cl_idx]-cl_phis[cl_idx]),color='dodgerblue')

                        for gcl_idx in range(len(gnn_cluster_cells_i)):
                            ax[1].scatter(gcl_etas[gcl_idx],gcl_phis[gcl_idx],marker=markers[gcl_idx],s=75,color=colors[gcl_idx],label=f'GCl {gcl_idx}: ({len(gnn_cluster_cells_i[gcl_idx])})')

                        ax[0].set(xlim=(MIN_CELLS_ETA,MAX_CELLS_ETA),ylim=(MIN_CELLS_PHI,MAX_CELLS_PHI),title=f'Box {truth_box_number}, ({truth_box_i[0]:.1f},{truth_box_i[1]:.1f}): {len(cluster_cells_i)} Topocluster(s)')
                        ax[1].grid()
                        ax[1].legend(bbox_to_anchor=(1.01, 0.25),loc='lower left',fontsize='x-small')
                        f.tight_layout()
                        plt.show()
                        plt.close()


                        #plot in x-y, x-z
                        fig,axes = plt.subplots(1,2,figsize=(10,5))
                        axes[0].scatter(cl_xs,cl_ys,marker='*',color='dodgerblue',ec='black',s=400,label='TC')
                        axes[0].scatter(cl_xs_2sig,cl_ys_2sig,marker='*',color='cyan',ec='black',s=400,label='TC >2sig')
                        axes[0].scatter(gcl_xs,gcl_ys,marker=markers[0],color='firebrick',s=100,label='GCl')

                        axes[1].scatter(cl_xs,cl_zs,marker='*',color='dodgerblue',ec='black',s=400,label='TC')
                        axes[1].scatter(cl_xs_2sig,cl_zs_2sig,marker='*',color='cyan',ec='black',s=400,label='TC >2sig')
                        axes[1].scatter(gcl_xs,gcl_zs,marker=markers[0],color='firebrick',s=100,label='GCl')
                        
                        axes[0].set(xlabel='x',ylabel='y')
                        legend = axes[0].legend()
                        legend.get_frame().set_linewidth(3.0)
                        axes[0].grid()
                        axes[1].set(xlabel='x',ylabel='z')
                        axes[1].grid()
                        fig.suptitle('Position in Cartesian Coordinates of Topoclusters and ML Clusters', fontsize=16)
                        fig.tight_layout()
                        plt.show()
                        plt.close()


                        #plot energy, energy-z
                        fig,axes = plt.subplots(1,2,figsize=(10,5))
                        
                        axes[0].hist(cl_Es,bins=20,histtype='step',color='dodgerblue',label='Topoclusters')
                        axes[0].hist(cl_Es_2sig,bins=20,histtype='step',color='cyan',label='Topoclusters >2sig')
                        axes[0].hist(gcl_Es,bins=20,histtype='step',color='firebrick',label='GNN Clusters >2sig')
                        
                        axes[1].scatter(cl_zs,cl_Es,marker='*',color='dodgerblue',ec='black',s=400,label='TC')
                        axes[1].scatter(cl_zs_2sig,cl_Es_2sig,marker='*',color='cyan',ec='black',s=400,label='TC (>2sig)')
                        axes[1].quiver(cl_zs, cl_Es, (np.array(cl_zs_2sig)-np.array(cl_zs)), (np.array(cl_Es_2sig)-np.array(cl_Es)),angles='xy',scale_units='xy',scale=1,width=0.0065,color='dodgerblue')
                        axes[1].scatter(gcl_zs,gcl_Es,marker=markers[0],color='firebrick',s=100,label='GCl')
                        
                        axes[0].set(xlabel='Energy (EM scale)')
                        axes[0].legend()
                        axes[0].grid()
                        axes[1].set(xlabel='z',ylabel='Energy (EM Scale)')
                        axes[1].legend(bbox_to_anchor=(0.7, 1.01),loc='lower left',fontsize='x-small')
                        axes[1].grid()                                         
                        fig.suptitle('Raw energy of Topoclusters and ML Clusters',y=0.9 ,fontsize=16)
                        fig.tight_layout()
                        plt.show()
                        plt.close()     


                    
    





    print('\nModel name:\n',model_name)