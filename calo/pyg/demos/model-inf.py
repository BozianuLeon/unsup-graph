import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import numpy.lib.recfunctions as rf
import scipy
import os

import torch
import torchvision
from torch.nn import Linear
import torch.nn.functional as F
import torch_geometric



import numpy as np
import scipy
import h5py
import matplotlib.pyplot as plt
import os
import argparse

import torch
import torch_geometric

# import utils
MIN_CELLS_PHI,MAX_CELLS_PHI = -3.1334076, 3.134037
MIN_CELLS_ETA,MAX_CELLS_ETA = -4.823496, 4.823496
image_format = "png"

parser = argparse.ArgumentParser()
parser.add_argument('-idx','--idx', required=False)
args = vars(parser.parse_args())

truth_box_number = int(args['idx']) if args['idx'] is not None else 0



def wrap_check_truth(boxes,ymin,ymax):
    #here we look at truth boxes, remove the (wrapped) "duplicates" 
    #and mitigate those crossing the discontinuity
    #input is a np.ndarray containing the boxes in xyxy coords, after multiplication of extent
    if isinstance(boxes,np.ndarray):
        boxes = torch.tensor(boxes)
    suppress = np.zeros(len(boxes))
    for j in range(len(boxes)):
        box_j = boxes[j]

        #case (A) the truth box lies entirely outside the true phi range
        if (box_j[1]>ymax) or (box_j[3]<ymin):
            suppress[j] = 1
        
        #case (B) the truth box has two corners outside the true phi range
        #check the IoU of the truth box with its duplicate, remove just one of these
        elif (box_j[1] < ymin) or (box_j[3] > ymax):
            modded_box_j = box_j + (-1*torch.sign(box_j[1])) * torch.tensor([0.0, 2*np.pi, 0.0, 2*np.pi])
            overlaps = torchvision.ops.box_iou(modded_box_j.unsqueeze(0), boxes)
            wrapped_box = boxes[torch.argmax(overlaps)]

            #keep the truth box with the largest area (can be different due to merging).
            suppress[j] = max(suppress[j],(box_j[2]-box_j[0])*(box_j[3]-box_j[1])<(wrapped_box[2]-wrapped_box[0])*(wrapped_box[3]-wrapped_box[1]))

    boxes = boxes.numpy()
    return boxes[np.where(suppress==0)]



def RetrieveCellIdsFromBox(cells,boxes):

    ymin,ymax = min(cells['cell_phi']),max(cells['cell_phi']) # get physical bounds of calo cells
    list_containing_all_cells = []
    for box in boxes:
        eta_min,phi_min,eta_max,phi_max = box
        x_condition = np.logical_and.reduce((cells['cell_eta']>=eta_min, cells['cell_eta']<=eta_max))
        
        #box straddles bottom of image
        if (phi_min < ymin) and (phi_max > ymin):
            modded_box = box + np.array([0.0, 2*np.pi, 0.0, 2*np.pi])
            top_of_top_box = min(modded_box[3],ymax)
            y_condtion1 = np.logical_and.reduce((cells['cell_phi']>=ymin, cells['cell_phi']<=phi_max))
            y_condtion2 = np.logical_and.reduce((cells['cell_phi']>=modded_box[1], cells['cell_phi']<=top_of_top_box))
            y_cond = np.logical_or(y_condtion1,y_condtion2)
        
        #box straddles top of image
        elif (phi_max > ymax) and (phi_min < ymax):
            modded_box = box - np.array([0.0, 2*np.pi, 0.0, 2*np.pi])
            bottom_of_bottom_box = min(modded_box[1],ymin)
            y_condtion1 = np.logical_and.reduce((cells['cell_phi'] >= phi_min, cells['cell_phi'] <= ymax))
            y_condtion2 = np.logical_and.reduce((cells['cell_phi'] >= bottom_of_bottom_box, cells['cell_phi'] <= modded_box[3]))
            y_cond = np.logical_or(y_condtion1,y_condtion2)

        #box is completely above top
        elif (phi_max < ymin):
            modded_box = box + np.array([0.0, 2*np.pi, 0.0, 2*np.pi])
            y_cond = np.logical_and.reduce((cells['cell_phi']>=modded_box[1], cells['cell_phi']>=modded_box[3]))

        elif (phi_min > ymax):
            modded_box = box - np.array([0.0, 2*np.pi, 0.0, 2*np.pi])
            y_cond = np.logical_and.reduce((cells['cell_phi']>=modded_box[1], cells['cell_phi']>=modded_box[3]))
        else:
            y_cond = np.logical_and.reduce((cells['cell_phi']>=phi_min, cells['cell_phi']<=phi_max)) #multiple conditions #could use np.all(x,axis)
        
        tot_cond = np.logical_and(x_condition,y_cond)
        cells_here = cells[np.where(tot_cond)][['cell_E','cell_eta','cell_phi','cell_Sigma','cell_IdCells','cell_DetCells','cell_xCells','cell_yCells','cell_zCells','cell_TimeCells']]
        if len(cells_here):
            list_containing_all_cells.append(cells_here)
        else:
            # so that we know where there were no cells!
            # placeholder_values = np.array([(-1.0,-99.9,-99.9,-1.0,-10.0)],
            #     dtype=[('cell_E', '<f4'), ('cell_eta', '<f4'), ('cell_phi', '<f4'), ('cell_Sigma', '<f4'), ('cell_IdCells', '<u4')])
            # list_containing_all_cells.append(placeholder_values)
            list_containing_all_cells.append(None)

    # Check that the list does not include None/placeholder values and if it does, remove it
    filtered_list = [x for x in list_containing_all_cells if x is not None]
    # somelist = [x for x in list_containing_all_cells if not min(x['cell_phi']) < -99.0]

    return filtered_list


def RetrieveCellIdsFromCluster(cells,cluster_cell_info):
    # Inputs
    # cluster_cell_info, structured array containing cluster *cell* information
    #                   including cell ids for cells in topocluster
    # cells, structured array containing all cells in event ['cell_*'] properties
    # Outputs
    # cell_ids, array containing cell ids of all cells inside this topocluster

    list_containing_all_cells = []
    for cluster in range(len(cluster_cell_info)):
        cell_mask = np.isin(cells['cell_IdCells'],cluster_cell_info[cluster]['cl_cell_IdCells'])
        desired_cells = cells[cell_mask][['cell_E','cell_eta','cell_phi','cell_Sigma','cell_IdCells','cell_DetCells','cell_xCells','cell_yCells','cell_zCells','cell_TimeCells']]
        list_containing_all_cells.append(desired_cells)
    # return desired_cells
    return list_containing_all_cells


def RetrieveClusterCellsFromBox(cluster_d, cluster_cell_d, cells_this_event, boxes):

    list_truth_cells = RetrieveCellIdsFromBox(cells_this_event,boxes)
    l_topo_cells = RetrieveCellIdsFromCluster(cells_this_event,cluster_cell_d)
    list_cl_cells = []
    for truth_box in boxes:
        clusters_this_box = []
        for cl_no in range(len(l_topo_cells)):
            cluster_cells = l_topo_cells[cl_no]
            #if x condition satisfied
            if truth_box[0] <= np.mean(cluster_cells['cell_eta']) <= truth_box[2]:
                #if y condition satisfied
                y_mean_circ = circular_mean(cluster_cells['cell_phi'])
                if (truth_box[1] <= y_mean_circ <= truth_box[3]) or (truth_box[1] <= (y_mean_circ + (-1*np.sign(y_mean_circ))*2*np.pi) <= truth_box[3]):
                    cell_mask = np.isin(cells_this_event['cell_IdCells'],cluster_cell_d[cl_no]['cl_cell_IdCells'])
                    desired_cells = cells_this_event[cell_mask][['cell_E','cell_eta','cell_phi','cell_Sigma','cell_IdCells','cell_DetCells','cell_xCells','cell_yCells','cell_zCells','cell_TimeCells']]
                    clusters_this_box.append(desired_cells)
        list_cl_cells.append(clusters_this_box)
        
    return list_truth_cells, list_cl_cells

    
def circular_mean(phi_values):
    """
    Calculate the circular mean (average) of a list of phi_values.
    Handles the periodicity of phi_values correctly.
    
    :param phi_values: List of phi_values in radians
    :return: Circular mean in radians
    """
    sin_sum = np.sum(np.sin(phi_values))
    cos_sum = np.sum(np.cos(phi_values))
    circular_mean = np.arctan2(sin_sum, cos_sum)
    return circular_mean


def var_knn_graph(
        x,
        k,
        quantiles,
        x_ranking,
        batch=None,
):
    # Finds for each element in x the k nearest points in x-space
    # k changes depending on the importance of the node as defined in 
    # x_ranking tensor
    
    if batch is None:
        batch = x.new_zeros(x.size(0), dtype=torch.long)


    x = x.view(-1, 1) if x.dim() == 1 else x
    assert x.dim() == 2 and batch.dim() == 1
    assert x.size(0) == batch.size(0)
    assert len(k) == len(quantiles)+1, "There should be a k value for each quantile interval"

    # Rescale x and y.
    def normalize(x):
        x_normed = x - x.min(0,keepdim=True)[0]
        x_normed = x_normed / x_normed.max(0, keepdim=True)[0]
        return x_normed 
    x = normalize(x)

    # Concat batch/features to ensure no cross-links between examples exist.
    x = torch.cat([x, 2 * x.size(1) * batch.view(-1, 1).to(x.dtype)], dim=-1)

    # Pre-calculate KNN tree
    tree = scipy.spatial.cKDTree(x.detach()) 

    quantile_values = torch.quantile(x_ranking, torch.tensor(quantiles))
    edges_list = torch.tensor([[],[]],dtype=torch.int64)
    for i in range(len(quantile_values)+1):
        # Create masks for each quantile
        if i==0:
            qua_mask = x_ranking<=quantile_values[i]
        elif i==len(quantile_values):
            qua_mask = x_ranking>=quantile_values[i-1]
        else:
            qua_mask = (quantile_values[i-1]<x_ranking) & (x_ranking <= quantile_values[i])
        
        # Extract indices for each quantile
        indices = torch.nonzero(qua_mask, as_tuple=False).squeeze()
        qua_mask = qua_mask.squeeze()
        nodes_q = x[qua_mask]

        dist, col = tree.query(nodes_q.detach(),k=k[i],distance_upper_bound=x.size(1)) 
        dist = torch.from_numpy(dist).to(x.dtype)
        col = torch.from_numpy(col).to(torch.long)
        row = torch.arange(col.size(0), dtype=torch.long).view(-1, 1).repeat(1, k[i])
        mask = torch.logical_not(torch.isinf(dist).view(-1))
        row, col = row.view(-1)[mask], col.view(-1)[mask]
        # Return to original indices
        row = torch.gather(indices,0,row)
        # pairs of source, dest node indices
        edges_q = torch.stack([row, col], dim=0)
        # edges_q = torch.stack([col,row], dim=0)
        edges_list = torch.cat([edges_list,edges_q],dim=1)

    return edges_list



value_mapping = {65: 0, 
                    81: 1, 
                    97: 2, 
                    113: 3, 
                    65544: 4, 
                    73736: 5, 
                    81928: 6, 
                    270344: 7, 
                    278536: 8, 
                    811016: 9, 
                    131080: 10, 
                    139272: 11,
                    147464: 12,
                    257: 13,
                    273: 14,
                    145: 15,
                    289: 16,
                    161: 17,
                    305: 18,
                    2: 19,
                    514: 20,
                    1026: 21,
                    1538: 22,
                    2052: 23,
                    4100: 24,
                    6148: 25
                    }

def encode_layers(value):
    return value_mapping.get(value, 0) 

#initialise model
class Net(torch.nn.Module):
    def __init__(self, 
                 in_channels,
                 out_channels,
                 ):
        super().__init__()

        self.conv1 = torch_geometric.nn.GCNConv(in_channels, 256)
        self.relu = torch.nn.ReLU()
        self.pool1 = torch_geometric.nn.DMoNPooling(256,out_channels)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = self.relu(x)

        x, mask = torch_geometric.utils.to_dense_batch(x, batch)
        adj = torch_geometric.utils.to_dense_adj(edge_index, batch)
        s, x, adj, sp1, o1, c1 = self.pool1(x, adj, mask)

        return F.log_softmax(x, dim=-1), sp1+o1+c1, s

class DynamicNet(torch.nn.Module):
    def __init__(self, 
                 in_channels,
                 out_channels,
                 ):
        super().__init__()

        self.conv1 = torch_geometric.nn.GCNConv(in_channels, 128)
        self.conv2 = torch_geometric.nn.GCNConv(128, 128)
        self.relu = torch.nn.ReLU()
        self.pool1 = torch_geometric.nn.DMoNPooling(128,out_channels)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = self.relu(x)

        #recompute adjacency
        edge_index2 = torch_geometric.nn.knn_graph(
            x=x[:,[0,1,2]],
            k=4,
            batch=batch,
        )
        x = self.conv2(x,edge_index2)
        x = self.relu(x)

        x, mask = torch_geometric.utils.to_dense_batch(x, batch)
        adj = torch_geometric.utils.to_dense_adj(edge_index, batch)
        s, x, adj, sp1, o1, c1 = self.pool1(x, adj, mask)

        return F.log_softmax(x, dim=-1), sp1+o1+c1, s

num_clusters = 20
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net(3, num_clusters).to(device)
# saved_model = "../../../blobs/pyg/demos/dmon_sig12_xyz_k3333_4clus_40e"
saved_model = "calo_dmon_20clus_25e_knn123onetenperc"
model.load_state_dict(torch.load(saved_model+".pth"))
model.eval()




with open("../../../../struc_array.npy", "rb") as file:
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
    tees = wrap_check_truth(tees,MIN_CELLS_PHI,MAX_CELLS_PHI)

    #get the cells
    h5f = inference[i]['h5file']
    event_no = inference[i]['event_no']
    if h5f.decode('utf-8')=="01":
        cells_file = "../../../../user.cantel.34126190._0000{}.calocellD3PD_mc16_JZ4W.r10788.h5".format(h5f.decode('utf-8'))
        with h5py.File(cells_file,"r") as f:
            h5group = f["caloCells"]
            cells = h5group["2d"][event_no]

        clusters_file = "../../../../user.cantel.34126190._0000{}.topoclusterD3PD_mc16_JZ4W.r10788.h5".format(h5f.decode('utf-8'))
        with h5py.File(clusters_file,"r") as f:
            cl_data = f["caloCells"] 
            event_data = cl_data["1d"][event_no]
            cluster_data = cl_data["2d"][event_no]
            cluster_cell_data = cl_data["3d"][event_no]

        # l_topo_cells = utils.RetrieveCellIdsFromCluster(cells,cluster_cell_data)
        # l_true_cells = utils.RetrieveCellIdsFromBox(cells,tees)

        list_truth_cells, list_cl_cells = RetrieveClusterCellsFromBox(cluster_data,cluster_cell_data,cells,tees)


        for truth_box_number in range(len(list_truth_cells)):
            print('\t',truth_box_number)
            truth_box_cells_i = list_truth_cells[truth_box_number]
            cluster_cells_i = list_cl_cells[truth_box_number]
            if np.abs(np.mean(truth_box_cells_i['cell_eta']))<1.5:
                # Choose edge indices (MUST MATCH DATA PIPELINE)

                mask = abs(truth_box_cells_i['cell_E'] / truth_box_cells_i['cell_Sigma']) > 2
                truth_box_cells_2sig_i = truth_box_cells_i[mask]
                truth_box_cells_i = truth_box_cells_2sig_i

                struc_array = truth_box_cells_i[['cell_xCells','cell_yCells','cell_zCells','cell_E','cell_eta','cell_phi','cell_Sigma','cell_IdCells','cell_TimeCells']].copy()
                feature_matrix =  rf.structured_to_unstructured(struc_array,dtype=np.float32)
                cell_radius = torch.tensor(np.sqrt(truth_box_cells_i['cell_yCells']**2+truth_box_cells_i['cell_xCells']**2))
                vectorized_mapping = np.vectorize(encode_layers)
                cell_layer = torch.tensor(vectorized_mapping(truth_box_cells_i['cell_DetCells']))
                cell_significance = torch.tensor(abs(truth_box_cells_i['cell_E'] / truth_box_cells_i['cell_Sigma']))
                feature_tensor = torch.column_stack((torch.tensor(feature_matrix),cell_radius,cell_layer,cell_significance))
                feature_tensor = feature_tensor[:,[0,1,2,4,5,-3,-2]]
                feature_tensor = feature_tensor / feature_tensor.max(0, keepdim=True)[0]


                #normalise features
                def normalize(x):
                    x_normed = x / x.max(0, keepdim=True)[0]
                    return x_normed
                feature_tensor_norm = normalize(feature_tensor)

                # edge_index = utils.var_knn_graph(feature_tensor_norm[:, [0,1,2]],k=[3,4,8,16],quantiles=[0.25,0.5,0.8],x_ranking=cell_significance)
                edge_index = var_knn_graph(feature_tensor_norm[:, [0,1,2]],k=[1,2,3,25],quantiles=[0.25,0.5,0.9],x_ranking=cell_significance)
                # edge_index = torch_geometric.nn.knn_graph(feature_tensor_norm[:, [0,1,2]],k=10,loop=False)

                # put in graph Data object
                # truth_box_graph_data = torch_geometric.data.Data(x=feature_tensor,edge_index=edge_index) 
                truth_box_graph_data = torch_geometric.data.Data(x=feature_tensor[:, [0,1,2]],edge_index=edge_index) 

                # truth_box_number = 1
                truth_box_cells_i = list_truth_cells[truth_box_number]
                mask = abs(truth_box_cells_i['cell_E'] / truth_box_cells_i['cell_Sigma']) > 2
                truth_box_cells_2sig_i = truth_box_cells_i[mask]
                cluster_cells_i = list_cl_cells[truth_box_number]




                ####################################################################################################
                # Make prediction using trained model
                eval_graph = truth_box_graph_data
                pred, tot_loss, clus_ass = model(eval_graph.x,eval_graph.edge_index,eval_graph.batch)
                predicted_classes = clus_ass.squeeze().argmax(dim=1).numpy()
                unique_values, counts = np.unique(predicted_classes, return_counts=True)
                print('Number of cells etc. :',sum(unique_values),len(predicted_classes),predicted_classes.shape) 




                ####################################################################################################
                # Plotting
                if os.path.exists(f"plots/event{i}/") is False: os.makedirs(f"plots/event{i}/")
                figure = plt.figure(figsize=(14, 8))
                #Input graph
                ax1 = figure.add_subplot(131, projection='3d')
                # ax1.scatter(truth_box_cells_i['cell_xCells'], truth_box_cells_i['cell_zCells'], truth_box_cells_i['cell_yCells'], marker='o',s=4,color='green')
                ax1.scatter(eval_graph.x[:, 0], eval_graph.x[:, 2], eval_graph.x[:, 1], c="b", marker='o',s=4)
                # ax1.scatter(truth_box_cells_i['cell_xCells'], truth_box_cells_i['cell_yCells'], truth_box_cells_i['cell_zCells'], c="b", marker='o',s=4*cell_significance)
                for src, dst in eval_graph.edge_index.t().tolist():
                    x_src, y_src, z_src = eval_graph.x[src][:3]
                    x_dst, y_dst, z_dst = eval_graph.x[dst][:3]
                    ax1.plot([x_src, x_dst], [z_src, z_dst], [y_src, y_dst], c='r',alpha=0.5)
                
                #Model output
                ax2 = figure.add_subplot(132, projection='3d')
                # ax2.set(xlim=ax1.get_xlim(),ylim=ax1.get_ylim(),zlim=ax1.get_zlim())
                scatter = ax2.scatter(eval_graph.x[:, 0], eval_graph.x[:, 2], eval_graph.x[:, 1], c=predicted_classes, marker='o',s=3)
                labels = [f"{value}: ({count})" for value,count in zip(unique_values, counts)]
                # ax2.legend(handles=scatter.legend_elements(num=None)[0],labels=labels,title=f"Classes {len(unique_values)}/{num_clusters}",bbox_to_anchor=(1.07, 0.25),loc='lower left')

                #Topoclusters
                ax3 = figure.add_subplot(133, projection='3d')
                ax3.set(xlim=ax1.get_xlim(),ylim=ax1.get_ylim(),zlim=ax1.get_zlim())
                for cl_idx in range(len(cluster_cells_i)):
                    cluster_inside_box = cluster_cells_i[cl_idx]
                    ax3.scatter(cluster_inside_box['cell_xCells'], cluster_inside_box['cell_zCells'], cluster_inside_box['cell_yCells'], marker='^',s=4)

                ax1.set(xlabel='X',ylabel='Z',zlabel='Y',title=f'Input Graph ({len(truth_box_cells_i)} Cells)')
                ax2.set(xlabel='X',ylabel='Z',zlabel='Y',title=f'Model Output ({len(unique_values)} Clusters)')
                ax3.set(xlabel='X',ylabel='Z',zlabel='Y',title=f'({len(np.concatenate(cluster_cells_i))}) Total Topocluster Cells, {len(cluster_cells_i)} Clusters')
                # figure.savefig(f'plots/event{i}/truthbox{truth_box_number}.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
                plt.show()
                plt.close()
                print()
                for value, count in zip(unique_values, counts):
                    print(f"\tCluster {value}: {count} occurrences")
                print(f"\tTotal: {len(truth_box_cells_i)}")


        quit()






