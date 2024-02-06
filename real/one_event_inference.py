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
import pickle
import random

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


def perpendicular_dists(points, mean_point, b=torch.tensor([0.0,0.0,0.0])):
    # Calculate the perpendicular distances from a tensor of points to a line emanating from the origin 
    # to some mean point
    # Handle case where p is a single point, i.e. 1d array.
    points = torch.atleast_2d(points)

    if torch.all(mean_point == b):
        return torch.linalg.norm(points - mean_point, axis=1)

    # normalized tangent vector
    d = torch.divide(b - mean_point, torch.linalg.norm(b - mean_point))

    # perpendicular distance component, as before
    # note that for the 3D case these will be vectors
    try:
        return torch.linalg.cross(points - mean_point, d.expand_as(points), dim=1)
    except RuntimeError:
        diff = points - mean_point
        d = torch.unsqueeze(d,0)
        diff_pad = torch.nn.functional.pad(diff,(0,1))
        d_pad = torch.nn.functional.pad(d,(0,1))
        return torch.linalg.cross(diff_pad,d_pad)[...,2]



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
                        print('\t',len(gnn_cluster_cells_i), 'there were', len(cluster_cells_i), 'TCs', len(coordinates), 'cells')
                    
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

            print(list_topo_cells[0].dtype)
            print(list_gnn_cells[0].dtype)
            quit()


        
        
        
        
        
        


