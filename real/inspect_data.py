import torch
import torchvision
import torch_geometric
import h5py
import os
import numpy as np
import numpy.lib.recfunctions as rf
import matplotlib.pyplot as plt

MIN_CELLS_PHI,MAX_CELLS_PHI = -3.1334076, 3.134037
MIN_CELLS_ETA,MAX_CELLS_ETA = -4.823496, 4.823496





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






def inspect_calo_data_list(
        inference_array,
        box_eta_cut=1.5,
        cell_significance_cut=2,
        norm=False,
        k=3,    

        #plotting:
        save_loc='',
        n_plots=2,
    ):
    '''
    Turn truth (green) boxes into point clouds in pytorch geometric, save these for later training
    '''

    data_list = list()
    cluster_list = list()
    for i in range(len(inference_array)):
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

                if np.abs(np.mean(truth_box_cells_i['cell_eta']))<box_eta_cut:
                    mask = abs(truth_box_cells_i['cell_E'] / truth_box_cells_i['cell_Sigma']) >= cell_significance_cut
                    truth_box_cells_2sig_i = truth_box_cells_i[mask]
                    truth_box_cells_i = truth_box_cells_2sig_i
                    # calculate edges
                    struc_array = truth_box_cells_i[['cell_xCells','cell_yCells','cell_zCells','cell_E','cell_eta','cell_phi','cell_Sigma','cell_IdCells','cell_TimeCells']].copy()
                    feature_matrix =  rf.structured_to_unstructured(struc_array,dtype=np.float32)
                    cell_radius = torch.tensor(np.sqrt(truth_box_cells_i['cell_yCells']**2+truth_box_cells_i['cell_xCells']**2))
                    cell_significance = torch.tensor(abs(truth_box_cells_i['cell_E'] / truth_box_cells_i['cell_Sigma']))
                    feature_tensor = torch.column_stack((torch.tensor(feature_matrix),cell_radius,cell_significance))
                    
                    if norm:
                        feature_tensor = feature_tensor / feature_tensor.max(0,keepdim=True)[0]


                    edge_indices = torch_geometric.nn.knn_graph(feature_tensor[:, [0,1,2]],k=k,loop=False)
                    edge_indicesxy = torch_geometric.nn.knn_graph(feature_tensor[:, [0,1]],k=k,loop=False)
                    event_graph  = torch_geometric.data.Data(x=feature_tensor,edge_index=edge_indices) 
                    data_list.append(event_graph)
                    cluster_list.append(cluster_cells_i)







                    ######################################################################################################
                    #Plotting
                    ######################################################################################################
                    if truth_box_number > n_plots:
                        break
                    
                    fig = plt.figure(figsize=(12, 8))
                    ax = fig.add_subplot(121, projection='3d')
                    # ax.scatter(truth_box_cells_i['cell_xCells'], truth_box_cells_i['cell_zCells'], truth_box_cells_i['cell_yCells'], c='b', marker='o')
                    ax.scatter(event_graph.x[:,0], event_graph.x[:,2], event_graph.x[:,1], c='b', marker='o')
                    for src, dst in edge_indices.t().tolist():
                        x_src, y_src, z_src = event_graph.x[src][:3]
                        x_dst, y_dst, z_dst = event_graph.x[dst][:3]
                        
                        ax.plot([x_src, x_dst], [z_src, z_dst], [y_src, y_dst], c='r')

                    ax.set(xlabel='X',ylabel='Z',zlabel='Y',title=f'Truth Box Graph (>2 cell signif.) {len(event_graph.x)} nodes, {len(event_graph.edge_index.t())} edges')
                    #topoclusters: 
                    ax3 = fig.add_subplot(122, projection='3d')
                    ax3.set(xlim=ax.get_xlim(),ylim=ax.get_ylim(),zlim=ax.get_zlim())
                    for cl_idx in range(len(cluster_cells_i)):
                        cluster_inside_box = cluster_cells_i[cl_idx]
                        ax3.scatter(cluster_inside_box['cell_xCells'], cluster_inside_box['cell_zCells'], cluster_inside_box['cell_yCells'], marker='*',s=4*abs(cluster_inside_box['cell_E']/cluster_inside_box['cell_Sigma']))
                    ax3.set(xlabel='X',ylabel='Z',zlabel='Y',title=f'{len(cluster_cells_i)} Topocluster(s), ({len(np.concatenate(cluster_cells_i))} tot. cells)')      
                    plt.show()
                    plt.close()

                    fig = plt.figure(figsize=(12, 8))
                    ax = fig.add_subplot(121)
                    # ax.scatter(truth_box_cells_i['cell_xCells'], truth_box_cells_i['cell_zCells'], truth_box_cells_i['cell_yCells'], c='b', marker='o')
                    ax.scatter(event_graph.x[:,0], event_graph.x[:,1], c='b', marker='o',s=7.5)
                    for src, dst in edge_indices.t().tolist():
                        x_src, y_src, z_src = event_graph.x[src][:3]
                        x_dst, y_dst, z_dst = event_graph.x[dst][:3]
                        
                        ax.plot([x_src, x_dst], [y_src, y_dst], c='r',lw=0.7,alpha=0.6)

                    ax.set(xlabel='X',ylabel='Y',title=f'Truth Box Graph (>2 cell signif.) {len(event_graph.x)} nodes, {len(event_graph.edge_index.t())} edges')
                    #topoclusters: 
                    ax3 = fig.add_subplot(122)
                    ax3.set(xlim=ax.get_xlim(),ylim=ax.get_ylim())
                    for cl_idx in range(len(cluster_cells_i)):
                        cluster_inside_box = cluster_cells_i[cl_idx]
                        ax3.scatter(cluster_inside_box['cell_xCells'], cluster_inside_box['cell_yCells'], marker='*',s=4*abs(cluster_inside_box['cell_E']/cluster_inside_box['cell_Sigma']))
                    ax3.set(xlabel='X',ylabel='Y',title=f'{len(cluster_cells_i)} Topocluster(s), ({len(np.concatenate(cluster_cells_i))} tot. cells)')      
                    plt.show()
                    plt.close()
                    
                    #KNN on x-y
                    fig = plt.figure(figsize=(12, 8))
                    ax = fig.add_subplot(121)
                    # ax.scatter(truth_box_cells_i['cell_xCells'], truth_box_cells_i['cell_zCells'], truth_box_cells_i['cell_yCells'], c='b', marker='o')
                    ax.scatter(event_graph.x[:,0], event_graph.x[:,1], c='b', marker='o',s=7.5)
                    print(edge_indicesxy.t())
                    for src, dst in edge_indicesxy.t().tolist():
                        x_src, y_src, z_src = event_graph.x[src][:3]
                        x_dst, y_dst, z_dst = event_graph.x[dst][:3]
                        ax.plot([x_dst, x_src], [y_dst,y_src], c='r',lw=0.7,alpha=0.6)
                        ax.text(x_dst,y_dst,s=f'{dst}',fontsize='small')

                    ax.set(xlabel='X',ylabel='Y',title=f'Truth Box Graph (>2 cell signif.) {len(event_graph.x)} nodes, {len(event_graph.edge_index.t())} edges (XY only)')
                    #topoclusters: 
                    ax3 = fig.add_subplot(122)
                    ax3.set(xlim=ax.get_xlim(),ylim=ax.get_ylim())
                    for cl_idx in range(len(cluster_cells_i)):
                        cluster_inside_box = cluster_cells_i[cl_idx]
                        ax3.scatter(cluster_inside_box['cell_xCells'], cluster_inside_box['cell_yCells'], marker='*',s=4*abs(cluster_inside_box['cell_E']/cluster_inside_box['cell_Sigma']))
                    ax3.set(xlabel='X',ylabel='Y',title=f'{len(cluster_cells_i)} Topocluster(s), ({len(np.concatenate(cluster_cells_i))} tot. cells)')      
                    plt.show()
                    plt.close()
                    # quit()                    



                    # if os.path.exists(f"../plots/{truth_box_number}/") is False: os.makedirs(f"../plots/{truth_box_number}/")


    return data_list, cluster_list





if __name__=="__main__":
    with open("../../struc_array.npy", "rb") as file:
        inference = np.load(file)


    inspect_calo_data_list(
            inference,
            box_eta_cut=1.5,
            cell_significance_cut=2,
            norm=False,
            k=3,    
            #plotting:
            save_loc='',
            n_plots=2)