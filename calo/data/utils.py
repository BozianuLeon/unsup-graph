import numpy as np
import scipy
import torch
import torchvision



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
        cells_here = cells[np.where(tot_cond)][['cell_E','cell_eta','cell_phi','cell_Sigma','cell_IdCells','cell_xCells','cell_yCells','cell_zCells','cell_TimeCells']]
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
        desired_cells = cells[cell_mask][['cell_E','cell_eta','cell_phi','cell_Sigma','cell_IdCells','cell_xCells','cell_yCells','cell_zCells','cell_TimeCells']]
        list_containing_all_cells.append(desired_cells)
    # return desired_cells
    return list_containing_all_cells




#for finding which clusters are in each box
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
                    desired_cells = cells_this_event[cell_mask][['cell_E','cell_eta','cell_phi','cell_Sigma','cell_IdCells','cell_xCells','cell_yCells','cell_zCells','cell_TimeCells']]
                    clusters_this_box.append(desired_cells)
        list_cl_cells.append(clusters_this_box)
        
    return list_truth_cells, list_cl_cells

    




def quartile_variable_knn(feature_matrix,point_importance):

    tree = scipy.spatial.cKDTree(feature_matrix)
    q1, q2, q3 = torch.quantile(point_importance, torch.tensor([0.25, 0.75, 0.95]))
    
    # Create masks for each quartile
    mask_q1 = point_importance <= q1
    mask_q2 = (q1 < point_importance) & (point_importance <= q2)
    mask_q3 = (q2 < point_importance) & (point_importance <= q3)
    mask_q4 = point_importance > q3

    # Extract indices for each quartile
    indices_q1 = torch.nonzero(mask_q1, as_tuple=False).squeeze(dim=1)
    indices_q2 = torch.nonzero(mask_q2, as_tuple=False).squeeze(dim=1)
    indices_q3 = torch.nonzero(mask_q3, as_tuple=False).squeeze(dim=1)
    indices_q4 = torch.nonzero(mask_q4, as_tuple=False).squeeze(dim=1)

    points_q1 = feature_matrix[mask_q1]
    points_q2 = feature_matrix[mask_q2]
    points_q3 = feature_matrix[mask_q3]
    points_q4 = feature_matrix[mask_q4]

    #variable k
    # k1,k2,k3,k4 = 3,6,9,12
    k1,k2,k3,k4 = 1,2,5,10

    distances1,indices1 = tree.query(points_q1, k=k1)
    distances1 = torch.from_numpy(distances1).to(torch.float32)
    col1 = torch.from_numpy(indices1).to(torch.long)
    row1 = torch.arange(col1.size(0),dtype=torch.long).view(-1,1).repeat(1,k1)
    mask = ~torch.isinf(distances1).view(-1).to(torch.bool)
    row1, col1 = row1.view(-1)[mask], col1.view(-1)[mask]
    #need to return to original indices:
    orig_row1 = torch.gather(indices_q1,0,row1)
    #return pairs of source,dest node indices.
    edges_q1 = torch.stack([orig_row1, col1], dim=0)

    distances2,indices2 = tree.query(points_q2, k=k2)
    distances2 = torch.from_numpy(distances2).to(torch.float32)
    col2 = torch.from_numpy(indices2).to(torch.long)
    row2 = torch.arange(col2.size(0),dtype=torch.long).view(-1,1).repeat(1,k2)
    mask = ~torch.isinf(distances2).view(-1).to(torch.bool)
    row2, col2 = row2.view(-1)[mask], col2.view(-1)[mask]
    orig_row2 = torch.gather(indices_q2,0,row2)
    edges_q2 = torch.stack([orig_row2, col2], dim=0)

    distances3,indices3 = tree.query(points_q3, k=k3)
    distances3 = torch.from_numpy(distances3).to(torch.float32)
    col3 = torch.from_numpy(indices3).to(torch.long)
    row3 = torch.arange(col3.size(0),dtype=torch.long).view(-1,1).repeat(1,k3)
    mask = ~torch.isinf(distances3).view(-1).to(torch.bool)
    row3, col3 = row3.view(-1)[mask], col3.view(-1)[mask]
    orig_row3 = torch.gather(indices_q3,0,row3)
    edges_q3 = torch.stack([orig_row3, col3], dim=0)

    distances4,indices4 = tree.query(points_q4, k=k4)
    distances4 = torch.from_numpy(distances4).to(torch.float32)
    col4 = torch.from_numpy(indices4).to(torch.long)
    row4 = torch.arange(col4.size(0),dtype=torch.long).view(-1,1).repeat(1,k4)
    mask = ~torch.isinf(distances4).view(-1).to(torch.bool)
    row4, col4 = row4.view(-1)[mask], col4.view(-1)[mask]
    orig_row4 = torch.gather(indices_q4,0,row4)
    edges_q4 = torch.stack([orig_row4, col4], dim=0)


    edges_total = torch.cat((edges_q1, edges_q2, edges_q3, edges_q4), dim=1)

    return edges_total