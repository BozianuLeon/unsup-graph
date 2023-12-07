import numpy as np
import scipy
import h5py
import matplotlib.pyplot as plt


import torch
import torch_geometric


# TODO: NEED WRAP CHECKS
# TODO: NEED TO GET CELLS FROM BOXES
# TODO: NEED TO FIND THE CLUSTERS THAT ARE INSIDE EACH TRUTH BOX

with open("/Users/leonbozianu/work/phd/graph/dmon/struc_array.npy", "rb") as file:
    inference = np.load(file)


for i in range(1):
    i=0
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


    #get the cells
    h5f = inference[i]['h5file']
    event_no = inference[i]['event_no']

    file = "/Users/leonbozianu/work/phd/graph/dmon/user.cantel.34126190._0000{}.calocellD3PD_mc16_JZ4W.r10788.h5".format(h5f.decode('utf-8'))
    with h5py.File(file,"r") as f:
        h5group = f["caloCells"]
        cells = h5group["2d"][event_no]

    clusters_file = "/Users/leonbozianu/work/phd/graph/dmon/user.cantel.34126190._0000{}.topoclusterD3PD_mc16_JZ4W.r10788.h5".format(h5f.decode('utf-8'))
    with h5py.File(clusters_file,"r") as f:
        cl_data = f["caloCells"] 
        event_data = cl_data["1d"][event_no]
        cluster_data = cl_data["2d"][event_no]
        cluster_cell_data = cl_data["3d"][event_no]






#now we have the boxes, the clusters and the cells we can begin making


#file structure is
#plots/index/total_event/
#plots/index/cluster_X/





