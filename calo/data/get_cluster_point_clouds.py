import numpy as np
import scipy
import matplotlib.pyplot as plt

import torch
import torch_geometric




with open("/Users/leonbozianu/work/phd/graph/dmon/struc_array.npy", "rb") as f:
    inference = np.load(f)


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







