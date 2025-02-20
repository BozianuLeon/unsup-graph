import torch
import torch_geometric
from torch_geometric.loader import DataLoader

import os
import time
import argparse
import numpy as np
from matplotlib import pyplot as plt

import models
import data



parser = argparse.ArgumentParser()
parser.add_argument('-e','--epochs', type=int, required=True, help='Number of training epochs',)
parser.add_argument('-bs','--batch_size', nargs='?', const=8, default=8, type=int, help='Batch size to be used')
parser.add_argument('-nw','--num_workers', nargs='?', const=2, default=2, type=int, help='Number of worker CPUs')
parser.add_argument('-nc','--num_clusters', nargs='?', const=5, default=5, type=int, help='Number of (max) clusters DMoN can predict')
parser.add_argument('-k', nargs='?', const=3, default=3, type=int, help='k nearest neighbours integer')
parser.add_argument('--model_dir', type=str, required=True, help='Path to saved models directory',)
parser.add_argument('-out','--output_dir',nargs='?', const='./cache/', default='./cache/', type=str, help='Path to directory containing struc_array.npy',)
args = parser.parse_args()





if __name__=='__main__':

    config = {
        "seed"       : 0,
        "device"     : torch.device("cuda" if torch.cuda.is_available() else "cpu"), # torch.device("mps" if torch.backends.mps.is_available() else "cpu"),
        "n_train"    : 1000,
        "val_frac"   : 0.25,
        "test_frac"  : 0.15,
        "n_nodes"    : 250,
        "k"          : args.k,
        "NW"         : args.num_workers,
        "BS"         : args.batch_size,
        "n_clus"     : int(args.num_clusters),
        "n_epochs"   : int(args.epochs),
    }
    torch.manual_seed(config["seed"])

    # generate data and place in geometric dataloaders
    n_train = int(config["n_train"])
    n_val   = int(config["n_train"]*config["val_frac"])
    n_test  = int(config["n_train"]*config["test_frac"])
    print('\ttrain / val / test size : ',n_train,'/',n_val,'/',n_test,'\n')
    train_data = data.synthetic_blobs_list(num_graphs=n_train,avg_num_nodes=config["n_nodes"],k=config["k"])
    valid_data = data.synthetic_blobs_list(num_graphs=n_val,avg_num_nodes=config["n_nodes"],k=config["k"])
    test_data  = data.synthetic_blobs_list(num_graphs=n_test,avg_num_nodes=config["n_nodes"],k=config["k"])

    train_loader = DataLoader(train_data, batch_size=config["BS"], num_workers=config["NW"])
    val_loader   = DataLoader(valid_data, batch_size=config["BS"], num_workers=config["NW"])
    test_loader  = DataLoader(test_data, batch_size=config["BS"], num_workers=config["NW"])

    # instantiate model
    model = models.Net(3, config["n_clus"]).to(config["device"])
    total_params = sum(p.numel() for p in model.parameters())
    print(f'DMoN (single conv layer) \t{total_params:,} total parameters.\n')
    model_name = "DMoN_blob_{}c_{}e".format(config["n_clus"],config["n_epochs"])
    model_save_path = args.model_dir + f"/{model_name}.pth"
    model.load_state_dict(torch.load(model_save_path, weights_only=True, map_location=torch.device(config["device"])))

    save_loc = args.output_dir + "/" + model_name + "/" + time.strftime("%Y%m%d-%H") + "/"
    print("Save location: ", save_loc)
    if not os.path.exists(save_loc): os.makedirs(save_loc)



    # instantiate numpy structured array to hold inference results
    dt = np.dtype([('event_no', 'i4'), ('node_features', 'f4', (350, 3)), ('model_output', 'f4', (config["n_clus"], 128)), ('cluster_assign', 'i4', (300,1))])
    inference = np.zeros((len(test_loader)*config["BS"]), dtype=dt)  
    model.eval()
    beginning = time.perf_counter()
    with torch.inference_mode():
        for step, data in enumerate(test_loader):
            print(step)
            data = data.to(config["device"])
            pred, tot_loss, clus_ass = model(data.x,data.edge_index,data.batch)
            print("pred",pred.shape)
            print("loss",tot_loss.shape,tot_loss)
            print("cluster",clus_ass.shape)
            node_features = torch_geometric.utils.unbatch(data.x,data.batch)

            # make lists for later padding
            total_pred, total_clus, total_loss = list(), list(), list()
            event_nos, feat_mats = list(), list()
            for b in range(config["BS"]):
                # remove from GPU 
                pred_i = pred[b].detach().cpu().numpy()
                clus_i = clus_ass[b].detach().cpu()
                # loss_i = tot_loss[b]
                x_i = node_features[b].detach().cpu().numpy()

                # force each node to its most likely cluster, no soft assignment
                predicted_classes = clus_i.squeeze().argmax(dim=1).numpy()
                print(predicted_classes.shape)
                total_pred.append(pred_i)
                total_clus.append(clus_i)
                # total_loss.append(loss_i)
                feat_mats.append(x_i)
                event_nos.append(step*config["BS"]+b)


            dataset_idx = step*config["BS"]

            cl_array   = [np.pad(cl_i, ((0,300-len(cl_i))), 'constant', constant_values=(0)) for cl_i in total_clus]
            out_array  = [np.pad(out, ((0,config["n_clus"]-len(out)),(0,0)), 'constant', constant_values=(0)) for out in total_pred]
            feat_array = [np.pad(x, ((0,350-len(x)),(0,0)), 'constant', constant_values=(0)) for x in feat_mats]
            # t_boxes = [np.pad(trub, ((0,250-len(trub)),(0,0)), 'constant', constant_values=(0)) for trub in tru_boxes]

            inference['event_no'][dataset_idx:dataset_idx+config["BS"]] = event_nos
            inference['node_features'][dataset_idx:dataset_idx+config["BS"]] = feat_array   
            inference['model_output'][dataset_idx:dataset_idx+config["BS"]] = out_array   
            inference['cluster_assign'][dataset_idx:dataset_idx+config["BS"]] = cl_array 

    end = time.perf_counter()      
    print(f"Time taken for entire test set: {(end-beginning)/60:.3f} mins, (or {(end-beginning):.3f}s), average {(end-beginning)/test_len:.4f} per image")


            