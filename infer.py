import torch
import torch_geometric
from torch_geometric.loader import DataLoader

import os
import time
import argparse
import numpy as np
import matplotlib
from matplotlib import pyplot as plt

import models
import data



parser = argparse.ArgumentParser()
parser.add_argument('--root', nargs='?', const='./', default='./', type=str, help='Path to top-level h5 directory',)
parser.add_argument('--name', type=str, required=True, help='Name of edge building scheme (knn, rad, bucket, custom)')
parser.add_argument('-k', nargs='?', const=None, default=None, type=int, help='K-nearest neighbours value to be used only in knn graph')
parser.add_argument('-r', nargs='?', const=None, default=None, type=int, help='Radius value to be used only in radial graph')
parser.add_argument('--graph_dir', nargs='?', const='./', default='./', type=str, help='Path to processed folder containing .pt graphs',)


parser.add_argument('-e','--epochs', type=int, required=True, help='Number of training epochs',)
parser.add_argument('-bs','--batch_size', nargs='?', const=8, default=8, type=int, help='Batch size to be used')
parser.add_argument('-nw','--num_workers', nargs='?', const=2, default=2, type=int, help='Number of worker CPUs')
parser.add_argument('-nc','--num_clusters', nargs='?', const=5, default=5, type=int, help='Number of (max) clusters DMoN can predict')
parser.add_argument('--model_dir', type=str, required=True, help='Path to saved models directory',)
parser.add_argument('-out','--output_dir',nargs='?', const='./cache/', default='./cache/', type=str, help='Path to directory containing struc_array.npy',)
args = parser.parse_args()



def weighted_circular_mean(phi_values, energy_values):
    """
    Calculate the weighted circular mean (average) of a list of angles.
    Handles the periodicity of phi correctly. http://palaeo.spb.ru/pmlibrary/pmbooks/mardia&jupp_2000.pdf

   Inputs:
        phi_values: numpy array, variable in question (periodic in phi)
        energy_values: numpy array, energy of each contributing element
    Outputs:
        weighted_circular_mean: np.array, result of circular weighted mean
    """
    if len(phi_values) != len(energy_values):
        raise ValueError("phi_values and energy_values must have the same length")

    weighted_sin_sum = np.sum(energy_values * np.sin(phi_values))
    weighted_cos_sum = np.sum(energy_values * np.cos(phi_values))
    weighted_circular_mean = np.arctan2(weighted_sin_sum, weighted_cos_sum)

    return weighted_circular_mean

def weighted_mean(values, energy_values):
    '''
    Function to calculate mean of values input, based on their energy
    Inputs:
        values: np.array, variable in question
        energy_values: np.array, energy of each contributing element

    Outputs:
        weighted_mean: np.array, result of weighted mean
    '''  
    if len(values) != len(energy_values):
        raise ValueError("values and energy_values must have the same length") 

    return np.dot(values,np.abs(energy_values)) / sum(np.abs(energy_values))







if __name__=='__main__':

    config = {
        "seed"       : 0,
        "device"     : torch.device("cuda" if torch.cuda.is_available() else "cpu"), # torch.device("mps" if torch.backends.mps.is_available() else "cpu"),
        "n_train"    : 1000,
        "val_frac"   : 0.25,
        "test_frac"  : 0.15,
        "builder"    : args.name,
        "graph_dir"  : args.graph_dir,
        "k"          : args.k,
        "r"          : args.r,
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

    train_data = data.CaloDataset(root=args.root, name=config["builder"], k=config["k"], rad=config["r"], graph_dir=config["graph_dir"])
    valid_data = data.CaloDataset(root=args.root, name=config["builder"], k=config["k"], rad=config["r"], graph_dir=config["graph_dir"])
    test_data  = data.CaloDataset(root=args.root, name=config["builder"], k=config["k"], rad=config["r"], graph_dir=config["graph_dir"])
    print('\ttrain / val / test size : ',len(train_data),'/',len(valid_data),'/',len(test_data),'\n')

    train_loader = DataLoader(train_data, batch_size=config["BS"], num_workers=config["NW"])
    val_loader   = DataLoader(valid_data, batch_size=config["BS"], num_workers=config["NW"])
    test_loader  = DataLoader(test_data, batch_size=config["BS"], num_workers=config["NW"])

    # instantiate model
    model = models.Net(6, config["n_clus"]).to(config["device"])
    total_params = sum(p.numel() for p in model.parameters())
    print(f'DMoN (single conv layer) \t{total_params:,} total parameters.\n')
    model_name = "DMoN_calo_{}_{}c_{}e".format(train_data.name, config["n_clus"], config["n_epochs"])
    model_save_path = args.model_dir + f"/{model_name}.pth"
    model.load_state_dict(torch.load(model_save_path, weights_only=True, map_location=torch.device(config["device"])))

    save_loc = args.output_dir + "/" + model_name + "/" + time.strftime("%Y%m%d-%H") + "/"
    print("Save location: ", save_loc)
    if not os.path.exists(save_loc): os.makedirs(save_loc)
    if not os.path.exists(f"{save_loc}/plots/results/"): os.makedirs(f"{save_loc}/plots/results/")


    # instantiate numpy structured array to hold inference results
    dt = np.dtype([('event_no', 'i4'), ('node_features', 'f4', (350, 3)), ('model_output', 'f4', (config["n_clus"], 128)), ('cluster_assign', 'i4', (350,1))])
    inference = np.zeros((len(test_loader)*config["BS"]), dtype=dt)  
    model.eval()
    beginning = time.perf_counter()
    with torch.inference_mode():
        for step, data in enumerate(test_loader):
            print(step)
            data = data.to(config["device"])
            pred, tot_loss, clus_ass = model(data.x,data.edge_index,data.batch)
            # print("loss",tot_loss.shape,tot_loss)
            # print("pred",pred.shape)
            # print("cluster",clus_ass.shape)
            node_features = torch_geometric.utils.unbatch(data.x,data.batch)
            # print("node",data.x.shape)
            # print('node 0',node_features[0].shape)
            # print('node 1',node_features[1].shape)
            # print('node 2',node_features[2].shape)

            # make lists for later padding
            total_pred, total_clus, total_loss = list(), list(), list()
            event_nos, feat_mats = list(), list()
            for batch_idx in range(config["BS"]):
                # remove from GPU 
                pred_i = pred[batch_idx].detach().cpu().numpy()
                clus_i = clus_ass[batch_idx].detach().cpu()
                # print('\tclus_i',clus_i.shape)
                # print('\tclus_1',clus_ass[batch_idx+1].detach().cpu().shape)
                # print('\tclus_2',clus_ass[batch_idx+2].detach().cpu().shape)
                clus_i = clus_i[~np.isnan(clus_i).any(axis=1).bool()] # remove nan values
                # print('\tclus_i',clus_i.shape)
                # print('\tclus_1',clus_ass[batch_idx+1].detach().cpu()[~np.isnan(clus_ass[batch_idx+1].detach().cpu()).any(axis=1).bool()].shape)
                # print('\tclus_2',clus_ass[batch_idx+2].detach().cpu()[~np.isnan(clus_ass[batch_idx+2].detach().cpu()).any(axis=1).bool()].shape)
                # loss_i = tot_loss[b]
                x_i = node_features[batch_idx].detach().cpu().numpy()

                # force each node to its most likely cluster, no soft assignment
                predicted_classes = clus_i.squeeze().argmax(dim=1).numpy()
                print("\tAlready out of shape here",x_i.shape,pred_i.shape,clus_i.shape,predicted_classes.shape)
                unique_values, counts = np.unique(predicted_classes, return_counts=True)
                print(f"\t{len(unique_values)} clusters formed, potential max {config['n_clus']}")
                for value, count in zip(unique_values, counts):
                    print(f"\tCluster {value}: {count} occurrences")

                ##########################################################################################################################################
                fig = plt.figure(figsize=(10, 6))
                edge_index = torch_geometric.utils.unbatch_edge_index(data.edge_index,data.batch)
                edg_i = edge_index[batch_idx]
                unique_values, counts = np.unique(predicted_classes, return_counts=True)
                ax = fig.add_subplot(121, projection='3d')
                ax.scatter(x_i[:, 0], x_i[:, 1], x_i[:, 2], s=x_i[:, -1]/3, c='b', marker='o', label='Nodes')
                for src, dst in edg_i.t().tolist():
                    x_src, y_src, z_src, *feat = x_i[src]
                    x_dst, y_dst, z_dst, *feat = x_i[dst]
                    ax.plot([x_src, x_dst], [y_src, y_dst], [z_src, z_dst], c='r')
                ax.set(xlabel='X',ylabel='Y',zlabel='Z',title=f'Input Graph with KNN 3 Edges')

                ax2 = fig.add_subplot(122, projection='3d')
                print('\tFinal here',x_i.shape, edg_i.shape,predicted_classes.shape)
                cmap = matplotlib.colormaps['Dark2']
                scatter = ax2.scatter(x_i[:, 0], x_i[:, 1], x_i[:, 2], s=x_i[:, -1]/3, c=predicted_classes, marker='o', cmap=cmap) #Dark2
                labels = [f"{value}: ({count})" for value,count in zip(unique_values, counts)]
                # ax2.legend(handles=scatter.legend_elements(num=None)[0],labels=labels,title=f"Classes {len(unique_values)}/{config['n_clus']}",bbox_to_anchor=(1.07, 0.25),loc='lower left')
                ax2.set(xlabel='X',ylabel='Y',zlabel='Z',title=f'DMoN Output Graph')
                x_axis = ax2.get_xlim()
                y_axis = ax2.get_ylim()
                z_axis = ax2.get_zlim()
                fig.savefig(f"{save_loc}/plots/results/data_{step*batch_idx+batch_idx}_3d.png", bbox_inches="tight")
                plt.close()
                
                f,a = plt.subplots(1,1,figsize=(10,12))   
                a.scatter(x_i[:, 3], x_i[:, 4], s=10, c=predicted_classes, cmap='tab20b')
                a.set(xlabel='eta',ylabel='phi',title=f'DMoN Output Clustered Cells')
                f.savefig(f"{save_loc}/plots/results/data_{step*batch_idx+batch_idx}_cells_2d.png")
                plt.close()

                fi,ax = plt.subplots(1,1,figsize=(10,12)) 
                ax.set_prop_cycle(color=plt.cm.tab20b(np.linspace(0, 1, 20)))
                for cl_idx in range(len(unique_values)):
                    cluster_i_mask = predicted_classes == cl_idx
                    cl_cell_i = x_i[cluster_i_mask, :]
                    cl_eta, cl_phi = weighted_mean(cl_cell_i[:,3], cl_cell_i[:,5]), weighted_circular_mean(cl_cell_i[:,4], cl_cell_i[:,5])
                    cl_e = np.mean(cl_cell_i[:,5])
                    ax.plot(cl_eta, cl_phi, markersize=8, marker='o',alpha=.6)
                ax.set(xlabel='eta',ylabel='phi',title=f'DMoN Output Clusters')
                fi.savefig(f"{save_loc}/plots/results/data_{step*batch_idx+batch_idx}_cl_2d.png")
                plt.close()

                # plot individual clusters
                i_want_cluster = input('Enter cluster number to inspect (type q for exit):')
                while i_want_cluster!="q":
                    cluster_i_mask = predicted_classes == int(i_want_cluster)
                    print(cluster_i_mask)
                    fig = plt.figure(figsize=(10, 6))
                    ax1 = fig.add_subplot(111, projection='3d')
                    scatter = ax1.scatter(x_i[cluster_i_mask, 0], x_i[cluster_i_mask, 1], x_i[cluster_i_mask, 2], s=x_i[cluster_i_mask, -1], marker='o', c="crimson")
                    ax1.set(xlabel='X',ylabel='Y',zlabel='Z',xlim=x_axis,ylim=y_axis,zlim=z_axis,title=f'DMoN Output Graph Cluster {i_want_cluster}')
                    plt.show()
                    i_want_cluster = input('Enter cluster number to inspect (type q for exit):')

                quit()
                ##########################################################################################################################################
                
                total_pred.append(pred_i)
                total_clus.append(clus_i)
                # total_loss.append(loss_i)
                feat_mats.append(x_i)
                event_nos.append(step*config["BS"]+b)


            dataset_idx = step*config["BS"]

            cl_array   = [np.pad(cl_i, ((0,350-len(cl_i))), 'constant', constant_values=(0)) for cl_i in total_clus]
            out_array  = [np.pad(out, ((0,config["n_clus"]-len(out)),(0,0)), 'constant', constant_values=(0)) for out in total_pred]
            feat_array = [np.pad(x, ((0,350-len(x)),(0,0)), 'constant', constant_values=(0)) for x in feat_mats]
            # t_boxes = [np.pad(trub, ((0,250-len(trub)),(0,0)), 'constant', constant_values=(0)) for trub in tru_boxes]

            inference['event_no'][dataset_idx:dataset_idx+config["BS"]] = event_nos
            inference['node_features'][dataset_idx:dataset_idx+config["BS"]] = feat_array   
            inference['model_output'][dataset_idx:dataset_idx+config["BS"]] = out_array   
            inference['cluster_assign'][dataset_idx:dataset_idx+config["BS"]] = cl_array 

    end = time.perf_counter()      
    print(f"Time taken for entire test set: {(end-beginning)/60:.3f} mins, (or {(end-beginning):.3f}s), average {(end-beginning)/test_len:.4f} per image")

    print('\n\n')
    with open(save_loc+'struc_array.npy', 'wb') as f:
        print('Saving...')
        np.save(f, Large)
