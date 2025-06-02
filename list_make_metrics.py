import torch
import torch_geometric
from torch_geometric.loader import DataLoader

import os
import time
import pickle
import argparse
import numpy as np
import h5py
import fastjet

import models
import data




parser = argparse.ArgumentParser()
parser.add_argument('--root', nargs='?', const='./', default='./', type=str, help='Path to top-level h5 directory',)
parser.add_argument('--name', type=str, required=True, help='Name of edge building scheme (knn, rad, bucket, custom)')
parser.add_argument('--feat', type=str, nargs='?', const="XYZ", default="XYZ", help='Which geometrical columns are in the feature matrix (XYZ or REP)')
parser.add_argument('-k', nargs='?', const=None, default=None, type=int, help='K-nearest neighbours value to be used only in knn graph')
parser.add_argument('-r', nargs='?', const=None, default=None, type=int, help='Radius value to be used only in radial graph')
parser.add_argument('--graph_dir', nargs='?', const='./data/', default='./data/', type=str, help='Path to processed folder containing .pt graphs',)


parser.add_argument('-e','--epochs', type=int, required=True, help='Number of training epochs',)
parser.add_argument('-bs','--batch_size', nargs='?', const=1, default=1, type=int, help='Batch size to be used')
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
    elif (len(phi_values)==0) or (len(energy_values)==0):
        return np.nan

    weighted_sin_sum = np.sum(abs(energy_values) * np.sin(phi_values))
    weighted_cos_sum = np.sum(abs(energy_values) * np.cos(phi_values))
    weighted_circular_mean = np.arctan2(weighted_sin_sum, weighted_cos_sum)

    return weighted_circular_mean

def circle_mean(phi_values):
    if (len(phi_values)==0):
        return np.nan
    return np.arctan2(np.sum(np.sin(phi_values)), np.sum(np.cos(phi_values)))

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
    elif (len(values)==0) or (len(energy_values)==0):
        return np.nan

    return np.dot(values,np.abs(energy_values)) / sum(np.abs(energy_values))


def clip_phi(phi_values):
    return phi_values - 2 * np.pi * np.floor((phi_values + np.pi) / (2 * np.pi))


def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp)






if __name__=="__main__":

    config = {
        "seed"       : 0,
        "device"     : torch.device("cuda" if torch.cuda.is_available() else "cpu"), # torch.device("mps" if torch.backends.mps.is_available() else "cpu"),
        "n_train"    : 1000,
        "val_frac"   : 0.25,
        "test_frac"  : 0.15,
        "builder"    : args.name,
        "features"   : args.feat,
        "graph_dir"  : args.graph_dir,
        "k"          : args.k,
        "r"          : args.r,
        "NW"         : args.num_workers,
        "BS"         : args.batch_size,
        "n_clus"     : int(args.num_clusters),
        "n_epochs"   : int(args.epochs),
        "max_num"    : 10000,
    }
    torch.manual_seed(config["seed"])

    # load in h5 files
    path_to_cl_h5_file = "/srv/beegfs/scratch/shares/atlas_caloM/mu_200_truthjets/clusters/JZ4/user.lbozianu/user.lbozianu.43589851._000117.topoClD3PD_mc21_14TeV_JZ4.r14365.h5"
    c_data = h5py.File(path_to_cl_h5_file,"r")
    cl_event_data = c_data["caloCells"]["1d"]
    cl_data = c_data["caloCells"]["2d"]
    path_to_jet_h5_file = "/srv/beegfs/scratch/shares/atlas_caloM/mu_200_truthjets/jets/JZ4/user.lbozianu/user.lbozianu.43589851._000117.jetD3PD_mc21_14TeV_JZ4.r14365.h5"
    j_data = h5py.File(path_to_jet_h5_file,"r")
    jet_data = j_data["caloCells"]["2d"]


    test_data  = data.CaloDataset(root=args.root, name=config["builder"], feat=config["features"], k=config["k"], rad=config["r"], graph_dir=config["graph_dir"],test=True)
    test_loader  = DataLoader(test_data, batch_size=config["BS"], num_workers=config["NW"])
    print('\ttest size : ',len(test_data),'\n')

    # instantiate model
    feat_dict = {"XYZ": 5, "REP": 5, "REPP": 6, "GEO": 3, "CYL": 3}
    # model = models.Net(feat_dict[config["features"]], config["n_clus"]).to(config["device"])
    model = models.CustomNet(feat_dict[config["features"]], config["n_clus"]).to(config["device"])
    total_params = sum(p.numel() for p in model.parameters())
    print(f'DMoN (single conv layer) \t{total_params:,} total parameters.\n')
    # model_name = "DMoN_calo{}_{}_{}c_{}e".format(config["features"],config["builder"],config["n_clus"],config["n_epochs"])
    model_name = "customDMoN_calo{}_{}_{}c_{}e".format(config["features"],config["builder"],config["n_clus"],config["n_epochs"])
    model_save_path = args.model_dir + f"/{model_name}.pth"
    model.load_state_dict(torch.load(model_save_path, weights_only=True, map_location=torch.device(config["device"])))

    save_loc = args.output_dir + "/" + model_name + "/" + time.strftime("%Y%m%d-%H") + "/"
    print("Save location: ", save_loc)
    if not os.path.exists(save_loc): os.makedirs(save_loc)



    model.eval()
    beginning = time.perf_counter()

    with torch.inference_mode():
        tot_n_tc, tot_n_gnn = [], []
        tot_gnn_pt,tot_gnn_eta,tot_gnn_phi,tot_gnn_e,tot_gnn_n_cell = [],[],[],[],[]
        tot_cl_pt,tot_cl_eta,tot_cl_phi,tot_cl_e,tot_cl_n_cell = [],[],[],[],[]
        tot_akt_pt,tot_akt_eta,tot_akt_phi,tot_akt_m = [],[],[],[]
        tot_tru_pt,tot_tru_eta,tot_tru_phi,tot_tru_e,tot_tru_m = [],[],[],[],[]
        tot_cell_met = []
        for step, data in enumerate(test_loader):
            print(step)
            data = data.to(config["device"])
            # pred, tot_loss, clus_ass = model(data.x,data.edge_index,data.n,data.batch)
            pred, spectral_loss, ortho_loss, cluster_loss, clus_ass = model(data.x,data.edge_index,data.n,data.batch)
            node_features = torch_geometric.utils.unbatch(data.x,data.batch)
            cell_ids = torch_geometric.utils.unbatch(data.y,data.batch)
            
            for batch_idx in range(config["BS"]):
                # remove from GPU 
                pred_i = pred[batch_idx].detach().cpu().numpy()
                clus_i = clus_ass[batch_idx].detach().cpu()

                clus_i = clus_i[~np.isnan(clus_i).any(axis=1).bool()] # remove nan values
                x_i = node_features[batch_idx].detach().cpu().numpy()
                y_i = cell_ids[batch_idx].detach().cpu().numpy()

                # force each node to its most likely cluster, no soft assignment
                predicted_classes = clus_i.squeeze().argmax(dim=1).numpy()
                unique_values, counts = np.unique(predicted_classes, return_counts=True)
                tot_n_gnn.append(len(unique_values))

                # loop over GNN clusters
                gnn_pt,gnn_eta,gnn_phi,gnn_e,gnn_n_cell = [],[],[],[],[]
                for cl_idx in range(len(unique_values)):
                    cluster_i_mask = predicted_classes == unique_values[cl_idx]
                    cl_feat_cell_i = x_i[cluster_i_mask, :] # no longer have eta-phi in the eval_graph.x when feat=XYZ
                    cl_cell_i = y_i[cluster_i_mask, :] 
                    #XYZ == x, y, z, pt, significance
                    #REP == r, eta, phi, pt, significance
                    #y_i == ID, eta, phi, E
                    cl_eta, cl_phi = weighted_mean(cl_cell_i[:,1], cl_cell_i[:,3]), weighted_circular_mean(cl_cell_i[:,2], cl_cell_i[:,3])
                    cl_theta = 2*np.arctan(np.exp(-cl_eta))
                    cl_e = np.sum(cl_cell_i[:,3]) if len(cl_cell_i) else np.nan
                    # cl_phi = clip_phi(cl_phi)
                    cl_et = np.sin(cl_theta)*cl_e
                    cl_pt = np.sum(cl_feat_cell_i[:,-2])
                    if np.isfinite(cl_e):
                        gnn_pt.append(cl_pt)
                        gnn_eta.append(cl_eta)
                        gnn_phi.append(cl_phi)
                        gnn_e.append(cl_e)
                        gnn_n_cell.append(len(cl_cell_i))
                                             
                tot_gnn_pt.append(gnn_pt)
                tot_gnn_eta.append(gnn_eta)
                tot_gnn_phi.append(gnn_phi)
                tot_gnn_e.append(gnn_e)
                tot_gnn_n_cell.append(gnn_n_cell)

                #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # get event topo-clusters and antikt jets
                event_no = int(step*config["BS"]+batch_idx)

                clusters_event_i = cl_data[event_no]
                clusters_event_i = clusters_event_i[np.isfinite(clusters_event_i['cl_eta'])]
                cl_event_i = cl_event_data[event_no]
                tot_n_tc.append(cl_event_i["cl_n"]) #!

                # let's make jets out of TOPOCLUSTERS
                tc_pt,tc_eta,tc_phi,tc_e,tc_n_cell = [],[],[],[],[]
                for cl_idx in range(len(clusters_event_i)):
                    topocl_i = clusters_event_i[cl_idx]
                    cl_eta,cl_phi = topocl_i['cl_eta'], topocl_i['cl_phi']
                    cl_theta = 2*np.arctan(np.exp(-cl_eta))
                    cl_e = topocl_i['cl_E_em'] + topocl_i['cl_E_had']
                    cl_phi = clip_phi(cl_phi)
                    cl_et = np.sin(cl_theta)*cl_e
                    cl_pt = topocl_i['cl_pt']
                    tc_pt.append(cl_pt)
                    tc_eta.append(cl_eta)
                    tc_phi.append(cl_phi)
                    tc_e.append(cl_e)
                    tc_n_cell.append(topocl_i['cl_cell_n'])

                tot_cl_pt.append(tc_pt)
                tot_cl_eta.append(tc_eta)
                tot_cl_phi.append(tc_phi)
                tot_cl_e.append(tc_e)
                tot_cl_n_cell.append(tc_n_cell)

                # get antikt jets
                jets_event_i = jet_data[event_no]
                jets_event_i = jets_event_i[np.isfinite(jets_event_i['AntiKt4EMTopoJets_JetConstitScaleMomentum_pt'])]
                akt_pt,akt_eta,akt_phi,akt_m = [], [], [], []
                for jet_idx in range(len(jets_event_i)):
                    aktemtopojet = jets_event_i[jet_idx]
                    akt_pt.append(aktemtopojet['AntiKt4EMTopoJets_JetConstitScaleMomentum_pt'])
                    akt_eta.append(aktemtopojet['AntiKt4EMTopoJets_JetConstitScaleMomentum_eta'])
                    akt_phi.append(aktemtopojet['AntiKt4EMTopoJets_JetConstitScaleMomentum_phi'])
                    akt_m.append(aktemtopojet['AntiKt4EMTopoJets_JetConstitScaleMomentum_m'])
                tot_akt_pt.append(akt_pt)
                tot_akt_eta.append(akt_eta)
                tot_akt_phi.append(akt_phi)
                tot_akt_m.append(akt_m)

                # get truth jets
                tru_jets_event_i = jet_data[event_no]
                tru_jets_event_i = tru_jets_event_i[np.isfinite(tru_jets_event_i['AntiKt4TruthJets_pt'])]
                tru_pt,tru_eta,tru_phi,tru_e,tru_m = [], [], [],[],[]
                for jet_idx in range(len(tru_jets_event_i)):
                    trujet = tru_jets_event_i[jet_idx]
                    tru_pt.append(trujet['AntiKt4TruthJets_pt'])
                    tru_eta.append(trujet['AntiKt4TruthJets_eta'])
                    tru_phi.append(trujet['AntiKt4TruthJets_phi'])
                    tru_e.append(trujet['AntiKt4TruthJets_E'])
                    tru_m.append(trujet['AntiKt4TruthJets_m'])
                tot_tru_pt.append(tru_pt)
                tot_tru_eta.append(tru_eta)
                tot_tru_phi.append(tru_phi)
                tot_tru_e.append(tru_e)
                tot_tru_m.append(tru_m)

                # get cell MET
                #y_i == ID, eta, phi, E
                cells_E = y_i[:,3]
                cells_eta = y_i[:,1]
                cells_phi = y_i[:,2]
                cells_theta = 2*np.arctan(np.exp(-cells_eta))
                
                E_x = cells_E * np.sin(cells_theta) * np.cos(cells_phi) # hadamard product
                E_x_miss = np.sum(E_x)
                E_y = cells_E * np.sin(cells_theta) * np.sin(cells_phi) # hadamard product
                E_y_miss = np.sum(E_y)
                E_T_miss = np.sqrt( (E_x_miss**2) + (E_y_miss**2))
                tot_cell_met.append(E_T_miss)

        end = time.perf_counter()      
        print(f"Time taken for entire test set: {(end-beginning)/60:.3f} mins, (or {(end-beginning):.3f}s)")

    

        print('Saving the clusters and jets in lists...')
        # gnn clusters
        save_object(tot_gnn_pt, save_loc+'tot_gnn_pt.pkl')
        save_object(tot_gnn_eta, save_loc+'tot_gnn_eta.pkl')
        save_object(tot_gnn_phi, save_loc+'tot_gnn_phi.pkl')
        save_object(tot_gnn_e, save_loc+'tot_gnn_e.pkl')
        save_object(tot_gnn_n_cell, save_loc+'tot_gnn_n_cell.pkl')
        # topoclusters
        save_object(tot_cl_pt, save_loc+'tot_cl_pt.pkl')
        save_object(tot_cl_eta, save_loc+'tot_cl_eta.pkl')
        save_object(tot_cl_phi, save_loc+'tot_cl_phi.pkl')
        save_object(tot_cl_e, save_loc+'tot_cl_e.pkl')
        save_object(tot_cl_n_cell, save_loc+'tot_cl_n_cell.pkl')
        # AKT jets
        save_object(tot_akt_pt, save_loc+'tot_akt_pt.pkl')
        save_object(tot_akt_eta, save_loc+'tot_akt_eta.pkl')
        save_object(tot_akt_phi, save_loc+'tot_akt_phi.pkl')
        save_object(tot_akt_m, save_loc+'tot_akt_m.pkl')
        # Truth jets
        save_object(tot_tru_pt, save_loc+'tot_tru_pt.pkl')
        save_object(tot_tru_eta, save_loc+'tot_tru_eta.pkl')
        save_object(tot_tru_phi, save_loc+'tot_tru_phi.pkl')
        save_object(tot_tru_e, save_loc+'tot_tru_e.pkl')
        save_object(tot_tru_m, save_loc+'tot_tru_m.pkl')
        # number of clusters per event
        save_object(tot_n_gnn, save_loc+'tot_n_gnn.pkl')
        save_object(tot_n_tc, save_loc+'tot_n_tc.pkl')
        # cell met per event
        save_object(tot_cell_met, save_loc+'tot_cell_met.pkl')
