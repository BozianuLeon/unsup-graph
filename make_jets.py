import numpy as np 
import os

import pickle
import time
import fastjet

# this script reads in lists of clusters made by GNN and topoclustering, and uses fastjet to make jets in each event

def load_object(fname):
    with open(fname,'rb') as file:
        return pickle.load(file)

def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp)




features = "XYZ"
builder = "custom"
n_clus = 1000
n_epochs = 32
model_name = "DMoN_calo{}_{}_{}c_{}e".format(features,builder,n_clus,n_epochs)
datetime = "20250502-11"
metrics_folder = f"/home/users/b/bozianu/work/calo-cluster/unsup-graph/cache/{model_name}/{datetime}/"

def clip_phi(phi_values):
    # map phi values to lie in (-pi,pi]
    return phi_values - 2 * np.pi * np.floor((phi_values + np.pi) / (2 * np.pi))


# gnn clusters
event_gnn_cl_pt = load_object(metrics_folder+'tot_gnn_pt.pkl')
event_gnn_cl_eta = load_object(metrics_folder+'tot_gnn_eta.pkl')
event_gnn_cl_phi = load_object(metrics_folder+'tot_gnn_phi.pkl')
event_gnn_cl_e = load_object(metrics_folder+'tot_gnn_e.pkl')
# event_gnn_cl_n_cell = load_object(metrics_folder+'tot_gnn_n_cell.pkl')
# topoclusters
event_tcl_pt = load_object(metrics_folder+'tot_cl_pt.pkl')
event_tcl_eta = load_object(metrics_folder+'tot_cl_eta.pkl')
event_tcl_phi = load_object(metrics_folder+'tot_cl_phi.pkl')
event_tcl_e = load_object(metrics_folder+'tot_cl_e.pkl')
# event_tcl_n_cell = load_object(metrics_folder+'tot_cl_n_cell.pkl')



print(f"There were {len(event_gnn_cl_pt)} events in the test set. Or another way we have {len(event_tcl_eta)} events to produce GNN and topocluster jets for!")
beginning = time.perf_counter()


gnnjetdef = fastjet.JetDefinition(fastjet.antikt_algorithm, 0.4)
tcjetdef = fastjet.JetDefinition(fastjet.antikt_algorithm, 0.4)

tot_gnn_jet_pt,tot_gnn_jet_eta,tot_gnn_jet_phi,tot_gnn_jet_e,tot_gnn_jet_m = [],[],[],[],[] 
tot_cl_jet_pt,tot_cl_jet_eta,tot_cl_jet_phi,tot_cl_jet_e,tot_cl_jet_m = [],[],[],[],[]
for event_i in range(len(event_gnn_cl_pt)):

    # 1. Make GNN jets
    # loop over GNN clusters
    m = 0 # clusters are considered massless
    gnn_jet_constituents = []
    for cl_idx in range(len(event_gnn_cl_pt[event_i])):
        cl_pt  = event_gnn_cl_pt[event_i][cl_idx]
        cl_eta = event_gnn_cl_eta[event_i][cl_idx]
        cl_phi = event_gnn_cl_phi[event_i][cl_idx]
        cl_e   = event_gnn_cl_e[event_i][cl_idx]
        cl_theta = 2*np.arctan(np.exp(-cl_eta))
        cl_phi = clip_phi(cl_phi)
        if cl_e > 0.0: # ensure that raw cluster energy is positive entering the jets
            # print(f"There are {len(cl_cell_i)} cells in this cluster ({cl_idx},{unique_values[cl_idx]}). Eta: {cl_eta:.3f}, Phi: {cl_phi:.3f}, E: {cl_e/1000:.3f} GeV, ET {cl_et/1000:.3f} GeV, (PT {cl_pt/1000:.3f})")
            gnn_jet_constituents.append(fastjet.PseudoJet(cl_e * np.sin(cl_theta)*np.cos(cl_phi),
                                                          cl_e * np.sin(cl_theta)*np.sin(cl_phi),
                                                          cl_e * np.cos(cl_theta),
                                                          m))     

    # Use Anti-kt to cluster the gnn clusters into jets                                                                    
    gnn_pred_jets = fastjet.ClusterSequence(gnn_jet_constituents,gnnjetdef)
    gnn_pred_jets_inc = gnn_pred_jets.inclusive_jets()

    gnn_jet_pt,gnn_jet_eta,gnn_jet_phi,gnn_jet_e,gnn_jet_m = [],[],[],[],[]
    for gnnjet_i in range(len(gnn_pred_jets_inc)):
        gnn_jet_in_question = gnn_pred_jets_inc[gnnjet_i]
        if gnn_jet_in_question.pt() > 0.0:
            gnn_jet_pt.append(gnn_jet_in_question.pt())
            gnn_jet_eta.append(gnn_jet_in_question.eta())
            gnn_jet_phi.append(gnn_jet_in_question.phi())
            gnn_jet_e.append(gnn_jet_in_question.E())
            gnn_jet_m.append(gnn_jet_in_question.m())

    tot_gnn_jet_pt.append(gnn_jet_pt)
    tot_gnn_jet_eta.append(gnn_jet_eta)
    tot_gnn_jet_phi.append(gnn_jet_phi)
    tot_gnn_jet_e.append(gnn_jet_e)
    tot_gnn_jet_m.append(gnn_jet_m)


    # 2. Make Topocluster jets
    # loop over topoclusters, from the esd
    m = 0
    tc_jet_constituents = []
    for cl_idx in range(len(event_tcl_pt[event_i])):
        cl_pt  = event_tcl_pt[event_i][cl_idx]
        cl_eta = event_tcl_eta[event_i][cl_idx]
        cl_phi = event_tcl_phi[event_i][cl_idx]
        cl_e   = event_tcl_e[event_i][cl_idx]
        cl_theta = 2*np.arctan(np.exp(-cl_eta))
        cl_phi = clip_phi(cl_phi)
        if cl_e > 0.0: # ensure that raw cluster energy is positive entering the jets
            tc_jet_constituents.append(fastjet.PseudoJet(cl_e * np.sin(cl_theta)*np.cos(cl_phi),
                                                          cl_e * np.sin(cl_theta)*np.sin(cl_phi),
                                                          cl_e * np.cos(cl_theta),
                                                          m))     

    # Use Anti-kt to cluster the topoclusters into jets                                                                    
    tc_jets = fastjet.ClusterSequence(tc_jet_constituents,tcjetdef)
    tc_jets_inc = tc_jets.inclusive_jets()

    tc_jet_pt,tc_jet_eta,tc_jet_phi,tc_jet_e,tc_jet_m = [],[],[],[],[]
    for tcjet in range(len(tc_jets_inc)):
        tc_jet_in_question = tc_jets_inc[tcjet]
        tc_jet_pt.append(tc_jet_in_question.pt())
        tc_jet_eta.append(tc_jet_in_question.eta())
        tc_jet_phi.append(tc_jet_in_question.phi())
        tc_jet_e.append(tc_jet_in_question.E())
        tc_jet_m.append(tc_jet_in_question.m())

    tot_cl_jet_pt.append(tc_jet_pt)
    tot_cl_jet_eta.append(tc_jet_eta)
    tot_cl_jet_phi.append(tc_jet_phi)
    tot_cl_jet_e.append(tc_jet_e)
    tot_cl_jet_m.append(tc_jet_m)
    print(event_i)


end = time.perf_counter()      
print(f"Time taken for entire test set: {(end-beginning)/60:.3f} mins, (or {(end-beginning):.3f}s)")

print('Saving the clusters and jets in lists...')
# gnn jets
save_object(tot_gnn_jet_pt, metrics_folder+'tot_gnn_jet_pt.pkl')
save_object(tot_gnn_jet_eta, metrics_folder+'tot_gnn_jet_eta.pkl')
save_object(tot_gnn_jet_phi, metrics_folder+'tot_gnn_jet_phi.pkl')
save_object(tot_gnn_jet_e, metrics_folder+'tot_gnn_jet_e.pkl')
save_object(tot_gnn_jet_m, metrics_folder+'tot_gnn_jet_m.pkl')
# topocluster jets
save_object(tot_cl_jet_pt, metrics_folder+'tot_cl_jet_pt.pkl')
save_object(tot_cl_jet_eta, metrics_folder+'tot_cl_jet_eta.pkl')
save_object(tot_cl_jet_phi, metrics_folder+'tot_cl_jet_phi.pkl')
save_object(tot_cl_jet_e, metrics_folder+'tot_cl_jet_e.pkl')
save_object(tot_cl_jet_m, metrics_folder+'tot_cl_jet_m.pkl')
