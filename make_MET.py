import numpy as np 
import os

import pickle
import time

# this script reads in lists of clusters made by GNN and topoclustering, and uses arxiv.org/abs/2005.09554 to calculate the missing transverse momentum
# also gets MHT as well
# proceeds in two parts:
# 1. Calculate MET using clusters (GNN and TC)
# 2. Calculate MET using jets (GNN, TC, AKT, MC Truth)

def load_object(fname):
    with open(fname,'rb') as file:
        return pickle.load(file)

def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp)


# get clusters from a particular trained model
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
# topoclusters
event_tcl_pt = load_object(metrics_folder+'tot_cl_pt.pkl')
event_tcl_eta = load_object(metrics_folder+'tot_cl_eta.pkl')
event_tcl_phi = load_object(metrics_folder+'tot_cl_phi.pkl')
event_tcl_e = load_object(metrics_folder+'tot_cl_e.pkl')

#gnn jets
event_gnn_jet_cl_pt = load_object(metrics_folder+'tot_gnn_jet_pt.pkl')
event_gnn_jet_cl_eta = load_object(metrics_folder+'tot_gnn_jet_eta.pkl')
event_gnn_jet_cl_phi = load_object(metrics_folder+'tot_gnn_jet_phi.pkl')
event_gnn_jet_cl_e = load_object(metrics_folder+'tot_gnn_jet_e.pkl')
event_gnn_jet_cl_m = load_object(metrics_folder+'tot_gnn_jet_m.pkl')
# topocluster jets 
event_tcl_jet_pt = load_object(metrics_folder+'tot_cl_jet_pt.pkl')
event_tcl_jet_eta = load_object(metrics_folder+'tot_cl_jet_eta.pkl')
event_tcl_jet_phi = load_object(metrics_folder+'tot_cl_jet_phi.pkl')
event_tcl_jet_e = load_object(metrics_folder+'tot_cl_jet_e.pkl')
event_tcl_jet_m = load_object(metrics_folder+'tot_cl_jet_m.pkl')
# akt jets
event_akt_jet_pt = load_object(metrics_folder+'tot_akt_pt.pkl')
event_akt_jet_eta = load_object(metrics_folder+'tot_akt_eta.pkl')
event_akt_jet_phi = load_object(metrics_folder+'tot_akt_phi.pkl')
event_akt_jet_m = load_object(metrics_folder+'tot_akt_m.pkl')
# truth jets
event_tru_jet_pt =  load_object(metrics_folder+'tot_tru_pt.pkl')
event_tru_jet_eta =  load_object(metrics_folder+'tot_tru_eta.pkl')
event_tru_jet_phi =  load_object(metrics_folder+'tot_tru_phi.pkl')
event_tru_jet_e =  load_object(metrics_folder+'tot_tru_e.pkl')
event_tru_jet_m =  load_object(metrics_folder+'tot_tru_m.pkl')



print(f"There were {len(event_gnn_cl_pt)} events in the test set. Or another way we have {len(event_tcl_eta)} events to find missing transverse momentum for!")
print(f"There were {len(event_gnn_jet_cl_eta)} events in the test set. Or another way we have {len(event_tru_jet_pt)} events to find missing transverse momentum for!")
beginning = time.perf_counter()


tot_gnn_cl_met,tot_tcl_met = [],[]
tot_gnn_cl_ht,tot_tcl_ht = [],[]
tot_gnn_jet_met, tot_tcl_jet_met, tot_tru_jet_met, tot_akt_jet_met = [],[],[],[]
tot_gnn_jet_ht, tot_tcl_jet_ht, tot_tru_jet_ht, tot_akt_jet_ht = [],[],[],[]
for event_i in range(len(event_gnn_cl_pt)):

    # 1. Make MET using clusters
    # First loop over GNN clusters
    gnn_E_x_miss, gnn_E_y_miss, gnn_H_T = np.zeros(len(event_gnn_cl_pt[event_i])), np.zeros(len(event_gnn_cl_pt[event_i])), np.zeros(len(event_gnn_cl_pt[event_i]))
    for cl_idx in range(len(event_gnn_cl_pt[event_i])):
        cl_eta = event_gnn_cl_eta[event_i][cl_idx]
        cl_phi = event_gnn_cl_phi[event_i][cl_idx]
        cl_e   = event_gnn_cl_e[event_i][cl_idx] / 1000 # GeV
        cl_theta = 2*np.arctan(np.exp(-cl_eta))
        cl_phi = clip_phi(cl_phi)
        E_x = cl_e * np.sin(cl_theta) * np.cos(cl_phi)
        E_y = cl_e * np.sin(cl_theta) * np.sin(cl_phi)
        gnn_E_x_miss[cl_idx] = E_x
        gnn_E_y_miss[cl_idx] = E_y
        E_T = cl_e * np.sin(cl_theta)
        gnn_H_T[cl_idx] = E_T
    # missing E_{x,y} in the event
    E_x_miss = - np.sum(gnn_E_x_miss)
    E_y_miss = - np.sum(gnn_E_y_miss)
    E_T_miss = np.sqrt(E_x_miss**2 + E_y_miss**2)
    tot_gnn_cl_met.append(E_T_miss)
    tot_gnn_cl_ht.append(np.sum(gnn_H_T))


    # Now loop over topoclusters
    tc_E_x_miss, tc_E_y_miss, tc_H_T = np.zeros(len(event_tcl_pt[event_i])), np.zeros(len(event_tcl_pt[event_i])), np.zeros(len(event_tcl_pt[event_i]))
    for cl_idx in range(len(event_tcl_pt[event_i])):
        cl_eta = event_tcl_eta[event_i][cl_idx]
        cl_phi = event_tcl_phi[event_i][cl_idx]
        cl_e   = event_tcl_e[event_i][cl_idx] / 1000 # GeV
        cl_theta = 2*np.arctan(np.exp(-cl_eta))
        cl_phi = clip_phi(cl_phi)
        E_x = cl_e * np.sin(cl_theta) * np.cos(cl_phi)
        E_y = cl_e * np.sin(cl_theta) * np.sin(cl_phi)
        tc_E_x_miss[cl_idx] = E_x
        tc_E_y_miss[cl_idx] = E_y
        E_T = cl_e * np.sin(cl_theta)
        tc_H_T[cl_idx] = E_T

    E_x_miss = - np.sum(tc_E_x_miss)
    E_y_miss = - np.sum(tc_E_y_miss)
    E_T_miss = np.sqrt(E_x_miss**2 + E_y_miss**2)
    tot_tcl_met.append(E_T_miss)
    tot_tcl_ht.append(np.sum(tc_H_T))
    
    
    # 2. Make MET using jets
    # (w/o ROOT TLorentzVector)

    # GNN jets
    gnn_jet_E_x_miss, gnn_jet_E_y_miss, gnn_jet_H_T = np.zeros(len(event_gnn_jet_cl_pt[event_i])), np.zeros(len(event_gnn_jet_cl_pt[event_i])), np.zeros(len(event_gnn_jet_cl_pt[event_i]))
    for jet_idx in range(len(event_gnn_jet_cl_pt[event_i])):
        jet_eta = event_gnn_jet_cl_eta[event_i][jet_idx]
        jet_phi = event_gnn_jet_cl_pt[event_i][jet_idx]
        jet_e   = event_gnn_jet_cl_pt[event_i][jet_idx] / 1000 # GeV
        jet_theta = 2*np.arctan(np.exp(-jet_eta))
        jet_phi = clip_phi(jet_phi)
        E_x = jet_e * np.sin(jet_theta) * np.cos(jet_phi)
        E_y = jet_e * np.sin(jet_theta) * np.sin(jet_phi)
        gnn_jet_E_x_miss[jet_idx] = E_x
        gnn_jet_E_y_miss[jet_idx] = E_y
        E_T = jet_e * np.sin(jet_theta)
        gnn_jet_H_T[jet_idx] = E_T
    # missing E_{x,y} in the jets in the event
    E_x_miss = - np.sum(gnn_jet_E_x_miss)
    E_y_miss = - np.sum(gnn_jet_E_y_miss)
    E_T_miss_gnn_jet = np.sqrt(E_x_miss**2 + E_y_miss**2)
    tot_gnn_jet_met.append(E_T_miss_gnn_jet)
    tot_gnn_jet_ht.append(np.sum(gnn_jet_H_T))

    # topocluster jets
    tcl_jet_E_x_miss, tcl_jet_E_y_miss, tcl_jet_H_T = np.zeros(len(event_tcl_jet_pt[event_i])), np.zeros(len(event_tcl_jet_pt[event_i])), np.zeros(len(event_tcl_jet_pt[event_i]))
    for jet_idx in range(len(event_tcl_jet_pt[event_i])):
        jet_eta = event_tcl_jet_eta[event_i][jet_idx]
        jet_phi = event_tcl_jet_phi[event_i][jet_idx]
        # jet_e   = event_tcl_jet_e[event_i][jet_idx] / 1000 # GeV # broken value, defaults to 0!
        jet_pt   = event_tcl_jet_pt[event_i][jet_idx] # GeV
        jet_m    = event_tcl_jet_m[event_i][jet_idx]
        jet_e   = np.sqrt(jet_pt**2 * np.cosh(jet_eta)**2 + jet_m**2) / 1000 # because tcl jets have buggy E
        jet_theta = 2*np.arctan(np.exp(-jet_eta))
        jet_phi = clip_phi(jet_phi)
        E_x = jet_e * np.sin(jet_theta) * np.cos(jet_phi)
        E_y = jet_e * np.sin(jet_theta) * np.sin(jet_phi)
        tcl_jet_E_x_miss[jet_idx] = E_x
        tcl_jet_E_y_miss[jet_idx] = E_y
        E_T = jet_e * np.sin(jet_theta)
        tcl_jet_H_T[jet_idx] = E_T
    E_x_miss = - np.sum(tcl_jet_E_x_miss)
    E_y_miss = - np.sum(tcl_jet_E_y_miss)
    E_T_miss_tcl_jet = np.sqrt(E_x_miss**2 + E_y_miss**2)
    tot_tcl_jet_met.append(E_T_miss_tcl_jet)
    tot_tcl_jet_ht.append(np.sum(tcl_jet_H_T))
    
    # akt4 jets
    akt_jet_E_x_miss, akt_jet_E_y_miss, akt_jet_H_T = np.zeros(len(event_akt_jet_pt[event_i])), np.zeros(len(event_akt_jet_pt[event_i])), np.zeros(len(event_akt_jet_pt[event_i]))
    for jet_idx in range(len(event_akt_jet_pt[event_i])):
        jet_eta = event_akt_jet_eta[event_i][jet_idx]
        jet_phi = event_akt_jet_phi[event_i][jet_idx]
        jet_m   = event_akt_jet_m[event_i][jet_idx]
        jet_pt  = event_akt_jet_pt[event_i][jet_idx]
        jet_e   = np.sqrt(jet_pt**2 * np.cosh(jet_eta)**2 + jet_m**2) / 1000 # because AKT jets don't have raw E, recalculate it!
        jet_theta = 2*np.arctan(np.exp(-jet_eta))
        jet_phi = clip_phi(jet_phi)
        E_x = jet_e * np.sin(jet_theta) * np.cos(jet_phi)
        E_y = jet_e * np.sin(jet_theta) * np.sin(jet_phi)
        akt_jet_E_x_miss[jet_idx] = E_x
        akt_jet_E_y_miss[jet_idx] = E_y
        E_T = jet_e * np.sin(jet_theta)
        akt_jet_H_T[jet_idx] = E_T
    E_x_miss = - np.sum(akt_jet_E_x_miss)
    E_y_miss = - np.sum(akt_jet_E_y_miss)
    E_T_miss_akt_jet = np.sqrt(E_x_miss**2 + E_y_miss**2)
    tot_tru_jet_met.append(E_T_miss_akt_jet)
    tot_tru_jet_ht.append(np.sum(akt_jet_H_T))

    # truth jets
    tru_jet_E_x_miss, tru_jet_E_y_miss, tru_jet_H_T = np.zeros(len(event_tru_jet_pt[event_i])), np.zeros(len(event_tru_jet_pt[event_i])), np.zeros(len(event_tru_jet_pt[event_i]))
    for jet_idx in range(len(event_tru_jet_pt[event_i])):
        jet_eta = event_tru_jet_eta[event_i][jet_idx]
        jet_phi = event_tru_jet_phi[event_i][jet_idx]
        jet_e   = event_tru_jet_e[event_i][jet_idx] / 1000 # GeV
        jet_theta = 2*np.arctan(np.exp(-jet_eta))
        jet_phi = clip_phi(jet_phi)
        E_x = jet_e * np.sin(jet_theta) * np.cos(jet_phi)
        E_y = jet_e * np.sin(jet_theta) * np.sin(jet_phi)
        tru_jet_E_x_miss[jet_idx] = E_x
        tru_jet_E_y_miss[jet_idx] = E_y
        E_T = jet_e * np.sin(jet_theta)
        tru_jet_H_T[jet_idx] = E_T
    E_x_miss = - np.sum(tru_jet_E_x_miss)
    E_y_miss = - np.sum(tru_jet_E_y_miss)
    E_T_miss_tru_jet = np.sqrt(E_x_miss**2 + E_y_miss**2)
    tot_tru_jet_met.append(E_T_miss_tru_jet)
    tot_tru_jet_ht.append(np.sum(tru_jet_H_T))
       
    print(event_i)



end = time.perf_counter()      
print(f"Time taken for entire test set: {(end-beginning)/60:.3f} mins, (or {(end-beginning):.3f}s)")

print('Saving the clusters and jets in lists...')
# 1. Cluster MET
save_object(tot_gnn_cl_met, metrics_folder+'tot_gnn_cl_met.pkl')
save_object(tot_tcl_met, metrics_folder+'tot_tcl_met.pkl')

save_object(tot_gnn_cl_ht, metrics_folder+'tot_gnn_cl_ht.pkl')
save_object(tot_tcl_ht, metrics_folder+'tot_tcl_ht.pkl')
# 2. Jet MET
save_object(tot_gnn_jet_met, metrics_folder+'tot_gnn_jet_met.pkl')
save_object(tot_tcl_jet_met, metrics_folder+'tot_tcl_jet_met.pkl')
save_object(tot_akt_jet_met, metrics_folder+'tot_akt_jet_met.pkl')
save_object(tot_tru_jet_met, metrics_folder+'tot_tru_jet_met.pkl')

save_object(tot_gnn_jet_ht, metrics_folder+'tot_gnn_jet_ht.pkl')
save_object(tot_tcl_jet_ht, metrics_folder+'tot_tcl_jet_ht.pkl')
save_object(tot_akt_jet_ht, metrics_folder+'tot_akt_jet_ht.pkl')
save_object(tot_tru_jet_ht, metrics_folder+'tot_tru_jet_ht.pkl')
