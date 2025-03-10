# THIS SCRIPT IS A TESTER< JUST DO ONE EVENT
import torch
import torch_geometric
import h5py
import numpy as np
import numpy.lib.recfunctions as rf
import scipy
import os
import matplotlib.pyplot as plt
import matplotlib
import plotly.graph_objects as go
import plotly.io as pio
import fastjet


import models
import data


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





if __name__=="__main__":


    # first we make the input graph
    path_to_h5_file = "/srv/beegfs/scratch/shares/atlas_caloM/mu_200_truthjets/cells/JZ4/user.lbozianu/user.lbozianu.43589851._000117.calocellD3PD_mc21_14TeV_JZ4.r14365.h5"
    print("\t",path_to_h5_file)
    f1 = h5py.File(path_to_h5_file,"r")

    kk = 3 # event numb
    cells = f1["caloCells"]["2d"][kk] # event 0 cells
    mask_2sigma = abs(cells['cell_E'] / cells['cell_Sigma']) >= 2
    mask_4sigma = abs(cells['cell_E'] / cells['cell_Sigma']) >= 4
    cells2sig = cells[mask_2sigma]
    cells4sig = cells[mask_4sigma]
    print(cells4sig.dtype)

    # get cell feature matrix from struct array 
    cell_significance = np.expand_dims(abs(cells2sig['cell_E'] / cells2sig['cell_Sigma']),axis=1)
    cell_features = cells2sig[['cell_xCells','cell_yCells','cell_zCells','cell_eta','cell_phi','cell_E','cell_Sigma','cell_pt']]
    feature_matrix = rf.structured_to_unstructured(cell_features,dtype=np.float32)
    feature_matrix = np.hstack((feature_matrix,cell_significance))
    feature_tensor = torch.tensor(feature_matrix)    

    # get cell IDs,we will also return the cell IDs in the "y" attribute of .Data object
    cell_id_array  = np.expand_dims(cells2sig['cell_IdCells'],axis=1)
    cell_id_tensor = torch.tensor(cell_id_array.astype(np.int64))

    # make sparse adjacency matrix using BUCKET graph
    neighbours_array = cell_neighbours = np.load('./data/pyg/cell_neighbours.npy')
    src_neighbours_array = src_cell_neighbours = np.load('./data/pyg/src_cell_neighbours.npy')

    # get cell IDs
    cell_ids_2 = np.array(cells2sig['cell_IdCells'].astype(int)) # THIS IS THE GRAND LIST OF CELLS WE CAN USE IN THIS EVENT
    cell_ids_2_array = np.expand_dims(cell_ids_2,axis=1) # we will also return the cell IDs in the "y" attribute pytorch geometric
    # get the neighbour arrays for the 2 sigma cells
    cell_neighb_2 = neighbours_array[mask_2sigma]
    src_cell_neighb_2 = src_neighbours_array[mask_2sigma]
    # filter cell neighbours, only >2sigma and remove padded -999 values
    actual_cell_neighb_2 = np.where(np.isin(cell_neighb_2,cell_ids_2), cell_neighb_2, np.nan) # actual cells we can use from cell_neighbours
    actual_src_cell_neighb_2 = np.where(np.isin(cell_neighb_2,cell_ids_2), src_cell_neighb_2, np.nan) 
    # find the cellID indices from cell_ids_2, what index are they in this event?
    neighb_2sig_indices = np.searchsorted(cell_ids_2,actual_cell_neighb_2)
    neighb_src_2sig_indices = np.searchsorted(cell_ids_2,actual_src_cell_neighb_2)
    # use the nan array to again extract just the valid node indices we want
    dst_node_indices = neighb_2sig_indices[~np.isnan(actual_cell_neighb_2)]
    src_node_indices = neighb_src_2sig_indices[~np.isnan(actual_src_cell_neighb_2)]

    edge_indices = np.stack((dst_node_indices,src_node_indices),axis=0)
    eval_graph   = torch_geometric.data.Data(x=feature_tensor[:,[0,1,2,3,4,-1]],edge_index=torch.tensor(edge_indices),y=cell_id_tensor) 




    ##########################################################################################################################################################################################


    # make scatter plot of model inference(?)
    # BUCKET GRAPH
    # instantiate model
    model_name = "DMoN_calo_bucket_150c_10e"
    model_save_path = f"/home/users/b/bozianu/work/calo-cluster/unsup-graph/saved_models/{model_name}.pth"
    model = models.Net(6, 150)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'DMoN (single conv layer) \t{total_params:,} total parameters.\n')
    model.load_state_dict(torch.load(model_save_path, weights_only=True, map_location=torch.device(torch.device("cuda" if torch.cuda.is_available() else "cpu"))))

    pred, tot_loss, clus_ass = model(eval_graph.x,eval_graph.edge_index,eval_graph.batch)
    predicted_classes = clus_ass.squeeze().argmax(dim=1).numpy()
    unique_values, counts = np.unique(predicted_classes, return_counts=True)


    save_loc = f"/home/users/b/bozianu/work/calo-cluster/unsup-graph/plots/{model_name}/{kk}/"
    if not os.path.exists(save_loc): os.makedirs(save_loc)

    # make input graph with edges from sparse adj matrix

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=eval_graph.x[:, 0],y=eval_graph.x[:, 2],z=eval_graph.x[:, 1],mode='markers',marker=dict(size=2.5,color='blue',opacity=0.8,symbol='circle'),name='Nodes'))

    edge_x,edge_y,edge_z = [],[],[]
    # For each edge, add the coordinates of both nodes and None to create a break
    for src, dst in eval_graph.edge_index.t().tolist():
        x_src, y_src, z_src, *feat = eval_graph.x[src]
        x_dst, y_dst, z_dst, *feat = eval_graph.x[dst]
        edge_x.extend([x_src, x_dst, None])
        edge_y.extend([y_src, y_dst, None])
        edge_z.extend([z_src, z_dst, None])

    # Add the edges as a single line trace
    fig.add_trace(go.Scatter3d(x=edge_x,y=edge_z,z=edge_y,mode='lines',line=dict(color='red',width=1),opacity=0.5,name='Edges',hoverinfo='none'))

    # Update the layout
    fig.update_layout(
        title={'text': f'Input (|s| > 2) cell point cloud Bucket (eta-phi) Edges','y':0.95,'x':0.5,'xanchor': 'center','yanchor': 'top'},
        scene=dict(xaxis_title='X',yaxis_title='Y',zaxis_title='Z',aspectmode='data'),
        width=1400,
        height=1000,
        margin=dict(l=100, r=50, b=200, t=0),
        showlegend=True,
        legend=dict(yanchor="top",y=0.5,xanchor="right",x=0.99)
        )

    html_file_path = save_loc + '/bucket_150_input_3d_plot.html'
    config = {'displayModeBar': True,'displaylogo': False}
    pio.write_html(fig, file=html_file_path, auto_open=True, include_plotlyjs='cdn', config=config)

    #####################################################



    fig_bucket = go.Figure()
    for idx in range(len(unique_values)):
        cluster_no = unique_values[idx]
        cluster_id = predicted_classes==cluster_no
        gnn_cluster_ids = eval_graph.y[cluster_id]
        gnn_cell_x = eval_graph.x[cluster_id]
        fig_bucket.add_trace(go.Scatter3d(x=gnn_cell_x[:, 0],y=gnn_cell_x[:, 2],z=gnn_cell_x[:, 1],mode='markers',marker=dict(size=3.0,opacity=0.6,symbol='circle'),name=f'GNN cluster {idx}'))

    # Update the layout
    fig_bucket.update_layout(
        title={'text': 'GNN Output (|s| > 2) BUCKET (eta-phi) Edges','y':0.95,'x':0.5,'xanchor': 'center','yanchor': 'top'},
        scene=dict(xaxis_title='X',yaxis_title='Y',zaxis_title='Z',aspectmode='data'),
        width=1400,
        height=1000,
        margin=dict(l=100, r=50, b=300, t=0),
        showlegend=True,
        legend=dict(yanchor="top",y=0.5,xanchor="right",x=0.99)
        )

    html_file_path = save_loc + '/bucket_150_inference_3d_plot.html'
    config = {'displayModeBar': True,'displaylogo': False}
    pio.write_html(fig_bucket, file=html_file_path, auto_open=True, include_plotlyjs='cdn', config=config)


    #######################################

    cell_etas = cell_features['cell_eta']
    cell_phis = cell_features['cell_phi']
    cell_pt = cell_features['cell_pt']/1000
    #np.linspace(start, stop, int((stop - start) / step + 1))
    bins_x = np.linspace(min(cell_etas), max(cell_etas), int((max(cell_etas) - min(cell_etas)) / 0.1 + 1))
    bins_y = np.linspace(min(cell_phis), max(cell_phis), int((max(cell_phis) - min(cell_phis)) / ((2*np.pi)/64) + 1))
    H_sum_pt, xedges, yedges, binnumber = scipy.stats.binned_statistic_2d(cell_etas, cell_phis,values=cell_pt,bins=(bins_x,bins_y),statistic='sum')
    H_sum_pt = H_sum_pt.T
    # Padding
    repeat_frac = 0.5
    repeat_rows = int(H_sum_pt.shape[0]*repeat_frac)
    one_box_height = (yedges[-1]-yedges[0])/H_sum_pt.shape[0]
    H_sum_pt = np.pad(H_sum_pt, ((repeat_rows,repeat_rows),(0,0)),'wrap')
    H_sum_pt[np.isnan(H_sum_pt)] = 0
    extent = (xedges[0],xedges[-1],yedges[0]-(repeat_rows*one_box_height),yedges[-1]+(repeat_rows*one_box_height)) 

    f,ax = plt.subplots(figsize=(10,12))
    ii = ax.imshow(H_sum_pt,cmap='YlOrRd',extent=extent,origin='lower')
    cbar = f.colorbar(ii,ax=ax)
    cbar.ax.get_yaxis().labelpad = 10
    cbar.set_label(f'cell pt', rotation=90)
    ax.set(xlabel='eta',ylabel='phi')
    f.savefig(f"{save_loc}/data_{kk}_examine_img_2d.png")


    #######################################
    f,a = plt.subplots(1,1,figsize=(10,12))   
    a.scatter(feature_tensor[:, 3], feature_tensor[:, 4], s=10, c=predicted_classes, cmap='tab20b')
    a.set(xlabel='eta',ylabel='phi',title=f'DMoN Output Clustered Cells')
    f.savefig(f"{save_loc}/data_{kk}_cells_2d.png")
    plt.close()

    #######################################
    fi,ax = plt.subplots(1,1,figsize=(10,12)) 
    ax.set_prop_cycle(color=plt.cm.tab20b(np.linspace(0, 1, 20)))
    for cl_idx in range(len(unique_values)):
        cluster_i_mask = predicted_classes == unique_values[cl_idx]
        cl_cell_i = feature_tensor[cluster_i_mask, :].numpy()
        cl_eta, cl_phi = weighted_mean(cl_cell_i[:,3], cl_cell_i[:,5]), weighted_circular_mean(cl_cell_i[:,4], cl_cell_i[:,5])
        cl_e = np.sum(cl_cell_i[:,5]) if len(cl_cell_i) else np.nan
        cl_phi = clip_phi(cl_phi)
        cl_theta = 2*np.arctan(np.exp(-cl_eta))
        cl_et = np.sin(cl_theta)*cl_e
        ax.plot(cl_eta, cl_phi, ms=(cl_et/1000)/5, marker='^',alpha=.6)
        if cl_et > 5000:
            ax.text(cl_eta+0.1,cl_phi+0.1, f"{cl_et/1000:.1f}",color='red',fontsize=8)
    ax.set(xlabel='eta',ylabel='phi',title=f'DMoN Output Clusters')
    fi.savefig(f"{save_loc}/data_{kk}_DMoNcl_2d.png")
    plt.close()

    for value, count in zip(unique_values, counts):
        print(f"\tCluster {value}: {count} occurrences")
    
    #######################################
    # let's make jets out of our clusters
    jetdef = fastjet.JetDefinition(fastjet.antikt_algorithm, 0.4)
    m = 0 # clusters are considered massless
    gnn_jet_constituents = []
    for cl_idx in range(len(unique_values)):
        cluster_i_mask = predicted_classes == unique_values[cl_idx]
        cl_cell_i = feature_tensor[cluster_i_mask, :].numpy()
        cl_eta, cl_phi = weighted_mean(cl_cell_i[:,3], cl_cell_i[:,5]), weighted_circular_mean(cl_cell_i[:,4], cl_cell_i[:,5])
        cl_theta = 2*np.arctan(np.exp(-cl_eta))
        cl_e = np.sum(cl_cell_i[:,5]) if len(cl_cell_i) else np.nan
        # cl_phi = clip_phi(cl_phi)
        cl_et = np.sin(cl_theta)*cl_e
        cl_pt = np.sum(cl_cell_i[:,-2])
        if np.isfinite(cl_e):
            print(f"There are {len(cl_cell_i)} cells in this cluster ({cl_idx},{unique_values[cl_idx]}). Eta: {cl_eta:.3f}, Phi: {cl_phi:.3f}, E: {cl_e/1000:.3f} GeV, ET {cl_et/1000:.3f} GeV, (PT {cl_pt/1000:.3f})")
            gnn_jet_constituents.append(fastjet.PseudoJet(cl_e * np.sin(cl_theta)*np.cos(cl_phi),
                                                        cl_e * np.sin(cl_theta)*np.sin(cl_phi),
                                                        cl_e * np.cos(cl_theta),
                                                        m))
        else:
            print(f"!There are {len(cl_cell_i)} cells in this cluster. Eta: {cl_eta:.3f}, Phi: {cl_phi:.3f}, E: {cl_e/1000:.3f} GeV, ET {cl_et/1000:.3f} GeV, (PT {cl_pt/1000:.3f})")
    gnn_pred_jets = fastjet.ClusterSequence(gnn_jet_constituents,jetdef)
    gnn_pred_jets_inc = gnn_pred_jets.inclusive_jets()

    fi,ax = plt.subplots(1,1,figsize=(10,12)) 
    for jet_idx in range(len(gnn_pred_jets_inc)):
        gnn_jet_i = gnn_pred_jets_inc[jet_idx]
        ax.plot(gnn_jet_i.eta(), clip_phi(gnn_jet_i.phi()), ms=(gnn_jet_i.pt()/1000)/10, marker='o',alpha=.6)
        ax.text(gnn_jet_i.eta()+0.1,clip_phi(gnn_jet_i.phi())+0.1, f"{gnn_jet_i.pt()/1000:.1f}",color='red',fontsize=8)
    ax.set(xlabel='eta',ylabel='phi',xlim=(-4.9,4.9),ylim=(-3.2,3.2),title=f'DMoN Output Clusters => Jets')
    fi.savefig(f"{save_loc}/data_{kk}_DMoNjets_2d.png")
    plt.close()

    fi,ax = plt.subplots(1,1,figsize=(10,12)) 
    for jet_idx in range(len(gnn_pred_jets_inc)):
        gnn_jet_i = gnn_pred_jets_inc[jet_idx]
        print(f"This GNN jet has Eta: {gnn_jet_i.eta():.3f}, Phi: {clip_phi(gnn_jet_i.phi()):.3f}, E: {gnn_jet_i.E()/1000:.3f} GeV, pT {gnn_jet_i.pt()/1000:.3f} GeV")
        if gnn_jet_i.pt() > 25000:
            ax.plot(gnn_jet_i.eta(), clip_phi(gnn_jet_i.phi()), ms=(gnn_jet_i.pt()/1000)/10, marker='o',alpha=.6)
            ax.text(gnn_jet_i.eta()+0.1,clip_phi(gnn_jet_i.phi())+0.1, f"{gnn_jet_i.pt()/1000:.1f}",color='red',fontsize=8)
    ax.set(xlabel='eta',ylabel='phi',xlim=(-4.9,4.9),ylim=(-3.2,3.2),title=f'DMoN Output Jets')
    fi.savefig(f"{save_loc}/data_{kk}_DMoNjets_20GeV_2d.png")
    plt.close()


    ##########################################################################################################################################################################################


    def remove_nan(array):
        # find the indices where there are not nan values
        good_indices = np.where(array==array) 
        return array[good_indices]



    path_to_cl_h5_file = "/srv/beegfs/scratch/shares/atlas_caloM/mu_200_truthjets/clusters/JZ4/user.lbozianu/user.lbozianu.43589851._000117.topoClD3PD_mc21_14TeV_JZ4.r14365.h5"
    with h5py.File(path_to_cl_h5_file,"r") as f:
        c_data = f["caloCells"]
        cl_data = c_data["2d"][kk]
        cl_data = remove_nan(cl_data)


    fi,ax = plt.subplots(1,1,figsize=(10,12)) 
    for cl_idx in range(len(cl_data)):
        topocl_i = cl_data[cl_idx]
        ax.plot(topocl_i['cl_eta'], topocl_i['cl_phi'], ms=(topocl_i['cl_pt']/1000)/5, marker='H',alpha=.6,color='darkslategrey')
        # ax.plot(topocl_i['cl_eta'], topocl_i['cl_phi'], ms=((topocl_i['cl_E_em'] + topocl_i['cl_E_had'])/1000)/5, marker='s',alpha=.6)
        if topocl_i['cl_pt'] > 5000:
        # if (topocl_i['cl_E_em'] + topocl_i['cl_E_had']) > 5000:
            # ax.text(topocl_i['cl_eta']+0.1,topocl_i['cl_phi']+0.1, f"{(topocl_i['cl_E_em'] + topocl_i['cl_E_had'])/1000:.1f}",color='red',fontsize=8)
            ax.text(topocl_i['cl_eta']+0.1,topocl_i['cl_phi']+0.1, f"{(topocl_i['cl_pt'])/1000:.1f}",color='red',fontsize=8)
    ax.set(xlabel='eta',ylabel='phi',xlim=(-4.9,4.9),ylim=(-3.2,3.2),title=f'Topoclusters')
    fi.savefig(f"{save_loc}/data_{kk}_topocl_2d.png")
    plt.close()
    print()


    #######################################
    # let's make jets out of TOPOCLUSTERS
    jetdef = fastjet.JetDefinition(fastjet.antikt_algorithm, 0.4)
    tc_jet_constituents = []
    for cl_idx in range(len(cl_data)):
        topocl_i = cl_data[cl_idx]
        cl_eta,cl_phi = topocl_i['cl_eta'], topocl_i['cl_phi']
        cl_theta = 2*np.arctan(np.exp(-cl_eta))
        cl_e = topocl_i['cl_E_em'] + topocl_i['cl_E_had']
        cl_phi = clip_phi(cl_phi)
        cl_et = np.sin(cl_theta)*cl_e
        cl_pt = topocl_i['cl_pt']
        # print(f"There are {len(cl_cell_i)} cells in this cluster. Eta: {cl_eta:.3f}, Phi: {cl_phi:.3f}, E: {cl_e/1000:.3f} GeV, ET {cl_et/1000:.3f} GeV, (PT {cl_pt/1000:.3f})")
        tc_jet_constituents.append(fastjet.PseudoJet(cl_e * np.sin(cl_theta)*np.cos(cl_phi),
                                                     cl_e * np.sin(cl_theta)*np.sin(cl_phi),
                                                     cl_e * np.cos(cl_theta),
                                                     m))
    tc_jets = fastjet.ClusterSequence(tc_jet_constituents,jetdef)
    tc_jets_inc = tc_jets.inclusive_jets()
    
    fi,ax = plt.subplots(1,1,figsize=(10,12)) 
    for jet_idx in range(len(tc_jets_inc)):
        topocl_jet_i = tc_jets_inc[jet_idx]
        print(f"This TC fastjet jet has Eta: {topocl_jet_i.eta():.3f}, Phi: {clip_phi(topocl_jet_i.phi()):.3f}, E: {topocl_jet_i.E()/1000:.3f} GeV, pT {topocl_jet_i.pt()/1000:.3f} GeV")
        if topocl_jet_i.pt() > 25000:
            ax.plot(topocl_jet_i.eta(), clip_phi(topocl_jet_i.phi()), ms=(topocl_jet_i.pt()/1000)/10, marker='o',alpha=.6,color='darkslategrey')
            ax.text(topocl_jet_i.eta()+0.1,clip_phi(topocl_jet_i.phi())+0.1, f"{topocl_jet_i.pt()/1000:.1f}",color='red',fontsize=8)
    ax.set(xlabel='eta',ylabel='phi',xlim=(-4.9,4.9),ylim=(-3.2,3.2),title=f'Topocluster Constituents through Fastjet')
    fi.savefig(f"{save_loc}/data_{kk}_topocl_fastjet_2d.png")
    plt.close()

    ##########################################################################################################################################################################################
    print()

    path_to_jet_h5_file = "/srv/beegfs/scratch/shares/atlas_caloM/mu_200_truthjets/jets/JZ4/user.lbozianu/user.lbozianu.43589851._000117.jetD3PD_mc21_14TeV_JZ4.r14365.h5"
    with h5py.File(path_to_jet_h5_file,"r") as f:
        j_data = f["caloCells"]
        jet_data = j_data["2d"][kk]


    
    fi,ax = plt.subplots(1,1,figsize=(10,12)) 
    for jet_idx in range(len(jet_data['AntiKt4EMTopoJets_JetConstitScaleMomentum_pt'])):
        aktemtopojet = jet_data[jet_idx]
        if np.isnan(aktemtopojet['AntiKt4EMTopoJets_JetConstitScaleMomentum_eta']):
            continue
        print(f"This AntiKt jet has Eta: {aktemtopojet['AntiKt4EMTopoJets_JetConstitScaleMomentum_eta']:.3f}, Phi: {aktemtopojet['AntiKt4EMTopoJets_JetConstitScaleMomentum_phi']:.3f}, ~E: {aktemtopojet['AntiKt4EMTopoJets_E']/1000:.3f} GeV, pT {aktemtopojet['AntiKt4EMTopoJets_JetConstitScaleMomentum_pt']/1000:.3f} GeV")
        ax.plot(aktemtopojet['AntiKt4EMTopoJets_JetConstitScaleMomentum_eta'], aktemtopojet['AntiKt4EMTopoJets_JetConstitScaleMomentum_phi'], ms=(aktemtopojet['AntiKt4EMTopoJets_JetConstitScaleMomentum_pt']/1000)/10, marker='o',alpha=.6,color='seagreen')
        ax.text(aktemtopojet['AntiKt4EMTopoJets_JetConstitScaleMomentum_eta']+0.1,aktemtopojet['AntiKt4EMTopoJets_JetConstitScaleMomentum_phi']+0.1, f"{aktemtopojet['AntiKt4EMTopoJets_JetConstitScaleMomentum_pt']/1000:.1f}",color='red',fontsize=8)
    ax.set(xlabel='eta',ylabel='phi',xlim=(-4.9,4.9),ylim=(-3.2,3.2),title=f'AntiKt4EMTopoJets JetConstitScale')
    fi.savefig(f"{save_loc}/data_{kk}_AntiKt4EMTopoJets_2d.png")
    plt.close()
    
    print()
    fi,ax = plt.subplots(1,1,figsize=(10,12)) 
    for jet_idx in range(len(jet_data['AntiKt4TruthJets_pt'])):
        aktjet = jet_data[jet_idx]
        if np.isnan(aktjet['AntiKt4TruthJets_pt']):
            continue
        print(f"This Truth jet has Eta: {aktjet['AntiKt4TruthJets_eta']:.3f}, Phi: {aktjet['AntiKt4TruthJets_phi']:.3f}, ~E: {aktjet['AntiKt4TruthJets_E']/1000:.3f} GeV, pT {aktjet['AntiKt4TruthJets_pt']/1000:.3f} GeV")
        ax.plot(aktjet['AntiKt4TruthJets_eta'], aktjet['AntiKt4TruthJets_phi'], ms=(aktjet['AntiKt4TruthJets_pt']/1000)/10, marker='o',alpha=.6,color='gold')
        ax.text(aktjet['AntiKt4TruthJets_eta']+0.1,aktjet['AntiKt4TruthJets_phi']+0.1, f"{aktjet['AntiKt4TruthJets_pt']/1000:.1f}",color='black',fontsize=8)
    ax.set(xlabel='eta',ylabel='phi',xlim=(-4.9,4.9),ylim=(-3.2,3.2),title=f'AntiKt4 Truth jets')
    fi.savefig(f"{save_loc}/data_{kk}_AntiKt4TruthJets_2d.png")
    plt.close()

    

