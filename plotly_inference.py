import torch
import torch_geometric
import h5py
import numpy as np
import numpy.lib.recfunctions as rf
import plotly.graph_objects as go
import plotly.io as pio

import models
import data



if __name__=="__main__":


    # first we make the input graph
    path_to_h5_file = "/Users/leonbozianu/work/phd/graph/dmon/user.cantel.34126190._000001.calocellD3PD_mc16_JZ4W.r10788.h5"
    print("\t",path_to_h5_file)
    f1 = h5py.File(path_to_h5_file,"r")

    kk = 0 # event numb
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
    model_name = "DMoN_calo_100c_5e"
    model_save_path = f"/Users/leonbozianu/work/phd/graph/dmon/unsup-graph/saved_models/{model_name}.pth"
    model = models.Net(6, 100)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'DMoN (single conv layer) \t{total_params:,} total parameters.\n')
    model.load_state_dict(torch.load(model_save_path, weights_only=True))

    pred, tot_loss, clus_ass = model(eval_graph.x,eval_graph.edge_index,eval_graph.batch)
    predicted_classes = clus_ass.squeeze().argmax(dim=1).numpy()
    unique_values, counts = np.unique(predicted_classes, return_counts=True)

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

    html_file_path = './data/GNN_calo_2sig_bucket_edges_3d_plot.html'
    config = {'displayModeBar': True,'displaylogo': False}
    pio.write_html(fig_bucket, file=html_file_path, auto_open=True, include_plotlyjs='cdn', config=config)


    ##########################################################################################################################################################################################


    # KNN GRAPH
    edge_indices = torch_geometric.nn.knn_graph(feature_tensor[:,:3],k=3,loop=False)
    eval_graph   = torch_geometric.data.Data(x=feature_tensor[:,[0,1,2,3,4,-1]],edge_index=edge_indices,y=cell_id_tensor) 
    # instantiate model
    model_name = "DMoN_calo_50c_5e"
    model_save_path = f"/Users/leonbozianu/work/phd/graph/dmon/unsup-graph/saved_models/{model_name}.pth"
    model = models.Net(6, 50)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'DMoN (single conv layer) \t{total_params:,} total parameters.\n')
    model.load_state_dict(torch.load(model_save_path, weights_only=True))

    pred, tot_loss, clus_ass = model(eval_graph.x,eval_graph.edge_index,eval_graph.batch)
    predicted_classes = clus_ass.squeeze().argmax(dim=1).numpy()
    unique_values, counts = np.unique(predicted_classes, return_counts=True)

    fig_knn = go.Figure()
    for idx in range(len(unique_values)):
        cluster_no = unique_values[idx]
        cluster_id = predicted_classes==cluster_no
        gnn_cluster_ids = eval_graph.y[cluster_id]
        gnn_cell_x = eval_graph.x[cluster_id]
        fig_knn.add_trace(go.Scatter3d(x=gnn_cell_x[:, 0],y=gnn_cell_x[:, 2],z=gnn_cell_x[:, 1],mode='markers',marker=dict(size=3.0,opacity=0.6,symbol='circle'),name=f'GNN cluster {idx}'))

    # Update the layout
    fig_knn.update_layout(
        title={'text': 'GNN Output (|s| > 2) knn (5) Edges','y':0.95,'x':0.5,'xanchor': 'center','yanchor': 'top'},
        scene=dict(xaxis_title='X',yaxis_title='Y',zaxis_title='Z',aspectmode='data'),
        width=1400,
        height=1000,
        margin=dict(l=100, r=50, b=300, t=0),
        showlegend=True,
        legend=dict(yanchor="top",y=0.5,xanchor="right",x=0.99)
        )

    html_file_path = './data/GNN_calo_2sig_knn_edges_3d_plot.html'
    config = {'displayModeBar': True,'displaylogo': False}
    pio.write_html(fig_knn, file=html_file_path, auto_open=True, include_plotlyjs='cdn', config=config)



    ##########################################################################################################################################################################################


    # RADIUS GRAPH
    quit()
    edge_indices = torch_geometric.nn.radius_graph(feature_tensor[:,:3],r=200,loop=False)
    eval_graph   = torch_geometric.data.Data(x=feature_tensor[:,[0,1,2,3,4,-1]],edge_index=edge_indices,y=cell_id_tensor) 
    # instantiate model
    model_name = "DMoN_calo_50c_2e"
    model_save_path = f"/Users/leonbozianu/work/phd/graph/dmon/unsup-graph/saved_models/{model_name}.pth"
    model = models.Net(6, 50)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'DMoN (single conv layer) \t{total_params:,} total parameters.\n')
    model.load_state_dict(torch.load(model_save_path, weights_only=True))

    pred, tot_loss, clus_ass = model(eval_graph.x,eval_graph.edge_index,eval_graph.batch)
    predicted_classes = clus_ass.squeeze().argmax(dim=1).numpy()
    unique_values, counts = np.unique(predicted_classes, return_counts=True)

    fig_knn = go.Figure()
    for idx in range(len(unique_values)):
        cluster_no = unique_values[idx]
        cluster_id = predicted_classes==cluster_no
        gnn_cluster_ids = eval_graph.y[cluster_id]
        gnn_cell_x = eval_graph.x[cluster_id]
        fig_knn.add_trace(go.Scatter3d(x=gnn_cell_x[:, 0],y=gnn_cell_x[:, 2],z=gnn_cell_x[:, 1],mode='markers',marker=dict(size=3.0,opacity=0.6,symbol='circle'),name=f'GNN cluster {idx}'))

    # Update the layout
    fig_knn.update_layout(
        title={'text': 'GNN Output (|s| > 2) rad (200) Edges','y':0.95,'x':0.5,'xanchor': 'center','yanchor': 'top'},
        scene=dict(xaxis_title='X',yaxis_title='Y',zaxis_title='Z',aspectmode='data'),
        width=1400,
        height=1000,
        margin=dict(l=100, r=50, b=300, t=0),
        showlegend=True,
        legend=dict(yanchor="top",y=0.5,xanchor="right",x=0.99)
        )

    html_file_path = './data/GNN_calo_2sig_rad_edges_3d_plot.html'
    config = {'displayModeBar': True,'displaylogo': False}
    pio.write_html(fig_knn, file=html_file_path, auto_open=True, include_plotlyjs='cdn', config=config)











