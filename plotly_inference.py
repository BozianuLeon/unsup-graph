import torch
import torch_geometric
from torch_geometric.loader import DataLoader

import h5py
import numpy as np
import numpy.lib.recfunctions as rf

import plotly.graph_objects as go
import plotly.io as pio
import os
import argparse

import models
import data



parser = argparse.ArgumentParser()
parser.add_argument('--root', nargs='?', const='./', default='./', type=str, help='Path to top-level h5 directory',)
parser.add_argument('--name', type=str, required=True, help='Name of edge building scheme (knn, rad, bucket, custom)')
parser.add_argument('-idx', nargs='?', const=0, default=0, type=int, help='Event number to examine')
parser.add_argument('-k', nargs='?', const=None, default=None, type=int, help='K-nearest neighbours value to be used only in knn graph')
parser.add_argument('-r', nargs='?', const=None, default=None, type=int, help='Radius value to be used only in radial graph')
parser.add_argument('--graph_dir', nargs='?', const='./data/', default='./data/', type=str, help='Path to processed folder containing .pt graphs',)


parser.add_argument('-e','--epochs', type=int, required=True, help='Number of training epochs',)
parser.add_argument('-bs','--batch_size', nargs='?', const=8, default=8, type=int, help='Batch size to be used')
parser.add_argument('-nw','--num_workers', nargs='?', const=2, default=2, type=int, help='Number of worker CPUs')
parser.add_argument('-nc','--num_clusters', nargs='?', const=5, default=5, type=int, help='Number of (max) clusters DMoN can predict')
parser.add_argument('--model_dir', type=str, required=True, help='Path to saved models directory',)
parser.add_argument('-out','--output_dir',nargs='?', const='./cache/', default='./cache/', type=str, help='Path to directory containing struc_array.npy',)
args = parser.parse_args()




if __name__=="__main__":

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
    test_data    = data.CaloDataset(root=args.root, name=config["builder"], k=config["k"], rad=config["r"], graph_dir=config["graph_dir"])
    test_loader  = DataLoader(test_data, batch_size=config["BS"], num_workers=config["NW"])
    eval_graph = test_data[args.idx].to(config["device"])



    ##########################################################################################################################################################################################

    # instantiate model
    model = models.Net(6, config["n_clus"]).to(config["device"])
    total_params = sum(p.numel() for p in model.parameters())
    print(f'DMoN (single conv layer) \t{total_params:,} total parameters.\n')
    model_name = "DMoN_calo_{}_{}c_{}e".format(config["builder"], config["n_clus"], config["n_epochs"])
    model_save_path = args.model_dir + f"/{model_name}.pth"
    print(f'Model saved here: {model_save_path}')
    model.load_state_dict(torch.load(model_save_path, weights_only=True, map_location=torch.device(config["device"])))

    print(eval_graph.x.shape)
    print(eval_graph.edge_index.shape)
    print(len(torch.unique(eval_graph.edge_index)))
    edge_indices = torch_geometric.nn.radius_graph(eval_graph.x[:,[0,1,2]], r=150)
    print('repeat dataset ',edge_indices.shape)
    adj = torch_geometric.utils.to_dense_adj(edge_indices)
    print('repeat dataset ',adj.shape)
    pred, tot_loss, clus_ass = model(eval_graph.x,eval_graph.edge_index,eval_graph.batch)
    # send back to cpu for plotting
    eval_graph = eval_graph.to("cpu")
    predicted_classes = clus_ass.squeeze().argmax(dim=1).cpu().numpy()
    unique_values, counts = np.unique(predicted_classes, return_counts=True)


    ##########################################################################################################################################################################################

    # make scatter plot of model inference

    if not os.path.exists(args.output_dir + model_name): os.makedirs(args.output_dir + model_name)
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

    # html_file_path = '/home/users/b/bozianu/work/calo-cluster/unsup-graph/plots/' + model_name + '/infer_3d_plot.html'
    html_file_path = args.output_dir + model_name + '/infer_3d_plot.html'
    config = {'displayModeBar': True,'displaylogo': False}
    pio.write_html(fig_bucket, file=html_file_path, auto_open=True, include_plotlyjs='cdn', config=config)


    ##########################################################################################################################################################################################

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
        title={'text': 'Input (|s| > 2) cell point cloud KNN 3 Edges','y':0.95,'x':0.5,'xanchor': 'center','yanchor': 'top'},
        scene=dict(xaxis_title='X',yaxis_title='Y',zaxis_title='Z',aspectmode='data'),
        width=1400,
        height=1000,
        margin=dict(l=100, r=50, b=200, t=0),
        showlegend=True,
        legend=dict(yanchor="top",y=0.5,xanchor="right",x=0.99)
        )

    html_file_path = args.output_dir + model_name + '/input_3d_plot.html'
    config = {'displayModeBar': True,'displaylogo': False}
    pio.write_html(fig, file=html_file_path, auto_open=True, include_plotlyjs='cdn', config=config)


    ##########################################################################################################################################################################################

    # TODO: get topoclusters from this event



