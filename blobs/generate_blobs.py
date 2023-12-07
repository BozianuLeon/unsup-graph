import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import time

import torch
from torch.nn import Linear
import torch.nn.functional as F
import torch_geometric
from torch_geometric.loader import DataLoader
from torch_geometric.nn import DMoNPooling, GCNConv



def make_data_list(num_graphs,avg_num_nodes=250,k=3):
    # Generate the synthetic data
    pyg_data_list = []
    for j in range(num_graphs):
        # np.random.seed(42)
        # num_points = torch.normal(mean=torch.tensor(avg_num_nodes,dtype=torch.int),std=torch.tensor(avg_num_nodes/10,dtype=torch.float32))
        num_points = int(np.random.normal(250,25))
        cluster_centers = torch.tensor([[2.0, 2.0, 2.0],
                                        [3.0, 3.0, 3.0],
                                        [2.0, 3.0, 2.0],
                                        [3.0, 2.0, 3.0]], dtype=torch.float32)

        coordinates = torch.zeros((num_points, 3), dtype=torch.float32)
        point_importance = torch.zeros(num_points, dtype=torch.float32)
        true_clusters = torch.zeros(num_points, dtype=torch.float32)

        for i in range(num_points):
            rnd_cluster_idx = torch.randint(low=0, high=4,size=(1,))
            center = cluster_centers[rnd_cluster_idx]
            std_dev = 0.125
            coordinates[i] = center + torch.randn(size=(3,)) * std_dev
            point_importance[i] = 1 / np.linalg.norm(coordinates[i] - center) 
            true_clusters[i] = rnd_cluster_idx

        feat_mat = torch.column_stack((coordinates,point_importance))

        # Create a PyTorch Geometric Data object
        data = torch_geometric.data.Data(x=feat_mat[:, :3], y=true_clusters)
        edge_index = torch_geometric.nn.knn_graph(feat_mat[:, :3],k=3)

        graph_data = torch_geometric.data.Data(x=feat_mat[:, :3],edge_index=edge_index, y=true_clusters) 
        pyg_data_list.append(graph_data)
            
    return pyg_data_list


if __name__=="__main__":
    avg_num_nodes = 250
    k = 3
    #generate data and put into pytorch geometric dataloaders
    train_data_list = make_data_list(1000,avg_num_nodes,k)
    val_data_list = make_data_list(200,avg_num_nodes,k)
    test_data_list = make_data_list(100,avg_num_nodes,k)

    train_loader = DataLoader(train_data_list, batch_size=20)
    val_loader = DataLoader(val_data_list, batch_size=20)
    test_loader = DataLoader(test_data_list, batch_size=20)


    eval_graph = test_data_list[0].to(device)
    edge_index = eval_graph.edge_index

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(eval_graph.x[:, 0], eval_graph.x[:, 1], eval_graph.x[:, 2], c='b', marker='o', label='Nodes')
    for src, dst in edge_index.t().tolist():
        x_src, y_src, z_src = eval_graph.x[src]
        x_dst, y_dst, z_dst = eval_graph.x[dst]
        
        ax.plot([x_src, x_dst], [y_src, y_dst], [z_src, z_dst], c='r')

    ax.set(xlabel='X',ylabel='Y',zlabel='Z',title=f'Graph with KNN {k} Edges')
    plt.legend()
    plt.savefig('/home/users/b/bozianu/work/graph/unsupervised_graph/unsup-graph/blobs/plots/synthetic-data.png')
