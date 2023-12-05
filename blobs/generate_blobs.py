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
        num_points = np.random.normal(loc=avg_num_nodes,scale=avg_num_nodes/10)
        cluster_centers = np.array([[2.0, 2.0, 2.0],
                                    [3.0, 3.0, 3.0],
                                    [2.0, 3.0, 2.0],
                                    [3.0, 2.0, 3.0]], dtype=np.float32)

        coordinates = np.zeros((num_points, 3), dtype=np.float32)
        energy_consumption = np.zeros(num_points, dtype=np.float32)
        true_cluster = np.zeros(num_points, dtype=np.float32)

        for i in range(num_points):
            cluster_idx = np.random.randint(0, 4)
            center = cluster_centers[cluster_idx]
            std_dev = 0.125
            coordinates[i] = center + np.random.randn(3) * std_dev
            energy_consumption[i] = 1 / np.linalg.norm(coordinates[i] - center) #np.random.uniform(0.1, 2.0)
            true_cluster[i] = cluster_idx

        data_np = np.column_stack((coordinates, energy_consumption))
        data_torch = torch.tensor(data_np, dtype=torch.float)

        # Create a PyTorch Geometric Data object
        data = torch_geometric.data.Data(x=data_torch[:, :3], y=torch.tensor(true_cluster))
        edge_index = torch_geometric.nn.knn_graph(data_torch[:, :3],k=3)

        graph_data = torch_geometric.data.Data(x=data_torch[:, :3],edge_index=edge_index, y=torch.tensor(true_cluster)) 
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
