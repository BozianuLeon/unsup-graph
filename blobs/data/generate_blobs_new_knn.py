import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import scipy
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
        num_points = int(torch.normal(mean=avg_num_nodes,std=torch.tensor(avg_num_nodes/10)))
        print(num_points,type(num_points))
        
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

        # new
        q1, q2, q3 = torch.quantile(point_importance, torch.tensor([0.25, 0.5, 0.75]))

        # Create masks for each quartile
        mask_q1 = point_importance <= q1
        mask_q2 = (q1 < point_importance) & (point_importance <= q2)
        mask_q3 = (q2 < point_importance) & (point_importance <= q3)
        mask_q4 = point_importance > q3

        # Extract indices for each quartile
        indices_q1 = torch.nonzero(mask_q1, as_tuple=False).squeeze(dim=1)
        indices_q2 = torch.nonzero(mask_q2, as_tuple=False).squeeze(dim=1)
        indices_q3 = torch.nonzero(mask_q3, as_tuple=False).squeeze(dim=1)
        indices_q4 = torch.nonzero(mask_q4, as_tuple=False).squeeze(dim=1)

        points_q1 = coordinates[mask_q1]
        points_q2 = coordinates[mask_q2]
        points_q3 = coordinates[mask_q3]
        points_q4 = coordinates[mask_q4]

        tree = scipy.spatial.cKDTree(coordinates)

        #variable k
        k1,k2,k3,k4 = 3,6,9,12

        distances1,indices1 = tree.query(points_q1, k=k1)
        distances1 = torch.from_numpy(distances1).to(torch.float32)
        col1 = torch.from_numpy(indices1).to(torch.long)
        row1 = torch.arange(col1.size(0),dtype=torch.long).view(-1,1).repeat(1,k1)
        mask = ~torch.isinf(distances1).view(-1).to(torch.bool)
        row1, col1 = row1.view(-1)[mask], col1.view(-1)[mask]
        #need to return to original indices:
        orig_row1 = torch.gather(indices_q1,0,row1)
        #return pairs of source,dest node indices.
        edges_q1 = torch.stack([orig_row1, col1], dim=0)

        distances2,indices2 = tree.query(points_q2, k=k2)
        distances2 = torch.from_numpy(distances2).to(torch.float32)
        col2 = torch.from_numpy(indices2).to(torch.long)
        row2 = torch.arange(col2.size(0),dtype=torch.long).view(-1,1).repeat(1,k2)
        mask = ~torch.isinf(distances2).view(-1).to(torch.bool)
        row2, col2 = row2.view(-1)[mask], col2.view(-1)[mask]
        #need to return to original indices:
        orig_row2 = torch.gather(indices_q2,0,row2)
        edges_q2 = torch.stack([orig_row2, col2], dim=0)

        distances3,indices3 = tree.query(points_q3, k=k3)
        distances3 = torch.from_numpy(distances3).to(torch.float32)
        col3 = torch.from_numpy(indices3).to(torch.long)
        row3 = torch.arange(col3.size(0),dtype=torch.long).view(-1,1).repeat(1,k3)
        mask = ~torch.isinf(distances3).view(-1).to(torch.bool)
        row3, col3 = row3.view(-1)[mask], col3.view(-1)[mask]
        #need to return to original indices:
        orig_row3 = torch.gather(indices_q3,0,row3)
        edges_q3 = torch.stack([orig_row3, col3], dim=0)

        distances4,indices4 = tree.query(points_q4, k=k4)
        distances4 = torch.from_numpy(distances4).to(torch.float32)
        col4 = torch.from_numpy(indices4).to(torch.long)
        row4 = torch.arange(col4.size(0),dtype=torch.long).view(-1,1).repeat(1,k4)
        mask = ~torch.isinf(distances4).view(-1).to(torch.bool)
        row4, col4 = row4.view(-1)[mask], col4.view(-1)[mask]
        #need to return to original indices:
        orig_row4 = torch.gather(indices_q4,0,row4)
        edges_q4 = torch.stack([orig_row4, col4], dim=0)


        edges_total = torch.cat((edges_q1, edges_q2, edges_q3, edges_q4), dim=1)

        # Create a PyTorch Geometric Data object
        data = torch_geometric.data.Data(x=feat_mat[:, :3], y=true_clusters)
        # edge_index = torch_geometric.nn.knn_graph(feat_mat[:, :3],k=3)

        graph_data = torch_geometric.data.Data(x=feat_mat[:, :3],edge_index=edges_total, y=true_clusters) 
        pyg_data_list.append(graph_data)
            
    return pyg_data_list






if __name__=="__main__":
    avg_num_nodes = 25
    k = 3
    #generate data and put into pytorch geometric dataloaders
    train_data_list = make_data_list(1000,avg_num_nodes,k)
    val_data_list = make_data_list(200,avg_num_nodes,k)
    test_data_list = make_data_list(100,avg_num_nodes,k)

    train_loader = DataLoader(train_data_list, batch_size=20)
    val_loader = DataLoader(val_data_list, batch_size=20)
    test_loader = DataLoader(test_data_list, batch_size=20)


    eval_graph = test_data_list[0]
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
    # plt.show()
    plt.savefig('../plots/synthetic-data-new-knn.png')
