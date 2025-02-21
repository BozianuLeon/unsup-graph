import torch
import torch_geometric
from torch_geometric.loader import DataLoader

from matplotlib import pyplot as plt


def synthetic_blobs_list(num_graphs,avg_num_nodes=250,k=3):
    '''
    Quick function to generate synthetic data in 3D
    Centred on 3 pre-defined clusters, variance given 
    by a Gaussian 
    Input: 
        num_graphs: int, number of synthetic "events" to generate
        avg_num_nodes: int, average number of nodes to include 
            in point cloud
        k: int, k-nearest neighbours value
    
    Returns:
        pyg_data_list: list, collection of torch_geometric.data.Data
            objects containing node features, edge indices and 
            synthetic labels
    '''
    
    pyg_data_list = []
    for j in range(num_graphs):
        num_points = int(torch.normal(mean=avg_num_nodes,std=torch.tensor(avg_num_nodes/10)))
        
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
            std_dev = 0.5
            coordinates[i] = center + torch.randn(size=(3,)) * std_dev
            point_importance[i] = 1 / torch.linalg.norm(coordinates[i] - center) 
            true_clusters[i] = rnd_cluster_idx

        feat_mat = torch.column_stack((coordinates,point_importance))

        # Create a PyTorch Geometric Data object
        data = torch_geometric.data.Data(x=feat_mat[:, :3], y=true_clusters)
        edge_index = torch_geometric.nn.knn_graph(feat_mat[:, :3],k=3)

        graph_data = torch_geometric.data.Data(x=feat_mat[:, :4],edge_index=edge_index, y=true_clusters) 
        pyg_data_list.append(graph_data)
            
    return pyg_data_list


if __name__=="__main__":
    avg_num_nodes = 250
    k = 3
    #generate data and put into pytorch geometric dataloaders
    train_data_list = synthetic_blobs_list(1000,avg_num_nodes,k)
    val_data_list = synthetic_blobs_list(200,avg_num_nodes,k)
    test_data_list = synthetic_blobs_list(100,avg_num_nodes,k)

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
    plt.show()
    # plt.savefig('../plots/synthetic-data.png')
