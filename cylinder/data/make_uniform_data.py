import matplotlib
from matplotlib import pyplot as plt

import torch
import torch_geometric
from torch_geometric.loader import DataLoader



def make_data_list(num_graphs,k=3):
    # Cylinder parameters
    mean_radius = 24.0
    mean_length = 72.0

    mean_num_nodes = 12.0
    mean_num_lid_nodes = 72.0
    mean_num_poles = 6.0


    pyg_data_list = []
    for j in range(num_graphs):
        num_nodes = int(torch.normal(mean=torch.tensor(mean_num_nodes),std=2.0))
        num_lids_nodes = int(torch.normal(mean=torch.tensor(mean_num_lid_nodes),std=8.0))
        num_poles = int(max(3,torch.normal(mean=torch.tensor(mean_num_poles),std=2.0)))
        length = int(torch.normal(mean=torch.tensor(mean_length),std=6.0))
        radius = int(torch.normal(mean=torch.tensor(mean_radius),std=2.0))


        # generate cylinder data
        z = torch.linspace(0.0, length, num_nodes)
        theta = torch.linspace(0.0, 2*torch.pi, num_nodes)
        x = radius * torch.cos(theta)
        y = radius * torch.sin(theta)
        cyl_points = torch.vstack((torch.tile(x, (num_poles,)), torch.tile(y, (num_poles,)), torch.repeat_interleave(z, num_poles))).T

        # generate lid data
        lid_theta = torch.rand(num_lids_nodes)*2*torch.pi #uniform in [0,1]
        r = torch.sqrt(torch.rand(num_lids_nodes)*radius**2)
        lid_points = torch.vstack([r * torch.cos(lid_theta), r * torch.sin(lid_theta), length*torch.ones(num_lids_nodes)]).T
        
        lid_theta = torch.rand(num_lids_nodes)*2*torch.pi #uniform in [0,1]
        r = torch.sqrt(torch.rand(num_lids_nodes)*radius**2)
        lid_points2 = torch.vstack([r * torch.cos(lid_theta), r * torch.sin(lid_theta), torch.zeros(num_lids_nodes)]).T

        #
        points = torch.vstack([cyl_points,lid_points,lid_points2])

        # Create a PyTorch Geometric Data object
        # data = torch_geometric.data.Data(x=points[:, :3])
        edge_index = torch_geometric.nn.knn_graph(points[:, :3],k=3)

        graph_data = torch_geometric.data.Data(x=points[:, :3],edge_index=edge_index) 
        pyg_data_list.append(graph_data)
            
    return pyg_data_list


if __name__=="__main__":
    k = 3
    #generate data and put into pytorch geometric dataloaders
    train_data_list = make_data_list(1000,k)
    val_data_list = make_data_list(200,k)
    test_data_list = make_data_list(100,k)

    train_loader = DataLoader(train_data_list, batch_size=20)
    val_loader = DataLoader(val_data_list, batch_size=20)
    test_loader = DataLoader(test_data_list, batch_size=20)


    eval_graph = test_data_list[0]
    edge_index = eval_graph.edge_index

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(121, projection='3d')
    ax.scatter(eval_graph.x[:, 0], eval_graph.x[:, 1], eval_graph.x[:, 2], c='b', marker='o', label='Nodes')
    for src, dst in edge_index.t().tolist():
        x_src, y_src, z_src = eval_graph.x[src]
        x_dst, y_dst, z_dst = eval_graph.x[dst]
        
        ax.plot([x_src, x_dst], [y_src, y_dst], [z_src, z_dst], c='r')

    ax.set(xlabel='X',ylabel='Y',zlabel='Z',title=f'Graph with KNN {k} Edges')

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(eval_graph.x[:, 0], eval_graph.x[:, 1], eval_graph.x[:, 2], c='b', marker='o', label='Nodes')
    ax2.set(xlabel='X',ylabel='Y',zlabel='Z',title=f'Graph with KNN {k} Edges')
    plt.legend()
    plt.show()
    # plt.savefig('../plots/synthetic-data.png')

