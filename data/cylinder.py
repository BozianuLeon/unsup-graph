import torch
import torch_geometric
from torch_geometric.loader import DataLoader

from matplotlib import pyplot as plt


def synthetic_cylinder_list(num_graphs,k=3):
    '''
    Quick function to generate synthetic data in a 3D
    cylinder, rough approximation to a calorimeter. 
    No "truth" labels, organised in endcaps and poles.
    Hard coded geometrical radius/length

    Input: 
        num_graphs: int, number of synthetic "events" to generate
        k: int, k-nearest neighbours value
    
    Returns:
        pyg_data_list: list, collection of torch_geometric.data.Data
            objects containing node features, edge indices and 
            synthetic labels
    '''
    mean_radius = 24.0
    mean_length = 72.0

    mean_num_bar_nodes = 12.0
    mean_num_ec_nodes = 72.0
    mean_num_poles = 6.0


    pyg_data_list = []
    for j in range(num_graphs):
        num_bar_nodes = int(torch.normal(mean=torch.tensor(mean_num_bar_nodes),std=2.0))
        num_ec_nodes = int(torch.normal(mean=torch.tensor(mean_num_ec_nodes),std=8.0))
        num_poles = int(max(3,torch.normal(mean=torch.tensor(mean_num_poles),std=2.0)))
        length = int(torch.normal(mean=torch.tensor(mean_length),std=6.0))
        radius = int(torch.normal(mean=torch.tensor(mean_radius),std=2.0))


        # generate cylinder data
        z = torch.linspace(0.0, length, num_bar_nodes)
        theta = torch.linspace(0.0, 2*torch.pi, num_bar_nodes)
        x = radius * torch.cos(theta)
        y = radius * torch.sin(theta)
        cyl_points = torch.vstack((torch.tile(x, (num_poles,)), torch.tile(y, (num_poles,)), torch.repeat_interleave(z, num_poles))).T

        # generate lid data
        ec_theta = torch.rand(num_ec_nodes)*2*torch.pi #uniform in [0,1]
        r = torch.sqrt(torch.rand(num_ec_nodes)*radius**2)
        ec_points = torch.vstack([r * torch.cos(ec_theta), r * torch.sin(ec_theta), length*torch.ones(num_ec_nodes)]).T
        
        ec_theta = torch.rand(num_ec_nodes)*2*torch.pi #uniform in [0,1]
        r = torch.sqrt(torch.rand(num_ec_nodes)*radius**2)
        ec_points2 = torch.vstack([r * torch.cos(ec_theta), r * torch.sin(ec_theta), torch.zeros(num_ec_nodes)]).T

        # gather points in torch tensor
        points = torch.vstack([cyl_points,ec_points,ec_points2])

        # Create a PyTorch Geometric Data object
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

