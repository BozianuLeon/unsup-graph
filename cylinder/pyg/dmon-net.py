import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import os
import time

import torch
from torch.nn import Linear
import torch.nn.functional as F
import torch_geometric
from torch_geometric.loader import DataLoader
from torch_geometric.nn import DMoNPooling, GCNConv



def make_data_list(num_graphs,k=3):
    # Cylinder parameters
    mean_radius = 24.0
    mean_length = 72.0

    mean_num_nodes = 20.0
    mean_num_lid_nodes = 64.0
    mean_num_poles = 14.0


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
        cyl2_points = torch.vstack((torch.tile(1.2*radius * torch.cos(theta+0.1), (int(num_poles/2),)), torch.tile(1.2*radius * torch.sin(theta+0.1), (int(num_poles/2),)), torch.repeat_interleave(z, int(num_poles/2)))).T
        cyl3_points = torch.vstack((torch.tile(2.2*radius * torch.cos(theta+0.1), (int(num_poles/3),)), torch.tile(2.2*radius * torch.sin(theta+0.1), (int(num_poles/3),)), torch.repeat_interleave(z, int(num_poles/3)))).T

        # generate lid data
        lid_theta = torch.rand(num_lids_nodes)*2*torch.pi #uniform in [0,1]
        r = torch.sqrt(torch.rand(num_lids_nodes)*radius**2)
        lid_points = torch.vstack([r * torch.cos(lid_theta), r * torch.sin(lid_theta), length*torch.ones(num_lids_nodes)]).T
        
        lid_theta = torch.rand(num_lids_nodes)*2*torch.pi #uniform in [0,1]
        r = torch.sqrt(torch.rand(num_lids_nodes)*radius**2)
        lid_points2 = torch.vstack([r * torch.cos(lid_theta), r * torch.sin(lid_theta), torch.zeros(num_lids_nodes)]).T

        #
        points = torch.vstack([cyl_points,cyl2_points,cyl3_points,lid_points,lid_points2])

        # Create a PyTorch Geometric Data object
        # data = torch_geometric.data.Data(x=points[:, :3])
        edge_index = torch_geometric.nn.knn_graph(points[:, :3],k=3)

        graph_data = torch_geometric.data.Data(x=points[:, :3],edge_index=edge_index) 
        pyg_data_list.append(graph_data)
            
    return pyg_data_list




#initialise model
class Net(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=32):
        super().__init__()

        self.conv1 = GCNConv(in_channels, 128)
        self.relu = torch.nn.ReLU()
        self.selu = torch.nn.SELU()
        self.pool1 = DMoNPooling(128,out_channels)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        # x = self.relu(x)
        x = self.selu(x)

        x, mask = torch_geometric.utils.to_dense_batch(x, batch)
        adj = torch_geometric.utils.to_dense_adj(edge_index, batch)
        s, x, adj, sp1, o1, c1 = self.pool1(x, adj, mask)

        return F.log_softmax(x, dim=-1), sp1+o1+c1, s





def train(train_loader):
    model.train()
    loss_all = 0

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out, tot_loss, _ = model(data.x, data.edge_index, data.batch)
        loss = tot_loss #+ F.nll_loss(out, data.y.view(-1))
        loss.backward()
        loss_all += data.x.size(0) * float(loss)
        optimizer.step()
    return loss_all / len(train_data_list)



@torch.no_grad()
def test(loader):
    model.eval()
    correct = 0
    loss_all = 0

    for data in loader:
        data = data.to(device)
        pred, tot_loss, _ = model(data.x, data.edge_index, data.batch)
        loss = tot_loss # + F.nll_loss(pred, data.y.view(-1))
        loss_all += data.x.size(0) * float(loss)
        # correct += int(pred.max(dim=1)[1].eq(data.y.view(-1)).sum())

    return loss_all / len(loader.dataset)#, correct / len(loader.dataset)




if __name__=='__main__':

    #generate data and put into pytorch geometric dataloaders
    train_data_list = make_data_list(1000)
    val_data_list = make_data_list(200)
    test_data_list = make_data_list(100)

    train_loader = DataLoader(train_data_list, batch_size=20)
    val_loader = DataLoader(val_data_list, batch_size=20)
    test_loader = DataLoader(test_data_list, batch_size=20)

    num_clusters = 5
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net(3, num_clusters).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    #run training
    for epoch in range(1, 14):
        start = time.perf_counter()
        train_loss = train(train_loader)
        train_loss2 = test(train_loader)
        val_loss = test(val_loader)
        test_loss = test(test_loader)
        print(f"Epoch: {epoch:03d}, Train Loss: {train_loss:.3f}, Train Loss2: {train_loss2:.3f}, Val Loss: {val_loss:.3f}, Test Loss: {test_loss:.3f}, Time: {time.perf_counter() - start:.3f}s")


    #evaluate using first graph
    eval_graph = test_data_list[0]
    pred,tot_loss,clus_ass = model(eval_graph.x, eval_graph.edge_index, eval_graph.batch)

    predicted_classes = clus_ass.squeeze().argmax(dim=1).numpy()
    unique_values, counts = np.unique(predicted_classes, return_counts=True)

    if os.path.exists(f"../plots/results/") is False: os.makedirs(f"../plots/results/")
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(121, projection='3d')
    ax.scatter(eval_graph.x[:, 0], eval_graph.x[:, 1], eval_graph.x[:, 2], c='b', marker='o', label='Nodes')
    for src, dst in eval_graph.edge_index.t().tolist():
        x_src, y_src, z_src = eval_graph.x[src]
        x_dst, y_dst, z_dst = eval_graph.x[dst]
        
        ax.plot([x_src, x_dst], [y_src, y_dst], [z_src, z_dst], c='r')

    ax.set(xlabel='X',ylabel='Y',zlabel='Z',title=f'Graph with KNN 3 Edges')

    ax2 = fig.add_subplot(122, projection='3d')
    scatter = ax2.scatter(eval_graph.x[:, 0], eval_graph.x[:, 1], eval_graph.x[:, 2], c=predicted_classes, marker='o')
    labels = [f"{value}: ({count})" for value,count in zip(unique_values, counts)]
    ax2.legend(handles=scatter.legend_elements(num=None)[0],labels=labels,title=f"Classes {len(unique_values)}/{num_clusters}",bbox_to_anchor=(1.07, 0.25),loc='lower left')
    ax2.set(xlabel='X',ylabel='Y',zlabel='Z',title=f'Graph with KNN Edges')
    plt.show()
    fig.savefig(f'../plots/results/synthetic_data_knn_dmon_{num_clusters}clus.png', bbox_inches="tight")
    print()
    for value, count in zip(unique_values, counts):
        print(f"Cluster {value}: {count} occurrences")