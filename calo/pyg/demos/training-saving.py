import numpy as np
import scipy
import time
import pickle

import torch
from torch.nn import Linear
import torch.nn.functional as F
import torch_geometric
from torch_geometric.loader import DataLoader
from torch_geometric.nn import DMoNPooling, GCNConv



#initialise model
class DynamicNet(torch.nn.Module):
    def __init__(self, 
                 in_channels,
                 out_channels,
                 ):
        super().__init__()

        self.conv1 = torch_geometric.nn.GCNConv(in_channels, 128)
        self.conv2 = torch_geometric.nn.GCNConv(128, 128)
        self.relu = torch.nn.ReLU()
        self.pool1 = torch_geometric.nn.DMoNPooling(128,out_channels)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = self.relu(x)

        #recompute adjacency
        edge_index2 = torch_geometric.nn.knn_graph(
            x=x[:,[0,1,2]],
            k=4,
            batch=batch,
        )
        x = self.conv2(x,edge_index2)
        x = self.relu(x)

        x, mask = torch_geometric.utils.to_dense_batch(x, batch)
        adj = torch_geometric.utils.to_dense_adj(edge_index, batch)
        s, x, adj, sp1, o1, c1 = self.pool1(x, adj, mask)

        return F.log_softmax(x, dim=-1), sp1+o1+c1, s

#initialise model
class Net(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=32):
        super().__init__()

        self.conv1 = torch_geometric.nn.GCNConv(in_channels, 128)
        self.relu = torch.nn.ReLU()
        self.pool1 = torch_geometric.nn.DMoNPooling(128,out_channels)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = self.relu(x)

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
    return loss_all / len(train_loader.dataset)


@torch.no_grad()
def test(loader):
    model.eval()
    loss_all = 0

    for data in loader:
        data = data.to(device)
        pred, tot_loss, _ = model(data.x, data.edge_index, data.batch)
        loss = tot_loss # + F.nll_loss(pred, data.y.view(-1))
        loss_all += data.x.size(0) * float(loss)

    return loss_all / len(loader.dataset)




if __name__=='__main__':
    #load data from pipeline lists
    with open('../../data/lists/truth_box_graphs_2sig_knn123onetenperc_xyz.pkl', 'rb') as f:
    # with open('../../data/lists/truth_box_graphs_2sig_knn12325_norm_xyz.pkl', 'rb') as f:
       data_list = pickle.load(f)
    print(len(data_list))
    print(data_list[0])
    print(data_list[0].x)

    # data_list = data_list[:1000]
    train_size = int(0.9 * len(data_list))
    train_loader = torch_geometric.loader.DataLoader(data_list[:train_size], batch_size=8)
    test_loader = torch_geometric.loader.DataLoader(data_list[train_size:], batch_size=8)
    print(train_size,len(data_list),len(train_loader.dataset))

    num_clusters = 20
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net(3, num_clusters).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    #run training
    num_epochs = 25
    for epoch in range(1, num_epochs):
        start = time.perf_counter()
        train_loss = train(train_loader)
        train_loss2 = test(train_loader)
        test_loss = test(test_loader)
        print(f"Epoch: {epoch:03d}, Train Loss: {train_loss:.3f}, Train Loss2: {train_loss2:.3f}, Test Loss: {test_loss:.3f}, Time: {time.perf_counter() - start:.3f}s")

    model_name = f"calo_dmon_{num_clusters}clus_{num_epochs}e_knn123onetenperc"
    torch.save(model.state_dict(), f"{model_name}.pth")
