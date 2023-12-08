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
        num_points = torch.normal(mean=avg_num_nodes,std=torch.tensor(avg_num_nodes/10))
        cluster_centers = torch.tensor([[2.0, 2.0, 2.0],
                                    [3.0, 3.0, 3.0],
                                    [2.0, 3.0, 2.0],
                                    [3.0, 2.0, 3.0]], dtype=torch.float32)

        coordinates = torch.zeros((num_points, 3), dtype=torch.float32)
        point_importance = torch.zeros(num_points, dtype=torch.float32)
        true_clusters = torch.zeros(num_points, dtype=torch.float32)

        for i in range(num_points):
            rnd_cluster_idx = torch.randint(low=0, high=4)
            center = cluster_centers[rnd_cluster_idx]
            std_dev = 0.125
            coordinates[i] = center + torch.randn(size=3) * std_dev
            point_importance[i] = 1 / np.linalg.norm(coordinates[i] - center) 
            true_clusters[i] = rnd_cluster_idx

        feat_mat = torch.column_stack((coordinates, point_importance))

        # Create a PyTorch Geometric Data object
        data = torch_geometric.data.Data(x=feat_mat[:, :3], y=true_clusters)
        edge_index = torch_geometric.nn.knn_graph(feat_mat[:, :3],k=3)

        graph_data = torch_geometric.data.Data(x=feat_mat[:, :3],edge_index=edge_index, y=true_clusters) 
        pyg_data_list.append(graph_data)
            
    return pyg_data_list




#initialise model
class Net(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=32):
        super().__init__()

        self.conv1 = GCNConv(in_channels, 128)
        self.relu = torch.nn.ReLU()
        self.pool1 = DMoNPooling(128,out_channels)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        # return F.log_softmax(x,dim=-1), 0

        x, mask = to_dense_batch(x, batch)
        adj = to_dense_adj(edge_index, batch)
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
        loss_all += data.y.size(0) * float(loss)
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
        loss_all += data.y.size(0) * float(loss)
        # correct += int(pred.max(dim=1)[1].eq(data.y.view(-1)).sum())

    return loss_all / len(loader.dataset)#, correct / len(loader.dataset)




if __name__=='__main__':
    avg_num_nodes = 250
    #generate data and put into pytorch geometric dataloaders
    train_data_list = make_data_list(1000)
    val_data_list = make_data_list(200)
    test_data_list = make_data_list(100)

    train_loader = DataLoader(train_data_list, batch_size=20)
    val_loader = DataLoader(val_data_list, batch_size=20)
    test_loader = DataLoader(test_data_list, batch_size=20)

    #run training
    for epoch in range(1, 40):
        start = time.perf_counter()
        train_loss = train(train_loader)
        train_loss2 = test(train_loader)
        val_loss = test(val_loader)
        test_loss = test(test_loader)
        timing = 0
        print(f"Epoch: {epoch:03d}, Train Loss: {train_loss:.3f}, Train Loss2: {train_loss2:.3f}, Val Loss: {val_loss:.3f}, Test Loss: {test_loss:.3f}, Time: {time.perf_counter() - start:.3f}s")


    #evakuate using first graph
    eval_graph = test_data_list[0].to(device)
    pred,tot_loss,clus_ass = model(eval_graph.x,eval_graph.edge_index,eval_graph.batch)

    predicted_classes = clus_ass.squeeze().argmax(dim=1).numpy()
    unique_values, counts = np.unique(predicted_classes, return_counts=True)

        