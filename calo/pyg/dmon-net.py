import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import time
import pickle
import os

import torch
import torch_geometric
import torch.nn.functional as F 






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
    correct = 0
    loss_all = 0

    for data in loader:
        data = data.to(device)
        pred, tot_loss, _ = model(data.x, data.edge_index, data.batch)
        loss = tot_loss # + F.nll_loss(pred, data.y.view(-1))
        loss_all += data.x.size(0) * float(loss)

    return loss_all / len(loader.dataset)





if __name__=='__main__':

    num_clusters = 3
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net(3, num_clusters).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    #load data from pipeline lists
    with open('../data/lists/truth_box_graphs.pkl', 'rb') as f:
       data_list = pickle.load(f)
    print(len(data_list))
    print(data_list[0])

    # eval_graph = data_list[0]
    # pred,tot_loss,clus_ass = model(eval_graph.x,eval_graph.edge_index,eval_graph.batch)
    # predicted_classes = clus_ass.squeeze().argmax(dim=1).numpy()
    # unique_values, counts = np.unique(predicted_classes, return_counts=True)
    # for value, count in zip(unique_values, counts):
    #     print(f"Cluster {value}: {count} occurrences")



    data_list = data_list[:1000]
    train_size = int(0.67 * len(data_list))
    train_loader = torch_geometric.loader.DataLoader(data_list[:train_size], batch_size=20)
    test_loader = torch_geometric.loader.DataLoader(data_list[train_size:], batch_size=20)
    print(train_size,len(data_list),len(train_loader.dataset))
    #run training
    num_epochs = 5
    for epoch in range(1, num_epochs):
        start = time.perf_counter()
        train_loss = train(train_loader)
        train_loss2 = test(train_loader)
        test_loss = test(test_loader)
        timing = 0
        print(f"Epoch: {epoch:03d}, Train Loss: {train_loss:.3f}, Train Loss2: {train_loss2:.3f}, Test Loss: {test_loss:.3f}, Time: {time.perf_counter() - start:.3f}s")

    model_name = f"dmon_{num_epochs}e_{num_clusters}clus"
    if os.path.exists(f"./models/") is False: os.makedirs(f"./models/") 
    torch.save(model.state_dict(), f"models/{model_name}.pth")


    #evaluate using first graph
    eval_graph = data_list[train_size+1]
    pred,tot_loss,clus_ass = model(eval_graph.x,eval_graph.edge_index,eval_graph.batch)

    predicted_classes = clus_ass.squeeze().argmax(dim=1).numpy()
    unique_values, counts = np.unique(predicted_classes, return_counts=True)

    if os.path.exists(f"../plots/results/") is False: os.makedirs(f"../plots/results/")
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(eval_graph.x[:, 0], eval_graph.x[:, 1], eval_graph.x[:, 2], c=predicted_classes, marker='o')
    labels = [f"{value}: ({count})" for value,count in zip(unique_values, counts)]
    ax.legend(handles=scatter.legend_elements(num=None)[0],labels=labels,title=f"Classes {len(unique_values)}/{num_clusters}",bbox_to_anchor=(1.07, 0.25),loc='lower left')
    ax.set(xlabel='X',ylabel='Y',zlabel='Z',title=f'Graph with KNN Edges')
    fig.savefig(f'../plots/results/synthetic_data_dmon_{num_clusters}clus.png', bbox_inches="tight")
    print()
    for value, count in zip(unique_values, counts):
        print(f"Cluster {value}: {count} occurrences")

