import torch
import torch_geometric
from torch_geometric.loader import DataLoader

import os
import time
import argparse
import numpy as np
from matplotlib import pyplot as plt

import models
import data



parser = argparse.ArgumentParser()
parser.add_argument('-e','--epochs', type=int, required=True, help='Number of training epochs',)
parser.add_argument('-bs','--batch_size', nargs='?', const=8, default=8, type=int, help='Batch size to be used')
parser.add_argument('-nw','--num_workers', nargs='?', const=2, default=2, type=int, help='Number of worker CPUs')
parser.add_argument('-nc','--num_clusters', nargs='?', const=5, default=5, type=int, help='Number of (max) clusters DMoN can predict')
args = parser.parse_args()





from torch_geometric.nn import DMoNPooling, GCNConv
# check which version of pytorch geometric - new updates mid-2024
class Net(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=128):
        super().__init__()

        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.relu = torch.nn.ReLU()
        self.pool1 = DMoNPooling(hidden_channels,out_channels)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = self.relu(x)

        x, mask = torch_geometric.utils.to_dense_batch(x, batch)
        adj = torch_geometric.utils.to_dense_adj(edge_index, batch)
        s, x, adj, sp1, o1, c1 = self.pool1(x, adj, mask)

        return torch.nn.functional.log_softmax(x, dim=-1), sp1+o1+c1, s




# simple train/test function see: https://github.com/pyg-team/pytorch_geometric/blob/master/examples/proteins_dmon_pool.py 
def train(train_loader, device):
    model.train()
    tot_loss = 0

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out, loss, _ = model(data.x, data.edge_index, data.batch)
        # loss += + F.nll_loss(out, data.y.view(-1)) # only relevant if we have labels
        loss.backward()
        tot_loss += float(loss) * data.x.size(0)
        optimizer.step()

    return tot_loss / len(train_loader.dataset) 



@torch.no_grad()
def test(loader, device):
    model.eval()
    tot_loss = 0

    for data in loader:
        data = data.to(device)
        pred, loss, _ = model(data.x, data.edge_index, data.batch)
        tot_loss += float(loss) * data.x.size(0) 

    return tot_loss / len(loader.dataset) 







if __name__=='__main__':

    config = {
        "seed"       : 0,
        "device"     : torch.device("cuda" if torch.cuda.is_available() else "cpu"), # torch.device("mps" if torch.backends.mps.is_available() else "cpu"),
        "n_train"    : 1000,
        "val_frac"   : 0.25,
        "test_frac"  : 0.15,
        "n_nodes"    : 250,
        "k"          : 3,
        "NW"         : args.num_workers,
        "BS"         : args.batch_size,
        "LR"         : 0.01,
        "WD"         : 0.01,
        "n_clus"     : int(args.num_clusters),
        "n_epochs"   : int(args.epochs),
    }

    # generate data and place in geometric dataloaders
    n_train = int(config["n_train"])
    n_val   = int(config["n_train"]*config["val_frac"])
    n_test  = int(config["n_train"]*config["test_frac"])
    print('\ttrain / val / test size : ',n_train,'/',n_val,'/',n_test,'\n')
    train_data = data.synthetic_blobs_list(num_graphs=n_train,avg_num_nodes=config["n_nodes"],k=config["k"])
    valid_data = data.synthetic_blobs_list(num_graphs=n_val,avg_num_nodes=config["n_nodes"],k=config["k"])
    test_data  = data.synthetic_blobs_list(num_graphs=n_test,avg_num_nodes=config["n_nodes"],k=config["k"])

    train_loader = DataLoader(train_data, batch_size=config["BS"], num_workers=config["NW"])
    val_loader   = DataLoader(valid_data, batch_size=config["BS"], num_workers=config["NW"])
    test_loader  = DataLoader(test_data, batch_size=config["BS"], num_workers=config["NW"])

    # instantiate model, optimizer
    model = Net(3, config["n_clus"]).to(config["device"])
    total_params = sum(p.numel() for p in model.parameters())
    print(f'DMoN (single conv layer) \t{total_params:,} total parameters.\n')
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["LR"], weight_decay=config["WD"], amsgrad=True)  

    # run training (simple vanilla torch)
    print(f"Starting training... on {config['device']}")
    for epoch in range(config["n_epochs"]):
        start = time.perf_counter()
        train_loss = train(train_loader, config["device"])
        val_loss   = test(val_loader, config["device"])
        print(f"Epoch: {epoch:03d}, Train Loss: {train_loss:.3f}, Val Loss: {val_loss:.3f}, Time: {time.perf_counter() - start:.3f}s")



    print(f"Finished training. Evaluating using first event of test set.")
    eval_graph = test_data[0] 
    # inference from a single forward pass
    model.eval()
    with torch.inference_mode():
        pred,tot_loss,clus_ass = model(eval_graph.x,eval_graph.edge_index,eval_graph.batch)
        for i in range(len(eval_graph.x)):
            print("coords:",eval_graph.x[i].detach().numpy(),"score",clus_ass[0][i].detach().numpy())

    # force each node to its most likely cluster, no soft assignment
    predicted_classes = clus_ass.squeeze().argmax(dim=1).numpy()
    unique_values, counts = np.unique(predicted_classes, return_counts=True)

    if os.path.exists(f"plots/results/") is False: os.makedirs(f"plots/results/")
    print("Plotting evaluation graph")

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(121, projection='3d')
    ax.scatter(eval_graph.x[:, 0], eval_graph.x[:, 1], eval_graph.x[:, 2], c='b', marker='o', label='Nodes')
    for src, dst in eval_graph.edge_index.t().tolist():
        x_src, y_src, z_src = eval_graph.x[src]
        x_dst, y_dst, z_dst = eval_graph.x[dst]
        ax.plot([x_src, x_dst], [y_src, y_dst], [z_src, z_dst], c='r')
    ax.set(xlabel='X',ylabel='Y',zlabel='Z',title=f'Input Graph with KNN 3 Edges')

    ax2 = fig.add_subplot(122, projection='3d')
    scatter = ax2.scatter(eval_graph.x[:, 0], eval_graph.x[:, 1], eval_graph.x[:, 2], c=predicted_classes, marker='o')
    labels = [f"{value}: ({count})" for value,count in zip(unique_values, counts)]
    ax2.legend(handles=scatter.legend_elements(num=None)[0],labels=labels,title=f"Classes {len(unique_values)}/{config['n_clus']}",bbox_to_anchor=(1.07, 0.25),loc='lower left')
    ax2.set(xlabel='X',ylabel='Y',zlabel='Z',title=f'DMoN Output Graph')
    plt.show()
    fig.savefig(f"plots/results/synthetic_data_knn_dmon_{config['n_clus']}clus.png", bbox_inches="tight")
    print()
    for value, count in zip(unique_values, counts):
        print(f"Cluster {value}: {count} occurrences")