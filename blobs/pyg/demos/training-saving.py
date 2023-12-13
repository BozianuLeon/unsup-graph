import numpy as np
import scipy
import time

import torch
from torch.nn import Linear
import torch.nn.functional as F
import torch_geometric
from torch_geometric.loader import DataLoader
from torch_geometric.nn import DMoNPooling, GCNConv



def knn_graph(
        x,
        k,
        batch=None,
):
    # Finds for each element in x the k nearest points in x-space
    
    if batch is None:
        batch = x.new_zeros(x.size(0), dtype=torch.long)

    x = x.view(-1, 1) if x.dim() == 1 else x
    assert x.dim() == 2 and batch.dim() == 1
    assert x.size(0) == batch.size(0)

    # Rescale x and y.
    def normalize(x):
        x_normed = x - x.min(0,keepdim=True)[0]
        x_normed = x_normed / x_normed.max(0, keepdim=True)[0]
        return x_normed 
    x = normalize(x)

    # Concat batch/features to ensure no cross-links between examples exist.
    x = torch.cat([x, 2 * x.size(1) * batch.view(-1, 1).to(x.dtype)], dim=-1)

    tree = scipy.spatial.cKDTree(x.detach())
    dist, col = tree.query(x.detach().cpu(), k=k, distance_upper_bound=x.size(1))
    dist = torch.from_numpy(dist).to(x.dtype)
    col = torch.from_numpy(col).to(torch.long)
    row = torch.arange(col.size(0), dtype=torch.long).view(-1, 1).repeat(1, k)
    mask = torch.logical_not(torch.isinf(dist).view(-1))
    row, col = row.view(-1)[mask], col.view(-1)[mask]

    return torch.stack([row, col], dim=0)

def var_knn_graph(
        x,
        k,
        quantiles,
        x_ranking,
        batch=None,
):
    # Finds for each element in x the k nearest points in x-space
    # k changes depending on the importance of the node as defined in 
    # x_ranking tensor
    
    if batch is None:
        batch = x.new_zeros(x.size(0), dtype=torch.long)


    x = x.view(-1, 1) if x.dim() == 1 else x
    assert x.dim() == 2 and batch.dim() == 1
    assert x.size(0) == batch.size(0)
    assert len(k) == len(quantiles)+1, "There should be a k value for each quantile interval"

    # Rescale x and y.
    def normalize(x):
        x_normed = x - x.min(0,keepdim=True)[0]
        x_normed = x_normed / x_normed.max(0, keepdim=True)[0]
        return x_normed 
    x = normalize(x)

    # Concat batch/features to ensure no cross-links between examples exist.
    x = torch.cat([x, 2 * x.size(1) * batch.view(-1, 1).to(x.dtype)], dim=-1)

    # Pre-calculate KNN tree
    tree = scipy.spatial.cKDTree(x.detach()) 

    quantile_values = torch.quantile(x_ranking, torch.tensor(quantiles))
    edges_list = torch.tensor([[],[]],dtype=torch.int64)
    for i in range(len(quantile_values)+1):
        # Create masks for each quantile
        if i==0:
            qua_mask = x_ranking<quantile_values[i]
        elif i==len(quantile_values):
            qua_mask = x_ranking>quantile_values[i-1]
        else:
            qua_mask = (quantile_values[i-1]<x_ranking) & (x_ranking <= quantile_values[i])
        
        # Extract indices for each quantile
        indices = torch.nonzero(qua_mask, as_tuple=False).squeeze()
        qua_mask = qua_mask.squeeze()
        nodes_q = x[qua_mask]

        dist, col = tree.query(nodes_q.detach(),k=k[i],distance_upper_bound=x.size(1)) 
        dist = torch.from_numpy(dist).to(x.dtype)
        col = torch.from_numpy(col).to(torch.long)
        row = torch.arange(col.size(0), dtype=torch.long).view(-1, 1).repeat(1, k[i])
        mask = torch.logical_not(torch.isinf(dist).view(-1))
        row, col = row.view(-1)[mask], col.view(-1)[mask]
        # Return to original indices
        row = torch.gather(indices,0,row)
        # pairs of source, dest node indices
        edges_q = torch.stack([row, col], dim=0)
        edges_list = torch.cat([edges_list,edges_q],dim=1)

    return edges_list





def make_data_list(num_graphs,sigma,avg_num_nodes=250,ks=[3,3,3,3]):
    # Generate the synthetic data
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
            std_dev = sigma
            coordinates[i] = center + torch.randn(size=(3,)) * std_dev
            point_importance[i] = 1 / np.linalg.norm(coordinates[i] - center) 
            true_clusters[i] = rnd_cluster_idx

        feat_mat = torch.column_stack((coordinates,point_importance))
        # edge_index = var_knn_graph(feat_mat[:,:3],k=ks,quantiles=[0.25,0.5,0.8],x_ranking=point_importance)
        edge_index = torch_geometric.nn.knn_graph(feat_mat[:, :3],k=3)

        graph_data = torch_geometric.data.Data(x=feat_mat,edge_index=edge_index, y=true_clusters) 
        pyg_data_list.append(graph_data)
            
    return pyg_data_list


#initialise model
class Net(torch.nn.Module):
    def __init__(self, 
                 in_channels,
                 out_channels,
                 ):
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
        loss_all += data.y.size(0) * float(loss)
        optimizer.step()
    return loss_all / len(train_data_list)


@torch.no_grad()
def test(loader):
    model.eval()
    loss_all = 0

    for data in loader:
        data = data.to(device)
        pred, tot_loss, _ = model(data.x, data.edge_index, data.batch)
        loss = tot_loss # + F.nll_loss(pred, data.y.view(-1))
        loss_all += data.y.size(0) * float(loss)

    return loss_all / len(loader.dataset)




if __name__=='__main__':
    avg_num_nodes = 250
    sigma = 0.12
    ks = [3,3,3,3]
   
    #generate data and put into pytorch geometric dataloaders
    train_data_list = make_data_list(1000,sigma=sigma,ks=ks)
    val_data_list = make_data_list(200,sigma=sigma,ks=ks)
    test_data_list = make_data_list(100,sigma=sigma,ks=ks)

    train_loader = torch_geometric.loader.DataLoader(train_data_list, batch_size=20)
    val_loader = torch_geometric.loader.DataLoader(val_data_list, batch_size=20)
    test_loader = torch_geometric.loader.DataLoader(test_data_list, batch_size=20)

    num_clusters = 4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net(train_data_list[0].x.size(1), num_clusters).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


    #run training
    num_epochs = 40
    for epoch in range(1, num_epochs):
        start = time.perf_counter()
        train_loss = train(train_loader)
        train_loss2 = test(train_loader)
        val_loss = test(val_loader)
        test_loss = test(test_loader)
        timing = 0
        print(f"Epoch: {epoch:03d}, Train Loss: {train_loss:.3f}, Train Loss2: {train_loss2:.3f}, Val Loss: {val_loss:.3f}, Test Loss: {test_loss:.3f}, Time: {time.perf_counter() - start:.3f}s")

    k_name = "".join(str(element) for element in ks)
    model_name = f"dmon_sig{int(100*sigma)}_xyzE_k{k_name}_{num_clusters}clus_{num_epochs}e"
    torch.save(model.state_dict(), f"{model_name}.pth")
