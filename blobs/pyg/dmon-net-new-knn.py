import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import scipy
import time
import os

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
    tree = scipy.spatial.cKDTree(x.detach()) #this needs to be restricted to coordinates/columns we actually want!

    quantile_values = torch.quantile(x_ranking, torch.tensor(quantiles))
    edges_list = torch.tensor([[],[]])
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




# x = torch.Tensor([[-100, -100], [-100, 100], [100, -100], [100, 100]])
# batch_x = torch.tensor([0, 0, 0, 0])
# assign_index = knn_graph(x, 2, batch_x)
# print(assign_index)
# batch_x = torch.tensor([0, 0, 0, 1])
# assign_index = knn_graph(x, 2, batch_x)
# print(assign_index)
# print()
# point_importance = torch.tensor([[1.5],[1.0],[0.8],[2.0]])
# batch_x = torch.tensor([0, 0, 0, 0])
# assign_index = var_knn_graph(x,[2,2,2,2],[0.25,0.5,0.75],point_importance,batch=batch_x)
# print(assign_index)
# batch_x = torch.tensor([0, 0, 0, 1])
# assign_index = var_knn_graph(x,[2,2,2,2],[0.25,0.5,0.75],point_importance,batch=batch_x)
# print(assign_index)

# quit()

cluster_centers = torch.tensor([[2.0, 2.0, 2.0],
                                [3.0, 3.0, 3.0],
                                [2.0, 3.0, 2.0],
                                [3.0, 2.0, 3.0]], dtype=torch.float32)
num_points = 16
coordinates = torch.zeros((num_points, 3), dtype=torch.float32)
point_importance = torch.zeros(num_points, dtype=torch.float32)
true_clusters = torch.zeros(num_points, dtype=torch.float32)

for i in range(num_points):
    rnd_cluster_idx = torch.randint(low=0, high=4,size=(1,))
    center = cluster_centers[rnd_cluster_idx]
    std_dev = 0.35
    coordinates[i] = center + torch.randn(size=(3,)) * std_dev
    point_importance[i] = 1 / np.linalg.norm(coordinates[i] - center) 
    true_clusters[i] = rnd_cluster_idx

batch = torch.ones((coordinates.size(0)))
edge_index = knn_graph(coordinates,k=3,batch=batch)
print(edge_index)
print(edge_index.shape)
print()
print()
print()
print()

# edge_index = var_knn_graph(coordinates,[2,4,6,8],[0.25,0.5,0.75],coordinates[:,0],batch)
edge_index = var_knn_graph(coordinates,[3,3,3,3],[0.25,0.5,0.75],coordinates[:,0],batch)
sorted_values, indices = torch.sort(edge_index[0])
sorted_corresponding_tensor = edge_index[1][indices]
print(torch.stack((sorted_values,sorted_corresponding_tensor),dim=0))
# print(edge_index)
print(edge_index.shape)


quit()


def make_data_list(num_graphs,avg_num_nodes=250,k=3):
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
            std_dev = 0.35
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


#initialise model
class Net(torch.nn.Module):
    def __init__(self, 
                 in_channels,
                 out_channels,
                 hidden_channels=32):
        super().__init__()

        self.conv1 = torch_geometric.nn.GCNConv(in_channels, 128)
        self.relu = torch.nn.ReLU()
        self.pool1 = torch_geometric.nn.DMoNPooling(128,out_channels)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        # return F.log_softmax(x,dim=-1), 0

        x, mask = torch_geometric.utils.to_dense_batch(x, batch)
        adj = torch_geometric.utils.to_dense_adj(edge_index, batch)
        s, x, adj, sp1, o1, c1 = self.pool1(x, adj, mask)

        return F.log_softmax(x, dim=-1), sp1+o1+c1, s

num_clusters = 6
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net(3, num_clusters).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

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
    #generate data and put into pytorch geometric dataloaders
    train_data_list = make_data_list(1000)
    val_data_list = make_data_list(200)
    test_data_list = make_data_list(100)

    train_loader = torch_geometric.loader.DataLoader(train_data_list, batch_size=20)
    val_loader = torch_geometric.loader.DataLoader(val_data_list, batch_size=20)
    test_loader = torch_geometric.loader.DataLoader(test_data_list, batch_size=20)

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

    model_name = f"dmon_{num_epochs}e_{num_clusters}clus"
    torch.save(model.state_dict(), f"models/{model_name}.pth")


    #evaluate using first graph
    eval_graph = test_data_list[0]
    pred,tot_loss,clus_ass = model(eval_graph.x,eval_graph.edge_index,eval_graph.batch)

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

    ax.set(xlabel='X',ylabel='Y',zlabel='Z',title=f'Graph with var KNN Edges')

    ax2 = fig.add_subplot(122, projection='3d')
    scatter = ax2.scatter(eval_graph.x[:, 0], eval_graph.x[:, 1], eval_graph.x[:, 2], c=predicted_classes, marker='o')
    labels = [f"{value}: ({count})" for value,count in zip(unique_values, counts)]
    ax2.legend(handles=scatter.legend_elements(num=None)[0],labels=labels,title=f"Classes {len(unique_values)}/{num_clusters}",bbox_to_anchor=(1.07, 0.25),loc='lower left')
    ax2.set(xlabel='X',ylabel='Y',zlabel='Z',title=f'Model Output')
    # fig.savefig(f'../plots/results/synthetic_data_var_knn_dmon_{num_clusters}clus.png', bbox_inches="tight")
    plt.show()
    print()
    for value, count in zip(unique_values, counts):
        print(f"Cluster {value}: {count} occurrences")