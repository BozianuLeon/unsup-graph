import torch
import torch_geometric
from torch_geometric.nn import DMoNPooling, GCNConv
from torch_geometric.nn.norm import GraphNorm
import torch.nn.functional as F





# check which version of pytorch geometric - new updates mid-2024
class Net(torch.nn.Module):
    '''
    Spectral modularity pooling operator from https://arxiv.org/abs/2006.16904
    Pooling operator based on learned cluster assignment soft scores. Returns the 
    learned cluster assignment matrix, the pooled node feature matrix, the coarse
    symmetric normalised adjacency matrix and the three(?) loss functions:
    spectral loss, orthogonality loss and cluster loss
    
    Returns:
        log softmax (x), output tensor of pooled node features
        sp1+o1+cl1, loss 
        s, learned cluster assignment
    '''
    def __init__(self, in_channels, out_channels, hidden_channels=128):
        super().__init__()

        self.norm  = GraphNorm(in_channels)
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.relu  = torch.nn.ReLU()
        self.selu  = torch.nn.SELU()
        self.pool1 = DMoNPooling(hidden_channels,out_channels)

    def forward(self, x, edge_index, batch):
        # print(f"1. {x.shape}")
        # print(f"1edge_index {edge_index.shape}")
        x = self.norm(x)
        # print(f"2. {x.shape}")
        x = self.conv1(x, edge_index)
        # print(f"3. {x.shape}")
        x = self.selu(x)
        # print(f"4. {x.shape}")

        x, mask = torch_geometric.utils.to_dense_batch(x, batch)
        # print(f"5. {x.shape}")
        adj = torch_geometric.utils.to_dense_adj(edge_index, batch, max_num_nodes=x.shape[1])
        # print(f"5adj. {adj.shape}")

        s, x, adj, sp1, o1, c1 = self.pool1(x, adj, mask)
        # print(f"6. {x.shape}")

        return F.log_softmax(x, dim=-1), sp1+o1+c1, s



if __name__=="__main__":

    torch.manual_seed(0)

    num_nodes = 1000
    in_channels = 4  # Input feature dimension
    out_channels = 10  # Number of clusters
    x = torch.randn(num_nodes, in_channels)

    # Create a graph using KNN
    data = torch_geometric.data.Data(x=x)
    data.edge_index = torch_geometric.nn.knn_graph(x[:, :3],k=3)
    # Add batch information (single graph, so all nodes belong to batch 0)
    data.batch = torch.zeros(num_nodes, dtype=torch.long)

    # Initialize the model
    model = Net(in_channels=in_channels, out_channels=out_channels)
    model.eval()  

    # Forward pass
    with torch.no_grad():
        out, loss, assignment = model(data.x, data.edge_index, data.batch)

    print(f"Output shape: {out.shape}")  # Shape: [1, out_channels]
    print(f"Clustering loss: {loss.item()}")
    print(f"Assignment matrix shape: {assignment.shape}")  # Shape: [1, num_nodes, out_channels]
