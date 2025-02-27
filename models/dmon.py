import torch
import torch_geometric
from torch_geometric.nn import DMoNPooling, GCNConv, norm
import torch.nn.functional as F

# check which version of pytorch geometric - new updates mid-2024
class Net(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=128):
        super().__init__()

        self.norm = norm.GraphNorm(in_channels)
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.relu = torch.nn.ReLU()
        self.pool1 = DMoNPooling(hidden_channels,out_channels)

    def forward(self, x, edge_index, batch):
        x = self.norm(x)
        x = self.conv1(x, edge_index)
        x = self.relu(x)

        x, mask = torch_geometric.utils.to_dense_batch(x, batch)
        adj = torch_geometric.utils.to_dense_adj(edge_index, batch)

        s, x, adj, sp1, o1, c1 = self.pool1(x, adj, mask)

        return F.log_softmax(x, dim=-1), sp1+o1+c1, s






if __name__=="__main__":

    torch.manual_seed(0)

    num_nodes = 1000
    in_channels = 16  # Input feature dimension
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
