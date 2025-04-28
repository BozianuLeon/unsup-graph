import torch
import torch_geometric
from torch_geometric.nn import DMoNPooling, GCNConv
from torch_geometric.nn.norm import GraphNorm
import torch.nn.functional as F


from torch_geometric.nn.dense.mincut_pool import _rank3_trace
EPS = 1e-15


class DMoNPooling(torch.nn.Module):
    def __init__(self, channels, k, dropout: float = 0.0):
        super().__init__()
        if isinstance(channels, int):
            channels = [channels]
        from torch_geometric.nn.models.mlp import MLP
        self.mlp = MLP(channels + [k], act=None, norm=None)
        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.mlp.reset_parameters()

    def forward(
        self,
        x,
        adj,
        mask,
    ):
        r"""Forward pass.
        Args:
            x (torch.Tensor): Node feature tensor
                :math:`\mathbf{X} \in \mathbb{R}^{B \times N \times F}`, with
                batch-size :math:`B`, (maximum) number of nodes :math:`N` for
                each graph, and feature dimension :math:`F`.
                Note that the cluster assignment matrix
                :math:`\mathbf{S} \in \mathbb{R}^{B \times N \times C}` is
                being created within this method.
            adj (torch.Tensor): Adjacency tensor
                :math:`\mathbf{A} \in \mathbb{R}^{B \times N \times N}`.
            mask (torch.Tensor, optional): Mask matrix
                :math:`\mathbf{M} \in {\{ 0, 1 \}}^{B \times N}` indicating
                the valid nodes for each graph. (default: :obj:`None`)

        :rtype: (:class:`torch.Tensor`, :class:`torch.Tensor`,
            :class:`torch.Tensor`, :class:`torch.Tensor`,
            :class:`torch.Tensor`, :class:`torch.Tensor`)
        """
        x = x.unsqueeze(0) if x.dim() == 2 else x
        adj = adj.unsqueeze(0) if adj.dim() == 2 else adj

        s = self.mlp(x)
        s = F.dropout(s, self.dropout, training=self.training)
        s = torch.softmax(s, dim=-1) # cluster assignments

        (batch_size, num_nodes, _), C = x.size(), s.size(-1)

        if mask is None:
            mask = torch.ones(batch_size, num_nodes, dtype=torch.bool,
                              device=x.device)

        mask = mask.view(batch_size, num_nodes, 1).to(x.dtype)
        x, s = x * mask, s * mask

        # Perhaps add something here, but need to match indices/nodes with batch size etc
        # if we turn this on from the start, it is likely to stunt the growth of certain 
        # columns/weights in the network. Turn on after some prelim training?
        print("mask\n",mask)
        print(s.shape, mask.shape)
        print("s\n",s)
        print()
        col_sum = torch.sum(s,dim=1,keepdim=True) # [B,1,num_clusters]
        print(col_sum)
        print(col_sum.shape)
        print()
        _, min_cols = torch.min(col_sum,dim=2,keepdim=True) # [B,1,1]
        print(min_cols)
        print(min_cols.shape)
        print()
        _, bottomk_cols = torch.topk(col_sum,k=2,dim=2,largest=False) # [B,1,topk]
        print(bottomk_cols)
        print(bottomk_cols.shape)
        print()
        batch_idx = torch.arange(s.shape[0]).view(s.shape[0],1,1)
        print(batch_idx)
        print(batch_idx.shape)
        print(s)
        s[batch_idx,:,bottomk_cols] = 0
        print(s)

        print()
        print()
        quit()
        print()
        out = F.selu(torch.matmul(s.transpose(1, 2), x)) # features pooled
        out_adj = torch.matmul(torch.matmul(s.transpose(1, 2), adj), s)

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        # Spectral loss:
        degrees = torch.einsum('ijk->ij', adj)  # B X N
        degrees = degrees.unsqueeze(-1) * mask  # B x N x 1
        degrees_t = degrees.transpose(1, 2)  # B x 1 x N

        m = torch.einsum('ijk->i', degrees) / 2  # B
        m_expand = m.view(-1, 1, 1).expand(-1, C, C)  # B x C x C

        ca = torch.matmul(s.transpose(1, 2), degrees)  # B x C x 1
        cb = torch.matmul(degrees_t, s)  # B x 1 x C

        normalizer = torch.matmul(ca, cb) / 2 / m_expand
        decompose = out_adj - normalizer
        spectral_loss = -_rank3_trace(decompose) / 2 / m
        spectral_loss = spectral_loss.mean()

        # Orthogonality regularization:
        ss = torch.matmul(s.transpose(1, 2), s)
        i_s = torch.eye(C).type_as(ss)
        ortho_loss = torch.norm(
            ss / torch.norm(ss, dim=(-1, -2), keepdim=True) -
            i_s / torch.norm(i_s), dim=(-1, -2))
        ortho_loss = ortho_loss.mean()

        # Cluster loss: (collapse regularization)
        cluster_size = torch.einsum('ijk->ik', s)  # B x C
        cluster_loss = torch.norm(input=cluster_size, dim=1)
        cluster_loss = cluster_loss / mask.sum(dim=1) * torch.norm(i_s) - 1
        cluster_loss = cluster_loss.mean()

        # Fix and normalize coarsened adjacency matrix:
        ind = torch.arange(C, device=out_adj.device)
        out_adj[:, ind, ind] = 0
        d = torch.einsum('ijk->ij', out_adj)
        d = torch.sqrt(d)[:, None] + EPS
        out_adj = (out_adj / d) / d.transpose(1, 2)

        return s, out, out_adj, spectral_loss, ortho_loss, cluster_loss







# check which version of pytorch geometric - new updates mid-2024
class Net(torch.nn.Module):
    '''
    Spectral modularity pooling operator from https://arxiv.org/abs/2006.16904
    Pooling operator based on learned cluster assignment soft scores. Returns the 
    learned cluster assignment matrix, the pooled node feature matrix, the coarse
    symmetric normalised adjacency matrix and the three(?) loss functions:
    spectral loss, orthogonality loss and cluster loss
    
    Returns:
        log softmax(x), output tensor of pooled node features
        sp1+o1+cl1, total loss term 
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
        print(f"1.x {x.shape}")
        print(f"1edge_index {edge_index.shape}")
        x = self.norm(x)
        print(f"2.x {x.shape}")
        x = self.conv1(x, edge_index)
        print(f"3.x {x.shape}")
        x = self.selu(x)
        print(f"4.x {x.shape}")

        x, mask = torch_geometric.utils.to_dense_batch(x, batch)
        print(f"5.x {x.shape}")
        adj = torch_geometric.utils.to_dense_adj(edge_index, batch, max_num_nodes=x.shape[1])
        print(f"5adj. {adj.shape}")

        s, x, adj, sp1, o1, c1 = self.pool1(x, adj, mask)
        print(f"6.x {x.shape}")
        print(f"7.s {s.shape}")
        print(s)
        print(f"8.loss {sp1:.7f}, {o1:.7f}, {c1:.7f}")

        return F.log_softmax(x, dim=-1), sp1+o1+c1, s



if __name__=="__main__":

    torch.manual_seed(0)

    # num_nodes = 10
    # in_channels = 4  # Input feature dimension
    # out_channels = 5  # Number of clusters
    # x = torch.randn(num_nodes, in_channels)

    # feature matrix (xyz+E coords)
    x = torch.tensor([[0.2, 0.1, 0.5, 10.0],
                      [0.0, 0.3, 0.2, 23.0],
                      [0.4, 0.0, 0.2, 12.0],
                      [0.1, 0.1, 0.3, 16.0],
                      [0.3, 0.5, 0.0, 20.0],
                      [0.2, 0.2, 0.1, 19.0],
                      [0.4, 0.1, 0.4, 18.0],
                      [0.0, 0.2, 0.3, 14.0],
                      [0.1, 0.5, 0.5, 16.0],
                      [0.5, 0.3, 0.2, 11.0],
                      ])
    num_nodes = x.size(0)
    in_channels = 4  # Input feature dimension
    out_channels = 4  # Number of clusters

    # Create a graph using KNN
    data = torch_geometric.data.Data(x=x)
    data.edge_index = torch_geometric.nn.knn_graph(x[:, :3],k=3)
    # Add batch information (single graph, so all nodes belong to batch 0)
    data.batch = torch.zeros(num_nodes, dtype=torch.long)
    data.batch = torch.tensor([0,0,0,0,0,1,1,1,1,1])

    # Initialize the model
    model = Net(in_channels=in_channels, out_channels=out_channels)
    model.eval()  

    # Forward pass
    with torch.no_grad():
        out, loss, assignment = model(data.x, data.edge_index, data.batch)

    print()
    print(f"Output shape: {out.shape}")  # Shape: [1, out_channels]
    print(f"Clustering loss: {loss.item()}")
    print(f"Assignment matrix shape: {assignment.shape}")  # Shape: [1, num_nodes, out_channels]

# In two events
0.0003385, 0.9183854, 0.0026324
0.9213563203811646
# All in same event/batch:
0.0002484, 0.9183720, 0.0026218
0.9212422370910645