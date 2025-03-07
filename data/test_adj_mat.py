import torch
import torch_geometric



feature_matrix = torch.tensor([[0.0, 0.0, 0.0],
                               [0.5, 0.5, 0.5],
                               [1.0, 1.0, 1.0],
                               [9.9, 9.9, 9.9],
                               [0.2, 0.2, 0.2]])
print(feature_matrix.shape)

edge_indices = torch_geometric.nn.radius_graph(feature_matrix, r=1.75)
print(edge_indices)
adj = torch_geometric.utils.to_dense_adj(edge_indices,max_num_nodes=feature_matrix.shape[0])
print(adj)
event_graph  = torch_geometric.data.Data(x=feature_matrix,edge_index=edge_indices) 
