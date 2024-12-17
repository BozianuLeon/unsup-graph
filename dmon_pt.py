import torch
from torch import nn


class DMoN(nn.Module):
    def __init__(self, input_shape, n_clusters, collapse_regularization=0.1,dropout_rate=0):
        super().__init__()
        self.input_shape = input_shape
        self.n_clusters = n_clusters
        self.collapse_regularization = collapse_regularization
        self.dropout_rate = dropout_rate

        self.linear = nn.Linear(self.input_shape, self.n_clusters)
        nn.init.orthogonal_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        
        self.transform = nn.Sequential(
            self.linear,
            nn.Dropout(p=self.dropout_rate)
            )

    def forward(self, inputs):
        """Performs DMoN clustering according to input features and input graph.

        Args:
        inputs: A tuple of Tensorflow tensors. First element is (n*d) node feature
            matrix and the second one is (n*n) sparse graph adjacency matrix.

        Returns:
        A tuple (features, clusters) with (k*d) cluster representations and
        (n*k) cluster assignment matrix, where k is the number of cluster,
        d is the dimensionality of the input, and n is the number of nodes in the
        input graph. If do_unpooling is True, returns (n*d) node representations
        instead of cluster representations.
        """
        features, adjacency = inputs








def distance_matrix(tensor_xyz):

    # Step 1: Compute the squared norms of each point 
    squared_norms = (tensor_xyz ** 2).sum(dim=1)
    # Step 2: Compute the dot product matrix (shape [15, 15])
    dot_product = torch.mm(tensor_xyz, tensor_xyz.t())
    # Step 3: Compute the squared distance matrix
    # Use the formula: squared_distances = ||p_i||^2 + ||p_j||^2 - 2 * p_i * p_j^T
    squared_distances = squared_norms.unsqueeze(0) + squared_norms.unsqueeze(1) - 2 * dot_product
    # Step 4: Take the square root to get the Euclidean distances
    distance_matrix = torch.sqrt(squared_distances)

    return distance_matrix



def distance_matrix2(tensor_xyz):
    # Step 1: Subtract the tensor from its transpose
    diff = tensor_xyz.unsqueeze(0) - tensor_xyz.unsqueeze(1)

    # Step 2: Square the differences (element-wise)
    squared_diff = diff ** 2
    
    # Step 3: Sum the squared differences along the last dimension (i.e., x, y, z)
    squared_distances = squared_diff.sum(dim=-1)
    return torch.sqrt(squared_distances)



def knn_adj_matrix(distance_mat,k=3):
    A = torch.zeros((distance_mat.shape[0],distance_mat.shape[0]))
    val,idx = torch.topk(-d_mat,k=k,dim=1)
    A.scatter_(1, idx, 1.0)
    return A




if __name__=="__main__":
    number_of_nodes = 5
    number_node_features_d = 3
    n_clusters = 2
    dropout_rate = 0.2
    #####################
    torch.manual_seed(0)

    linear = nn.Linear(number_node_features_d, n_clusters)
    nn.init.orthogonal_(linear.weight)
    nn.init.zeros_(linear.bias)
    
    transform = nn.Sequential(
        linear,
        nn.Dropout(p=dropout_rate)
        )
    selu = torch.nn.SELU() 
    
    input_tensor = torch.randn(number_of_nodes, number_node_features_d)
    dist_mat = distance_matrix(input_tensor)
    d_mat = distance_matrix2(input_tensor)
    input_adj_matrix = knn_adj_matrix(d_mat,k=2)
    
    # import matplotlib
    # import matplotlib.pyplot as plt
    # fig = plt.figure(figsize=(12, 8),dpi=100)
    # ax1 = fig.add_subplot(111, projection='3d')
    # ax1.scatter(input_tensor[:,0],input_tensor[:,1],input_tensor[:,2], color='pink',marker='o',s=150)
    # plt.show()


    output_tensor = transform(input_tensor)
    print("Input Tensor:",input_tensor.shape)
    print(input_tensor)
    print("Transform Tensor:",output_tensor.shape)
    print(output_tensor)


    sm = torch.nn.Softmax(dim=1)
    sm_tensor = sm(output_tensor)
    cluster_sizes = torch.sum(sm_tensor,dim=0)
    predicted_classes = sm_tensor.squeeze().argmax(dim=1)
    print("Softmaxed Tensor")
    print(sm_tensor)
    print("Predicted classes")
    print(predicted_classes)
    print("Cluster sizes")
    print(cluster_sizes)

    assignments = sm_tensor
    assignments_pooling = assignments / cluster_sizes
    print("Assignments pooling")
    print(assignments_pooling)

    degrees = torch.sum(input_adj_matrix,dim=0)
    print("Degrees")
    print(degrees)
    degrees = torch.reshape(degrees,(-1,1))

    n_nodes = input_adj_matrix.shape[0]
    n_edges = torch.sum(degrees)
    print("Number of nodes",n_nodes)
    print("Number of edeges",n_edges)

    # Computes the size [k, k] pooled graph as S^T*A*S in two multiplications.
    graph_pooled = torch.transpose(torch.matmul(input_adj_matrix, assignments),dim0=0,dim1=1)
    graph_pooled = torch.matmul(graph_pooled,assignments)
    print("Graph pooled")
    print(graph_pooled)

    # We compute the rank-1 normaizer matrix S^T*d*d^T*S efficiently
    # in three matrix multiplications by first processing the left part S^T*d
    # and then multyplying it by the right part d^T*S.
    # Left part is [k, 1] tensor.

    normalizer_left = torch.matmul(torch.transpose(assignments,dim0=0,dim1=1),degrees)
    # Right part is [1, k] tensor.
    normalizer_right = torch.matmul(torch.transpose(degrees,dim0=0,dim1=1),assignments)
    print("Normalization right/left")
    print(normalizer_left)
    print(normalizer_right)
    # Normalizer is rank-1 correction for degree distribution for degrees of the
    # nodes in the original graph, casted to the pooled graph.
    normalizer = torch.matmul(normalizer_left,normalizer_right) / 2 / n_edges
    spectral_loss = -torch.trace(graph_pooled-normalizer) / 2 /n_edges
        
    collapse_loss = torch.norm(cluster_sizes) / number_of_nodes * torch.sqrt(torch.tensor(n_clusters)) - 1

    features_pooled = torch.matmul(torch.transpose(assignments_pooling,dim0=0,dim1=1), input_tensor)
    features_pooled = selu(features_pooled)
    print()
    print("Features pooled")
    print(features_pooled)
    print(input_tensor)
    print()
    print("Spectral loss & collapse loss")
    print(spectral_loss)
    print(collapse_loss)

    # unique_values, counts = torch.unique(predicted_classes, return_counts=True)
    # colors = matplotlib.cm.jet(torch.linspace(0, 1, n_clusters))
    # fig = plt.figure(figsize=(12, 8),dpi=100)
    # ax2 = fig.add_subplot(111, projection='3d')
    # xs,ys,zs = input_tensor[:,0],input_tensor[:,1],input_tensor[:,2]
    # for i in unique_values:
    #     mask = predicted_classes==i
    #     ax2.scatter(xs[mask],zs[mask],ys[mask],color=colors[i],s=200)
    # plt.show()





