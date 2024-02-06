import torch
import torch_geometric
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import time
import os
import pickle
import random



class Net(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=128):
        super().__init__()

        self.norm = torch_geometric.nn.GraphNorm(in_channels)
        self.conv1 = torch_geometric.nn.GCNConv(in_channels, hidden_channels)
        self.relu = torch.nn.ReLU()
        self.selu = torch.nn.SELU()
        self.pool1 = torch_geometric.nn.DMoNPooling(hidden_channels,out_channels)

    def forward(self, x, edge_index, batch):

        x = self.norm(x)
        x = self.conv1(x, edge_index)
        # x = self.relu(x)
        x = self.selu(x)
        # return F.log_softmax(x,dim=-1), 0

        x, mask = torch_geometric.utils.to_dense_batch(x, batch)
        adj = torch_geometric.utils.to_dense_adj(edge_index, batch)
        s, x, adj, sp1, o1, c1 = self.pool1(x, adj, mask)

        return torch.nn.functional.log_softmax(x, dim=-1), sp1+o1+c1, s








if __name__=='__main__':

    #get data
    dataset_name = "xyzdeltaR_604_2.1_2_4"
    with open(f'datasets/data_{dataset_name}.pkl', 'rb') as f:
       data_list = pickle.load(f)

    with open(f'datasets/topocl_data_604.pkl', 'rb') as f:
       topocl_list = pickle.load(f)

    train_size = 0.8
    train_data_list = data_list[:int(train_size*len(data_list))]
    test_data_list = data_list[int(train_size*len(data_list)):]
    test_topocl_list = topocl_list[int(train_size*len(data_list)):]
    name = "calo"

    train_loader = torch_geometric.loader.DataLoader(train_data_list, batch_size=1)
    test_loader = torch_geometric.loader.DataLoader(test_data_list, batch_size=1)
    print(f'Starting {name} inference...\n\t{len(test_data_list)} validation graphs, with {train_data_list[0].x.shape[1]} attributes per node')

    #initialise model
    num_clusters = 10
    hidden_channels = 256
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net(in_channels=train_data_list[0].x.shape[1],
                hidden_channels=hidden_channels, 
                out_channels=num_clusters).to(device)

    num_epochs = 30 
    #load model
    model_name = f"{name}_dmon_{dataset_name}_data_{hidden_channels}nn_{num_clusters}c_{num_epochs}e"
    model.load_state_dict(torch.load("models/"+model_name + ".pt"))
    model.eval()

    save_loc = "plots/" + name +"/" + model_name + "/" + time.strftime("%Y%m%d-%H") + "/"
    os.makedirs(save_loc) if not os.path.exists(save_loc) else None


    #evaluate using some test graphs
    for evt in range(10):
        eval_graph = test_data_list[evt]
        pred,tot_loss,clus_ass = model(eval_graph.x,eval_graph.edge_index,eval_graph.batch)

        predicted_classes = clus_ass.squeeze().argmax(dim=1).numpy()
        unique_values, counts = np.unique(predicted_classes, return_counts=True)
        max_values,_ = torch.max(clus_ass.squeeze(),dim=1)
        below_threshold = max_values < 0.9


        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Plotting
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        fig = plt.figure(figsize=(12, 8))
        # 1. Model Input
        ax1 = fig.add_subplot(131, projection='3d')
        ax1.scatter(eval_graph.x[:, 0], eval_graph.x[:, 2], eval_graph.x[:, 1], c="b", marker='o',s=4)
        for src, dst in eval_graph.edge_index.t().tolist():
            x_src, y_src, z_src = eval_graph.x[src][:3]
            x_dst, y_dst, z_dst = eval_graph.x[dst][:3]
            ax1.plot([x_src, x_dst], [z_src, z_dst], [y_src, y_dst], c='r',alpha=0.5)
        
        #New:
        weighted_means = torch.sum(eval_graph.x[:, :3], dim=0) / eval_graph.x.shape[0]
        weighted_means1 = torch.mean(eval_graph.x[:, :3], dim=0)
        print('weighty means: ',weighted_means,weighted_means1)
        ax1.scatter(xs=weighted_means[0], ys=weighted_means[2], zs=weighted_means[1], color='gold',ec='black', marker='*',s=100)
        # parametrise 3d line
        slopey = weighted_means[1] / weighted_means[0] if weighted_means[0] != 0 else torch.inf
        slopez = weighted_means[2] / weighted_means[0] if weighted_means[0] != 0 else torch.inf

        y_value = slopey * (max(eval_graph.x[:,0]) - weighted_means[0]) + weighted_means[1]
        z_value = slopez * (max(eval_graph.x[:,0]) - weighted_means[0]) + weighted_means[2]
        ax1.plot([0,max(eval_graph.x[:,0])],[0,z_value],[0,y_value],ls='--',color='gold',lw=1.5)
        ax1.set(xlabel='X',ylabel='Z',zlabel='Y',title=f'Input Graph ({len(eval_graph.x)} 2sig Cells, {len(eval_graph.edge_index.t())} Edges)')

        # 2. Model Output
        ax2 = fig.add_subplot(132, projection='3d')
        ax2.set(xlim=ax1.get_xlim(),ylim=ax1.get_ylim(),zlim=ax1.get_zlim())
        # scatter = ax2.scatter(eval_graph.x[:, 0], eval_graph.x[:, 2], eval_graph.x[:, 1], c=predicted_classes, marker='o',s=3)
        # labels = [f"{value}: ({count})" for value,count in zip(unique_values, counts)]

        xs = eval_graph.x[:, 0]
        ys = eval_graph.x[:, 1]
        zs = eval_graph.x[:, 2]
        markers = [".",",","o","v","^","<",">","1","2","3","4","8","s","p","P","*","h","H","+","x","X","D","d","|","_"]
        random.shuffle(markers)
        colors = matplotlib.cm.jet(np.linspace(0, 1, num_clusters))
        for i in unique_values:
            mask = predicted_classes==i
            ax2.scatter(xs[mask],zs[mask],ys[mask],color=colors[i],marker=markers[i],s=8,label=f'{i}: {counts[np.where(unique_values==i)[0]][0]}')
        #Highlight uncertain points
        # ax2.scatter(xs[below_threshold],zs[below_threshold],ys[below_threshold],color='red',marker='o',s=1)
        ax2.legend(bbox_to_anchor=(1.02, 0.25),loc='lower left',fontsize='x-small')
        ax2.set(xlabel='X',ylabel='Z',zlabel='Y',title=f'Model Output {len(unique_values)} clusters')

        # 3. TopoClusters
        ax3 = fig.add_subplot(133, projection='3d')
        ax3.set(xlim=ax1.get_xlim(),ylim=ax1.get_ylim(),zlim=ax1.get_zlim())
        eval_topocl = test_topocl_list[evt]
        for cl_idx in range(len(eval_topocl)):
            cluster_inside_box = eval_topocl[cl_idx]
            ax3.scatter(cluster_inside_box['cell_xCells'], cluster_inside_box['cell_zCells'], cluster_inside_box['cell_yCells'], marker=markers[i],s=4,label=f'TC {cl_idx}: {len(cluster_inside_box)}')

        ax3.legend(bbox_to_anchor=(1.02, 0.25),loc='lower left',fontsize='x-small')
        ax3.set(xlabel='X',ylabel='Z',zlabel='Y',title=f'{len(eval_topocl)} Topocluster(s), {len(np.concatenate(eval_topocl))} cells')

        fig.tight_layout()
        # plt.show()
        # plt.close()
        fig.savefig(save_loc+f'/toy_data_event{evt}.png', bbox_inches="tight")
        print(f"Event number {evt}")
        for value, count in zip(unique_values, counts):
            print(f"\tCluster {value}: {count} occurrences")   


    print('\nModel name:\n',model_name)