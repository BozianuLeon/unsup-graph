import torch
import torch_geometric
import os
import pickle
import time


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



def train(train_loader):
    model.train()
    loss_all = 0

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out, tot_loss, _ = model(data.x, data.edge_index, data.batch)
        loss = tot_loss
        # print('loss',loss.item(),data.x.size(0))
        if loss!=loss:
            print(data.x.size())
            break
        loss.backward()
        loss_all += data.x.size(0) * float(loss)
        optimizer.step()
    return loss_all / len(train_loader.dataset)


@torch.no_grad()
def test(loader):
    model.eval()
    loss_all = 0

    for data in loader:
        data = data.to(device)
        pred, tot_loss, _ = model(data.x, data.edge_index, data.batch)
        loss = tot_loss 
        loss_all += data.x.size(0) * float(loss)

    return loss_all / len(loader.dataset)




if __name__=='__main__':

    #get data
    dataset_name = "xyzdeltaR_604_2.1_2_4"
    with open(f'datasets/data_{dataset_name}.pkl', 'rb') as f:
       data_list = pickle.load(f)
    train_size = 0.9
    train_data_list = data_list[:int(train_size*len(data_list))]
    test_data_list = data_list[int(train_size*len(data_list)):]
    name = "calo"

    train_loader = torch_geometric.loader.DataLoader(train_data_list, batch_size=1)
    test_loader = torch_geometric.loader.DataLoader(test_data_list, batch_size=1)
    print(f'Starting {name} training...\n\t{len(train_data_list)} training graphs, {len(test_data_list)} validation graphs, with {train_data_list[0].x.shape[1]} attributes per node')

    #initialise model
    num_clusters = 15
    hidden_channels = 256
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model = Net(in_channels=train_data_list[0].x.shape[1],
                hidden_channels=hidden_channels, 
                out_channels=num_clusters).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    #run training
    num_epochs = 30
    for epoch in range(1, num_epochs):
        start = time.perf_counter()
        train_loss = train(train_loader)
        train_loss2 = test(train_loader)
        test_loss = test(test_loader)
        timing = 0
        print(f"Epoch: {epoch:03d}, Train Loss: {train_loss:.3f}, Train Loss2: {train_loss2:.3f}, Test Loss: {test_loss:.3f}, Time: {time.perf_counter() - start:.3f}s")


    model_name = f"{name}_dmon_{dataset_name}_data_{hidden_channels}nn_{num_clusters}c_{num_epochs}e"
    torch.save(model.state_dict(), f"models/{model_name}.pt")
    print(f'Training finished for model {model_name}, enjoy!')





