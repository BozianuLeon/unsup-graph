import torch
import torch_geometric
import os
import pickle
import time
import tqdm
import wandb
wandb.login()


class Net(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=128):
        super().__init__()
        self.n_clusters = out_channels
        self.n_conv = 1
        self.activation = "relu"

        self.norm = torch_geometric.nn.GraphNorm(in_channels)
        self.conv1 = torch_geometric.nn.GCNConv(in_channels, hidden_channels)
        self.relu = torch.nn.ReLU()
        self.selu = torch.nn.SELU()
        self.pool1 = torch_geometric.nn.DMoNPooling(hidden_channels,out_channels)

    def forward(self, x, edge_index, batch):

        x = self.norm(x)
        x = self.conv1(x, edge_index)
        x = self.relu(x)
   
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
    dataset_name = "xyzdeltaR_604_2.1_2_3"
    with open(f'datasets/data_{dataset_name}.pkl', 'rb') as f:
       data_list = pickle.load(f)
    train_size = 0.9
    train_data_list = data_list[:int(train_size*len(data_list))]
    test_data_list = data_list[int(train_size*len(data_list)):]
    name = "calo"
    bs = 1

    train_loader = torch_geometric.loader.DataLoader(train_data_list, batch_size=bs)
    test_loader = torch_geometric.loader.DataLoader(test_data_list, batch_size=bs)
    print(f'Starting {name} training...\n\t{len(train_data_list)} training graphs, {len(test_data_list)} validation graphs, with {train_data_list[0].x.shape[1]} attributes per node')

    #initialise model
    num_clusters = 3
    hidden_channels = 256
    lr = 0.0005
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model = Net(in_channels=train_data_list[0].x.shape[1],
                hidden_channels=hidden_channels, 
                out_channels=num_clusters).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    #run training
    num_epochs = 30



    run = wandb.init(
        # Set the project where this run will be logged
        project="calo-dmon",
        # Track hyperparameters and run metadata
        config={
            "dataset": dataset_name,
            "train_size": train_size,
            "batch_size": bs,
            "in_channels": train_data_list[0].x.shape[1],
            "n_clusters": num_clusters,
            "hidden_channels": hidden_channels,
            "n_conv": model.n_conv,
            "activation": model.activation,
            "learning_rate": lr,
            "epochs": num_epochs,
            }
        )

    wandb.watch(model, log_freq=100)
    for epoch in range(1, num_epochs):
        start = time.perf_counter()
        train_loss = train(train_loader)
        train_loss2 = test(train_loader)
        test_loss = test(test_loader)
        timing = 0
        print(f"Epoch: {epoch:03d}, Train Loss: {train_loss:.3f}, Train Loss2: {train_loss2:.3f}, Test Loss: {test_loss:.3f}, Time: {time.perf_counter() - start:.3f}s")
        wandb.log({"Train loss": train_loss, "Val loss": test_loss})

    model_name = f"{name}_dmon_{dataset_name}_data_{hidden_channels}nn_{num_clusters}c_{num_epochs}e"
    torch.save(model.state_dict(), f"models/{model_name}.pt")
    print(f'Training finished for model {model_name}, enjoy!')





