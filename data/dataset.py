import torch
import torch_geometric

import h5py
import numpy as np
import numpy.lib.recfunctions as rf

import os.path as osp






def get_features_edges(event_no, h5group_cells, neighbours_array, src_neighbours_array, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):

    # neighbours_array = cell_neighbours = np.load('./pyg/cell_neighbours.npy')
    # src_neighbours_array = src_cell_neighbours = np.load('./pyg/src_cell_neighbours.npy')
    # h5group_cells = f1["caloCells"]["2d"]
    
    cells = h5group_cells[event_no] 
    mask_2sigma = abs(cells['cell_E'] / cells['cell_Sigma']) >= 2
    mask_4sigma = abs(cells['cell_E'] / cells['cell_Sigma']) >= 4
    cells2sig = cells[mask_2sigma]
    cells4sig = cells[mask_4sigma]

    # get cell feature matrix from struct array 
    cell_significance = np.expand_dims(abs(cells2sig['cell_E'] / cells2sig['cell_Sigma']),axis=1)
    cell_features = cells2sig[['cell_xCells','cell_yCells','cell_zCells','cell_eta','cell_phi','cell_E','cell_Sigma','cell_pt']]
    feature_matrix = rf.structured_to_unstructured(cell_features,dtype=np.float32)
    feature_matrix = np.hstack((feature_matrix,cell_significance))

    # get cell IDs
    cell_ids_2 = np.array(cells2sig['cell_IdCells'].astype(int)) # THIS IS THE GRAND LIST OF CELLS WE CAN USE IN THIS EVENT
    cell_ids_4 = np.array(cells4sig['cell_IdCells'].astype(int))

    # get the neighbour arrays for the 2 sigma cells
    cell_neighb_2 = neighbours_array[mask_2sigma]
    src_cell_neighb_2 = src_neighbours_array[mask_2sigma]

    # filter cell neighbours, only >2sigma and remove padded -999 values
    actual_cell_neighb_2 = np.where(np.isin(cell_neighb_2,cell_ids_2), cell_neighb_2, np.nan) # actual cells we can use from cell_neighbours
    actual_src_cell_neighb_2 = np.where(np.isin(cell_neighb_2,cell_ids_2), src_cell_neighb_2, np.nan) 

    # find the cellID indices from cell_ids_2, what index are they in this event?
    neighb_2sig_indices = np.searchsorted(cell_ids_2,actual_cell_neighb_2)
    neighb_src_2sig_indices = np.searchsorted(cell_ids_2,actual_src_cell_neighb_2)

    # use the nan array to again extract just the valid node indices we want
    dst_node_indices = neighb_2sig_indices[~np.isnan(actual_cell_neighb_2)]
    src_node_indices = neighb_src_2sig_indices[~np.isnan(actual_src_cell_neighb_2)]

    edge_indices = np.stack((dst_node_indices,src_node_indices),axis=0)

    return torch.tensor(feature_matrix,device=device), torch.tensor(edge_indices,device=device)






class MyOwnDataset(torch_geometric.data.Dataset):
    """The Custom Calorimeter Cells Dataset
    Dataset to cluster point clouds of cells into distinct clusters.

    Args:
        root (str): Root directory where the dataset should be saved.
                    If root is not specified (None), no processing
        k    (int): K-nearest neighbour edhes. Degree of each node
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
    """

    def __init__(self, root, k=3, transform=None):
        self.k = k
        print('1.',self.__dict__)
        print('2.',self.raw_dir)
        print('3.',root)
        super().__init__(root, transform)


    @property
    def raw_file_names(self):
        '''
        List of the h5 files to be opened during processing
        '''
        return ["user.cantel.34126190._000001.calocellD3PD_mc16_JZ4W.r10788.h5"]

    @property
    def raw_dir(self):
        '''
        Path to the raw cell data folder containing h5 files
        Later on, raw_paths = raw_dir + / + raw_file_names
        '''
        # return osp.join(self.root, 'cells')
        return "../"

    @property
    def processed_file_names(self):
        '''
        List of the names of the pytorch geometric Data objects
        Unique id appended to file name
        '''
        output_file_list = [f"event_graph_{i}.pt" for i in range(self.len())]
        return output_file_list

    @property
    def processed_dir(self):
        '''
        Path to the output folder containing pyg graphs .pt files
        Later on, we save event graphs to processed_dir + *.pt
        Checks made on this dir, if exists and full no processing
        '''
        # return osp.join(self.root, 'pyg')
        return "./data/pyg2sig/"
    
    def len(self):
        n_total_events = 0
        for file in self.raw_paths:
            f1 = h5py.File(file,"r")
            n_events_in_file = len(f1["caloCells"]["2d"])
            n_total_events += n_events_in_file
            f1.close()
        return n_total_events
   
    def __len__(self):
        n_total_events = 0
        for file in self.raw_paths:
            f1 = h5py.File(file,"r")
            n_events_in_file = len(f1["caloCells"]["2d"])
            n_total_events += n_events_in_file
            f1.close()
        return n_total_events

    def process(self):
        idx = 0
        for file in self.raw_paths:
            print("\t",file)
            f1 = h5py.File(file,"r")
            n_events_in_file = len(f1["caloCells"]["2d"])
            cells_h5group = f1["caloCells"]["2d"]
            for event_no in range(n_events_in_file):
                cells = cells_h5group[event_no]
                mask_2sigma = abs(cells['cell_E'] / cells['cell_Sigma']) >= 2
                mask_4sigma = abs(cells['cell_E'] / cells['cell_Sigma']) >= 4
                cells2sig = cells[mask_2sigma]
                cells4sig = cells[mask_4sigma]

                # get cell feature matrix from struct array 
                cell_significance = np.expand_dims(abs(cells2sig['cell_E'] / cells2sig['cell_Sigma']),axis=1)
                cell_features = cells2sig[['cell_xCells','cell_yCells','cell_zCells','cell_eta','cell_phi','cell_E','cell_Sigma','cell_pt']]
                
                feature_matrix = rf.structured_to_unstructured(cell_features,dtype=np.float32)
                feature_matrix = np.hstack((feature_matrix,cell_significance))
                feature_tensor = torch.tensor(feature_matrix)
                print("\tX tensor",feature_tensor.shape)
                
                cell_id_array  = np.expand_dims(cells2sig['cell_IdCells'],axis=1)
                cell_id_tensor = torch.tensor(cell_id_array.astype(np.int64))

                # make sparse adjacency matrix using xyz coords
                edge_indices = torch_geometric.nn.knn_graph(feature_tensor[:,:3],k=self.k,loop=False)
                print("\t",edge_indices.shape)
                # create pyg Data object for saving
                event_graph  = torch_geometric.data.Data(x=feature_tensor[:,[0,1,2,3,4,-1]],edge_index=edge_indices,y=cell_id_tensor) 

                print("\tEvent graph made, saving... in here:", osp.join(self.processed_dir, f'event_graph_{idx}.pt'))
                torch.save(event_graph, osp.join(self.processed_dir, f'event_graph_{idx}.pt'))
                idx += 1
            f1.close()

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f'event_graph_{idx}.pt'), weights_only=False)
        return data



if __name__ == "__main__":

    mydata = MyOwnDataset("root_dir",k=2) # unhash to recreate dataset
    # mydata = MyOwnDataset(None) # do not execute process 
    print("len",mydata.len(),len(mydata))
    print()

    event_no = 2
    event0 = mydata[event_no]
    print(event0)

    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    # remember that XZY makes the calorimeter appear the "correct" way up
    ax.scatter(event0.x[:, 0], event0.x[:, 2], event0.x[:, 1], s=event0.x[:, -1], c='b', marker='o')
    for src, dst in event0.edge_index.t().tolist():
        x_src, y_src, z_src, *feat = event0.x[src]
        x_dst, y_dst, z_dst, *feat = event0.x[dst]
        ax.plot([x_src, x_dst], [z_src, z_dst], [y_src, y_dst], c='r')
    ax.set(xlabel='X',ylabel='Y',zlabel='Z',title=f'Event Graph with KNN 3 Edges')
    plt.show()
    fig.savefig(f"../plots/inputs/ex-event-{event_no}.png", bbox_inches="tight")

