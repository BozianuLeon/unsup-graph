import torch
import torch_geometric

import h5py
import numpy as np
import numpy.lib.recfunctions as rf

import argparse
import os.path as osp




def get_bucket_edges(cells2sig, mask2sig, neighbours_array, src_neighbours_array):
    '''
    Function to calculate edges between nodes in neighbouring buckets of eta,phi.
    No limit on number of edges, inputs are a subset of all cells. Max number of 
    neighbours is ~750.
    Inputs:
        cells2sig: numpy struct array, containing cell information for cells with 
            |significance|>2 
        mask2sig: numpy boolean array, True/False array used to mask cells failing
            the significance threshold
        neighbours_array: numpy.array, LUT calculating fixed cell neighbours based on 
            eta-phi buckets
        src_neighbours_array: numpy.array, LUT as neighbours_array containing the source
            nodes to match the dest nodes to make edge_indices in sparse tensor format
    Outputs:
        edge_indices: torch.tensor, tensor containing sparse adjacency matrix indices for
            cells passing significance threshold, shape [2,num_edges]
    '''

    # get cell IDs, used to mask the cells we have access to for this event
    cell_ids_2 = np.array(cells2sig['cell_IdCells'].astype(int))

    # get the neighbour arrays for the 2 sigma cells
    cell_neighb_2 = neighbours_array[mask2sig]
    src_cell_neighb_2 = src_neighbours_array[mask2sig]

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
    return torch.tensor(edge_indices)





class EdgeBuilder(torch.nn.Module):
    def __init__(self, name, signif_cut=2, k=None, rad=None, graph_dir=None):
        super().__init__()
        self.name = name # knn, rad, bucket, custom
        self.signif_cut = signif_cut
        self.k = k
        self.rad = rad
        self.graph_dir = graph_dir

        if self.name=="knn" and self.k is not None:
            self.builder = torch_geometric.nn.knn_graph
            self.args = {"k" : self.k}

        elif self.name=="rad" and self.rad is not None:
            self.builder = torch_geometric.nn.radius_graph
            self.args = {"r" : self.rad}

        elif self.name=="bucket" and self.graph_dir is not None:
            self.builder = get_bucket_edges
            self.args = {"neighbours_array"     : np.load(self.graph_dir+'/pyg/cell_neighbours.npy'),
                         "src_neighbours_array" : np.load(self.graph_dir+'/pyg/src_cell_neighbours.npy')}

        elif self.name=="custom":
            # to be implemented 
            self.builder = get_bucket_edges

        else:
            print("Please specify a valid builder with sufficient arguments")

    
    def forward(self, event_no, h5group_cells):

        cells = h5group_cells[event_no] 
        mask_2sigma = abs(cells['cell_E'] / cells['cell_Sigma']) >= 2
        cells2sig = cells[mask_2sigma]

        # get cell feature matrix from struct array 
        # TODO: instead of x,y,z coords give radius (or bucketized radius) instead
        cell_significance = np.expand_dims(abs(cells2sig['cell_E'] / cells2sig['cell_Sigma']),axis=1)
        cell_features = cells2sig[['cell_xCells','cell_yCells','cell_zCells','cell_eta','cell_phi','cell_E','cell_Sigma','cell_pt']]
        feature_matrix = rf.structured_to_unstructured(cell_features,dtype=np.float32)
        feature_matrix = np.hstack((feature_matrix,cell_significance))
        feature_tensor = torch.tensor(feature_matrix)    

        # get cell IDs,we will also return the cell IDs in the "y" attribute of .Data object
        cell_id_array  = np.expand_dims(cells2sig['cell_IdCells'],axis=1)
        cell_id_tensor = torch.tensor(cell_id_array.astype(np.int64))

        # make sparse adjacency matrix 
        if self.name == "bucket":
            edge_indices = self.builder(cells2sig, mask_2sigma, **self.args)
        else:
            edge_indices = self.builder(feature_tensor[:,[0,1,2]], **self.args)

        return feature_tensor[:,[0,1,2,3,4,-1]], edge_indices, cell_id_tensor




class CaloDataset(torch_geometric.data.Dataset):
    """The Custom Calorimeter Cells Dataset
    Dataset to cluster point clouds of cells into distinct clusters.
    Be thread safe wrt CUDA, see:
    https://discuss.pytorch.org/t/w-cudaipctypes-cpp-22-producer-process-has-been-terminated-before-all-shared-cuda-tensors-released-see-note-sharing-cuda-tensors/124445/14

    Args:
        root (str): Root directory where the dataset should be saved.
                    If root is not specified (None), no processing
        k    (int): K-nearest neighbour edhes. Degree of each node
        rad  (float): Threshold used in radial graph
        out  (str): Path to output directory, will have /data/.../ appended
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
    """

    def __init__(self, root, name="knn", k=None, rad=None, graph_dir=None, transform=None):
        self.name = name
        self.root = root
        self.k = k
        self.rad = rad
        self.graph_dir = graph_dir
        self.builder = EdgeBuilder(name=self.name,k=self.k,rad=self.rad,graph_dir=self.graph_dir)
        self.transform = transform if transform!=None else torch_geometric.transforms.RemoveDuplicatedEdges() # https://github.com/pyg-team/pytorch_geometric/discussions/7427
        print('1.',self.__dict__)
        print('2. root dir',root)
        print('3. raw  dir',self.raw_dir)
        super().__init__(self.root, self.transform)


    @property
    def raw_file_names(self):
        '''
        List of the h5 files to be opened during processing
        '''
        # return ['user.lbozianu.42998779._000026.calocellD3PD_mc21_14TeV_JZ4.r14365.h5']
        return ["user.lbozianu.42998779._000085.calocellD3PD_mc21_14TeV_JZ4.r14365.h5"]

    @property
    def raw_dir(self):
        '''
        Path to the raw cell data folder containing h5 files
        Later on, raw_paths = raw_dir + / + raw_file_names
        '''
        return osp.join(self.root, 'cells/JZ4/user.lbozianu')

    @property
    def raw_cl_file_names(self):
        '''
        List of the CLUSTER h5 files to be opened during processing
        '''
        return ["user.lbozianu.42998779._000085.topoClD3PD_mc21_14TeV_JZ4.r14365.h5"]

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
        Later on, we save event graphs to processed_dir + processed_file + *.pt
        Checks made on this dir, if exists and full no processing
        '''
        file_structure = {
            "custom" :   f"/custom/pyg2sig",
            "bucket" :   f"/bucket/pyg2sig",
            "knn"    :   f"/knn/{self.k}/pyg2sig",
            "rad"    :   f"/rad/{self.rad}/pyg2sig" 
        }

        return self.graph_dir + file_structure[self.name]
    
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
                feature_tensor, edge_indices, cell_ids = self.builder(event_no, cells_h5group)

                # create pyg Data object for saving
                event_graph  = torch_geometric.data.Data(x=feature_tensor,edge_index=edge_indices,y=cell_ids) 
                self.transform(event_graph)

                print("\tEvent graph made, saving... in here:", osp.join(self.processed_dir, f'event_graph_{idx}.pt'))
                torch.save(event_graph, osp.join(self.processed_dir, f'event_graph_{idx}.pt'))
                idx += 1
            f1.close()

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f'event_graph_{idx}.pt'), weights_only=False)
        return data
    
    def get_clusteres(self, idx):
        # idx tells us which event from all h5 files,
        # need to find the file first, then get the event no

        for j in range(len(self.raw_paths)):
            file = self.raw_paths[j]
            f1 = h5py.File(file,"r")
            n_events_in_file = len(f1["caloCells"]["2d"])
            if idx < n_events_in_file:
                cl_file = osp.join(self.root, 'clusters/JZ4/user.lbozianu', self.raw_cl_file_names[j])
                f2 = h5py.File(cl_file,"r")
                cl_data = f2["caloCells"] 
                event_data   = cl_data["1d"][idx]
                cluster_data = cl_data["2d"][idx]
                print(event_data.dtype)
                print()
                print(cluster_data.dtype)

                cl_pts = cluster_data['cl_pt'][np.isfinite(cluster_data['cl_pt'])] # [~np.isnan(cl_pts)]
                cl_E_em  = cluster_data['cl_E_em'][np.isfinite(cluster_data['cl_E_em'])]
                cl_E_had = cluster_data['cl_E_had'][np.isfinite(cluster_data['cl_E_had'])]
                cl_cell_n = cluster_data['cl_cell_n'][np.isfinite(cluster_data['cl_cell_n'])]
                cl_cellmaxfrac = cluster_data['cl_cellmaxfrac'][np.isfinite(cluster_data['cl_cellmaxfrac'])]
                # cl_etas = cluster_data['cl_eta'][np.isfinite(cluster_data['cl_eta'])] # no eta/phi YET
                topocluster_dict = {
                    "cl_pt":          cl_pts,
                    "cl_E" :          cl_E_em+cl_E_had,
                    "cl_cell_n":      cl_cell_n,
                    "cl_cellmaxfrac": cl_cellmaxfrac,
                    "cl_n":           event_data["cl_n"]
                }
                f2.close()
                return topocluster_dict
            else:
                idx = idx - n_events_in_file



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True, help='Path to top-level h5 directory',)
    parser.add_argument('--name', type=str, required=True, help='Name of edge building scheme (knn, rad, bucket, custom)')
    parser.add_argument('-k', nargs='?', const=None, default=None, type=int, help='K-nearest neighbours value to be used only in knn graph')
    parser.add_argument('-r', nargs='?', const=None, default=None, type=int, help='Radius value to be used only in radial graph')
    parser.add_argument('-o','--out',nargs='?', const='./cache/', default='./cache/', type=str, help='Path to processed folder containing .pt graphs',)
    args = parser.parse_args()

    # instantiate a dataset, if not already present will be created via process() call
    mydata = CaloDataset(root=args.root, name=args.name, k=args.k, rad=args.r, graph_dir=args.out)
    print("len",mydata.len(),len(mydata))
    print()

    event_no = 2
    event0 = mydata[event_no]
    print(event0)
    event0_cl = mydata.get_clusteres(event_no)
    print(event0_cl.keys())
    quit()

    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(event0.x[:, 0], event0.x[:, 2], event0.x[:, 1], s=event0.x[:, -1], c='b', marker='o')
    for src, dst in event0.edge_index.t().tolist():
        x_src, y_src, z_src, *feat = event0.x[src]
        x_dst, y_dst, z_dst, *feat = event0.x[dst]
        ax.plot([x_src, x_dst], [z_src, z_dst], [y_src, y_dst], c='r')
    ax.set(xlabel='X',ylabel='Y',zlabel='Z',title=f'Example Event Graph')
    plt.show()
    fig.savefig(f"./plots/inputs/ex-event-{event_no}.png", bbox_inches="tight")

