import torch
import torch_geometric

import h5py
import numpy as np
import numpy.lib.recfunctions as rf

import argparse
import os
import os.path as osp




def get_bucket_edges(cells2sig, mask2sig, neighbours_array, src_neighbours_array):
    '''
    Function to calculate edges between nodes in neighbouring buckets of eta,phi.
    No limit on number of edges, inputs are a subset of all cells. Max number of 
    edges for a single node is ~750.
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


def get_custom_edges(cells, neigh_3x3, src_neigh_3x3, neigh_cross, src_neigh_cross, neigh_1x1, src_neigh_1x1):
    '''
    Function to calculate edges between nodes in neighbouring buckets of eta,phi.
    Take all of 2 sigma cells, then connect them to all >3 sigma cells in the
    neighbouring buckets.
    Inputs:
        cells: numpy struct array, containing all cell information, 
            to be masked differently for |significance| > 2 or 3
        neigh_3x3: numpy.array, LUT calculating fixed cell neighbours based on 
            3x3 windows in eta-phi buckets
        src_neigh_3x3: numpy.array, LUT same as neigh_3x3 containing the source
            nodes to match the dest nodes to make edge_indices in sparse tensor format
        neigh_cross: numpy.array, LUT calculating fixed cell neighbours based on 
            cross-like windows in eta-phi buckets
        src_neigh_cross: numpy.array, LUT same as neigh_cross containing the source
            nodes to match the dest nodes to make edge_indices in sparse tensor format
        neigh_1x1: numpy.array, LUT calculating fixed cell neighbours based on 
            1x1 windows in eta-phi buckets
        src_neigh_1x1: numpy.array, LUT same as neigh_1x1 containing the source
            nodes to match the dest nodes to make edge_indices in sparse tensor format
    Outputs:
        edge_indices: torch.tensor, tensor containing sparse adjacency matrix indices for
            cells passing significance threshold, shape [2,num_edges]
    '''

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # First, deal with the seed cells:
    mask4sig = abs(cells['cell_E'] / cells['cell_Sigma']) >= 4
    cells4sig = cells[mask4sig]
    cell_ids_4 = np.array(cells4sig['cell_IdCells'].astype(int))

    # get the neighbours from the 3x3 LUT
    cell_neighb_3x3_4 = neigh_3x3[mask4sig]
    src_cell_neighb_3x3_4 = src_neigh_3x3[mask4sig]

    # but not all of the cells that are neighbours exceed 4 sigma. Filter out low sig cells (+ 999 padded values)
    actual_cell_neighb_4 = np.where(np.isin(cell_neighb_3x3_4,cell_ids_4), cell_neighb_3x3_4, np.nan) # actual cells we can use from cell_neighbours
    actual_src_cell_neighb_4 = np.where(np.isin(cell_neighb_3x3_4,cell_ids_4), src_cell_neighb_3x3_4, np.nan) 

    # translate from cell ID to index, used in this event
    neighb_4sig_indices = np.searchsorted(cell_ids_4,actual_cell_neighb_4)
    neighb_src_4sig_indices = np.searchsorted(cell_ids_4,actual_src_cell_neighb_4)

    # use the nan array to again extract just the valid node indices we want
    dst_node_4_indices = neighb_4sig_indices[~np.isnan(actual_cell_neighb_4)]
    src_node_4_indices = neighb_src_4sig_indices[~np.isnan(actual_src_cell_neighb_4)]
    edge_indices_4 = np.stack((dst_node_4_indices,src_node_4_indices),axis=0)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Second, the cells with 3 < |significance| < 4
    # only connect to > 4 sigma cells in "cross-like" window
    mask3sig = (abs(cells['cell_E'] / cells['cell_Sigma']) >= 3) & (abs(cells['cell_E'] / cells['cell_Sigma']) < 4)
    cells3sig = cells[mask3sig]

    # get cell IDs, used to mask the cells we have access to for this event
    cell_ids_3 = np.array(cells3sig['cell_IdCells'].astype(int))

    # get the neighbour arrays for the 3 sigma cells, from the cross-like LUT
    cell_neighb_3 = neigh_cross[mask3sig]
    src_cell_neighb_3 = src_neigh_cross[mask3sig]

    # again, not all neighbours pass the 4(!) sigma threshold (+ remove 999 pad values)
    # importantly, we have ALL possible cells in the LUT, but we only want those above 4sigma (some will be even below 2sigma!) 
    actual_cell_neighb_3 = np.where(np.isin(cell_neighb_3,cell_ids_4), cell_neighb_3, np.nan) # actual cells we can use from cell_neighbours
    actual_src_cell_neighb_3 = np.where(np.isin(cell_neighb_3,cell_ids_4), src_cell_neighb_3, np.nan) 

    # find the cellID indices from cell_ids_3, what index are they in this event?
    # this transforms from cells3sig indices to cell index per event
    neighb_3sig_indices = np.searchsorted(cell_ids_3,actual_cell_neighb_3)
    neighb_src_3sig_indices = np.searchsorted(cell_ids_3,actual_src_cell_neighb_3)

    # use the nan array to again extract just the valid node indices we want
    dst_node_3_indices = neighb_3sig_indices[~np.isnan(actual_cell_neighb_3)]
    src_node_3_indices = neighb_src_3sig_indices[~np.isnan(actual_src_cell_neighb_3)]
    edge_indices_3 = np.stack((dst_node_3_indices,src_node_3_indices),axis=0)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # third, the cells with 2 < |significance| < 3
    # only connect to > 4 sigma cells in a single 1x1 bin
    # go through same process as above
    mask2sig = (abs(cells['cell_E'] / cells['cell_Sigma']) >= 2) & (abs(cells['cell_E'] / cells['cell_Sigma']) < 3)
    cells2sig = cells[mask2sig]

    # get cell IDs, used to mask the cells we have access to for this event
    cell_ids_2 = np.array(cells2sig['cell_IdCells'].astype(int))

    # get the neighbour arrays for the 2 sigma cells
    cell_neighb_2 = neigh_1x1[mask2sig]
    src_cell_neighb_2 = src_neigh_1x1[mask2sig]

    # again, not all neighbours pass the 4(!) sigma threshold (+ remove 999 pad values)
    actual_cell_neighb_2 = np.where(np.isin(cell_neighb_2,cell_ids_4), cell_neighb_2, np.nan) # actual cells we can use from cell_neighbours
    actual_src_cell_neighb_2 = np.where(np.isin(cell_neighb_2,cell_ids_4), src_cell_neighb_2, np.nan) 

    # find the cellID indices from cell_ids_2, what index are they in this event?
    neighb_2sig_indices = np.searchsorted(cell_ids_2,actual_cell_neighb_2)
    neighb_src_2sig_indices = np.searchsorted(cell_ids_2,actual_src_cell_neighb_2)

    # use the nan array to again extract just the valid node indices we want
    dst_node_2_indices = neighb_2sig_indices[~np.isnan(actual_cell_neighb_2)]
    src_node_2_indices = neighb_src_2sig_indices[~np.isnan(actual_src_cell_neighb_2)]

    edge_indices_2 = np.stack((dst_node_2_indices,src_node_2_indices),axis=0)
    # print('====')
    # print(cells4sig.shape,cells3sig.shape,cells2sig.shape)
    # print(edge_indices_4.shape,edge_indices_3.shape,edge_indices_2.shape)  
    # print(np.hstack((edge_indices_4, edge_indices_3, edge_indices_2)).shape)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Now, deal with 2 sigma cells:
    # they only connect to >3 sigma cells in a smaller 3x3 window
    # mask2sig = abs(cells['cell_E'] / cells['cell_Sigma']) >= 2
    # cells2sig = cells[mask2sig]
    # mask3sig = abs(cells['cell_E'] / cells['cell_Sigma']) >= 3
    # cells3sig = cells[mask3sig]

    # # get cell IDs, used to mask the cells we have access to for this event
    # cell_ids_2 = np.array(cells2sig['cell_IdCells'].astype(int))
    # cell_ids_3 = np.array(cells3sig['cell_IdCells'].astype(int))

    # # get the neighbour arrays for the 2 sigma cells
    # cell_neighb_2 = neigh_3x3[mask2sig]
    # src_cell_neighb_2 = src_neigh_3x3[mask2sig]

    # # again, not all neighbours pass the 3(!) sigma threshold (+ remove 999 pad values)
    # actual_cell_neighb_2 = np.where(np.isin(cell_neighb_2,cell_ids_3), cell_neighb_2, np.nan) # actual cells we can use from cell_neighbours
    # actual_src_cell_neighb_2 = np.where(np.isin(cell_neighb_2,cell_ids_3), src_cell_neighb_2, np.nan) 

    # # find the cellID indices from cell_ids_2, what index are they in this event?
    # neighb_2sig_indices = np.searchsorted(cell_ids_2,actual_cell_neighb_2)
    # neighb_src_2sig_indices = np.searchsorted(cell_ids_2,actual_src_cell_neighb_2)

    # # use the nan array to again extract just the valid node indices we want
    # dst_node_2_indices = neighb_2sig_indices[~np.isnan(actual_cell_neighb_2)]
    # src_node_2_indices = neighb_src_2sig_indices[~np.isnan(actual_src_cell_neighb_2)]

    # edge_indices_2 = np.stack((dst_node_2_indices,src_node_2_indices),axis=0)
    # print(edge_indices_4.shape,edge_indices_2.shape)

    return torch.tensor(np.hstack((edge_indices_4, edge_indices_3, edge_indices_2)))





class EdgeBuilder(torch.nn.Module):
    def __init__(self, name, feat, signif_cut=2, k=None, rad=None, graph_dir=None):
        super().__init__()
        self.name = name # knn, rad, bucket, custom
        self.feat = feat
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
            self.builder = get_custom_edges
            self.args = {"neigh_3x3"       : np.load(self.graph_dir+'/pyg/cell_neighbours.npy'),
                         "src_neigh_3x3"   : np.load(self.graph_dir+'/pyg/src_cell_neighbours.npy'),
                         "neigh_cross"     : np.load(self.graph_dir+'/pyg/cell_cross_neighbours.npy'),
                         "src_neigh_cross" : np.load(self.graph_dir+'/pyg/src_cell_cross_neighbours.npy'),
                         "neigh_1x1"       : np.load(self.graph_dir+'/pyg/cell_1x1_neighbours.npy'),
                         "src_neigh_1x1"   : np.load(self.graph_dir+'/pyg/src_cell_1x1_neighbours.npy'),
                         }
        else:
            print("Please specify a valid builder (knn, rad, bucket, custom) with sufficient arguments")

    
    def forward(self, event_no, h5group_cells):

        cells = h5group_cells[event_no] 
        mask_2sigma = abs(cells['cell_E'] / cells['cell_Sigma']) >= 2
        cells2sig = cells[mask_2sigma]

        # get cell feature matrix from struct array 
        # TODO: instead of x,y,z coords give radius (or bucketized radius) instead
        cell_significance = np.expand_dims(abs(cells2sig['cell_E'] / cells2sig['cell_Sigma']),axis=1)
        cell_phi_2pi = np.expand_dims(cells2sig['cell_phi']%(2*np.pi),axis=1)
        cell_radius = np.sqrt(np.power(cells2sig['cell_xCells'],2) + np.power(cells2sig['cell_yCells'],2) + np.power(cells2sig['cell_zCells'],2))
        cell_radius = np.expand_dims(cell_radius, axis=1)
        cell_features = cells2sig[['cell_xCells','cell_yCells','cell_zCells','cell_eta','cell_phi','cell_E','cell_Sigma','cell_pt']]
        feature_matrix = rf.structured_to_unstructured(cell_features,dtype=np.float32)
        feature_matrix = np.hstack((feature_matrix,cell_radius,cell_phi_2pi,cell_significance))
        feature_tensor = torch.tensor(feature_matrix)    

        # get cell IDs,we will also return the cell IDs in the "y" attribute of .Data object
        # cell_id_array  = np.expand_dims(cells2sig['cell_IdCells'],axis=1)
        # cell_id_tensor = torch.tensor(cell_id_array.astype(np.int64))
        cell_y_array = cells2sig[['cell_IdCells','cell_eta','cell_phi','cell_E']]
        y_matrix = rf.structured_to_unstructured(cell_y_array,dtype=np.float32)
        y_tensor = torch.tensor(y_matrix)    

        # get number of cells above |significance| threshold, to be stored in .n attribute
        cells5sig = cells2sig[abs(cells2sig['cell_E'] / cells2sig['cell_Sigma']) >= 5]
        n_5sig_cells = len(cells5sig)

        # make sparse adjacency matrix 
        if self.name == "bucket":
            edge_indices = self.builder(cells2sig, mask_2sigma, **self.args)
        elif self.name == "custom":
            edge_indices = self.builder(cells, **self.args)
        else:
            edge_indices = self.builder(feature_tensor[:,[0,1,2]], **self.args)
        
        if self.feat=="XYZ":
            cols = [0,1,2,7,-1] # x, y, z, pt, significance
        elif self.feat=="GEO":
            cols = [0,1,2] # x, y, z
        elif self.feat=="CYL":
            cols = [8,3,4] # r, eta, phi
        elif self.feat=="REP":
            cols = [8,3,4,7,-1]   # r, eta, phi, pt, significance
        elif self.feat=="REPP":
            cols = [8,3,4,9,7,-1]   # r, eta, phi, phi(mod2pi), pt, significance

        return feature_tensor[:,cols], edge_indices, y_tensor, n_5sig_cells




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

    def __init__(self, root, name="knn", feat="XYZ", k=None, rad=None, graph_dir=None, transform=None, test=False):
        self.name = name
        self.feat = feat
        self.root = root
        self.k = k
        self.rad = rad
        self.graph_dir = graph_dir
        self.builder = EdgeBuilder(name=self.name,feat=self.feat,k=self.k,rad=self.rad,graph_dir=self.graph_dir)
        self.transform = transform if transform!=None else torch_geometric.transforms.RemoveDuplicatedEdges() # https://github.com/pyg-team/pytorch_geometric/discussions/7427
        # TODO: Look into  -  torch_geometric.transforms.RemoveIsolatedNodes, 
        self.test = test
        print('1.',self.__dict__)
        print('2. root dir',self.root, ' raw dir', self.raw_dir)
        super().__init__(self.root, self.transform)


    @property
    def raw_file_names(self):
        '''
        List of the h5 files to be opened during processing
        '''
        if not self.test:
            file_ids = ["01", "02", "03"] #, "04", "05", "06", "07"]
        else:
            file_ids = ["10"]
        return [f"user.lbozianu.44670103._0000{file_id}.calocellD3PD_mc21_14TeV_ttbar.r15583.h5" for file_id in file_ids]

    @property
    def raw_dir(self):
        '''
        Path to the raw cell data folder containing h5 files
        Later on, raw_paths = raw_dir + / + raw_file_names
        '''
        return osp.join(self.root, 'cells/ttbar/user.lbozianu.mc21_14TeV.601229.PhPy8EG_A14_ttbar_hdamp258p75_SingleLep.h5_calocellD3PD_mc21_14TeV_ttbar.r15583.h5')
        # return osp.join(self.root, 'cells/JZ4/user.lbozianu')

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
        if not self.test:
            file_structure = {
                "custom" :   f"/custom/ttbar/pyg2sig{self.feat}",
                "bucket" :   f"/bucket/ttbar/pyg2sig{self.feat}",
                "knn"    :   f"/knn/ttbar/{self.k}/pyg2sig{self.feat}",
                "rad"    :   f"/rad/ttbar/{self.rad}/pyg2sig{self.feat}" 
            }
        else:
            file_structure = {
                "custom" :   f"/custom/ttbar_test/pyg2sig{self.feat}",
                "bucket" :   f"/bucket/ttbar_test/pyg2sig{self.feat}",
                "knn"    :   f"/knn/ttbar_test/{self.k}/pyg2sig{self.feat}",
                "rad"    :   f"/rad/ttbar_test/{self.rad}/pyg2sig{self.feat}" 
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
                feature_tensor, edge_indices, y_tensor, n_tensor = self.builder(event_no, cells_h5group)

                # create pyg Data object for saving
                event_graph  = torch_geometric.data.Data(x=feature_tensor,edge_index=edge_indices,y=y_tensor,n=n_tensor) 
                self.transform(event_graph)

                print("\tEvent graph made, saving... in here:", osp.join(self.processed_dir, f'event_graph_{idx}.pt'))
                torch.save(event_graph, osp.join(self.processed_dir, f'event_graph_{idx}.pt'))
                idx += 1
            f1.close()

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f'event_graph_{idx}.pt'), weights_only=False)
        return data



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True, help='Path to top-level h5 directory',)
    parser.add_argument('--name', type=str, required=True, help='Name of edge building scheme (knn, rad, bucket, custom)')
    parser.add_argument('--feat', type=str, nargs='?', const="XYZ", default="XYZ", help='Which geometrical columns are in the feature matrix (XYZ or REP)')
    parser.add_argument('--test', action="store_true", help='Bool for train or test set')
    parser.add_argument('-k', nargs='?', const=None, default=None, type=int, help='K-nearest neighbours value to be used only in knn graph')
    parser.add_argument('-r', nargs='?', const=None, default=None, type=int, help='Radius value to be used only in radial graph')
    parser.add_argument('-o','--out',nargs='?', const='./cache/', default='./cache/', type=str, help='Path to processed folder containing .pt graphs',)
    args = parser.parse_args()
    print("\t\t",args.test)
    # instantiate a dataset, if not already present will be created via process() call
    mydata = CaloDataset(root=args.root, name=args.name, feat=args.feat, k=args.k, rad=args.r, graph_dir=args.out, test=args.test)
    print("len",mydata.len(),len(mydata))
    print()

    event_no = 2
    event0 = mydata[event_no]
    print(event0)
    print(event0.n)
    # event0_cl = mydata.get_clusters(event_no)
    # print(event0_cl.keys())


    save_loc = osp.join(args.out,osp.pardir) + "/plots/inputs/"
    if not os.path.exists(save_loc): os.makedirs(save_loc)

    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(event0.x[:, 0], event0.x[:, 2], event0.x[:, 1], s=event0.x[:, -1], c='b', marker='o')
    for src, dst in event0.edge_index.t().tolist():
        x_src, y_src, z_src, *feat = event0.x[src]
        x_dst, y_dst, z_dst, *feat = event0.x[dst]
        ax.plot([x_src, x_dst], [z_src, z_dst], [y_src, y_dst], c='r')
    ax.set(xlabel='X',ylabel='Y',zlabel='Z',title=f'Example Event Graph ({event0.edge_index.shape[1]} edges)')
    plt.show()
    fig.savefig(save_loc+f"/ex-{args.name}-{args.feat}-event-{event_no}.png", bbox_inches="tight")
