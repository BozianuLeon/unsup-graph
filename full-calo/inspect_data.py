import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import h5py
import os


def remove_nan(array):
    #find the indices where there are not nan values
    good_indices = np.where(array==array) 
    return array[good_indices]



if __name__=="__main__":


    with open("../../struc_array.npy", "rb") as file:
        inference = np.load(file)


    for i in range(len(inference)):
        h5f = inference[i]['h5file']
        event_no = inference[i]['event_no']
        if h5f.decode('utf-8')=="01":
            extent_i = inference[i]['extent']

            print(i)
            cells_file = "../../user.cantel.34126190._0000{}.calocellD3PD_mc16_JZ4W.r10788.h5".format(h5f.decode('utf-8'))
            with h5py.File(cells_file,"r") as f:
                h5group = f["caloCells"]
                cells = h5group["2d"][event_no]
                mask_2sigma = abs(cells['cell_E'] / cells['cell_Sigma']) >= 2
                mask_4sigma = abs(cells['cell_E'] / cells['cell_Sigma']) >= 4
                cells2sig = cells[mask_2sigma]
                cells4sig = cells[mask_4sigma]


            clusters_file = "../../user.cantel.34126190._0000{}.topoclusterD3PD_mc16_JZ4W.r10788.h5".format(h5f.decode('utf-8'))
            with h5py.File(clusters_file,"r") as f:
                cl_data = f["caloCells"] 
                event_data = cl_data["1d"][event_no]
                cluster_data = cl_data["2d"][event_no]
                cluster_cell_data = cl_data["3d"][event_no]    



    
            ##  1. Calorimeter in 3d
            fig = plt.figure(figsize=(12, 8),dpi=100)
            ax1 = fig.add_subplot(111, projection='3d')
            ax1.scatter(cells['cell_xCells'][np.isin(cells['cell_DetCells'],[65,81,97,113])], cells['cell_zCells'][np.isin(cells['cell_DetCells'],[65,81,97,113])], cells['cell_yCells'][np.isin(cells['cell_DetCells'],[65,81,97,113])], color='coral',alpha=0.1,marker='o',s=1,label='EMBar')
            ax1.scatter(cells['cell_xCells'][np.isin(cells['cell_DetCells'],[257,273,289,305])], cells['cell_zCells'][np.isin(cells['cell_DetCells'],[257,273,289,305])], cells['cell_yCells'][np.isin(cells['cell_DetCells'],[257,273,289,305])], color='khaki',alpha=0.25,marker='o',s=1,label='EMEC')
            ax1.scatter(cells['cell_xCells'][np.isin(cells['cell_DetCells'],[145,161])], cells['cell_zCells'][np.isin(cells['cell_DetCells'],[145,161])], cells['cell_yCells'][np.isin(cells['cell_DetCells'],[145,161])], color='olivedrab',alpha=0.4,marker='o',s=1,label='EMIW')
            ax1.scatter(cells['cell_xCells'][np.isin(cells['cell_DetCells'],[2052])], cells['cell_zCells'][np.isin(cells['cell_DetCells'],[2052])], cells['cell_yCells'][np.isin(cells['cell_DetCells'],[2052])], color='saddlebrown',alpha=0.4,marker='o',s=1,label='EMFCAL')
            ax1.scatter(cells['cell_xCells'][np.isin(cells['cell_DetCells'],[65544,73736,81928])], cells['cell_zCells'][np.isin(cells['cell_DetCells'],[65544,73736,81928])], cells['cell_yCells'][np.isin(cells['cell_DetCells'],[65544,73736,81928])], color='greenyellow',alpha=0.55,marker='o',s=5.5,label='TileBar')
            ax1.scatter(cells['cell_xCells'][np.isin(cells['cell_DetCells'],[131080,139272,147464])], cells['cell_zCells'][np.isin(cells['cell_DetCells'],[131080,139272,147464])], cells['cell_yCells'][np.isin(cells['cell_DetCells'],[131080,139272,147464])], color='slateblue',alpha=0.4,marker='o',s=5.5,label='TileEC')
            ax1.scatter(cells['cell_xCells'][np.isin(cells['cell_DetCells'],[811016,278536,270344])], cells['cell_zCells'][np.isin(cells['cell_DetCells'],[811016,278536,270344])], cells['cell_yCells'][np.isin(cells['cell_DetCells'],[811016,278536,270344])], color='cornflowerblue',alpha=0.4,marker='o',s=1,label='TileGap')
            ax1.scatter(cells['cell_xCells'][np.isin(cells['cell_DetCells'],[2,514,1026,1538])], cells['cell_zCells'][np.isin(cells['cell_DetCells'],[2,514,1026,1538])], cells['cell_yCells'][np.isin(cells['cell_DetCells'],[2,514,1026,1538])], color='darkturquoise',alpha=0.4,marker='o',s=2,label='HEC')
            ax1.scatter(cells['cell_xCells'][np.isin(cells['cell_DetCells'],[2052,4100,6148])], cells['cell_zCells'][np.isin(cells['cell_DetCells'],[2052,4100,6148])], cells['cell_yCells'][np.isin(cells['cell_DetCells'],[2052,4100,6148])], color='black',alpha=0.4,marker='o',s=1,label='FCAL')

            legend_elements = [matplotlib.lines.Line2D([],[], marker='o', color='coral', label='EMBar',linestyle='None',markersize=15),
                               matplotlib.lines.Line2D([],[], marker='o', color='khaki', label='EMEC',linestyle='None',markersize=15),
                               matplotlib.lines.Line2D([],[], marker='o', color='olivedrab', label='EMIW',linestyle='None',markersize=15),
                               matplotlib.lines.Line2D([],[], marker='o', color='saddlebrown', label='EMFCAL',linestyle='None',markersize=15),
                               matplotlib.lines.Line2D([],[], marker='o', color='greenyellow', label='TileBar',linestyle='None',markersize=15),
                               matplotlib.lines.Line2D([],[], marker='o', color='slateblue', label='TileEC',linestyle='None',markersize=15),
                               matplotlib.lines.Line2D([],[], marker='o', color='cornflowerblue', label='TileGap',linestyle='None',markersize=15),
                               matplotlib.lines.Line2D([],[], marker='o', color='darkturquoise', label='HEC',linestyle='None',markersize=15),
                               matplotlib.lines.Line2D([],[], marker='o', color='black', label='FCAL',linestyle='None',markersize=15),]
            ax1.legend(handles=legend_elements,frameon=False,bbox_to_anchor=(0.87, 0.3),loc='lower left',prop={'family':'serif','style':'normal','size':12})
            ax1.set(xlabel='X',ylabel='Z',zlabel='Y')
            x_lim = ax1.get_xlim()
            y_lim = ax1.get_ylim()
            z_lim = ax1.get_zlim()
            ax1.axis('off')
            fig.tight_layout()
            plt.show()
            plt.close() 

            fig = plt.figure(figsize=(12, 8),dpi=100)
            ax1 = fig.add_subplot(111, projection='3d')
            ax1.scatter(cells2sig['cell_xCells'], cells2sig['cell_zCells'], cells2sig['cell_yCells'], color='red',alpha=0.9,marker='o',s=2)
            ax1.set(xlabel='X',ylabel='Z',zlabel='Y',title='|significance|>2',xlim=x_lim,ylim=y_lim,zlim=z_lim)
            ax1.axis('off')
            fig.tight_layout()
            plt.show()
            plt.close() 

            # fig = plt.figure(figsize=(12, 8),dpi=100)
            # ax1 = fig.add_subplot(111, projection='3d')
            # ax1.scatter(cells4sig['cell_xCells'], cells4sig['cell_zCells'], cells4sig['cell_yCells'], color='dodgerblue',alpha=0.9,marker='o',s=8)
            # ax1.set(xlabel='X',ylabel='Z',zlabel='Y',title='|significance|>4',xlim=x_lim,ylim=y_lim,zlim=z_lim)
            # ax1.axis('off')
            # fig.tight_layout()
            # plt.show()
            # plt.close() 

            # fig = plt.figure(figsize=(12, 8),dpi=100)
            # ax3 = fig.add_subplot(111, projection='3d')
            # ax3.set(xlim=ax1.get_xlim(),ylim=ax1.get_ylim(),zlim=ax1.get_zlim())
            # for cl_idx in range(len(cluster_cell_data)):
            #     cluster_idx = cluster_cell_data[cl_idx]
            #     cluster_i   = cluster_idx[np.where(cluster_idx==cluster_idx)] # get rid of nan values (padded in h5 files)
            #     ax3.scatter(cluster_i['cl_cell_xCells'], cluster_i['cl_cell_zCells'], cluster_i['cl_cell_yCells'],s=12,alpha=0.5,label=f'TC {cl_idx}: {len(cluster_i)}')
            # ax3.axis('off')
            # ax3.legend()
            # fig.tight_layout()
            # plt.show()
            # plt.close() 


            fig = plt.figure(figsize=(12,8))
            ax1 = fig.add_subplot(121, projection='3d')
            ax1.scatter(cells4sig['cell_xCells'], cells4sig['cell_zCells'], cells4sig['cell_yCells'], color='dodgerblue',alpha=0.9,marker='o',s=2)
            ax1.set(xlabel='X',ylabel='Z',zlabel='Y',title=f'|significance|>4 ({len(cells4sig)} cells)',xlim=x_lim,ylim=y_lim,zlim=z_lim)

            ax2 = fig.add_subplot(122, projection='3d')
            ax2.set(xlim=ax1.get_xlim(),ylim=ax1.get_ylim(),zlim=ax1.get_zlim())
            for cl_idx in range(len(cluster_cell_data)):
                cluster_idx = cluster_cell_data[cl_idx]
                cluster_i   = cluster_idx[np.where(cluster_idx==cluster_idx)] # get rid of nan values (padded in h5 files)
                ax2.scatter(cluster_i['cl_cell_xCells'], cluster_i['cl_cell_zCells'], cluster_i['cl_cell_yCells'],s=4,alpha=0.5,label=f'TC {cl_idx}: {len(cluster_i)}')
            ax2.set(xlabel='X',ylabel='Z',zlabel='Y',title=f'Topoclusters {len(cluster_cell_data[np.where(cluster_cell_data==cluster_cell_data)])}',xlim=x_lim,ylim=y_lim,zlim=z_lim)   
            plt.show()
            plt.close()

            print(cluster_data["cl_E_em"]+cluster_data["cl_E_had"])
            print(cluster_data["cl_pt"])
            print(cl_idx)
            print(len(cluster_data))
            print(len(cluster_cell_data))
            print([len(x) for x in cluster_cell_data])
            print(len([len(x) for x in cluster_cell_data]))
            print([x for x in cluster_data["cl_cell_n"]])
            print(event_data["cl_n"])
            print(len(cluster_data["cl_pt"]))



            if input("Do you want to quit? (y/n): ").strip().lower() == 'y': quit()













            