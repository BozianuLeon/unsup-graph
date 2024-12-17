import torch
import torchvision
import torch_geometric
import h5py
import os
import pickle
import scipy
import numpy as np
import numpy.lib.recfunctions as rf
import matplotlib
import matplotlib.pyplot as plt
import plotly
import plotly.graph_objects as go

from utils import wrap_check_truth, circular_mean, RetrieveCellIdsFromCluster, RetrieveClusterCellsFromBox

MIN_CELLS_PHI,MAX_CELLS_PHI = -3.1334076, 3.134037
MIN_CELLS_ETA,MAX_CELLS_ETA = -4.823496, 4.823496




def calo_visualisation(
        inference_array,
        box_eta_cut=1.5,
        cell_significance_cut=2,

    ):
    '''
    Look at calorimeter cells/clusters and plot for visualisation
    '''



    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1, projection='3d')
    # u = np.linspace(0, np.pi/2, 200)
    # v = np.linspace(-1, 1, 200)
    # U, V = np.meshgrid(u, v)

    # radius = 1.0  # Adjust the radius as desired
    # height = 5.0  # Adjust the height as desired

    # X = radius * np.cos(U)
    # Y = V * height
    # Z = radius * np.sin(U)

    # ax.plot_surface(X, Y, Z, alpha=0.2, color='none',ec='black', rstride=4, cstride=4)
    # ax.plot_surface(2.5*np.cos(U), Y, 2.5*np.sin(U), alpha=0.2, color='none',ec='black', fc='none', rstride=25, cstride=25)
    # ax.plot(np.zeros(2),np.array([-7,7]),np.zeros(2),color='red',lw=3)
    # ax.set(xlim=(-1,5),zlim=(-1,5))

    # ax.set(xlabel='X',ylabel='Y',zlabel='Z')
    # plt.show()




    n_clusters_per_box, n_cells_per_box, n_cells2sig_per_box, n_cells4sig_per_box, cluster_significance, box_significance, n_cells15sig_per_box, n_cells1sig_per_box, box_etas, box_areas = list(),list(), list(), list(), list(), list(), list(), list(), list(), list()
    for i in range(len(inference_array)):
        h5f = inference[i]['h5file']
        event_no = inference[i]['event_no']
        if h5f.decode('utf-8')=="01":
            extent_i = inference[i]['extent']
            preds = inference[i]['p_boxes']
            trues = inference[i]['t_boxes']
            scores = inference[i]['p_scores']
            pees = preds[np.where(preds[:,0] != 0)]
            tees = trues[np.where(trues[:,0] != 0)]
            pees = torch.tensor(pees)
            tees = torch.tensor(tees)

            #make boxes cover extent
            tees[:,(0,2)] = (tees[:,(0,2)]*(extent_i[1]-extent_i[0]))+extent_i[0]
            tees[:,(1,3)] = (tees[:,(1,3)]*(extent_i[3]-extent_i[2]))+extent_i[2]

            pees[:,(0,2)] = (pees[:,(0,2)]*(extent_i[1]-extent_i[0]))+extent_i[0]
            pees[:,(1,3)] = (pees[:,(1,3)]*(extent_i[3]-extent_i[2]))+extent_i[2]


            #wrap check
            # pees = wrap_check_NMS(pees,scores,MIN_CELLS_PHI,MAX_CELLS_PHI,threshold=0.2)
            tees = wrap_check_truth(tees,MIN_CELLS_PHI,MAX_CELLS_PHI)


            print(i)
            cells_file = "../../user.cantel.34126190._0000{}.calocellD3PD_mc16_JZ4W.r10788.h5".format(h5f.decode('utf-8'))
            with h5py.File(cells_file,"r") as f:
                h5group = f["caloCells"]
                cells = h5group["2d"][event_no]


            clusters_file = "../../user.cantel.34126190._0000{}.topoclusterD3PD_mc16_JZ4W.r10788.h5".format(h5f.decode('utf-8'))
            with h5py.File(clusters_file,"r") as f:
                cl_data = f["caloCells"] 
                event_data = cl_data["1d"][event_no]
                cluster_data = cl_data["2d"][event_no]
                cluster_cell_data = cl_data["3d"][event_no]    

            print('Making cells plots')
            print(cells.dtype)
                        
                        
            def data_for_cylinder_along_y(center_x,center_z,radius,height_y):
                y = np.linspace(0, height_y, 50)
                theta = np.linspace(0, 2*np.pi, 50)
                theta_grid, y_grid=np.meshgrid(theta, y)
                x_grid = radius*np.cos(theta_grid) + center_x
                z_grid = radius*np.sin(theta_grid) + center_z
                return x_grid,y_grid,z_grid

    
            ##  1. Calorimeter in 3d
            fig = plt.figure(figsize=(12, 8),dpi=100)
            ax1 = fig.add_subplot(111, projection='3d')

            ax1.scatter(cells['cell_xCells'][np.isin(cells['cell_DetCells'],[65,81,97,113])], cells['cell_zCells'][np.isin(cells['cell_DetCells'],[65,81,97,113])], cells['cell_yCells'][np.isin(cells['cell_DetCells'],[65,81,97,113])], color='coral',alpha=0.1,marker='o',s=1,label='EMBar')
            ax1.scatter(cells['cell_xCells'][np.isin(cells['cell_DetCells'],[257,273,289,305])], cells['cell_zCells'][np.isin(cells['cell_DetCells'],[257,273,289,305])], cells['cell_yCells'][np.isin(cells['cell_DetCells'],[257,273,289,305])], color='khaki',alpha=0.25,marker='o',s=1,label='EMEC')
            ax1.scatter(cells['cell_xCells'][np.isin(cells['cell_DetCells'],[145,161])], cells['cell_zCells'][np.isin(cells['cell_DetCells'],[145,161])], cells['cell_yCells'][np.isin(cells['cell_DetCells'],[145,161])], color='olivedrab',alpha=0.4,marker='o',s=1,label='EMIW')
            # ax1.scatter(cells['cell_xCells'][np.isin(cells['cell_DetCells'],[2052])], cells['cell_zCells'][np.isin(cells['cell_DetCells'],[2052])], cells['cell_yCells'][np.isin(cells['cell_DetCells'],[2052])], color='saddlebrown',alpha=0.4,marker='o',s=1,label='EMFCAL')
            ax1.scatter(cells['cell_xCells'][np.isin(cells['cell_DetCells'],[65544,73736,81928])], cells['cell_zCells'][np.isin(cells['cell_DetCells'],[65544,73736,81928])], cells['cell_yCells'][np.isin(cells['cell_DetCells'],[65544,73736,81928])], color='greenyellow',alpha=0.55,marker='o',s=5.5,label='TileBar')
            ax1.scatter(cells['cell_xCells'][np.isin(cells['cell_DetCells'],[131080,139272,147464])], cells['cell_zCells'][np.isin(cells['cell_DetCells'],[131080,139272,147464])], cells['cell_yCells'][np.isin(cells['cell_DetCells'],[131080,139272,147464])], color='slateblue',alpha=0.4,marker='o',s=5.5,label='TileEC')
            ax1.scatter(cells['cell_xCells'][np.isin(cells['cell_DetCells'],[811016,278536,270344])], cells['cell_zCells'][np.isin(cells['cell_DetCells'],[811016,278536,270344])], cells['cell_yCells'][np.isin(cells['cell_DetCells'],[811016,278536,270344])], color='cornflowerblue',alpha=0.4,marker='o',s=1,label='TileGap')
            ax1.scatter(cells['cell_xCells'][np.isin(cells['cell_DetCells'],[2,514,1026,1538])], cells['cell_zCells'][np.isin(cells['cell_DetCells'],[2,514,1026,1538])], cells['cell_yCells'][np.isin(cells['cell_DetCells'],[2,514,1026,1538])], color='darkturquoise',alpha=0.4,marker='o',s=2,label='HEC')
            ax1.scatter(cells['cell_xCells'][np.isin(cells['cell_DetCells'],[2052,4100,6148])], cells['cell_zCells'][np.isin(cells['cell_DetCells'],[2052,4100,6148])], cells['cell_yCells'][np.isin(cells['cell_DetCells'],[2052,4100,6148])], color='black',alpha=0.4,marker='o',s=1,label='FCAL')
            # # ax1.plot(np.zeros(2),np.array([-7000,0]),np.zeros(2),color='black',lw=25)
            # # Xc,Yc,Zc = data_for_cylinder_along_y(0.0,0.0,250,7_000)
            # # ax1.plot_surface(Xc, Yc, Zc, alpha=0.75)
            # # Xc,Yc,Zc = data_for_cylinder_along_y(0.0,0.0,250,-7_000)
            # # ax1.plot_surface(Xc, Yc, Zc, alpha=0.75,color='goldenrod')

            legend_elements = [matplotlib.lines.Line2D([],[], marker='o', color='coral', label='EMBar',linestyle='None',markersize=15),
                               matplotlib.lines.Line2D([],[], marker='o', color='khaki', label='EMEC',linestyle='None',markersize=15),
                               matplotlib.lines.Line2D([],[], marker='o', color='olivedrab', label='EMIW',linestyle='None',markersize=15),
                            #    matplotlib.lines.Line2D([],[], marker='o', color='saddlebrown', label='EMFCAL',linestyle='None',markersize=15),
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
            

            print('EMBar  ',sum(np.isin(cells['cell_DetCells'],[65,81,97,113])))
            print('EMEC   ',sum(np.isin(cells['cell_DetCells'],[257,273,289,305])))
            print('EMIW   ',sum(np.isin(cells['cell_DetCells'],[145,161])))
            print('EMFCAL ',sum(np.isin(cells['cell_DetCells'],[2052])))
            print('TileBar',sum(np.isin(cells['cell_DetCells'],[65544,73736,81928])))
            print('TileEC ',sum(np.isin(cells['cell_DetCells'],[131080,139272,147464])))
            print('TileGap',sum(np.isin(cells['cell_DetCells'],[811016,278536,270344])))
            print('HEC    ',sum(np.isin(cells['cell_DetCells'],[2,514,1026,1538])))
            print('FCAL   ',sum(np.isin(cells['cell_DetCells'],[2052,4100,6148])))
            print('Total  ',len(cells))
            print()
            
            presam_idx = np.isin(cells['cell_DetCells'],[65])
            emb1_idx = np.isin(cells['cell_DetCells'],[81])
            emb2_idx = np.isin(cells['cell_DetCells'],[97])
            emb3_idx = np.isin(cells['cell_DetCells'],[113])
            embar_idx = np.isin(cells['cell_DetCells'],[65,81,97,113])
            emec_idx = np.isin(cells['cell_DetCells'],[257,273,289,305])
            emiw_idx = np.isin(cells['cell_DetCells'],[145,161])
            efcal_idx = np.isin(cells['cell_DetCells'],[2052])
            tileec_idx = np.isin(cells['cell_DetCells'],[131080,139272,147464])
            tilegap_idx = np.isin(cells['cell_DetCells'],[811016,278536,270344])
            hec_idx = np.isin(cells['cell_DetCells'],[2,514,1026,1538])
            fcal_idx = np.isin(cells['cell_DetCells'],[2052,4100,6148])

            presam_cells = cells[presam_idx]
            emb1_cells = cells[emb1_idx]
            emb2_cells = cells[emb2_idx]
            emb3_cells = cells[emb3_idx]
            emec_cells = cells[emec_idx]
            emiw_cells = cells[emiw_idx]
            emfcal_cells = cells[efcal_idx]
            tileec_cells = cells[tileec_idx]
            tilegap_cells = cells[tilegap_idx]
            hec_cells = cells[hec_idx]
            fcal_cells = cells[fcal_idx]
            print('presam   ',len(presam_cells),len(presam_cells[presam_cells['cell_eta']>0]))
            print('emb1   ',len(emb1_cells),len(emb1_cells[emb1_cells['cell_eta']>0]))
            print('emb2   ',len(emb2_cells),len(emb2_cells[emb2_cells['cell_eta']>0]))
            print('emb3   ',len(emb3_cells),len(emb3_cells[emb3_cells['cell_eta']>0]))
            print('EMB',len(presam_cells)+len(emb1_cells)+len(emb2_cells)+len(emb3_cells))
            print('EMEC   ',len(emec_cells),len(emec_cells[emec_cells['cell_eta']>0]))
            print('EMIW   ',len(emiw_cells),len(emiw_cells[emiw_cells['cell_eta']>0]))
            print('EMFCAL   ',len(emfcal_cells),len(emfcal_cells[emfcal_cells['cell_eta']>0]))
            print('TILE EC   ',len(tileec_cells),len(tileec_cells[tileec_cells['cell_eta']>0]))
            print('TILE GAP   ',len(tilegap_cells),len(tilegap_cells[tilegap_cells['cell_eta']>0]))
            print('HEC   ',len(hec_cells),len(hec_cells[hec_cells['cell_eta']>0]))
            print('FCAL   ',len(fcal_cells),len(fcal_cells[fcal_cells['cell_eta']>0]))



            
            fig = plt.figure(figsize=(12, 8),dpi=100)
            ax1 = fig.add_subplot(111, projection='3d')

            ax1.scatter(cells['cell_xCells'][embar_idx], cells['cell_zCells'][embar_idx], cells['cell_yCells'][embar_idx], color='coral',alpha=0.1,marker='o',s=1,label='EMBar')
            # ax1.scatter(cells['cell_xCells'][emec_idx], cells['cell_zCells'][emec_idx], cells['cell_yCells'][emec_idx], color='khaki',alpha=0.25,marker='o',s=1,label='EMEC')
            # ax1.scatter(cells['cell_xCells'][np.isin(cells['cell_DetCells'],[145,161])], cells['cell_zCells'][np.isin(cells['cell_DetCells'],[145,161])], cells['cell_yCells'][np.isin(cells['cell_DetCells'],[145,161])], color='olivedrab',alpha=0.4,marker='o',s=1,label='EMIW')
            # ax1.scatter(cells['cell_xCells'][np.isin(cells['cell_DetCells'],[2052])], cells['cell_zCells'][np.isin(cells['cell_DetCells'],[2052])], cells['cell_yCells'][np.isin(cells['cell_DetCells'],[2052])], color='saddlebrown',alpha=0.4,marker='o',s=1,label='EMFCAL')
            # ax1.scatter(cells['cell_xCells'][np.isin(cells['cell_DetCells'],[65544,73736,81928])], cells['cell_zCells'][np.isin(cells['cell_DetCells'],[65544,73736,81928])], cells['cell_yCells'][np.isin(cells['cell_DetCells'],[65544,73736,81928])], color='greenyellow',alpha=0.55,marker='o',s=5.5,label='TileBar')
            # ax1.scatter(cells['cell_xCells'][np.isin(cells['cell_DetCells'],[131080,139272,147464])], cells['cell_zCells'][np.isin(cells['cell_DetCells'],[131080,139272,147464])], cells['cell_yCells'][np.isin(cells['cell_DetCells'],[131080,139272,147464])], color='slateblue',alpha=0.4,marker='o',s=5.5,label='TileEC')
            # ax1.scatter(cells['cell_xCells'][np.isin(cells['cell_DetCells'],[811016,278536,270344])], cells['cell_zCells'][np.isin(cells['cell_DetCells'],[811016,278536,270344])], cells['cell_yCells'][np.isin(cells['cell_DetCells'],[811016,278536,270344])], color='cornflowerblue',alpha=0.4,marker='o',s=1,label='TileGap')
            # ax1.scatter(cells['cell_xCells'][np.isin(cells['cell_DetCells'],[2,514,1026,1538])], cells['cell_zCells'][np.isin(cells['cell_DetCells'],[2,514,1026,1538])], cells['cell_yCells'][np.isin(cells['cell_DetCells'],[2,514,1026,1538])], color='darkturquoise',alpha=0.4,marker='o',s=2,label='HEC')
            # ax1.scatter(cells['cell_xCells'][np.isin(cells['cell_DetCells'],[2052,4100,6148])], cells['cell_zCells'][np.isin(cells['cell_DetCells'],[2052,4100,6148])], cells['cell_yCells'][np.isin(cells['cell_DetCells'],[2052,4100,6148])], color='black',alpha=0.4,marker='o',s=1,label='FCAL')

            legend_elements = [
                               matplotlib.lines.Line2D([],[], marker='o', color='coral', label='EMBar',linestyle='None',markersize=15),
                            #    matplotlib.lines.Line2D([],[], marker='o', color='khaki', label='EMEC',linestyle='None',markersize=15),
                            #    matplotlib.lines.Line2D([],[], marker='o', color='olivedrab', label='EMIW',linestyle='None',markersize=15),
                            #    matplotlib.lines.Line2D([],[], marker='o', color='saddlebrown', label='EMFCAL',linestyle='None',markersize=15),
                            #    matplotlib.lines.Line2D([],[], marker='o', color='greenyellow', label='TileBar',linestyle='None',markersize=15),
                            #    matplotlib.lines.Line2D([],[], marker='o', color='slateblue', label='TileEC',linestyle='None',markersize=15),
                            #    matplotlib.lines.Line2D([],[], marker='o', color='cornflowerblue', label='TileGap',linestyle='None',markersize=15),
                            #    matplotlib.lines.Line2D([],[], marker='o', color='darkturquoise', label='HEC',linestyle='None',markersize=15),
                            #    matplotlib.lines.Line2D([],[], marker='o', color='black', label='FCAL',linestyle='None',markersize=15)
                               ]
            ax1.legend(handles=legend_elements,frameon=False,bbox_to_anchor=(0.87, 0.3),loc='lower left',prop={'family':'serif','style':'normal','size':12})

            ax1.set(xlabel='X',ylabel='Z',zlabel='Y',xlim=x_lim,ylim=y_lim,zlim=z_lim)
            ax1.axis('off')
            fig.tight_layout()
            plt.show()
            plt.close() 
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            quit()



            
            # fig = go.Figure()
            # fig.add_trace(go.Scatter3d(x=cells['cell_xCells'][np.isin(cells['cell_DetCells'],[65,81,97,113])], y=cells['cell_zCells'][np.isin(cells['cell_DetCells'],[65,81,97,113])], z=cells['cell_yCells'][np.isin(cells['cell_DetCells'],[65,81,97,113])], mode='markers', name='EMBar', marker=dict(color='coral', size=1.0, opacity=0.25)))
            # fig.add_trace(go.Scatter3d(x=cells['cell_xCells'][np.isin(cells['cell_DetCells'],[257,273,289,305])], y=cells['cell_zCells'][np.isin(cells['cell_DetCells'],[257,273,289,305])], z=cells['cell_yCells'][np.isin(cells['cell_DetCells'],[257,273,289,305])], mode='markers', name='EMEC', marker=dict(color='khaki', size=1.5, opacity=0.25)))
            # fig.add_trace(go.Scatter3d(x=cells['cell_xCells'][np.isin(cells['cell_DetCells'],[145,161])], y=cells['cell_zCells'][np.isin(cells['cell_DetCells'],[145,161])], z=cells['cell_yCells'][np.isin(cells['cell_DetCells'],[145,161])], mode='markers', name='EMIW', marker=dict(color='olivedrab', size=2.5, opacity=0.25)))
            # fig.add_trace(go.Scatter3d(x=cells['cell_xCells'][np.isin(cells['cell_DetCells'],[2052])], y=cells['cell_zCells'][np.isin(cells['cell_DetCells'],[2052])], z=cells['cell_yCells'][np.isin(cells['cell_DetCells'],[2052])], mode='markers', name='EMFCAL', marker=dict(color='saddlebrown', size=2.5, opacity=0.25)))
            
            # fig.add_trace(go.Scatter3d(x=cells['cell_xCells'][np.isin(cells['cell_DetCells'],[65544,73736,81928])], y=cells['cell_zCells'][np.isin(cells['cell_DetCells'],[65544,73736,81928])], z=cells['cell_yCells'][np.isin(cells['cell_DetCells'],[65544,73736,81928])], mode='markers', name='TileBar', marker=dict(color='greenyellow', size=2.5, opacity=0.5)))
            # fig.add_trace(go.Scatter3d(x=cells['cell_xCells'][np.isin(cells['cell_DetCells'],[131080,139272,147464])], y=cells['cell_zCells'][np.isin(cells['cell_DetCells'],[131080,139272,147464])], z=cells['cell_yCells'][np.isin(cells['cell_DetCells'],[131080,139272,147464])], mode='markers', name='TileEC', marker=dict(color='slateblue', size=2.5, opacity=0.5)))
            # fig.add_trace(go.Scatter3d(x=cells['cell_xCells'][np.isin(cells['cell_DetCells'],[811016,278536,270344])], y=cells['cell_zCells'][np.isin(cells['cell_DetCells'],[811016,278536,270344])], z=cells['cell_yCells'][np.isin(cells['cell_DetCells'],[811016,278536,270344])], mode='markers', name='TileGap', marker=dict(color='cornflowerblue', size=1.0, opacity=0.5)))
            # fig.add_trace(go.Scatter3d(x=cells['cell_xCells'][np.isin(cells['cell_DetCells'],[2,514,1026,1538])], y=cells['cell_zCells'][np.isin(cells['cell_DetCells'],[2,514,1026,1538])], z=cells['cell_yCells'][np.isin(cells['cell_DetCells'],[2,514,1026,1538])], mode='markers', name='HEC', marker=dict(color='darkturquoise', size=2.0, opacity=0.5)))
            # fig.add_trace(go.Scatter3d(x=cells['cell_xCells'][np.isin(cells['cell_DetCells'],[4100,6148])], y=cells['cell_zCells'][np.isin(cells['cell_DetCells'],[4100,6148])], z=cells['cell_yCells'][np.isin(cells['cell_DetCells'],[4100,6148])], mode='markers', name='HFCAL', marker=dict(color='black', size=1.5, opacity=0.5)))
            # fig.add_trace(go.Scatter3d(x=[0,0], y=[-7000,7000], z=[0,0], mode='lines',name='Beam Line',line=dict(color='red', width=10)))
            # # fig.add_trace(go.Scatter3d(x=[0,0], y=[0,7000], z=[0,0], mode='lines',showlegend=False,line=dict(color='red', width=10)))

            # fig.update_layout(legend=dict(
            #     orientation="h",
            #     yanchor="bottom",
            #     y=1.02,
            #     xanchor="right",
            #     x=1,
            #     itemsizing='constant'))
            # fig.update_layout(scene = dict(
            #     xaxis = dict(visible=False),
            #     yaxis = dict(visible=False),
            #     zaxis =dict(visible=False)))
            # fig.show()


            #2. Calorimeter cell density in eta
            h, bin_edges = np.histogram(cells['cell_eta'], bins=25)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            # f, ax = plt.subplots()
            # ax.hist(cells['cell_eta'],bins=bin_edges,density=True,histtype='step',linewidth=2,color='red',label='Cells')
            # ax.set(xlabel='eta',ylabel='Number of cells')
            # ax.legend()
            # f.tight_layout()
            # plt.show()
            # plt.close()

            emb = cells['cell_eta'][np.isin(cells['cell_DetCells'],[65,81,97,113])]
            emec = cells['cell_eta'][np.isin(cells['cell_DetCells'],[257,273,289,305])]
            emiw = cells['cell_eta'][np.isin(cells['cell_DetCells'],[145,161])]
            emfcal = cells['cell_eta'][np.isin(cells['cell_DetCells'],[2052])]

            tilebar = cells['cell_eta'][np.isin(cells['cell_DetCells'],[65544,73736,81928])]
            tileec = cells['cell_eta'][np.isin(cells['cell_DetCells'],[131080,139272,147464])]
            tilegap = cells['cell_eta'][np.isin(cells['cell_DetCells'],[811016,278536,270344])]
            hec = cells['cell_eta'][np.isin(cells['cell_DetCells'],[2,514,1026,1538])]
            fcal = cells['cell_eta'][np.isin(cells['cell_DetCells'],[2052,4100,6148])]

            # colors = ['coral','khaki','olivedrab','greenyellow','slateblue','cornflowerblue','darkturquoise','black']
            # labels = ['EMB','EMEC','EMIW','TileBar','TileEC','TileGap','HEC','FCAL']

            colors = ['black','darkturquoise','cornflowerblue','slateblue','greenyellow','coral','khaki','olivedrab']
            labels = ['FCAL','HEC','TileGap','TileEC','TileBar','EMB','EMEC','EMIW']
            
            f, ax = plt.subplots(figsize=(7,5))
            # ax.hist([emb,emec,emiw,emfcal,tilebar,tileec,tilegap,hec,hfcal],bins=bin_edges,density=True,stacked=True,histtype='bar',color=colors,alpha=0.95,label=labels)
            # ax.hist([emb,emec,emiw,tilebar,tileec,tilegap,hec,fcal],bins=bin_edges,density=True,stacked=True,histtype='bar',color=colors,alpha=0.95,label=labels)
            ax.hist([fcal,hec,tilegap,tileec,tilebar,emb,emec,emiw],bins=bin_edges,density=False,stacked=True,histtype='bar',color=colors,alpha=0.95,label=labels)

            ax.set_xlabel(f'Pseudorapidity ($\eta$)',fontdict={'family':'serif','color':'k','weight':'normal','size':16})
            ax.set_ylabel(f'Number of Calorimeter Cells',fontdict={'family':'serif','color':'k','weight':'light','style':'normal','size':16})
            ax.legend(frameon=False,bbox_to_anchor=(0.8, 0.2),loc='lower left',prop={'family':'serif','weight':'light','style':'normal','size':12})
            ax.spines[['right', 'top']].set_visible(False)
            ax.set_xticks([-5,-4,-3,-2,-1,0,1,2,3,4,5]) #ax.set_xticks(ax.get_xticks())
            # ax.set_yticks([0.0,0.02,0.04,0.06,0.08]) #
            ax.set_yticks(ax.get_yticks())
            ax.set_xticklabels(ax.get_xticklabels(), fontdict={'family':'serif','size': 14,'weight':'light'})
            ax.set_yticklabels(ax.get_yticklabels(), fontdict={'family':'serif','size': 14})
            # ax.set_ylim((0,0.15))# ax.set_ylim((0,0.095))
            ax.set_xlim((-5.25,5.25))

            f.tight_layout()
            plt.show()
            plt.close()


            h1,_ = np.histogram(emb,bins=bin_edges,density=False)
            h2,_ = np.histogram(emec,bins=bin_edges,density=False)
            h3,_ = np.histogram(emiw,bins=bin_edges,density=False)
            # h4,_ = np.histogram(emfcal,bins=bin_edges,density=False)
            h5,_ = np.histogram(tilebar,bins=bin_edges,density=False)
            h6,_ = np.histogram(tileec,bins=bin_edges,density=False)
            h7,_ = np.histogram(tilegap,bins=bin_edges,density=False)
            h8,_ = np.histogram(hec,bins=bin_edges,density=False)
            h9,_ = np.histogram(fcal,bins=bin_edges,density=False)

            print(h1)
            print(h2)
            # y = np.vstack([h1,h2,h3,h4,h5,h6,h7,h8,h9])
            # f, ax = plt.subplots()
            # ax.stackplot(bin_centers,y,alpha=0.95,labels=labels,colors=colors)
            # ax.set(xlabel='eta',ylabel='Number of cells')
            # ax.legend(frameon=False)
            # ax.spines[['right', 'top']].set_visible(False)
            # f.tight_layout()
            # plt.show()
            # plt.close()

            print('\n\n')

 

            #eta plot
            # f, ax = plt.subplots()
            # ax.stackplot(bin_centers,y/len(cells),alpha=0.95,labels=labels,colors=colors)
            # ax.arrow(0,0,0,0.07,head_width=0.55,head_length=0.0052,fc='grey',ec='grey')
            # ax.annotate('', xy=(0.0,0.04), xytext=(0.0,0.00),arrowprops={'arrowstyle': '->'}, va='center')
            # ax.annotate('', xy=(0.0,0.02), xytext=(0.0,0.0),arrowprops={'arrowstyle': '->', 'lw': 4, 'color': 'blue'},va='center')
            # ax.set_xlabel(f'$\eta$',fontdict={'family':'serif','color':'darkred','weight':'semibold','size':16})
            # ax.set_ylabel(f'% of total cels',fontdict={'family':'serif','color':'lightblue','style':'italic','size':16})
            # ax.legend(frameon=False,bbox_to_anchor=(0.88, 0.2),loc='lower left',prop={'family':'serif','weight':'bold','style':'normal','size':12})
            # ax.spines[['right', 'top']].set_visible(False)
            # ax.set_xticks([-5,-4,-3,-2,-1,0,1,2,3,4,5]) #ax.set_xticks(ax.get_xticks())
            # ax.set_yticks([0.0,0.02,0.04,0.06,0.08]) #ax.set_yticks(ax.get_yticks())
            # ax.set_xticklabels(ax.get_xticklabels(), fontdict={'family':'Arial','size': 12})
            # ax.set_yticklabels(ax.get_yticklabels(), fontdict={'family':'Arial','size': 14})
            # ax.set_ylim((0,0.15))# ax.set_ylim((0,0.095))
            # ax.set_xlim((-5.25,5.25))
            # f.tight_layout()
            # plt.show()
            # plt.close()


            def rainbowarrow(ax, start, end, cmap="viridis", n=50,lw=3):
                cmap = plt.get_cmap(cmap,n)
                # Arrow shaft: LineCollection
                x = np.linspace(start[0],end[0],n)
                y = np.linspace(start[1],end[1],n)
                points = np.array([x,y]).T.reshape(-1,1,2)
                segments = np.concatenate([points[:-1],points[1:]], axis=1)
                lc = matplotlib.collections.LineCollection(segments, cmap=cmap, linewidth=lw)
                lc.set_array(np.linspace(0,1,n))
                ax.add_collection(lc)
                # Arrow head: Triangle
                tricoords = [(0,-0.4),(0.5,0),(0,0.4),(0,-0.4)]
                angle = np.arctan2(end[1]-start[1],end[0]-start[0])
                rot = matplotlib.transforms.Affine2D().rotate(angle)
                tricoords2 = rot.transform(tricoords)
                tri = matplotlib.path.Path(tricoords2, closed=True)
                ax.scatter(end[0],end[1], c=1, s=(2*lw)**2, marker=tri, cmap=cmap,vmin=0)


            # f2, ax2 = plt.subplots()
            # r = 3.5
            # ax2.annotate('', xytext=(0.0,0.0), xy=(r*np.cos(0),r*np.sin(0)),arrowprops={'arrowstyle': 'fancy', 'lw': 0.5, 'color': 'red'},va='center')
            # ax2.text(s=f'$\eta=\inf$',x=r*np.cos(0),y=r*np.sin(0))

            # ax2.annotate('', xytext=(0.0,0.0), xy=(r*np.cos(np.pi/12),r*np.sin(np.pi/12)),arrowprops={'arrowstyle': 'fancy', 'lw': 0.5, 'color': 'red'},va='center')
            # ax2.annotate('', xytext=(0.0,0.0), xy=(r*np.cos(np.pi/6),r*np.sin(np.pi/6)),arrowprops={'arrowstyle': 'fancy', 'lw': 0.5, 'color': 'red'},va='center')
            # ax2.annotate('', xytext=(0.0,0.0), xy=(r*np.cos(np.pi/4),r*np.sin(np.pi/4)),arrowprops={'arrowstyle': 'fancy', 'lw': 0.5, 'color': 'red'},va='center')
            # ax2.annotate('', xytext=(0.0,0.0), xy=(r*np.cos(np.pi/3),r*np.sin(np.pi/3)),arrowprops={'arrowstyle': 'fancy', 'lw': 0.5, 'color': 'red'},va='center')
            # ax2.annotate('', xytext=(0.0,0.0), xy=(r*np.cos(5*np.pi/12),r*np.sin(5*np.pi/12)),arrowprops={'arrowstyle': 'fancy', 'lw': 0.5, 'color': 'red'},va='center')

            # ax2.annotate('', xytext=(0.0,0.0), xy=(r*np.cos(np.pi/2),r*np.sin(np.pi/2)),arrowprops={'arrowstyle': 'fancy', 'lw': 0.5, 'color': 'red'},va='center')
            # ax2.text(s=f'$\eta=0$',x=r*np.cos(np.pi/2),y=r*np.sin(np.pi/2),horizontalalignment='center')

            
            # ax2.annotate('', xytext=(0.0,0.0), xy=(r*np.cos(7*np.pi/12),r*np.sin(7*np.pi/12)),arrowprops={'arrowstyle': 'fancy', 'lw': 0.5, 'color': 'red'},va='center')
            # ax2.annotate('', xytext=(0.0,0.0), xy=(r*np.cos(2*np.pi/3),r*np.sin(2*np.pi/3)),arrowprops={'arrowstyle': 'fancy', 'lw': 0.5, 'color': 'red'},va='center')
            # ax2.annotate('', xytext=(0.0,0.0), xy=(r*np.cos(3*np.pi/4),r*np.sin(3*np.pi/4)),arrowprops={'arrowstyle': 'fancy', 'lw': 0.5, 'color': 'red'},va='center')
            # ax2.annotate('', xytext=(0.0,0.0), xy=(r*np.cos(5*np.pi/6),r*np.sin(5*np.pi/6)),arrowprops={'arrowstyle': 'fancy', 'lw': 0.5, 'color': 'red'},va='center')
            # ax2.annotate('', xytext=(0.0,0.0), xy=(r*np.cos(11*np.pi/12),r*np.sin(11*np.pi/12)),arrowprops={'arrowstyle': 'fancy', 'lw': 0.5, 'color': 'red'},va='center')
            
            # ax2.annotate('', xytext=(0.0,0.0), xy=(r*np.cos(np.pi),r*np.sin(np.pi)),arrowprops={'arrowstyle': 'fancy', 'lw': 0.5, 'color': 'red'},va='center')


            # # ax2.annotate('', xytext=(-3.5,-0.25), xy=(0,-0.25),arrowprops={'arrowstyle': 'simple', 'lw': 3.5, 'color': 'pink'},va='center')
            # # ax2.annotate('', xytext=(3.5,-0.25), xy=(0,-0.25),arrowprops={'arrowstyle': 'simple', 'lw': 3.5, 'color': 'pink'},va='center')
            # rainbowarrow(ax2, (-3.5,-0.25), (-0.15,-0.25), cmap="inferno_r", n=1000,lw=6)
            # rainbowarrow(ax2, (3.5,-0.25), (0.15,-0.25), cmap="inferno_r", n=1000,lw=6)

            # ax2.spines[['bottom','right','left','top']].set_visible(False)
            # ax2.set_xticks([])
            # ax2.set_yticks([])
            # ax2.set(xlim=(-5,5),ylim=(-5,5))
            # f2.tight_layout()
            # plt.show()
            # plt.close()


            # f3, ax3 = plt.subplots()
            # r = 3.5
            # #eta=1
            # theta1 = 2*np.arctan(np.exp(-1))
            # ax3.annotate('', xytext=(0.0,0.0), xy=(r*np.cos(theta1),r*np.sin(theta1)),arrowprops={'arrowstyle': 'fancy', 'lw': 0.5, 'color': 'red'},va='center')
            # ax3.text(s=f'$\eta=1$',x=r*np.cos(theta1),y=r*np.sin(theta1)+0.03,horizontalalignment='center',fontweight='extra bold',color='blue',fontstyle='italic',fontfamily='monospace')
            # theta2 = 2*np.arctan(np.exp(-2))
            # ax3.annotate('', xytext=(0.0,0.0), xy=(r*np.cos(theta2),r*np.sin(theta2)),arrowprops={'arrowstyle': 'fancy', 'lw': 0.5, 'color': 'red'},va='center')
            # ax3.text(s=f'$\eta=2$',x=r*np.cos(theta2)+0.175,y=r*np.sin(theta2)+0.08,horizontalalignment='center')
            # theta3 = 2*np.arctan(np.exp(-3))
            # ax3.annotate('', xytext=(0.0,0.0), xy=(r*np.cos(theta3),r*np.sin(theta3)),arrowprops={'arrowstyle': 'fancy', 'lw': 0.5, 'color': 'red'},va='center')
            # ax3.text(s=f'$\eta=3$',x=r*np.cos(theta3)+0.22,y=r*np.sin(theta3)+0.05,horizontalalignment='center')
            # theta4 = 2*np.arctan(np.exp(-4))
            # ax3.annotate('', xytext=(0.0,0.0), xy=(r*np.cos(theta4),r*np.sin(theta4)),arrowprops={'arrowstyle': 'fancy', 'lw': 0.5, 'color': 'red'},va='center')
            # ax3.text(s=f'$\eta=4$',x=r*np.cos(theta4)+0.26,y=r*np.sin(theta4)-0.05,horizontalalignment='center',style='italic')

            # ax3.annotate('', xytext=(0.0,0.0), xy=(r*np.cos(np.pi/2),r*np.sin(np.pi/2)),arrowprops={'arrowstyle': 'fancy', 'lw': 0.5, 'color': 'red'},va='center')
            # ax3.text(s=f'$\eta=0$',x=r*np.cos(np.pi/2),y=r*np.sin(np.pi/2),horizontalalignment='center')

            # theta_1 = 2*np.arctan(np.exp(1))
            # ax3.annotate('', xytext=(0.0,0.0), xy=(r*np.cos(theta_1),r*np.sin(theta_1)),arrowprops={'arrowstyle': 'fancy', 'lw': 0.5, 'color': 'red'},va='center')
            # ax3.text(s=f'$\eta=\minus 1$',x=r*np.cos(theta_1)+0.1,y=r*np.sin(theta_1)+0.03,horizontalalignment='center',fontweight='heavy',color='blue',style='italic',fontfamily='monospace')
            # theta_2 = 2*np.arctan(np.exp(2))
            # ax3.annotate('', xytext=(0.0,0.0), xy=(r*np.cos(theta_2),r*np.sin(theta_2)),arrowprops={'arrowstyle': 'fancy', 'lw': 0.5, 'color': 'red'},va='center')
            # ax3.text(s=f'$\eta=\minus 2$',x=r*np.cos(theta_2)-0.24,y=r*np.sin(theta_2)+0.08,horizontalalignment='center')
            # theta_3 = 2*np.arctan(np.exp(3))
            # ax3.annotate('', xytext=(0.0,0.0), xy=(r*np.cos(theta_3),r*np.sin(theta_3)),arrowprops={'arrowstyle': 'fancy', 'lw': 0.5, 'color': 'red'},va='center')
            # ax3.text(s=f'$\eta=\minus 3$',x=r*np.cos(theta_3)-0.32,y=r*np.sin(theta_3)+0.035,horizontalalignment='center')
            # theta_4 = 2*np.arctan(np.exp(4))
            # ax3.annotate('', xytext=(0.0,0.0), xy=(r*np.cos(theta_4),r*np.sin(theta_4)),arrowprops={'arrowstyle': 'fancy', 'lw': 0.5, 'color': 'red'},va='center')
            # ax3.text(s=f'$\eta=\minus 4$',x=r*np.cos(theta_4)-0.36,y=r*np.sin(theta_4)-0.06,horizontalalignment='center')

            # # ax3.annotate('', xytext=(0.0,0.0), xy=(r*np.cos(np.pi),r*np.sin(np.pi)),arrowprops={'arrowstyle': 'fancy', 'lw': 0.5, 'color': 'red'},va='center')
            # rainbowarrow(ax3, (-3.5,-0.25), (-0.15,-0.25), cmap="inferno_r", n=1000,lw=6)
            # rainbowarrow(ax3, (3.5,-0.25), (0.15,-0.25), cmap="inferno_r", n=1000,lw=6)

            # ax3.spines[['bottom','right','left','top']].set_visible(False)
            # ax3.set_xticks([])
            # ax3.set_yticks([])
            # ax3.set(xlim=(-5,5),ylim=(-5,5))
            # f3.tight_layout()
            # plt.show()
            # plt.close()




            image_format = "png"
            r = 5.0
            theta1 = 2*np.arctan(np.exp(-1))
            theta2 = 2*np.arctan(np.exp(-2))
            theta3 = 2*np.arctan(np.exp(-3))
            theta4 = 2*np.arctan(np.exp(-4))
            theta_1 = 2*np.arctan(np.exp(1))
            theta_2 = 2*np.arctan(np.exp(2))
            theta_3 = 2*np.arctan(np.exp(3))
            theta_4 = 2*np.arctan(np.exp(4))

            # f3b, ax3b = plt.subplots(1,1,figsize=(8,5))
            # ax3b.annotate('', xytext=(0.0,0.0), xy=(r*np.cos(theta1),r*np.sin(theta1)),arrowprops={'arrowstyle': 'simple', 'lw': 2.5, 'color': 'red'},va='center')
            # ax3b.annotate('', xytext=(0.0,0.0), xy=(r*np.cos(theta2),r*np.sin(theta2)),arrowprops={'arrowstyle': 'simple', 'lw': 2.5, 'color': 'red'},va='center')
            # ax3b.annotate('', xytext=(0.0,0.0), xy=(r*np.cos(theta3),r*np.sin(theta3)),arrowprops={'arrowstyle': 'simple', 'lw': 2.5, 'color': 'red'},va='center')
            # ax3b.annotate('', xytext=(0.0,0.0), xy=(r*np.cos(theta4),r*np.sin(theta4)),arrowprops={'arrowstyle': 'simple', 'lw': 2.5, 'color': 'red'},va='center')
            # ax3b.annotate('', xytext=(0.0,0.0), xy=(0.85*r*np.cos(np.pi/2),0.85*r*np.sin(np.pi/2)),arrowprops={'arrowstyle': 'simple', 'lw': 2.5, 'color': 'red'},va='center')
            # ax3b.annotate('', xytext=(0.0,0.0), xy=(r*np.cos(theta_1),r*np.sin(theta_1)),arrowprops={'arrowstyle': 'simple', 'lw': 2.5, 'color': 'red'},va='center')
            # ax3b.annotate('', xytext=(0.0,0.0), xy=(r*np.cos(theta_2),r*np.sin(theta_2)),arrowprops={'arrowstyle': 'simple', 'lw': 2.5, 'color': 'red'},va='center')
            # ax3b.annotate('', xytext=(0.0,0.0), xy=(r*np.cos(theta_3),r*np.sin(theta_3)),arrowprops={'arrowstyle': 'simple', 'lw': 2.5, 'color': 'red'},va='center')
            # ax3b.annotate('', xytext=(0.0,0.0), xy=(r*np.cos(theta_4),r*np.sin(theta_4)),arrowprops={'arrowstyle': 'simple', 'lw': 2.5, 'color': 'red'},va='center')
            # # ax3b.text(s=f'$\eta=1$',x=r*np.cos(theta1),y=r*np.sin(theta1)+0.03,horizontalalignment='center',fontweight='extra bold',color='blue',fontstyle='italic',fontfamily='monospace')
            # ax3b.text(s=f'$\eta=1$',x=r*np.cos(theta1),y=r*np.sin(theta1)+0.03,horizontalalignment='center',fontdict={'family':'serif','weight':'normal','size':18})
            # ax3b.text(s=f'$\eta=2$',x=r*np.cos(theta2)+0.36,y=r*np.sin(theta2)+0.08,horizontalalignment='center',fontdict={'family':'serif','weight':'normal','size':18})
            # ax3b.text(s=f'$\eta=3$',x=r*np.cos(theta3)+0.44,y=r*np.sin(theta3)+0.08,horizontalalignment='center',fontdict={'family':'serif','weight':'medium','size':18})
            # ax3b.text(s=f'$\eta=4$',x=r*np.cos(theta4)+0.56,y=r*np.sin(theta4)-0.06,horizontalalignment='center',fontdict={'family':'serif','weight':'normal','size':18})
            # ax3b.text(s=f'$\eta=0$',x=0.85*r*np.cos(np.pi/2),y=0.85*r*np.sin(np.pi/2),horizontalalignment='center',fontdict={'family':'serif','weight':'normal','size':18})
            # ax3b.text(s=f'$\eta=\minus 1$',x=r*np.cos(theta_1)+0.1,y=r*np.sin(theta_1)+0.03,horizontalalignment='center',fontdict={'family':'serif','weight':'normal','size':18})
            # ax3b.text(s=f'$\eta=\minus 2$',x=r*np.cos(theta_2)-0.30,y=r*np.sin(theta_2)+0.08,horizontalalignment='center',fontdict={'family':'serif','weight':'normal','size':18})
            # ax3b.text(s=f'$\eta=\minus 3$',x=r*np.cos(theta_3)-0.52,y=r*np.sin(theta_3)+0.08,horizontalalignment='center',fontdict={'family':'serif','weight':'normal','size':18})
            # ax3b.text(s=f'$\eta=\minus 4$',x=r*np.cos(theta_4)-0.66,y=r*np.sin(theta_4)-0.06,horizontalalignment='center',fontdict={'family':'serif','weight':'normal','size':18})
            # ax3b.spines[['bottom','right','left','top']].set_visible(False)
            # ax3b.set_xticks([])
            # ax3b.set_yticks([])
            # ax3b.set(xlim=(-6.25,6.25),ylim=(-0.85,5.2))
            # rainbowarrow(ax3b, (-6.5,-0.35), (-0.5,-0.35), cmap="inferno_r", n=100000,lw=15)
            # rainbowarrow(ax3b, (6.5,-0.35), (0.5,-0.35), cmap="inferno_r", n=100000,lw=15)
            # f3b.tight_layout()
            # f3b.savefig(f'/Users/leonbozianu/work/phd/data/eta_lines.{image_format}',dpi=500,format=image_format,bbox_inches="tight")
            # plt.show()
            # plt.close()





            # f4, ax4 = plt.subplots(1,1,figsize=(8,5),dpi=150)
            # ax4.hist([emb,emec,emiw,tilebar,tileec,tilegap,hec,fcal],bins=bin_edges,stacked=True,histtype='bar',color=colors,alpha=0.95,label=labels)
            
            # ax4.set_xlabel(f'Pseudorapidity ($\eta$)',fontdict={'family':'serif','color':'black','size':18})
            # ax4.set_ylabel(f'Number of Calorimeter Cells',fontdict={'family':'serif', 'color':'black','style':'normal','size':18})
            # ax4.legend(frameon=False,bbox_to_anchor=(0.8, 0.2),loc='lower left',prop={'family':'serif','style':'normal','size':12})
            # ax4.spines[['right', 'top']].set_visible(False)
            # ax4.set_xticks([-5,-4,-3,-2,-1,0,1,2,3,4,5]) #ax4.set_xticks(ax4.get_xticks())
            # ax4.set_yticks(ax4.get_yticks())
            # ax4.set_xticklabels(ax4.get_xticklabels(), fontdict={'family':'serif','size': 14})
            # ax4.set_yticklabels(ax4.get_yticklabels(), fontdict={'family':'serif','size': 14})
            # ax4.set_xlim((-5.25,5.25))
            # f4.tight_layout()
            # f4.savefig(f'/Users/leonbozianu/work/phd/data/ncells_eta2.{image_format}',dpi=500,format=image_format,bbox_inches="tight")
            # plt.show()
            # plt.close()



            # f5, ax5 = plt.subplots(1,1,figsize=(8,5))
            # x = np.random.randn(1000)
            # y = np.random.randn(1000)
            # hist, xedges, yedges = np.histogram2d(x, y, bins=50)
            # plt.imshow(hist.T, cmap='rainbow', aspect='auto', origin='lower')
            # plt.colorbar()
            # plt.xlabel('X')
            # plt.ylabel('Y')
            # plt.show()
            # # ax5.spines[['right', 'top']].set_visible(False)
            # # f5.tight_layout()
            # # f5.savefig(f'/Users/leonbozianu/work/phd/data/ncells_eta2.{image_format}',dpi=500,format=image_format,bbox_inches="tight")
            # plt.show()
            # plt.close()


            Z = np.arange(15, 0, -1)[:, np.newaxis] * np.ones((1, 15))
            np.random.shuffle(Z.flat)
            fig, ax1= plt.subplots(1, 1,figsize=(5,6))
            c = ax1.pcolor(Z, edgecolors='k',cmap='OrRd', linewidths=1.5)
            # ax1.spines[['right', 'top']].set_visible(False)
   
            # circle1 = Ellipse((x1, y1), width=3, height=3, fill=False, edgecolor='black')
            # circle2 = Ellipse((x2, y2), width=4, height=2, fill=False, edgecolor='black')
            # circle3 = Ellipse((x3, y3), width=2, height=4, fill=False, edgecolor='black')

            # # Add the shapes to the plot
            # ax1.add_patch(circle1)
            # ax1.add_patch(circle2)
            # ax1.add_patch(circle3)

            ax1.get_xaxis().set_ticks([])
            ax1.get_yaxis().set_ticks([])
            for side in ax1.spines.keys():  
                ax1.spines[side].set_linewidth(5)
            fig.tight_layout()
            plt.show()


            quit()
            list_truth_cells, list_cl_cells = RetrieveClusterCellsFromBox(cluster_data,cluster_cell_data,cells,tees)
               
            for truth_box_number in range(len(list_truth_cells)):
                truth_box_cells_i = list_truth_cells[truth_box_number]
                cluster_cells_i = list_cl_cells[truth_box_number]

                n_clusters_per_box.append(len(cluster_cells_i))
                n_cells_per_box.append(len(truth_box_cells_i))
                n_cells2sig_per_box.append(len(truth_box_cells_i[abs(truth_box_cells_i['cell_E'] / truth_box_cells_i['cell_Sigma']) >= cell_significance_cut]))
                n_cells4sig_per_box.append(len(truth_box_cells_i[abs(truth_box_cells_i['cell_E'] / truth_box_cells_i['cell_Sigma']) >= 4]))
                n_cells15sig_per_box.append(len(truth_box_cells_i[abs(truth_box_cells_i['cell_E'] / truth_box_cells_i['cell_Sigma']) >= 1.5]))
                n_cells1sig_per_box.append(len(truth_box_cells_i[abs(truth_box_cells_i['cell_E'] / truth_box_cells_i['cell_Sigma']) >= 1]))

                for cl_idx in range(len(cluster_cells_i)):
                    cluster_inside_box = cluster_cells_i[cl_idx]
                    cluster_significance.append(sum(cluster_inside_box['cell_E'] / np.sqrt(sum(cluster_inside_box['cell_Sigma']**2))))
                
                box_etas.append(np.dot(truth_box_cells_i['cell_eta'],np.abs(truth_box_cells_i['cell_E'])) / sum(np.abs(truth_box_cells_i['cell_E'])))
                box_significance.append(sum(truth_box_cells_i['cell_E'] / np.sqrt(sum(truth_box_cells_i['cell_Sigma']**2))))
                box_areas.append((tees[truth_box_number][2]-tees[truth_box_number][0])*(tees[truth_box_number][3]-tees[truth_box_number][1]))







if __name__=="__main__":
    with open("../../struc_array.npy", "rb") as file:
        inference = np.load(file)


    calo_visualisation(
            inference,
            box_eta_cut=1.5,
            cell_significance_cut=2)
        





















