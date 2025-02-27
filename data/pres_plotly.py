import torch
import torch_geometric

import h5py
import numpy as np
import numpy.lib.recfunctions as rf
import matplotlib.pyplot as plt

import os.path as osp

import plotly.graph_objects as go
import plotly.io as pio



if __name__ == "__main__":

    save_here = "/Users/leonbozianu/work/phd/graph/dmon/unsup-graph/data/"
    path_to_h5_file = "/Users/leonbozianu/work/phd/graph/dmon/user.cantel.34126190._000001.calocellD3PD_mc16_JZ4W.r10788.h5"
    print("\t",path_to_h5_file)
    f1 = h5py.File(path_to_h5_file,"r")

    kk = 0 # event numb
    cells = f1["caloCells"]["2d"][kk] # event 0 cells
    mask_2sigma = abs(cells['cell_E'] / cells['cell_Sigma']) >= 2
    mask_4sigma = abs(cells['cell_E'] / cells['cell_Sigma']) >= 4
    cells2sig = cells[mask_2sigma]
    cells4sig = cells[mask_4sigma]
    print(cells4sig.dtype)




    cells_EMBar = cells[np.isin(cells['cell_DetCells'],[65,81,97,113])]
    cells_EMEC = cells[np.isin(cells['cell_DetCells'],[257,273,289,305])]
    cells_EMIW = cells[np.isin(cells['cell_DetCells'],[145,161])]
    cells_EMFCAL = cells[np.isin(cells['cell_DetCells'],[2052])]

    cells_HEC = cells[np.isin(cells['cell_DetCells'],[2,514,1026,1538])]
    cells_HFCAL = cells[np.isin(cells['cell_DetCells'],[4100,6148])]
    cells_TileBar = cells[np.isin(cells['cell_DetCells'],[65544, 73736,81928,])]
    cells_TileEC = cells[np.isin(cells['cell_DetCells'],[131080,139272,147464])]
    cells_TileGap = cells[np.isin(cells['cell_DetCells'],[811016,278536,270344])]



    ##########################################################################################################################################################################################


    # Create a 3D scatter plot with Plotly
    fig = go.Figure()

    # Add each scatter trace corresponding to your matplotlib scatter plots
    # EM components (marker style '.')
    fig.add_trace(go.Scatter3d(x=cells_EMBar['cell_xCells'],y=cells_EMBar['cell_zCells'],z=cells_EMBar['cell_yCells'],mode='markers',marker=dict(size=2,color='royalblue',opacity=0.175,symbol='circle'),name='EM Bar'))
    fig.add_trace(go.Scatter3d(x=cells_EMEC['cell_xCells'],y=cells_EMEC['cell_zCells'],z=cells_EMEC['cell_yCells'],mode='markers',marker=dict(size=2,color='turquoise',opacity=0.175,symbol='circle'),name='EM EC'))
    fig.add_trace(go.Scatter3d(x=cells_EMIW['cell_xCells'],y=cells_EMIW['cell_zCells'],z=cells_EMIW['cell_yCells'],mode='markers',marker=dict(size=2,color='springgreen',opacity=0.175,symbol='circle'),name='EM IW'))
    fig.add_trace(go.Scatter3d(x=cells_EMFCAL['cell_xCells'],y=cells_EMFCAL['cell_zCells'],z=cells_EMFCAL['cell_yCells'],mode='markers',marker=dict(size=2,color='springgreen',opacity=0.175,symbol='circle'),name='EM FCAL'))

    # HAD components (marker style 'o')
    fig.add_trace(go.Scatter3d(x=cells_HEC['cell_xCells'],y=cells_HEC['cell_zCells'],z=cells_HEC['cell_yCells'],mode='markers',marker=dict(size=2,color='orange',opacity=0.175,symbol='circle'),name='HAD EC'))
    fig.add_trace(go.Scatter3d(x=cells_HFCAL['cell_xCells'],y=cells_HFCAL['cell_zCells'],z=cells_HFCAL['cell_yCells'],mode='markers',marker=dict(size=2,color='yellow',opacity=0.175,symbol='circle'),name='HAD FCAL'))
    fig.add_trace(go.Scatter3d(x=cells_TileBar['cell_xCells'],y=cells_TileBar['cell_zCells'],z=cells_TileBar['cell_yCells'],mode='markers',marker=dict(size=2,color='tomato',opacity=0.175,symbol='circle'),name='Tile Bar'))
    fig.add_trace(go.Scatter3d(x=cells_TileEC['cell_xCells'],y=cells_TileEC['cell_zCells'],z=cells_TileEC['cell_yCells'],mode='markers',marker=dict(size=2,color='red',opacity=0.175,symbol='circle'),name='Tile EC'))
    fig.add_trace(go.Scatter3d(x=cells_TileGap['cell_xCells'],y=cells_TileGap['cell_zCells'],z=cells_TileGap['cell_yCells'],mode='markers',marker=dict(size=2,color='peru',opacity=0.175,symbol='circle'),name='Tile Gap'))


    # Update the layout
    fig.update_layout(
        title={'text': 'All calo. cells','y':0.95,'x':0.5,'xanchor': 'center','yanchor': 'top'},
        scene=dict(xaxis_title='X',yaxis_title='Z',zaxis_title='Y',aspectmode='data'),
        width=1400,
        height=1000,  # Matching your 10x10 figure size
        margin=dict(l=100, r=50, b=400, t=0),
        showlegend=True,  # Ensure legend is visible
        legend=dict(yanchor="top",y=0.5,xanchor="right",x=1.05,itemsizing='constant')
    )

    # Save the plot as an HTML file
    html_file_path = save_here + 'calorimeter_subdet_3d_plot.html'
    config = {'displayModeBar': True, 'displaylogo': False, 'modeBarButtonsToAdd': ['toggleHover']}
    pio.write_html(fig, file=html_file_path, auto_open=True, include_plotlyjs='cdn', config=config)

    print(f"Plot saved and opened as {html_file_path}")




    ##########################################################################################################################################################################################

    cells_EMBar = cells2sig[np.isin(cells2sig['cell_DetCells'],[65,81,97,113])]
    cells_EMEC = cells2sig[np.isin(cells2sig['cell_DetCells'],[257,273,289,305])]
    cells_EMIW = cells2sig[np.isin(cells2sig['cell_DetCells'],[145,161])]
    cells_EMFCAL = cells2sig[np.isin(cells2sig['cell_DetCells'],[2052])]

    cells_HEC = cells2sig[np.isin(cells2sig['cell_DetCells'],[2,514,1026,1538])]
    cells_HFCAL = cells2sig[np.isin(cells2sig['cell_DetCells'],[4100,6148])]
    cells_TileBar = cells2sig[np.isin(cells2sig['cell_DetCells'],[65544, 73736,81928,])]
    cells_TileEC = cells2sig[np.isin(cells2sig['cell_DetCells'],[131080,139272,147464])]
    cells_TileGap = cells2sig[np.isin(cells2sig['cell_DetCells'],[811016,278536,270344])]

    # Create a 3D scatter plot with Plotly
    fig = go.Figure()

    # Add each scatter trace corresponding to your matplotlib scatter plots
    # EM components (marker style '.')
    fig.add_trace(go.Scatter3d(x=cells_EMBar['cell_xCells'],y=cells_EMBar['cell_zCells'],z=cells_EMBar['cell_yCells'],mode='markers',marker=dict(size=4,color='royalblue',opacity=0.3,symbol='circle'),name='EM Bar'))
    fig.add_trace(go.Scatter3d(x=cells_EMEC['cell_xCells'],y=cells_EMEC['cell_zCells'],z=cells_EMEC['cell_yCells'],mode='markers',marker=dict(size=4,color='turquoise',opacity=0.3,symbol='circle'),name='EM EC'))
    fig.add_trace(go.Scatter3d(x=cells_EMIW['cell_xCells'],y=cells_EMIW['cell_zCells'],z=cells_EMIW['cell_yCells'],mode='markers',marker=dict(size=4,color='springgreen',opacity=0.3,symbol='circle'),name='EM IW'))
    fig.add_trace(go.Scatter3d(x=cells_EMFCAL['cell_xCells'],y=cells_EMFCAL['cell_zCells'],z=cells_EMFCAL['cell_yCells'],mode='markers',marker=dict(size=4,color='springgreen',opacity=0.3,symbol='circle'),name='EM FCAL'))

    # HAD components (marker style 'o')
    fig.add_trace(go.Scatter3d(x=cells_HEC['cell_xCells'],y=cells_HEC['cell_zCells'],z=cells_HEC['cell_yCells'],mode='markers',marker=dict(size=4,color='orange',opacity=0.3,symbol='circle'),name='HAD EC'))
    fig.add_trace(go.Scatter3d(x=cells_HFCAL['cell_xCells'],y=cells_HFCAL['cell_zCells'],z=cells_HFCAL['cell_yCells'],mode='markers',marker=dict(size=4,color='yellow',opacity=0.3,symbol='circle'),name='HAD FCAL'))
    fig.add_trace(go.Scatter3d(x=cells_TileBar['cell_xCells'],y=cells_TileBar['cell_zCells'],z=cells_TileBar['cell_yCells'],mode='markers',marker=dict(size=4,color='tomato',opacity=0.3,symbol='circle'),name='Tile Bar'))
    fig.add_trace(go.Scatter3d(x=cells_TileEC['cell_xCells'],y=cells_TileEC['cell_zCells'],z=cells_TileEC['cell_yCells'],mode='markers',marker=dict(size=4,color='red',opacity=0.3,symbol='circle'),name='Tile EC'))
    fig.add_trace(go.Scatter3d(x=cells_TileGap['cell_xCells'],y=cells_TileGap['cell_zCells'],z=cells_TileGap['cell_yCells'],mode='markers',marker=dict(size=4,color='peru',opacity=0.3,symbol='circle'),name='Tile Gap'))


    # Update the layout
    fig.update_layout(
        title={'text': 'Cell |signif.|>2 calo. cells','y':0.95,'x':0.5,'xanchor': 'center','yanchor': 'top'},
        scene=dict(xaxis_title='X',yaxis_title='Z',zaxis_title='Y',aspectmode='data'),
        width=1400,
        height=1000,  # Matching your 10x10 figure size
        margin=dict(l=100, r=50, b=400, t=0),
        showlegend=True,  # Ensure legend is visible
        legend=dict(yanchor="top",y=0.5,xanchor="right",x=1.05,itemsizing='constant')
    )

    # Save the plot as an HTML file
    html_file_path = save_here + 'calo_2sig_subdet_3d_plot.html'
    config = {'displayModeBar': True, 'displaylogo': False, 'modeBarButtonsToAdd': ['toggleHover']}
    pio.write_html(fig, file=html_file_path, auto_open=True, include_plotlyjs='cdn', config=config)

    print(f"Plot saved and opened as {html_file_path}")

    ##########################################################################################################################################################################################

    cells_EMBar = cells4sig[np.isin(cells4sig['cell_DetCells'],[65,81,97,113])]
    cells_EMEC = cells4sig[np.isin(cells4sig['cell_DetCells'],[257,273,289,305])]
    cells_EMIW = cells4sig[np.isin(cells4sig['cell_DetCells'],[145,161])]
    cells_EMFCAL = cells4sig[np.isin(cells4sig['cell_DetCells'],[2052])]

    cells_HEC = cells4sig[np.isin(cells4sig['cell_DetCells'],[2,514,1026,1538])]
    cells_HFCAL = cells4sig[np.isin(cells4sig['cell_DetCells'],[4100,6148])]
    cells_TileBar = cells4sig[np.isin(cells4sig['cell_DetCells'],[65544, 73736,81928,])]
    cells_TileEC = cells4sig[np.isin(cells4sig['cell_DetCells'],[131080,139272,147464])]
    cells_TileGap = cells4sig[np.isin(cells4sig['cell_DetCells'],[811016,278536,270344])]

    # Create a 3D scatter plot with Plotly
    fig = go.Figure()

    # Add each scatter trace corresponding to your matplotlib scatter plots
    # EM components (marker style '.')
    fig.add_trace(go.Scatter3d(x=cells_EMBar['cell_xCells'],y=cells_EMBar['cell_zCells'],z=cells_EMBar['cell_yCells'],mode='markers',marker=dict(size=6,color='royalblue',opacity=0.5,symbol='circle'),name='EM Bar'))
    fig.add_trace(go.Scatter3d(x=cells_EMEC['cell_xCells'],y=cells_EMEC['cell_zCells'],z=cells_EMEC['cell_yCells'],mode='markers',marker=dict(size=6,color='turquoise',opacity=0.5,symbol='circle'),name='EM EC'))
    fig.add_trace(go.Scatter3d(x=cells_EMIW['cell_xCells'],y=cells_EMIW['cell_zCells'],z=cells_EMIW['cell_yCells'],mode='markers',marker=dict(size=6,color='springgreen',opacity=0.5,symbol='circle'),name='EM IW'))
    fig.add_trace(go.Scatter3d(x=cells_EMFCAL['cell_xCells'],y=cells_EMFCAL['cell_zCells'],z=cells_EMFCAL['cell_yCells'],mode='markers',marker=dict(size=6,color='springgreen',opacity=0.5,symbol='circle'),name='EM FCAL'))

    # HAD components (marker style 'o')
    fig.add_trace(go.Scatter3d(x=cells_HEC['cell_xCells'],y=cells_HEC['cell_zCells'],z=cells_HEC['cell_yCells'],mode='markers',marker=dict(size=6,color='orange',opacity=0.5,symbol='circle'),name='HAD EC'))
    fig.add_trace(go.Scatter3d(x=cells_HFCAL['cell_xCells'],y=cells_HFCAL['cell_zCells'],z=cells_HFCAL['cell_yCells'],mode='markers',marker=dict(size=6,color='yellow',opacity=0.5,symbol='circle'),name='HAD FCAL'))
    fig.add_trace(go.Scatter3d(x=cells_TileBar['cell_xCells'],y=cells_TileBar['cell_zCells'],z=cells_TileBar['cell_yCells'],mode='markers',marker=dict(size=6,color='tomato',opacity=0.5,symbol='circle'),name='Tile Bar'))
    fig.add_trace(go.Scatter3d(x=cells_TileEC['cell_xCells'],y=cells_TileEC['cell_zCells'],z=cells_TileEC['cell_yCells'],mode='markers',marker=dict(size=6,color='red',opacity=0.5,symbol='circle'),name='Tile EC'))
    fig.add_trace(go.Scatter3d(x=cells_TileGap['cell_xCells'],y=cells_TileGap['cell_zCells'],z=cells_TileGap['cell_yCells'],mode='markers',marker=dict(size=6,color='peru',opacity=0.5,symbol='circle'),name='Tile Gap'))


    # Update the layout
    fig.update_layout(
        title={'text': 'Cell |signif.|>4 calo. cells','y':0.95,'x':0.5,'xanchor': 'center','yanchor': 'top'},
        scene=dict(xaxis_title='X',yaxis_title='Z',zaxis_title='Y',aspectmode='data'),
        width=1400,
        height=1000,  # Matching your 10x10 figure size
        margin=dict(l=100, r=50, b=400, t=0),
        showlegend=True,  # Ensure legend is visible
        legend=dict(yanchor="top",y=0.5,xanchor="right",x=1.05,itemsizing='constant')
    )

    # Save the plot as an HTML file
    html_file_path = save_here + 'calo_4sig_subdet_3d_plot.html'
    config = {'displayModeBar': True, 'displaylogo': False, 'modeBarButtonsToAdd': ['toggleHover']}
    pio.write_html(fig, file=html_file_path, auto_open=True, include_plotlyjs='cdn', config=config)

    print(f"Plot saved and opened as {html_file_path}")



    ##########################################################################################################################################################################################


    # get cell feature matrix from struct array 
    cell_significance = np.expand_dims(abs(cells2sig['cell_E'] / cells2sig['cell_Sigma']),axis=1)
    cell_features = cells2sig[['cell_xCells','cell_yCells','cell_zCells','cell_eta','cell_phi','cell_E','cell_Sigma','cell_pt']]
    feature_matrix = rf.structured_to_unstructured(cell_features,dtype=np.float32)
    feature_matrix = np.hstack((feature_matrix,cell_significance))
    feature_tensor = torch.tensor(feature_matrix)    

    # get cell IDs,we will also return the cell IDs in the "y" attribute of .Data object
    cell_id_array  = np.expand_dims(cells2sig['cell_IdCells'],axis=1)
    cell_id_tensor = torch.tensor(cell_id_array.astype(np.int64))


    ##########################################################################################################################################################################################


    # make sparse adjacency matrix using KNN graph
    edge_indices = torch_geometric.nn.knn_graph(feature_tensor[:,:3],k=3,loop=False)
    eval_graph   = torch_geometric.data.Data(x=feature_tensor[:,[0,1,2,3,4,-1]],edge_index=edge_indices,y=cell_id_tensor) 

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=eval_graph.x[:, 0],y=eval_graph.x[:, 2],z=eval_graph.x[:, 1],mode='markers',marker=dict(size=2.5,color='blue',opacity=0.8,symbol='circle'),name='Nodes'))

    edge_x,edge_y,edge_z = [],[],[]
    # For each edge, add the coordinates of both nodes and None to create a break
    for src, dst in eval_graph.edge_index.t().tolist():
        x_src, y_src, z_src, *feat = eval_graph.x[src]
        x_dst, y_dst, z_dst, *feat = eval_graph.x[dst]
        edge_x.extend([x_src, x_dst, None])
        edge_y.extend([y_src, y_dst, None])
        edge_z.extend([z_src, z_dst, None])

    # Add the edges as a single line trace
    fig.add_trace(go.Scatter3d(x=edge_x,y=edge_z,z=edge_y,mode='lines',line=dict(color='red',width=1),opacity=0.5,name='Edges',hoverinfo='none'))

    # Update the layout
    fig.update_layout(
        title={'text': 'Input (|s| > 2) cell point cloud KNN 3 Edges','y':0.95,'x':0.5,'xanchor': 'center','yanchor': 'top'},
        scene=dict(xaxis_title='X',yaxis_title='Y',zaxis_title='Z',aspectmode='data'),
        width=1400,
        height=1000,
        margin=dict(l=100, r=50, b=200, t=0),
        showlegend=True,
        legend=dict(yanchor="top",y=0.5,xanchor="right",x=0.99)
        )

    html_file_path = save_here + 'calo_2sig_knn_edges_3d-plot.html'
    config = {'displayModeBar': True,'displaylogo': False}
    pio.write_html(fig, file=html_file_path, auto_open=True, include_plotlyjs='cdn', config=config)

    print(f"Plot saved and opened as {html_file_path}")

    ##########################################################################################################################################################################################

    # make sparse adjacency matrix using RADIUS graph
    edge_indices = torch_geometric.nn.radius_graph(feature_tensor[:,:3],r=250.0,loop=False)
    eval_graph   = torch_geometric.data.Data(x=feature_tensor[:,[0,1,2,3,4,-1]],edge_index=edge_indices,y=cell_id_tensor) 

    fig_rad = go.Figure()
    fig_rad.add_trace(go.Scatter3d(x=eval_graph.x[:, 0],y=eval_graph.x[:, 2],z=eval_graph.x[:, 1],mode='markers',marker=dict(size=2.5,color='blue',opacity=0.8,symbol='circle'),name='Nodes'))

    rad_edge_x,rad_edge_y,rad_edge_z = [],[],[]
    # For each edge, add the coordinates of both nodes and None to create a break
    for src, dst in eval_graph.edge_index.t().tolist():
        x_src, y_src, z_src, *feat = eval_graph.x[src]
        x_dst, y_dst, z_dst, *feat = eval_graph.x[dst]
        rad_edge_x.extend([x_src, x_dst, None])
        rad_edge_y.extend([y_src, y_dst, None])
        rad_edge_z.extend([z_src, z_dst, None])

    # Add the edges as a single line trace
    fig_rad.add_trace(go.Scatter3d(x=rad_edge_x,y=rad_edge_z,z=rad_edge_y,mode='lines',line=dict(color='red',width=1),opacity=0.5,name='Edges',hoverinfo='none'))

    # Update the layout
    fig_rad.update_layout(
        title={'text': 'Input (|s| > 2) cell point cloud RAD 250(xyz) Edges','y':0.95,'x':0.5,'xanchor': 'center','yanchor': 'top'},
        scene=dict(xaxis_title='X',yaxis_title='Y',zaxis_title='Z',aspectmode='data'),
        width=1400,
        height=1000,
        margin=dict(l=100, r=50, b=200, t=0),
        showlegend=True,
        legend=dict(yanchor="top",y=0.5,xanchor="right",x=0.99)
        )

    html_file_path = save_here + 'calo_2sig_rad_edges_3d-plot.html'
    config = {'displayModeBar': True,'displaylogo': False}
    pio.write_html(fig_rad, file=html_file_path, auto_open=True, include_plotlyjs='cdn', config=config)

    print(f"Plot saved and opened as {html_file_path}")


    ##########################################################################################################################################################################################

    # make sparse adjacency matrix using BUCKET graph
    ##      ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    neighbours_array = cell_neighbours = np.load(save_here+'/pyg/cell_neighbours.npy')
    src_neighbours_array = src_cell_neighbours = np.load(save_here+'/pyg/src_cell_neighbours.npy')

    # get cell IDs
    cell_ids_2 = np.array(cells2sig['cell_IdCells'].astype(int)) # THIS IS THE GRAND LIST OF CELLS WE CAN USE IN THIS EVENT
    cell_ids_2_array = np.expand_dims(cell_ids_2,axis=1) # we will also return the cell IDs in the "y" attribute pytorch geometric
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
    eval_graph   = torch_geometric.data.Data(x=feature_tensor[:,[0,1,2,3,4,-1]],edge_index=torch.tensor(edge_indices),y=cell_id_tensor) 
    ##      ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    fig_rad = go.Figure()
    fig_rad.add_trace(go.Scatter3d(x=eval_graph.x[:, 0],y=eval_graph.x[:, 2],z=eval_graph.x[:, 1],mode='markers',marker=dict(size=2.5,color='blue',opacity=0.8,symbol='circle'),name='Nodes'))

    rad_edge_x,rad_edge_y,rad_edge_z = [],[],[]
    # For each edge, add the coordinates of both nodes and None to create a break
    for src, dst in eval_graph.edge_index.t().tolist():
        x_src, y_src, z_src, *feat = eval_graph.x[src]
        x_dst, y_dst, z_dst, *feat = eval_graph.x[dst]
        rad_edge_x.extend([x_src, x_dst, None])
        rad_edge_y.extend([y_src, y_dst, None])
        rad_edge_z.extend([z_src, z_dst, None])

    # Add the edges as a single line trace
    fig_rad.add_trace(go.Scatter3d(x=rad_edge_x,y=rad_edge_z,z=rad_edge_y,mode='lines',line=dict(color='red',width=0.6),opacity=0.175,name='Edges',hoverinfo='none'))

    # Update the layout
    fig_rad.update_layout(
        title={'text': 'Input (|s| > 2) cell point cloud BUCKET (eta-phi) Edges','y':0.95,'x':0.5,'xanchor': 'center','yanchor': 'top'},
        scene=dict(xaxis_title='X',yaxis_title='Y',zaxis_title='Z',aspectmode='data'),
        width=1400,
        height=1000,
        margin=dict(l=100, r=50, b=200, t=0),
        showlegend=True,
        legend=dict(yanchor="top",y=0.5,xanchor="right",x=0.99)
        )

    html_file_path = save_here + 'calo_2sig_bucket_edges_3d-plot.html'
    config = {'displayModeBar': True,'displaylogo': False}
    pio.write_html(fig_rad, file=html_file_path, auto_open=True, include_plotlyjs='cdn', config=config)

    print(f"Plot saved and opened as {html_file_path}")











    ##########################################################################################################################################################################################


    # make scatter plot of topoclusters
    path_to_topocl_file = "/Users/leonbozianu/work/phd/graph/dmon/user.cantel.34126190._000001.topoclusterD3PD_mc16_JZ4W.r10788.h5"
    print("\t",path_to_topocl_file)
    cl_file = h5py.File(path_to_topocl_file,"r")
    cl_data = cl_file["caloCells"] 
    event_data = cl_data["1d"][kk]
    cluster_data = cl_data["2d"][kk]
    cl_cell_data = cl_data["3d"][kk]


    fig = go.Figure()
    for i in range(len(cl_cell_data)):
        fig.add_trace(go.Scatter3d(x=cl_cell_data[i]['cl_cell_xCells'],y=cl_cell_data[i]['cl_cell_zCells'],z=cl_cell_data[i]['cl_cell_yCells'],mode='markers',marker=dict(size=3.0,opacity=0.5,symbol='circle'),name=f'Cluster {i}'))

    fig.update_layout(
        # title={'text': f'All calo. cells in Topoclusters ({len(cl_cell_data)})','y':0.98,'x':0.5,'xanchor': 'center','yanchor': 'top'},
        title={'text': f'All calo. cells in Topoclusters','y':0.98,'x':0.5,'xanchor': 'center','yanchor': 'top'},
        scene=dict(xaxis_title='X',yaxis_title='Z',zaxis_title='Y',),
        width=1400,
        height=1000,
        margin=dict(l=100, r=50, b=300, t=0),
    )

    html_file_path = save_here + 'calo_2sig_topocl_3d_plot.html'
    pio.write_html(fig, file=html_file_path, auto_open=True)
    print(f"Plot saved and opened as {html_file_path}")



