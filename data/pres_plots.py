import torch
import torch_geometric

import h5py
import numpy as np
import numpy.lib.recfunctions as rf
import matplotlib.pyplot as plt

import os.path as osp




if __name__ == "__main__":


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



    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(cells4sig['cell_xCells'], cells4sig['cell_zCells'], cells4sig['cell_yCells'], c='b', marker='o', label='Nodes')
    ax.set(xlabel='X',ylabel='Z',zlabel='Y',title=f'Calorimeter Cell Point Cloud')
    # plt.show()
    fig.savefig(f"pyg/data_calo_cells.png", bbox_inches="tight")


    cells_EMBar = cells[np.isin(cells['cell_DetCells'],[65,81,97,113])]
    cells_EMEC = cells[np.isin(cells['cell_DetCells'],[257,273,289,305])]
    cells_EMIW = cells[np.isin(cells['cell_DetCells'],[145,161])]
    cells_EMFCAL = cells[np.isin(cells['cell_DetCells'],[2052])]

    cells_HEC = cells[np.isin(cells['cell_DetCells'],[2,514,1026,1538])]
    cells_HFCAL = cells[np.isin(cells['cell_DetCells'],[4100,6148])]
    cells_TileBar = cells[np.isin(cells['cell_DetCells'],[65544, 73736,81928,])]
    cells_TileEC = cells[np.isin(cells['cell_DetCells'],[131080,139272,147464])]
    cells_TileGap = cells[np.isin(cells['cell_DetCells'],[811016,278536,270344])]

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(cells_EMBar['cell_xCells'],cells_EMBar['cell_zCells'],cells_EMBar['cell_yCells'],s=1.25,marker='.',color='royalblue',alpha=.175,label='EM Bar')
    ax.scatter(cells_EMEC['cell_xCells'],cells_EMEC['cell_zCells'],cells_EMEC['cell_yCells'],s=1.25,marker='.',color='turquoise',alpha=.175,label='EM EC')
    ax.scatter(cells_EMIW['cell_xCells'],cells_EMIW['cell_zCells'],cells_EMIW['cell_yCells'],s=1.25,marker='.',color='springgreen',alpha=.65,label='EM IW')
    ax.scatter(cells_EMFCAL['cell_xCells'],cells_EMFCAL['cell_zCells'],cells_EMFCAL['cell_yCells'],s=1.25,marker='.',color='forestgreen',alpha=.5,label='EM FCAL')
    
    ax.scatter(cells_HEC['cell_xCells'],cells_HEC['cell_zCells'],cells_HEC['cell_yCells'],s=2.0,marker='o',color='tab:orange',alpha=.5,label='HAD EC')
    ax.scatter(cells_HFCAL['cell_xCells'],cells_HFCAL['cell_zCells'],cells_HFCAL['cell_yCells'],s=2.0,marker='o',color='yellow',alpha=.5,label='HAD FCAL')
    ax.scatter(cells_TileBar['cell_xCells'],cells_TileBar['cell_zCells'],cells_TileBar['cell_yCells'],s=2.0,marker='o',color='tomato',alpha=.5,label='HAD Tile Bar')
    ax.scatter(cells_TileEC['cell_xCells'],cells_TileEC['cell_zCells'],cells_TileEC['cell_yCells'],s=2.0,marker='o',color='red',alpha=.5,label='HAD Tile Bar')
    ax.scatter(cells_TileGap['cell_xCells'],cells_TileGap['cell_zCells'],cells_TileGap['cell_yCells'],s=2.0,marker='o',color='peru',alpha=.5,label='HAD Tile Gap')
    fig.savefig('pyg/calo-cells-3d.png',dpi=400, bbox_inches='tight')



    # get cell feature matrix from struct array 
    cell_significance = np.expand_dims(abs(cells2sig['cell_E'] / cells2sig['cell_Sigma']),axis=1)
    cell_features = cells2sig[['cell_xCells','cell_yCells','cell_zCells','cell_eta','cell_phi','cell_E','cell_Sigma','cell_pt']]
    feature_matrix = rf.structured_to_unstructured(cell_features,dtype=np.float32)
    feature_matrix = np.hstack((feature_matrix,cell_significance))
    feature_tensor = torch.tensor(feature_matrix)    

    # get cell IDs,we will also return the cell IDs in the "y" attribute of .Data object
    cell_id_array  = np.expand_dims(cells2sig['cell_IdCells'],axis=1)
    cell_id_tensor = torch.tensor(cell_id_array.astype(np.int64))

    # make sparse adjacency matrix using xyz coords
    edge_indices = torch_geometric.nn.knn_graph(feature_tensor[:,:3],k=3,loop=False)
    eval_graph   = torch_geometric.data.Data(x=feature_tensor[:,[0,1,2,3,4,-1]],edge_index=edge_indices,y=cell_id_tensor) 

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(131, projection='3d')
    ax.scatter(eval_graph.x[:, 0], eval_graph.x[:, 1], eval_graph.x[:, 2], c='b', marker='o', label='Nodes')
    for src, dst in eval_graph.edge_index.t().tolist():
        x_src, y_src, z_src, *feat = eval_graph.x[src]
        x_dst, y_dst, z_dst, *feat = eval_graph.x[dst]
        ax.plot([x_src, x_dst], [y_src, y_dst], [z_src, z_dst], c='r')
    ax.set(xlabel='X',ylabel='Y',zlabel='Z',title=f'Input Graph with KNN 3 Edges')
    # plt.show()
    fig.savefig(f"pyg/data_calo_knn_input.png", bbox_inches="tight")




##########################################################################################################################################################################################

    import plotly.graph_objects as go
    import plotly.io as pio



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
        scene=dict(xaxis_title='X',yaxis_title='Z',zaxis_title='Y',aspectmode='data'),
        width=1400,
        height=1000,  # Matching your 10x10 figure size
        margin=dict(l=100, r=50, b=400, t=0),
        showlegend=True,  # Ensure legend is visible
        legend=dict(yanchor="top",y=0.5,xanchor="right",x=1.05,itemsizing='constant')
    )

    # Save the plot as an HTML file
    html_file_path = 'calorimeter_subdet_3d_plot.html'
    config = {'displayModeBar': True, 'displaylogo': False, 'modeBarButtonsToAdd': ['toggleHover']}
    pio.write_html(fig, file=html_file_path, auto_open=True, include_plotlyjs='cdn', config=config)

    print(f"Plot saved and opened as {html_file_path}")







