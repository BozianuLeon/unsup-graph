import torch
import torchvision
import torch_geometric
import numpy as np
import numpy.lib.recfunctions as rf
import matplotlib.pyplot as plt
import matplotlib
import h5py
import os
import pandas as pd
import sklearn as sk
import xgboost as xgb

from utils import wrap_check_truth, perpendicular_dists, RetrieveCellIdsFromCluster, RetrieveClusterCellsFromBox

MIN_CELLS_PHI,MAX_CELLS_PHI = -3.1334076, 3.134037
MIN_CELLS_ETA,MAX_CELLS_ETA = -4.823496, 4.823496




def make_bdt_training(
        inference_array,
        box_eta_cut=1.5,
    ):
    '''
    Turn truth (green) boxes into point clouds in pytorch geometric, save these for later training
    '''

    n_clusters_per_box, n_cells_per_box, n_cells2sig_per_box, n_cells4sig_per_box, box_significance, n_cells15sig_per_box, n_cells1sig_per_box, box_etas, box_areas = list(), list(), list(), list(), list(), list(), list(), list(), list()
    for i in range(len(inference_array)):
        h5f = inference_array[i]['h5file']
        event_no = inference_array[i]['event_no']
        if h5f.decode('utf-8')=="01":
            extent_i = inference_array[i]['extent']
            preds = inference_array[i]['p_boxes']
            trues = inference_array[i]['t_boxes']
            scores = inference_array[i]['p_scores']
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

            list_truth_cells, list_cl_cells = RetrieveClusterCellsFromBox(cluster_data,cluster_cell_data,cells,tees)
               
            for truth_box_number in range(len(list_truth_cells)):
                truth_box_cells_i = list_truth_cells[truth_box_number]
                cluster_cells_i = list_cl_cells[truth_box_number]

                # Take boxes in central region
                if (np.abs(np.mean(truth_box_cells_i['cell_eta']))<box_eta_cut):
                    mask = abs(truth_box_cells_i['cell_E'] / truth_box_cells_i['cell_Sigma']) >= 2
                    truth_box_cells_2sig_i = truth_box_cells_i[mask]
                    # Take boxes with >50 2sigma cells
                    if (len(truth_box_cells_2sig_i['cell_eta'])>50):
                        n_clusters_per_box.append(len(cluster_cells_i))
                        n_cells_per_box.append(len(truth_box_cells_i))
                        n_cells2sig_per_box.append(len(truth_box_cells_i[abs(truth_box_cells_i['cell_E'] / truth_box_cells_i['cell_Sigma']) >= 2]))
                        n_cells4sig_per_box.append(len(truth_box_cells_i[abs(truth_box_cells_i['cell_E'] / truth_box_cells_i['cell_Sigma']) >= 4]))
                        n_cells15sig_per_box.append(len(truth_box_cells_i[abs(truth_box_cells_i['cell_E'] / truth_box_cells_i['cell_Sigma']) >= 1.5]))
                        n_cells1sig_per_box.append(len(truth_box_cells_i[abs(truth_box_cells_i['cell_E'] / truth_box_cells_i['cell_Sigma']) >= 1]))

                        box_etas.append(np.dot(truth_box_cells_i['cell_eta'],np.abs(truth_box_cells_i['cell_E'])) / sum(np.abs(truth_box_cells_i['cell_E'])))
                        box_significance.append(sum(truth_box_cells_i['cell_E'] / np.sqrt(sum(truth_box_cells_i['cell_Sigma']**2))))
                        box_areas.append((tees[truth_box_number][2]-tees[truth_box_number][0])*(tees[truth_box_number][3]-tees[truth_box_number][1]))

    df = pd.DataFrame({
        'n_clusters' : n_clusters_per_box,
        'n_cells' : n_cells_per_box,
        'n_2sigcells' : n_cells2sig_per_box,
        'n_4sigcells' : n_cells4sig_per_box,
        'eta' : box_etas,
        'significance' : box_significance,
        'area' : box_areas,
    })
    return df




if __name__=="__main__":

    box_eta_cut=2.1

    with open("../../struc_array.npy", "rb") as file:
        inference = np.load(file)

    if input("Would you like to make/save pandas dataframe (y/n)?") == 'y':
        data_df = make_bdt_training(inference,box_eta_cut)
        # data_df.to_parquet("datasets/overall_stats/bdt_n_cluster.parquet")
        data_df.to_pickle("datasets/overall_stats/bdt_n_cluster.pkl")

    print('Loading in already saved pandas dataframe...')
    # df = pd.read_pickle("datasets/overall_stats/bdt_n_cluster.pkl")
    df = pd.read_pickle("datasets/overall_stats/ALL_bdt_n_cluster.pkl")
    print(df.columns)
    print(df.shape)
    print(df.head)
    labels = df['n_clusters'].to_numpy()
    features = df.drop('n_clusters', axis=1).to_numpy()

    # X_train, X_test, y_train, y_test = sk.model_selection.train_test_split(
        # df.drop('n_clusters', axis=1), df['n_clusters'], test_size=0.2, random_state=2)
    X_train, X_test, y_train, y_test = sk.model_selection.train_test_split(
        features, labels, test_size=0.2, random_state=2)
    
    xgb_train = xgb.DMatrix(X_train, label=y_train)
    xgb_test = xgb.DMatrix(X_test)


    def quartic_error_obj(y_pred, dtrain):
        y_true = dtrain.get_label()
        # error = (y_true - y_pred) ** 4
        grad = -4 * (y_true - y_pred) ** 3  # Gradient of the loss function
        hess = 12 * (y_true - y_pred) ** 2  # Second derivative of the loss function
        return grad, hess

    # Train 
    params = {
        # 'objective': 'reg:squarederror', #pseudohubererror, absoluteerror
        'min_child_weight': 2,
        'max_depth': 7,
        'alpha': 10,
        'learning_rate': 0.1,
        'nfold':5,
        'seed': 0
    }
    xgb_regressor = xgb.train(params=params, dtrain=xgb_train,obj=quartic_error_obj, num_boost_round=45)
    # xgb_regressor = xgb.XGBRegressor(objective='reg:squarederror', random_state=2)
    # xgb_regressor.fit(X_train, y_train)

    # Predict
    y_pred = xgb_regressor.predict(xgb_test)
    # y_pred_series = pd.Series(y_pred, index=X_test.index, name='n_clusters_pred')
    # y_pred_int = pd.Series(np.rint(y_pred), index=X_test.index, name='n_clusters_pred_int')


    # Compare the predicted and actual values (optional)
    # pred_df = pd.concat([y_test,y_pred_int, y_pred_series], axis=1)
    # print(pred_df)
    pred_np = np.column_stack((y_test, np.rint(y_pred), y_pred))
    print(pred_np)
    # for jk in zip(y_test,y_pred_int):
    #     print(jk)


    # Plot prediction against truth
    plt.figure()
    plt.plot(y_test,y_test,ls='--',color='red')
    plt.scatter(x=y_test, y=y_pred)
    plt.xlabel('Num. clusters per box')
    plt.ylabel('XGB Prediction')
    plt.show()

    plt.figure()
    plt.plot(y_test,y_test,ls='--',color='red')
    plt.scatter(x=y_test, y=np.rint(y_pred))
    plt.xlabel('Num. clusters per box')
    plt.ylabel('XGB Prediction (rounded)')
    plt.show()


    # Plot the feature importance
    xgb.plot_importance(xgb_regressor)
    plt.tight_layout()
    plt.show()
    feature_important = xgb_regressor.get_score(importance_type='weight')
    keys = list(feature_important.keys())
    values = list(feature_important.values())
    print(keys)
    print(values)
    print()

    # Plot the tree (needs graphviz)
    # plt.figure()
    # xgb.to_graphviz(xgb_regressor, num_trees=2)
    # plt.tight_layout()
    # plt.show()
    # plt.close()

    features = np.array([[2000, 350, 175, 0.0, 100.0, 0.46]])
    features = np.array([[128, 7, 2, 2.92, 0.52, 0.35]])
    prediction = xgb_regressor.predict(xgb.DMatrix(features))
    print(prediction)

    xgb_regressor.save_model(f'models/xgb_calo_n_clusters.json')
    print('Model saved in ./models/ dir ') 



