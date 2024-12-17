import torch
import torchvision
import torch_geometric
import h5py
import os
import pickle
import numpy as np
import numpy.lib.recfunctions as rf
import matplotlib.pyplot as plt


from utils import wrap_check_truth, circular_mean, RetrieveCellIdsFromCluster, RetrieveClusterCellsFromBox

MIN_CELLS_PHI,MAX_CELLS_PHI = -3.1334076, 3.134037
MIN_CELLS_ETA,MAX_CELLS_ETA = -4.823496, 4.823496








def inspect_calo_data_list(
        inference_array,
        box_eta_cut=1.5,
        cell_significance_cut=2,
        norm=False,
        k=3,    

        #plotting:
        save_loc='',
        n_plots=2,
    ):
    '''
    Turn truth (green) boxes into point clouds in pytorch geometric, save these for later training
    '''

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


                # if np.abs(np.mean(truth_box_cells_i['cell_eta']))<box_eta_cut:
                #     mask = abs(truth_box_cells_i['cell_E'] / truth_box_cells_i['cell_Sigma']) >= cell_significance_cut
                #     truth_box_cells_2sig_i = truth_box_cells_i[mask]
                #     truth_box_cells_i = truth_box_cells_2sig_i


                        
    with open('datasets/overall_stats/n_clusters_per_box.pkl', 'wb') as f1:
        pickle.dump(n_clusters_per_box, f1)
                
    with open('datasets/overall_stats/n_cells_per_box.pkl', 'wb') as f1:
        pickle.dump(n_cells_per_box, f1)
                
    with open('datasets/overall_stats/n_cells2sig_per_box.pkl', 'wb') as f1:
        pickle.dump(n_cells2sig_per_box, f1)
                
    with open('datasets/overall_stats/n_cells4sig_per_box.pkl', 'wb') as f1:
        pickle.dump(n_cells4sig_per_box, f1)
                
    with open('datasets/overall_stats/n_cells15sig_per_box.pkl', 'wb') as f1:
        pickle.dump(n_cells15sig_per_box, f1)
                
    with open('datasets/overall_stats/n_cells1sig_per_box.pkl', 'wb') as f1:
        pickle.dump(n_cells1sig_per_box, f1)
                
    with open('datasets/overall_stats/cluster_significance.pkl', 'wb') as f1:
        pickle.dump(cluster_significance, f1)
                
    with open('datasets/overall_stats/box_significance.pkl', 'wb') as f1:
        pickle.dump(box_significance, f1)
                
    with open('datasets/overall_stats/box_etas.pkl', 'wb') as f1:
        pickle.dump(box_etas, f1)
                
    with open('datasets/overall_stats/box_areas.pkl', 'wb') as f1:
        pickle.dump(box_areas, f1)

    return n_clusters_per_box, n_cells_per_box, n_cells2sig_per_box, cluster_significance, box_significance

                # if os.path.exists(f"../plots/{truth_box_number}/") is False: os.makedirs(f"../plots/{truth_box_number}/")







if __name__=="__main__":
    with open("../../struc_array.npy", "rb") as file:
        inference = np.load(file)


    # a,b,c,d,e = inspect_calo_data_list(inference,
    #                             box_eta_cut=1.5,
    #                             cell_significance_cut=2)
    # quit()
        



    
    with open(f'datasets/overall_stats/n_clusters_per_box.pkl', 'rb') as f:
       n_clusters_per_box = pickle.load(f)
    
    with open(f'datasets/overall_stats/n_cells_per_box.pkl', 'rb') as f:
       n_cells_per_box = pickle.load(f)
    
    with open(f'datasets/overall_stats/n_cells2sig_per_box.pkl', 'rb') as f:
       n_cells2sig_per_box = pickle.load(f)
    
    with open(f'datasets/overall_stats/n_cells4sig_per_box.pkl', 'rb') as f:
       n_cells4sig_per_box = pickle.load(f)
    
    with open(f'datasets/overall_stats/n_cells15sig_per_box.pkl', 'rb') as f:
       n_cells15sig_per_box = pickle.load(f)
    
    with open(f'datasets/overall_stats/n_cells1sig_per_box.pkl', 'rb') as f:
       n_cells1sig_per_box = pickle.load(f)
    
    with open(f'datasets/overall_stats/cluster_significance.pkl', 'rb') as f:
       cluster_significance = pickle.load(f)
    
    with open(f'datasets/overall_stats/box_significance.pkl', 'rb') as f:
       box_significance = pickle.load(f)
    
    with open(f'datasets/overall_stats/box_etas.pkl', 'rb') as f:
       box_etas = pickle.load(f)
    
    with open(f'datasets/overall_stats/box_areas.pkl', 'rb') as f:
       box_areas = pickle.load(f)
    



    
    n_clusters_per_box = np.array(n_clusters_per_box)
    plt.figure()
    plt.hist(n_clusters_per_box,histtype='step',bins=max(n_clusters_per_box),color='purple',lw=2)
    plt.xlabel('N clusters per box')
    plt.yscale('log')
    plt.grid(color="0.95")
    plt.title("Cluster-Box Merging Test Set")
    plt.show()
    plt.close()

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #### ETA PLOTS:
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    eta_plots = False
    if eta_plots:
        box_etas = np.array(box_etas)
        etabin1 = n_clusters_per_box[np.abs(box_etas)<=1.0]
        etabin2 = n_clusters_per_box[(np.abs(box_etas)>1.0) & (np.abs(box_etas)<1.5)]
        etabin3 = n_clusters_per_box[(np.abs(box_etas)>1.5) & (np.abs(box_etas)<2.5)]
        etabin4 = n_clusters_per_box[(np.abs(box_etas)>2.5) & (np.abs(box_etas)<3.0)]
        etabin5 = n_clusters_per_box[np.abs(box_etas)>3.0]
        print(max(n_clusters_per_box),max(etabin1),max(etabin2),max(etabin3),max(etabin4),max(etabin5))
        print(len(n_clusters_per_box),len(etabin1),len(etabin2),len(etabin3),len(etabin4),len(etabin5))
        fig,ax = plt.subplots(1,1,figsize=(8,5))
        bins = range(0,max(n_clusters_per_box)+1)
        ax.hist(etabin1,histtype='step',bins=bins,lw=1.5,alpha=0.7,label='|eta|<=1.0')
        ax.hist(etabin2,histtype='step',bins=bins,lw=1.5,alpha=0.7,label='1.0<|eta|<=1.5')
        ax.hist(etabin3,histtype='step',bins=bins,lw=1.5,alpha=0.7,label='1.5<|eta|<=2.5')
        ax.hist(etabin4,histtype='step',bins=bins,lw=1.5,alpha=0.7,label='2.5<|eta|<=3.0')
        ax.hist(etabin5,histtype='step',bins=bins,lw=1.5,alpha=0.7,label='3.0<|eta|')
        ax.legend()
        ax.set(xlabel='n clusters per box',yscale='log')
        fig.tight_layout()
        plt.show()
        plt.close()

        plt.figure()
        plt.errorbar(0.5,np.mean(etabin1),yerr=np.std(etabin1),xerr=0.5)
        plt.errorbar(1.25,np.mean(etabin2),yerr=np.std(etabin2),xerr=0.25)
        plt.errorbar(2.0,np.mean(etabin3),yerr=np.std(etabin3),xerr=0.5)
        plt.errorbar(2.75,np.mean(etabin4),yerr=np.std(etabin4),xerr=0.25)
        plt.errorbar(3.5,np.mean(etabin5),yerr=np.std(etabin5),xerr=0.5)
        plt.xlabel('eta')
        plt.ylabel('avg. number of clusters per box')
        plt.grid()
        plt.show()
        plt.close()

        plt.figure()
        plt.boxplot(etabin1,positions=[0.5],widths=0.5)
        plt.boxplot(etabin2,positions=[1.25],widths=0.25)
        plt.boxplot(etabin3,positions=[2.0],widths=0.5)
        plt.boxplot(etabin4,positions=[2.75],widths=0.25)
        plt.boxplot(etabin5,positions=[3.5],widths=0.5)
        plt.xlabel('eta')
        plt.ylabel('number of clusters per box')
        plt.grid()
        plt.show()
        plt.close()

        fig,ax = plt.subplots(1,1,figsize=(6,4))
        h = ax.hist2d(box_etas,n_clusters_per_box,bins=[40,40],norm='log')
        fig.colorbar(h[3], ax=ax)
        ax.set(xlabel='box eta',ylabel='n clusters per box')
        ax.grid()
        fig.tight_layout()
        plt.show()
        plt.close()


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #### n cells PLOTS:
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    n_cells_plots = True
    if n_cells_plots:
        plt.figure()
        n,bins,patches = plt.hist(n_cells_per_box,histtype='step',bins=50,color='blue',label='all')
        plt.hist(n_cells1sig_per_box,histtype='step',bins=bins,color='purple',label='1sig')
        plt.hist(n_cells15sig_per_box,histtype='step',bins=bins,color='red',label='1.5sig')
        plt.hist(n_cells2sig_per_box,histtype='step',bins=bins,color='orange',label='2sig')
        plt.hist(n_cells4sig_per_box,histtype='step',bins=bins,color='gold',label='4sig')
        plt.xlabel('n cells per box')
        plt.legend()
        plt.yscale('log')
        plt.show()
        plt.close()


        nc_per_box = np.array(n_cells_per_box)
        ncbin0 = n_clusters_per_box[nc_per_box<=1000]
        ncbin1 = n_clusters_per_box[(nc_per_box>1000) & (nc_per_box<=2000)]
        ncbin2 = n_clusters_per_box[(nc_per_box>2000) & (nc_per_box<=3000)]
        ncbin3 = n_clusters_per_box[(nc_per_box>3000) & (nc_per_box<=4000)]
        ncbin4 = n_clusters_per_box[(nc_per_box>4000) & (nc_per_box<=5000)]
        ncbin5 = n_clusters_per_box[nc_per_box>5000]

        fig,ax = plt.subplots(1,1,figsize=(8,5))
        bins = range(0,max(n_clusters_per_box)+1)
        ax.hist(ncbin0,histtype='step',bins=bins,lw=1.5,alpha=0.7,label='# cells <=1000')
        ax.hist(ncbin1,histtype='step',bins=bins,lw=1.5,alpha=0.7,label='1000< # cells <=2000')
        ax.hist(ncbin2,histtype='step',bins=bins,lw=1.5,alpha=0.7,label='2000< # cells <=3000')
        ax.hist(ncbin3,histtype='step',bins=bins,lw=1.5,alpha=0.7,label='3000< # cells <=4000')
        ax.hist(ncbin4,histtype='step',bins=bins,lw=1.5,alpha=0.7,label='4000< # cells <=5000')
        ax.hist(ncbin5,histtype='step',bins=bins,lw=1.5,alpha=0.7,label='5000< # cells')
        ax.legend()
        ax.set(xlabel='n clusters per box',yscale='log')
        fig.tight_layout()
        plt.show()
        plt.close()

        plt.figure()
        bp0 = plt.boxplot(ncbin0,positions=[500],widths=500)
        plt.text(x=400,y=70.0,s=f"{bp0['medians'][0].get_ydata()[1]:.1f}",color='red')
        bp1 = plt.boxplot(ncbin1,positions=[1500],widths=500)
        plt.text(x=1400,y=70.0,s=f"{bp1['medians'][0].get_ydata()[1]:.1f}",color='red')
        bp2 = plt.boxplot(ncbin2,positions=[2500],widths=500)
        plt.text(x=2400,y=70.0,s=f"{bp2['medians'][0].get_ydata()[1]:.1f}",color='red')
        bp3 = plt.boxplot(ncbin3,positions=[3500],widths=500)
        plt.text(x=3400,y=70.0,s=f"{bp3['medians'][0].get_ydata()[1]:.1f}",color='red')
        bp4 = plt.boxplot(ncbin4,positions=[4500],widths=500)
        plt.text(x=4400,y=70.0,s=f"{bp4['medians'][0].get_ydata()[1]:.1f}",color='red')
        bp5 = plt.boxplot(ncbin5,positions=[5500],widths=500)
        plt.text(x=5400,y=70.0,s=f"{bp5['medians'][0].get_ydata()[1]:.1f}",color='red')
        plt.vlines(x=[1000,2000,3000,4000,5000],ymin=0,ymax=max(n_clusters_per_box),color='red',ls='--')
        plt.xlabel('# cells per box')
        plt.ylabel('number of clusters per box')
        plt.grid()
        plt.show()
        plt.close()

        nc4_per_box = np.array(n_cells4sig_per_box)
        nc4bin0 = n_clusters_per_box[nc4_per_box<=50]
        nc4bin1 = n_clusters_per_box[(nc4_per_box>50) & (nc4_per_box<=100)]
        nc4bin2 = n_clusters_per_box[(nc4_per_box>100) & (nc4_per_box<=200)]
        nc4bin3 = n_clusters_per_box[(nc4_per_box>200) & (nc4_per_box<=300)]
        nc4bin4 = n_clusters_per_box[nc4_per_box>300]

        plt.figure()
        bp0 = plt.boxplot(nc4bin0,positions=[25],widths=25)
        plt.text(x=15,y=70.0,s=f"{bp0['medians'][0].get_ydata()[1]:.1f}",color='red')
        bp1 = plt.boxplot(nc4bin1,positions=[75],widths=25)
        plt.text(x=65,y=70.0,s=f"{bp1['medians'][0].get_ydata()[1]:.1f}",color='red')
        bp2 = plt.boxplot(nc4bin2,positions=[150],widths=50)
        plt.text(x=140,y=70.0,s=f"{bp2['medians'][0].get_ydata()[1]:.1f}",color='red')
        bp3 = plt.boxplot(nc4bin3,positions=[250],widths=50)
        plt.text(x=240,y=70.0,s=f"{bp3['medians'][0].get_ydata()[1]:.1f}",color='red')
        bp4 = plt.boxplot(nc4bin4,positions=[350],widths=50)
        plt.text(x=340,y=70.0,s=f"{bp4['medians'][0].get_ydata()[1]:.1f}",color='red')
        plt.vlines(x=[50,100,200,300],ymin=0,ymax=max(n_clusters_per_box),color='red',ls='--')
        plt.xlabel('# 4sig cells per box')
        plt.ylabel('number of clusters per box')
        plt.grid()
        plt.show()
        plt.close()


        fig,ax = plt.subplots(1,3,figsize=(15,4))
        ax[0].hist2d(n_cells_per_box,n_clusters_per_box,bins=[40,max(n_clusters_per_box)],norm='log')
        ax[0].set(xlabel='n cells per box',ylabel='n clusters per box')
        # Perform linear regression
        coeffs = np.polyfit(n_cells_per_box,n_clusters_per_box, deg=1)
        ax[0].annotate(f'Grad.: {coeffs[0]:.5f}\nIntercept: {coeffs[1]:.5f}', xy=(0.05, 0.9), xycoords='axes fraction')
        x_line = np.linspace(np.min(n_cells_per_box), np.max(n_cells_per_box), 100)
        y_line = coeffs[0] * x_line + coeffs[1]
        ax[0].plot(x_line, y_line, color='red')
        ax[0].grid()
        
        #
        ax[1].hist2d(n_cells2sig_per_box,n_clusters_per_box,bins=[40,max(n_clusters_per_box)],norm='log')
        ax[1].set(xlabel='n 2sig cells per box',ylabel='n clusters per box')
        coeffs = np.polyfit(n_cells2sig_per_box,n_clusters_per_box, deg=1)
        ax[1].annotate(f'Grad.: {coeffs[0]:.5f}\nIntercept: {coeffs[1]:.5f}', xy=(0.05, 0.9), xycoords='axes fraction')
        x_line = np.linspace(np.min(n_cells2sig_per_box), np.max(n_cells2sig_per_box), 100)
        y_line = coeffs[0] * x_line + coeffs[1]
        ax[1].plot(x_line, y_line, color='red')
        ax[1].grid()
        
        #
        ax[2].hist2d(n_cells4sig_per_box,n_clusters_per_box,bins=[40,max(n_clusters_per_box)],norm='log')
        ax[2].set(xlabel='n 4sig cells per box',ylabel='n clusters per box')
        coeffs = np.polyfit(n_cells4sig_per_box,n_clusters_per_box, deg=1)
        ax[2].annotate(f'Grad.: {coeffs[0]:.5f}\nIntercept: {coeffs[1]:.5f}', xy=(0.05, 0.9), xycoords='axes fraction')
        x_line = np.linspace(np.min(n_cells4sig_per_box), np.max(n_cells4sig_per_box), 100)
        y_line = coeffs[0] * x_line + coeffs[1]
        ax[2].plot(x_line, y_line, color='red')
        ax[2].grid()
        fig.tight_layout()
        plt.show()
        plt.close()

        fig,ax = plt.subplots(1,2,figsize=(10,4))
        h = ax[0].hist2d(box_etas,n_cells_per_box,bins=[40,40],norm='log')
        fig.colorbar(h[3], ax=ax[0])
        ax[0].set(ylabel='n cells per box',xlabel='box eta')
        ax[0].grid()
        
        h = ax[1].hist2d(box_etas,n_cells4sig_per_box,bins=[40,40],norm='log')
        fig.colorbar(h[3], ax=ax[1])
        ax[1].set(xlabel='box eta',ylabel='n 4sig cells per box')
        ax[1].grid()
        fig.tight_layout()
        plt.show()
        plt.close()

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #### box area PLOTS:
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    box_area_plots = False
    if box_area_plots:
        box_areas = np.array(box_areas)
        plt.figure()
        n,bins,patches = plt.hist(box_areas,histtype='step',bins=50)
        plt.xlabel('box area')
        plt.yscale('log')
        plt.show()
        plt.close()

        plt.figure()
        n,bins,patches = plt.hist(n_clusters_per_box/box_areas,histtype='step',bins=50)   
        plt.xlabel('(number clusters per box) / (box area)')
        plt.yscale('log')
        plt.show()
        plt.close()

        fig,ax = plt.subplots(1,1,figsize=(6,4))
        h = ax.hist2d(box_areas,n_clusters_per_box,bins=[40,40],norm='log')
        fig.colorbar(h[3], ax=ax)
        coeffs = np.polyfit(box_areas,n_clusters_per_box, deg=1)
        ax.annotate(f'Grad.: {coeffs[0]:.5f}\nIntercept: {coeffs[1]:.5f}', xy=(0.05, 0.9), xycoords='axes fraction')
        x_line = np.linspace(np.min(n_cells4sig_per_box), np.max(n_cells4sig_per_box), 100)
        y_line = coeffs[0] * x_line + coeffs[1]
        ax.plot(x_line, y_line, color='red')
        ax.set(xlabel='box area',ylabel='n clusters per box')
        ax.grid()
        fig.tight_layout()
        plt.show()
        plt.close()

        babin0 = n_clusters_per_box[box_areas<=0.25]
        babin1 = n_clusters_per_box[(box_areas>0.25) & (box_areas<=0.75)]
        babin2 = n_clusters_per_box[(box_areas>0.75) & (box_areas<=1.25)]
        babin3 = n_clusters_per_box[(box_areas>1.25) & (box_areas<=1.5)]
        babin4 = n_clusters_per_box[box_areas>1.5]

        fig,ax = plt.subplots(5,1,figsize=(8,8),sharex=True,gridspec_kw={'wspace':0,'hspace':0})
        bins = range(0,max(n_clusters_per_box)+1)
        ax[0].hist(babin0,histtype='step',bins=bins,lw=1.5,alpha=0.7,color='tab:blue',label='box area <=0.25')
        ax[1].hist(babin1,histtype='step',bins=bins,lw=1.5,alpha=0.7,color='tab:orange',label='0.25< box area <=0.75')
        ax[2].hist(babin2,histtype='step',bins=bins,lw=1.5,alpha=0.7,color='tab:green',label='0.75< box area <=1.25')
        ax[3].hist(babin3,histtype='step',bins=bins,lw=1.5,alpha=0.7,color='tab:red',label='1.25< box area <=1.5')
        ax[4].hist(babin4,histtype='step',bins=bins,lw=1.5,alpha=0.7,color='tab:purple',label='1.5< box area')
        for axe in ax:
            axe.legend()
            axe.set(xlabel='n clusters per box',yscale='log')
        # ax[0].legend()
        # ax[3].set(xlabel='n clusters per box',yscale='log')
        fig.tight_layout()
        plt.show()
        plt.close()

        fig,ax = plt.subplots(5,1,figsize=(8,8),sharex=True,gridspec_kw={'wspace':0,'hspace':0})
        bins = range(0,max(n_clusters_per_box)+1)
        ax[0].hist(babin0/box_areas[box_areas<=0.25],histtype='step',bins=bins,lw=1.5,alpha=0.7,color='tab:blue',label='# box area <=0.25')
        ax[1].hist(babin1/box_areas[(box_areas>0.25) & (box_areas<=0.75)],histtype='step',bins=bins,lw=1.5,alpha=0.7,color='tab:orange',label='0.25< # box area <=0.75')
        ax[2].hist(babin2/box_areas[(box_areas>0.75) & (box_areas<=1.25)],histtype='step',bins=bins,lw=1.5,alpha=0.7,color='tab:green',label='0.75< # box area <=1.25')
        ax[3].hist(babin3/box_areas[(box_areas>1.25) & (box_areas<=1.5)],histtype='step',bins=bins,lw=1.5,alpha=0.7,color='tab:red',label='1.25< # box area <=1.5')
        ax[4].hist(babin4/box_areas[box_areas>1.5],histtype='step',bins=bins,lw=1.5,alpha=0.7,color='tab:purple',label='1.5< # box area')
        for axe in ax:
            axe.legend()
            axe.set(xlabel='n clusters per box / box area',yscale='log')
        # ax[0].legend()
        # ax[3].set(xlabel='n clusters per box',yscale='log')
        fig.tight_layout()
        plt.show()
        plt.close()






















