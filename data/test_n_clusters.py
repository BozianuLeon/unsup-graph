import torch
import torch_geometric

import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import pickle

def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp)

def load_object(fname):
    with open(fname,'rb') as file:
        return pickle.load(file)



if __name__=="__main__":

    make_lists = False
    make_cl_lists = False
    if make_lists:
        path_to_h5_file = "/srv/beegfs/scratch/shares/atlas_caloM/mu_200_truthjets/cells/JZ4/user.lbozianu/user.lbozianu.43589851._000117.calocellD3PD_mc21_14TeV_JZ4.r14365.h5"
        print("\t",path_to_h5_file)
        f1 = h5py.File(path_to_h5_file,"r")
        cell_data = f1["caloCells"]["2d"]

        tot_n_cell_1,tot_n_cell_2,tot_n_cell_3,tot_n_cell_4,tot_n_cell_5,tot_n_cell_6,tot_n_cell_8,tot_n_cell_10 = [],[],[],[],[],[],[],[]
        for kk in range(len(cell_data)):
            print(kk)
            cells = cell_data[kk]
            cells1sig = cells[abs(cells['cell_E'] / cells['cell_Sigma']) >= 1]
            tot_n_cell_1.append(len(cells1sig))
            cells2sig = cells1sig[abs(cells1sig['cell_E'] / cells1sig['cell_Sigma']) >= 2]
            tot_n_cell_2.append(len(cells2sig))
            cells3sig = cells2sig[abs(cells2sig['cell_E'] / cells2sig['cell_Sigma']) >= 3]
            tot_n_cell_3.append(len(cells3sig))
            cells4sig = cells3sig[abs(cells3sig['cell_E'] / cells3sig['cell_Sigma']) >= 4]
            tot_n_cell_4.append(len(cells4sig))
            cells5sig = cells4sig[abs(cells4sig['cell_E'] / cells4sig['cell_Sigma']) >= 5]
            tot_n_cell_5.append(len(cells5sig))
            cells6sig = cells5sig[abs(cells5sig['cell_E'] / cells5sig['cell_Sigma']) >= 6]
            tot_n_cell_6.append(len(cells6sig))
            cells8sig = cells6sig[abs(cells6sig['cell_E'] / cells6sig['cell_Sigma']) >= 8]
            tot_n_cell_8.append(len(cells8sig))
            cells10sig = cells8sig[abs(cells8sig['cell_E'] / cells8sig['cell_Sigma']) >= 10]
            tot_n_cell_10.append(len(cells10sig))

        f1.close()

        save_object(tot_n_cell_1, '../cache/inputs/tot_n_cell_1.pkl')
        save_object(tot_n_cell_2, '../cache/inputs/tot_n_cell_2.pkl')
        save_object(tot_n_cell_3, '../cache/inputs/tot_n_cell_3.pkl')
        save_object(tot_n_cell_4, '../cache/inputs/tot_n_cell_4.pkl')
        save_object(tot_n_cell_5, '../cache/inputs/tot_n_cell_5.pkl')
        save_object(tot_n_cell_6, '../cache/inputs/tot_n_cell_6.pkl')
        save_object(tot_n_cell_8, '../cache/inputs/tot_n_cell_8.pkl')
        save_object(tot_n_cell_10, '../cache/inputs/tot_n_cell_10.pkl')

    elif make_cl_lists:
        path_to_cl_file = "/srv/beegfs/scratch/shares/atlas_caloM/mu_200_truthjets/clusters/JZ4/user.lbozianu/user.lbozianu.43589851._000117.topoClD3PD_mc21_14TeV_JZ4.r14365.h5"
        print("\t",path_to_cl_file)
        f2 = h5py.File(path_to_cl_file,"r")
        cl_data = f2["caloCells"] 

        tot_n_cl, tot_n_tcl, tot_n_tcl_2 = [], [], []
        for idx in range(len(cl_data["2d"])):
            print(idx)
            event_data   = cl_data["1d"][idx]
            cluster_data = cl_data["2d"][idx]
            tot_n_cl.append(event_data["cl_n"])
            tot_n_tcl.append(len(cluster_data['cl_pt'][np.isfinite(cluster_data['cl_pt'])] ))
            tot_n_tcl_2.append(len(cluster_data['cl_pt'][cluster_data['cl_pt'] > 2000]))

            # cl_pts = cluster_data['cl_pt'][np.isfinite(cluster_data['cl_pt'])] # [~np.isnan(cl_pts)]
            # # cl_etas = cluster_data['cl_eta'][np.isfinite(cluster_data['cl_eta'])] # no eta/phi YET
            # cl_n = event_data["cl_n"]

        f2.close()

        save_object(tot_n_cl, '../cache/inputs/tot_n_cl.pkl')
        save_object(tot_n_tcl, '../cache/inputs/tot_n_tcl.pkl')
        save_object(tot_n_tcl_2, '../cache/inputs/tot_n_tcl_2.pkl')


    else:
        tot_n_cell_1 = load_object('../cache/inputs/tot_n_cell_1.pkl')
        tot_n_cell_2 = load_object('../cache/inputs/tot_n_cell_2.pkl')
        tot_n_cell_3 = load_object('../cache/inputs/tot_n_cell_3.pkl')
        tot_n_cell_4 = load_object('../cache/inputs/tot_n_cell_4.pkl')
        tot_n_cell_5 = load_object('../cache/inputs/tot_n_cell_5.pkl')
        tot_n_cell_6 = load_object('../cache/inputs/tot_n_cell_6.pkl')
        tot_n_cell_8 = load_object('../cache/inputs/tot_n_cell_8.pkl')
        tot_n_cell_10 = load_object('../cache/inputs/tot_n_cell_10.pkl')
        tot_n_cl = load_object('../cache/inputs/tot_n_cl.pkl')
        tot_n_tcl = load_object('../cache/inputs/tot_n_tcl.pkl')
        tot_n_tcl_2 = load_object('../cache/inputs/tot_n_tcl_2.pkl')

    #######################################################################################################################

    f,ax0 = plt.subplots(1,1,figsize=(9, 6))
    bins = np.linspace(0,4000,num=100)
    freq_tcl, bins, _    = ax0.hist(tot_n_cl,bins=bins,histtype='step',color='green',lw=0.9,label='TC')
    freq_tcl, bins, _    = ax0.hist(tot_n_tcl,bins=bins,histtype='step',color='lime',lw=0.7,label='ToppCl')
    freq_tcl, bins, _    = ax0.hist(tot_n_tcl_2,bins=bins,histtype='step',color='aquamarine',lw=0.99,label='ToppCl > 2 GeV')

    freq_3, bins, _   = ax0.hist(tot_n_cell_3,bins=bins,histtype='step',color='orange',lw=0.7,label='cells |signif| > 3')
    freq_4, bins, _   = ax0.hist(tot_n_cell_4,bins=bins,histtype='step',color='peru',lw=0.7,label='cells |signif| > 4')
    freq_5, bins, _   = ax0.hist(tot_n_cell_5,bins=bins,histtype='step',color='sienna',lw=0.7,label='cells |signif| > 5')
    freq_6, bins, _   = ax0.hist(tot_n_cell_6,bins=bins,histtype='step',color='grey',lw=0.7,label='cells |signif| > 6')
    freq_8, bins, _   = ax0.hist(tot_n_cell_8,bins=bins,histtype='step',color='tomato',lw=0.7,label='cells |signif| > 8')

    ax0.set_title('A single JZ slice', fontsize=16)
    ax0.legend(loc='lower left',bbox_to_anchor=(0.75, 0.65),fontsize="small")
    ax0.set(yscale='log',ylabel='Events',xlabel='Number of X per event')
    f.savefig(f'../plots/inputs/number_of_x.png',dpi=400,bbox_inches="tight")
    plt.close()

    f,ax0 = plt.subplots(1,1,figsize=(9, 6))
    bins = np.linspace(0,7500,num=100)
    freq_tcl, bins, _    = ax0.hist(tot_n_cl,bins=bins,histtype='step',color='green',lw=1.0,label='TC')
    freq_tcl, bins, _    = ax0.hist(tot_n_tcl,bins=bins,histtype='step',color='lime',lw=1.0,label='ToppCl')
    freq_tcl, bins, _    = ax0.hist(tot_n_tcl_2,bins=bins,histtype='step',color='aquamarine',lw=1.0,label='ToppCl > 2 GeV')

    freq_2, bins, _   = ax0.hist(tot_n_cell_2,bins=bins,histtype='step',color='gold',lw=0.7,label='cells |signif| > 2')
    freq_3, bins, _   = ax0.hist(tot_n_cell_3,bins=bins,histtype='step',color='orange',lw=0.7,label='cells |signif| > 3')
    freq_4, bins, _   = ax0.hist(tot_n_cell_4,bins=bins,histtype='step',color='peru',lw=0.7,label='cells |signif| > 4')
    freq_5, bins, _   = ax0.hist(tot_n_cell_5,bins=bins,histtype='step',color='sienna',lw=0.7,label='cells |signif| > 5')
    freq_6, bins, _   = ax0.hist(tot_n_cell_6,bins=bins,histtype='step',color='grey',lw=0.7,label='cells |signif| > 6')
    freq_8, bins, _   = ax0.hist(tot_n_cell_8,bins=bins,histtype='step',color='tomato',lw=0.7,label='cells |signif| > 8')
    freq_10, bins, _   = ax0.hist(tot_n_cell_10,bins=bins,histtype='step',color='red',lw=0.7,label='cells |signif| > 10')

    ax0.set_title('A single JZ slice', fontsize=16)
    ax0.legend(loc='lower left',bbox_to_anchor=(0.7, 0.6),fontsize="small")
    ax0.set(yscale='log',ylabel='Events',xlabel='Number of X per event')
    f.savefig(f'../plots/inputs/number_of_x_2.png',dpi=400,bbox_inches="tight")
    plt.close()


    f,ax0 = plt.subplots(1,1,figsize=(9, 6))
    bins = np.linspace(0,20_000,num=100)
    freq_tcl, bins, _    = ax0.hist(tot_n_cl,bins=100,histtype='step',color='green',lw=0.7,label='TC')
    freq_tcl, bins, _    = ax0.hist(tot_n_tcl,bins=100,histtype='step',color='lime',lw=0.7,label='ToppCl')
    freq_tcl, bins, _    = ax0.hist(tot_n_tcl_2,bins=100,histtype='step',color='aquamarine',lw=0.7,label='ToppCl > 2')

    freq_1, bins, _   = ax0.hist(tot_n_cell_1,bins=100,histtype='step',color='darkkhaki',lw=0.7,label='cells |signif| > 1')
    freq_2, bins, _   = ax0.hist(tot_n_cell_2,bins=100,histtype='step',color='gold',lw=0.7,label='cells |signif| > 2')
    freq_3, bins, _   = ax0.hist(tot_n_cell_3,bins=100,histtype='step',color='orange',lw=0.7,label='cells |signif| > 3')
    freq_4, bins, _   = ax0.hist(tot_n_cell_4,bins=100,histtype='step',color='peru',lw=0.7,label='cells |signif| > 4')
    freq_5, bins, _   = ax0.hist(tot_n_cell_5,bins=100,histtype='step',color='sienna',lw=0.7,label='cells |signif| > 5')
    freq_6, bins, _   = ax0.hist(tot_n_cell_6,bins=100,histtype='step',color='grey',lw=0.7,label='cells |signif| > 6')
    freq_8, bins, _   = ax0.hist(tot_n_cell_8,bins=100,histtype='step',color='tomato',lw=0.7,label='cells |signif| > 8')
    freq_10, bins, _   = ax0.hist(tot_n_cell_10,bins=100,histtype='step',color='red',lw=0.7,label='cells |signif| > 10')

    ax0.set_title('A single JZ slice', fontsize=16, fontfamily="TeX Gyre Heros")
    ax0.legend(loc='lower left',bbox_to_anchor=(0.7, 0.6),fontsize="small")
    ax0.set(yscale='log',ylabel='Events',xlabel='Number of X per event')
    f.savefig(f'../plots/inputs/number_of_x_binned.png',dpi=400,bbox_inches="tight")
    plt.close()
