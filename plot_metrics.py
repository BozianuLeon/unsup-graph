import numpy as np 
import scipy
import math
import os

import pickle

import matplotlib.pyplot as plt
import matplotlib

import mplhep as hep
hep.style.use(hep.style.ATLAS)


def load_object(fname):
    with open(fname,'rb') as file:
        return pickle.load(file)




metrics_folder = "/home/users/b/bozianu/work/calo-cluster/unsup-graph/cache/DMoN_caloXYZ_bucket_200c_32e/20250314-12/"
save_folder = metrics_folder + "/plots/"
image_format="png"
print("Save location: ", save_folder)
if not os.path.exists(save_folder): os.makedirs(save_folder)

def clip_phi(phi_values):
    return phi_values - 2 * np.pi * np.floor((phi_values + np.pi) / (2 * np.pi))




# gnn clusters
total_gnn_cl_pt = np.concatenate(load_object(metrics_folder+'tot_gnn_pt.pkl'))/1000
total_gnn_cl_eta = np.concatenate(load_object(metrics_folder+'tot_gnn_eta.pkl'))
total_gnn_cl_phi = clip_phi(np.concatenate(load_object(metrics_folder+'tot_gnn_phi.pkl')))
#gnn jets
event_gnn_jet_cl_pt = load_object(metrics_folder+'tot_gnn_jet_pt.pkl')
total_gnn_jet_cl_pt = np.concatenate(event_gnn_jet_cl_pt)/1000
total_gnn_jet_cl_eta = np.concatenate(load_object(metrics_folder+'tot_gnn_jet_eta.pkl'))
total_gnn_jet_cl_phi = clip_phi(np.concatenate(load_object(metrics_folder+'tot_gnn_jet_phi.pkl')))
# topoclusters
total_tcl_pt = np.concatenate(load_object(metrics_folder+'tot_cl_pt.pkl'))/1000
total_tcl_eta = np.concatenate(load_object(metrics_folder+'tot_cl_eta.pkl'))
total_tcl_phi = clip_phi(np.concatenate(load_object(metrics_folder+'tot_cl_phi.pkl')))
# topocluster jets 
event_tcl_jet_pt = load_object(metrics_folder+'tot_cl_jet_pt.pkl')
total_tcl_jet_pt = np.concatenate(event_tcl_jet_pt)/1000
total_tcl_jet_eta = np.concatenate(load_object(metrics_folder+'tot_cl_jet_eta.pkl'))
total_tcl_jet_phi = clip_phi(np.concatenate(load_object(metrics_folder+'tot_cl_jet_phi.pkl')))
# akt jets
event_akt_jet_pt = load_object(metrics_folder+'tot_akt_pt.pkl')
total_akt_jet_pt = np.concatenate(event_akt_jet_pt)/1000
total_akt_jet_eta = np.concatenate(load_object(metrics_folder+'tot_akt_eta.pkl'))
total_akt_jet_phi = clip_phi(np.concatenate(load_object(metrics_folder+'tot_akt_phi.pkl')))
# truth jets
event_tru_jet_pt =  load_object(metrics_folder+'tot_tru_pt.pkl')
total_tru_jet_pt = np.concatenate(event_tru_jet_pt)/1000
total_tru_jet_eta =  np.concatenate(load_object(metrics_folder+'tot_tru_eta.pkl'))
total_tru_jet_phi =  clip_phi(np.concatenate(load_object(metrics_folder+'tot_tru_phi.pkl')))



print("=======================================================================================================")
print(f"Plotting cluster kinematics, saving to {save_folder}")
print("=======================================================================================================\n")

f,ax0 = plt.subplots(1,1,figsize=(9, 6))
freq_pred, bins, _   = ax0.hist(total_gnn_cl_pt,bins=100,density=True,histtype='step',color='dodgerblue',lw=1.5,label='DMoN')
freq_tar, bins, _    = ax0.hist(total_tcl_pt,bins=bins,density=True,histtype='step',color='green',lw=1.5,label='TC')
ax0.set_title('Transverse Momentum', fontsize=16, fontfamily="TeX Gyre Heros")
ax0.legend(loc='lower left',bbox_to_anchor=(0.75, 0.8),fontsize="medium")
hep.atlas.label(ax=ax0,label='Work in Progress',data=False,lumi=None,loc=1)
ax0.set(yscale='log',xlabel='Cluster $p_{\mathrm{T}}$ EM Scale [GeV]')
f.savefig(save_folder + f'/cluster_pt_total.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
plt.close()

f,ax0 = plt.subplots(1,1,figsize=(9, 6))
freq_pred, bins, _   = ax0.hist(total_gnn_cl_eta,bins=50,density=True,histtype='step',color='dodgerblue',lw=1.5,label='DMoN')
freq_tar, bins, _    = ax0.hist(total_tcl_eta,bins=bins,density=True,histtype='step',color='green',lw=1.5,label='TC')
ax0.legend(loc='lower left',bbox_to_anchor=(0.75, 0.8),fontsize="medium")
hep.atlas.label(ax=ax0,label='Work in Progress',data=False,lumi=None,loc=1)
ax0.set(yscale='log',xlabel='Cluster $\eta$')
f.savefig(save_folder + f'/cluster_eta_total.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
plt.close()

f,ax0 = plt.subplots(1,1,figsize=(9, 6))
freq_pred, bins, _   = ax0.hist(total_gnn_cl_phi,bins=50,density=True,histtype='step',color='dodgerblue',lw=1.5,label='DMoN')
freq_tar, bins, _    = ax0.hist(total_tcl_phi,bins=bins,density=True,histtype='step',color='green',lw=1.5,label='TC')
ax0.legend(loc='lower left',bbox_to_anchor=(0.75, 0.8),fontsize="medium")
hep.atlas.label(ax=ax0,label='Work in Progress',data=False,lumi=None,loc=1)
ax0.set(yscale='log',xlabel='Cluster $\phi$')
f.savefig(save_folder + f'/cluster_phi_total.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
plt.close()


print("=======================================================================================================")
print(f"Plotting jet kinematics, saving to {save_folder}")
print("=======================================================================================================\n")

f,ax0 = plt.subplots(1,1,figsize=(9, 6))
bin_edges = [20, 40, 60, 80, 100, 120, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 850]
bin_edges = np.arange(20,600,step=20)
freq_pred, bins, _   = ax0.hist(total_gnn_jet_cl_pt,bins=bin_edges,density=True,histtype='step',color='dodgerblue',lw=1.5,label='DMoN')
freq_tar, bins, _    = ax0.hist(total_tcl_jet_pt,bins=bins,density=True,histtype='step',color='green',lw=1.5,label='TC')
freq_akt, bins, _    = ax0.hist(total_akt_jet_pt,bins=bins,density=True,histtype='step',color='slategrey',lw=1.5,label='AKT4EmTopo')
freq_tru, bins, _    = ax0.hist(total_tru_jet_pt,bins=bins,density=True,histtype='step',color='gold',lw=1.5,label='Truth')
ax0.legend(loc='lower left',bbox_to_anchor=(0.7, 0.7),fontsize="medium")
ax0.set_title('A single JZ slice', fontsize=16, fontfamily="TeX Gyre Heros")
hep.atlas.label(ax=ax0,label='Work in Progress',data=False,lumi=None,loc=1)
ax0.set(yscale='log',xlabel='Jet $p_{\mathrm{T}}$ [GeV]')
f.savefig(save_folder + f'/jet_pt_total.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
plt.close()

f,ax0 = plt.subplots(1,1,figsize=(9, 6))
freq_pred, bins, _   = ax0.hist(total_gnn_jet_cl_eta,bins=50,density=True,histtype='step',color='dodgerblue',lw=1.5,label='DMoN')
freq_tar, bins, _    = ax0.hist(total_tcl_jet_eta,bins=bins,density=True,histtype='step',color='green',lw=1.5,label='TC')
freq_akt, bins, _    = ax0.hist(total_akt_jet_eta,bins=bins,density=True,histtype='step',color='slategrey',lw=1.5,label='AKT4EmTopo')
freq_tru, bins, _    = ax0.hist(total_tru_jet_eta,bins=bins,density=True,histtype='step',color='gold',lw=1.5,label='Truth')
ax0.legend(loc='lower left',bbox_to_anchor=(0.7, 0.7),fontsize="medium")
hep.atlas.label(ax=ax0,label='Work in Progress',data=False,lumi=None,loc=1)
ax0.set(yscale='log',xlabel='Jet $\eta$ [GeV]')
f.savefig(save_folder + f'/jet_eta_total.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
plt.close()

f,ax0 = plt.subplots(1,1,figsize=(9, 6))
freq_pred, bins, _   = ax0.hist(total_gnn_jet_cl_phi,bins=50,density=True,histtype='step',color='dodgerblue',lw=1.5,label='DMoN')
freq_tar, bins, _    = ax0.hist(total_tcl_jet_phi,bins=bins,density=True,histtype='step',color='green',lw=1.5,label='TC')
freq_akt, bins, _    = ax0.hist(total_akt_jet_phi,bins=bins,density=True,histtype='step',color='slategrey',lw=1.5,label='AKT4EmTopo')
freq_tru, bins, _    = ax0.hist(total_tru_jet_phi,bins=bins,density=True,histtype='step',color='gold',lw=1.5,label='Truth')
ax0.legend(loc='lower left',bbox_to_anchor=(0.7, 0.7),fontsize="medium")
hep.atlas.label(ax=ax0,label='Work in Progress',data=False,lumi=None,loc=1)
ax0.set(yscale='log',xlabel='Jet $\phi$ [GeV]')
f.savefig(save_folder + f'/jet_phi_total.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
plt.close()




gnn_jet_lead_pt = np.array([max(x)/1000 for x in event_gnn_jet_cl_pt])
tcl_jet_lead_pt = np.array([max(x)/1000 for x in event_tcl_jet_pt])
akt_jet_lead_pt = np.array([max(x)/1000 for x in event_akt_jet_pt])
tru_jet_lead_pt = np.array([max(x)/1000 for x in event_tru_jet_pt])


f,ax0 = plt.subplots(1,1,figsize=(9, 6))
freq_tcl, bins, _    = ax0.hist(tcl_jet_lead_pt,bins=50,histtype='step',color='green',lw=1.5,label='TC')
freq_pred, bins, _   = ax0.hist(gnn_jet_lead_pt,bins=bins,histtype='step',color='dodgerblue',lw=1.5,label='DMoN')
freq_akt, bins, _   = ax0.hist(akt_jet_lead_pt,bins=bins,histtype='step',color='slategrey',lw=1.5,label='AKT4EmTopo')
# freq_tru, bins, _   = ax0.hist(tru_jet_lead_pt,bins=bins,histtype='step',color='gold',lw=1.5,label='Truth')
ax0.set_title('A single JZ slice', fontsize=16, fontfamily="TeX Gyre Heros")
ax0.legend(loc='lower left',bbox_to_anchor=(0.7, 0.7),fontsize="medium")
# hep.atlas.label(ax=ax0,label='Work in Progress',data=False,lumi=None,loc=1)
ax0.set(yscale='log',ylabel='Num. events',xlabel='Leading Jet $p_{\mathrm{T}}$ [GeV]')
f.savefig(save_folder + f'/jet_pt_lead.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
plt.close()





print("=======================================================================================================")
print(f"Making leading jet trigger decision")
print("=======================================================================================================\n")


def get_ratio(numer,denom):
    result = np.zeros_like(numer, dtype=float)
    non_zero_indices = denom != 0
    # Perform element-wise division, handling zeros in the denominator
    result[non_zero_indices] = numer[non_zero_indices] / denom[non_zero_indices]
    return result

def clopper_pearson(x, n, alpha=0.05):
    """
    Estimate the confidence interval for a sampled Bernoulli random
    variable.
    `x` is the number of successes and `n` is the number trials (x <=
    n). `alpha` is the confidence level (i.e., the true probability is
    inside the confidence interval with probability 1-alpha). The
    function returns a `(low, high)` pair of numbers indicating the
    interval on the probability.
    https://root.cern.ch/doc/master/classTEfficiency.html#ae80c3189bac22b7ad15f57a1476ef75b
    """

    lo = scipy.stats.beta.ppf(alpha / 2, x, n - x + 1)
    hi = scipy.stats.beta.ppf(1 - alpha / 2, x + 1, n - x)
    return 0.0 if math.isnan(lo) else lo, 1.0 if math.isnan(hi) else hi

def get_errorbars(success_array, total_array, alpha=0.05):
    """
    Function to calculate and return errorbars in matplotlib preferred format.
    Current usage of Clopper-Pearon may generalise later. Function currently
    returns interval.
    'success_array' is the count of each histogram bins after(!) cut applied
    'total_array' is the count of each histogram before trigger applied
    'alpha' is the confidence level
    Returns errors array to be used in ax.errorbars kwarg yerr
    """
    confidence_intervals = []
    
    lo, hi = np.vectorize(clopper_pearson)(success_array, total_array, alpha)
    
    confidence_intervals = np.array([lo, hi]).T
    
    zeros_mask = total_array == 0
    lower_error_bars = np.where(zeros_mask, lo, success_array/total_array - lo)
    upper_error_bars = np.where(zeros_mask, hi, hi - success_array/total_array)
    
    errors = np.array([lower_error_bars, upper_error_bars])
    
    return errors

lead_jet_pt_cut = 450 # GeV
trig_decision_akt = np.argwhere(akt_jet_lead_pt>lead_jet_pt_cut).T[0]
trig_decision_gnn_jet = np.argwhere(gnn_jet_lead_pt>lead_jet_pt_cut).T[0]
trig_decision_tcl_jet = np.argwhere(tcl_jet_lead_pt>lead_jet_pt_cut).T[0]

# Set the x-axis and binning
start,end = 300,600
step = 10
bins = np.arange(start, end, step)

f,ax = plt.subplots(3,1,figsize=(8,14))
n_tru,bins,_ = ax[0].hist(tru_jet_lead_pt,bins=bins,histtype='step',color='gold',label='Truth')
bin_centers = (bins[:-1] + bins[1:]) / 2
bin_width = bins[1] - bins[0]
ax[0].set_title(f'Before {lead_jet_pt_cut:.0f}GeV Cut',fontsize=16, fontfamily="TeX Gyre Heros")
ax[0].set(xlabel="Leading EMTopo (Offline) jet pT (GeV)",ylabel='Events')
ax[0].legend()

n2_akt,bins,_ = ax[1].hist(tru_jet_lead_pt[trig_decision_akt],bins=bins,histtype='step',label='AKT',color="slategrey")
n2_gnn,_,_ = ax[1].hist(tru_jet_lead_pt[trig_decision_gnn_jet],bins=bins,histtype='step',label='GNN',color="dodgerblue")
n2_tcl,_,_ = ax[1].hist(tru_jet_lead_pt[trig_decision_tcl_jet],bins=bins,histtype='step',label='TCL',color="green")
ax[1].axvline(x=lead_jet_pt_cut,ymin=0,ymax=1,ls='--',color='red',alpha=0.3,label='Cut')
ax[1].set_title(f'After {lead_jet_pt_cut:.0f}GeV Cut',fontsize=16, fontfamily="TeX Gyre Heros")
ax[1].set(xlabel="Leading jet pT (GeV)",ylabel='Events')
ax[1].legend()

ax[2].axvline(x=lead_jet_pt_cut,ymin=0,ymax=1,ls='--',color='red',alpha=0.3,label='Cut')
with np.errstate(divide='ignore', invalid='ignore'):
    akt_eff = get_ratio(n2_akt,n_tru)
    akt_err = get_errorbars(n2_akt,n_tru)

    gnn_eff = get_ratio(n2_gnn,n_tru)
    gnn_err = get_errorbars(n2_gnn,n_tru)
    
    tcl_eff = get_ratio(n2_tcl,n_tru)
    tcl_err = get_errorbars(n2_tcl,n_tru)


ax[2].errorbar(bin_centers,akt_eff,xerr=bin_width/2,yerr=akt_err,elinewidth=0.4,marker='.',ls='none',label='Anti-kt',color='slategrey')
ax[2].errorbar(bin_centers,gnn_eff,xerr=bin_width/2,yerr=gnn_err,elinewidth=0.4,marker='.',ls='none',label='DMoN',color='dodgerblue')
ax[2].errorbar(bin_centers,tcl_eff,xerr=bin_width/2,yerr=tcl_err,elinewidth=0.4,marker='.',ls='none',label='TCL',color='green')
ax[2].set(xlabel="Leading jet pT (GeV)",ylabel='Efficiency')
ax[2].legend(loc='lower right')
# hep.atlas.label(ax=ax[2],label='Work in Progress',data=False,lumi=None,loc=1)
f.subplots_adjust(hspace=0.4)
ax[0].set_yscale('log')
ax[1].set_yscale('log')
f.savefig(save_folder + f'/leading_{lead_jet_pt_cut:.0f}GeV_cuts.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
