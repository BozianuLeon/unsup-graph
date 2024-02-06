import numpy as np
import torch
import torchvision

def weighted_circular_mean(phi_values, energy_values):
    """
    Calculate the weighted circular mean (average) of a list of angles.
    Handles the periodicity of phi correctly. http://palaeo.spb.ru/pmlibrary/pmbooks/mardia&jupp_2000.pdf

    :param phi_values: List of angles in radians
    :param energy_values: List of weights corresponding to each phi value
    :return: Weighted circular mean in radians
    """
    if len(phi_values) != len(energy_values):
        raise ValueError("phi_values and energy_values must have the same length")

    weighted_sin_sum = np.sum(energy_values * np.sin(phi_values))
    weighted_cos_sum = np.sum(energy_values * np.cos(phi_values))
    weighted_circular_mean = np.arctan2(weighted_sin_sum, weighted_cos_sum)
    return weighted_circular_mean




def CalculateEnergyFromCells(desired_cells):
    # Inputs
    # desired_cells, structured array containing all cells in event ['cell_*'] properties
    # Outputs
    # total energy,  for this collection of cells
    
    if (desired_cells is None) or len(desired_cells)==0:
        return np.nan
    return sum(desired_cells['cell_E']) 

def CalculateEtaFromCells(desired_cells):
    # Inputs
    # desired_cells, structured array containing all cells in event ['cell_*'] properties
    # Outputs
    # eta, (absolute) energy weighted value
    
    if (desired_cells is None) or len(desired_cells)==0:
        return np.nan    
    energy_weighted_eta = np.dot(desired_cells['cell_eta'],np.abs(desired_cells['cell_E'])) / sum(np.abs(desired_cells['cell_E']))
    return energy_weighted_eta 

def CalculatePhiFromCells(desired_cells):
    # Inputs
    # desired_cells, structured array containing all cells in event ['cell_*'] properties
    # Outputs
    # phi, (absolute) energy weighted value
    
    if desired_cells is None:
        return np.nan
    energy_weighted_phi = weighted_circular_mean(desired_cells['cell_phi'],np.abs(desired_cells['cell_E']))
    return energy_weighted_phi 

def CalculateEtFromCells(desired_cells):
    # Inputs
    # desired_cells, structured array containing all cells in event ['cell_*'] properties
    # Outputs
    # E_T, Transverse energy
    
    if desired_cells is None:
        return np.nan
    total_energy = sum(desired_cells['cell_E']) 
    energy_weighted_eta = np.dot(desired_cells['cell_eta'],np.abs(desired_cells['cell_E'])) / sum(np.abs(desired_cells['cell_E']))
    return total_energy / np.cosh(energy_weighted_eta) 

def CalculateNCellsFromCells(desired_cells):
    # Inputs
    # desired_cells, structured array containing all cells in event ['cell_*'] properties
    # Outputs
    # Number of cells in the box/cluster

    if desired_cells is None:
        return np.nan
    return len(desired_cells) #if cell_ids is not None else np.nan

def CalculateNoiseFromCells(desired_cells):
    # Inputs
    # cell_ids, list of cell ids that we want to calculate 
    # cells, structured array containing all cells in event ['cell_*'] properties
    # Outputs
    # total noise of the cluster/box (see TC paper p. 26)
    
    if desired_cells is None:
        return np.nan
    total_noise = np.sqrt(sum(desired_cells['cell_Sigma']**2))
    return total_noise

def CalculateSignficanceFromCells(desired_cells):
    # Inputs
    # cell_ids, list of cell ids that we want to calculate 
    # cells, structured array containing all cells in event ['cell_*'] properties
    # Outputs
    # total significance of cells

    if desired_cells is None:
        return np.nan    
    total_energy = sum(desired_cells['cell_E']) 
    total_noise = np.sqrt(sum(desired_cells['cell_Sigma']**2))

    return total_energy / total_noise

def CalculateNegativeEnergyFromCells(desired_cells):
    # Inputs
    # desired_cells, structured array containing all cells in event ['cell_*'] properties
    # Outputs
    # fraction of total energy that is negative

    if desired_cells is None:
        return np.nan
    # total_energy = sum(abs(desired_cells['cell_E'])) 
    negative_energy_cells = desired_cells[desired_cells['cell_E']<0]['cell_E']
    positive_energy_cells = desired_cells[desired_cells['cell_E']>0]['cell_E']
    if len(positive_energy_cells)==0:
        return np.nan

    return abs(sum(negative_energy_cells)) / sum(positive_energy_cells) 

def CalculateMaxEnergyFracFromCells(desired_cells):
    # Inputs
    # desired_cells, structured array containing all cells in event ['cell_*'] properties
    # Outputs
    # fraction of total energy in one cell
    
    if (desired_cells is None) or (len(desired_cells)==0):
        return np.nan
    total_energy = sum(desired_cells['cell_E'])
    return abs(max(desired_cells['cell_E'])) / total_energy 

def CalculateEnergyFromTwoSigmaCells(desired_cells):
    # Inputs
    # cell_ids, list of cell ids that we want to calculate 
    # cells, structured array containing all cells in event ['cell_*'] properties
    # Outputs
    # total energy,  for cells with signif > 2
    
    if desired_cells is None:
        return np.nan
    cell_signif = desired_cells['cell_E'] / desired_cells['cell_Sigma']
    two_sigma_cells = desired_cells[abs(cell_signif)>=2]
    if len(two_sigma_cells)==0:
        # print('No 2 sigma cells',sum(desired_cells['cell_E']))
        return np.nan
    return sum(two_sigma_cells['cell_E']) 

def CalculateEtFromTwoSigmaCells(desired_cells):
    # Inputs
    # cell_ids, list of cell ids that we want to calculate 
    # cells, structured array containing all cells in event ['cell_*'] properties
    # Outputs
    # E_T, Transverse energy for cells with signif > 2
    
    if desired_cells is None:
        return np.nan
    cell_signif = desired_cells['cell_E'] / desired_cells['cell_Sigma']
    two_sigma_cells = desired_cells[abs(cell_signif)>=2]
    if len(two_sigma_cells)==0:
        # print('No 2 sigma cells',sum(desired_cells['cell_E']))
        return np.nan
    total_energy = sum(two_sigma_cells['cell_E']) 
    energy_weighted_eta = np.dot(two_sigma_cells['cell_eta'],np.abs(two_sigma_cells['cell_E'])) / sum(np.abs(two_sigma_cells['cell_E']))
    return total_energy / np.cosh(energy_weighted_eta) 

def CalculateNCellsFromTwoSigmaCells(desired_cells):
    # Inputs
    # cell_ids, list of cell ids that we want to calculate 
    # cells, structured array containing all cells in event ['cell_*'] properties
    # Outputs
    # Number of cells in the box/cluster with signif > 2

    if desired_cells is None:
        return np.nan
    cell_signif = desired_cells['cell_E'] / desired_cells['cell_Sigma']
    two_sigma_cells = desired_cells[abs(cell_signif)>=2]
    return len(two_sigma_cells) 



def get_physics_dictionary(list_cluster_cells):
    output_dict = {
        'energy'       : [CalculateEnergyFromCells(cells_i) for cells_i in list_cluster_cells],
        'eta'          : [CalculateEtaFromCells(cells_i) for cells_i in list_cluster_cells],
        'phi'          : [CalculatePhiFromCells(cells_i) for cells_i in list_cluster_cells],
        'eT'           : [CalculateEtFromCells(cells_i) for cells_i in list_cluster_cells],
        'n_cells'      : [CalculateNCellsFromCells(cells_i) for cells_i in list_cluster_cells],
        'noise'        : [CalculateNoiseFromCells(cells_i) for cells_i in list_cluster_cells],
        'significance' : [CalculateSignficanceFromCells(cells_i) for cells_i in list_cluster_cells],
        'neg_frac'     : [CalculateNegativeEnergyFromCells(cells_i) for cells_i in list_cluster_cells],
        'max_frac'     : [CalculateMaxEnergyFracFromCells(cells_i) for cells_i in list_cluster_cells],

        'energy2sig'   : [CalculateEnergyFromTwoSigmaCells(cells_i) for cells_i in list_cluster_cells],
        'eT2sig'       : [CalculateEtFromTwoSigmaCells(cells_i) for cells_i in list_cluster_cells],
        'n_cells2sig'  : [CalculateNCellsFromTwoSigmaCells(cells_i) for cells_i in list_cluster_cells],
    }

    return output_dict