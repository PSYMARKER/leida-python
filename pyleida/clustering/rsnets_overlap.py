"""Compute overlap between phase-locking states and Yeo resting-state networks"""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import nibabel as nib
import matplotlib.pyplot as plt
import seaborn as sns
import os

def compute_overlap(centroids,parcellation=None,n_areas=None):
    """
    Compute the overlap between the 7 resting-state networks
    defined in Yeo et al. (2011) and the brain cortical
    regions/parcels of the phase-locking states that were
    identified with K-Means clustering.

    Params:
    --------
    centroids : pd.dataframe with shape (209,n_rois+2)
        Contains the clusters centroids of
        each k partition. 
        1st column ('k') specifies the partition,
        2nd column the 'state', and rest of columns
        each brain region/parcel value.

    parcellation : str
        Specify path to your parcellation .nii file.
        Note: the parcellation must be of 2mm resolution.

    n_areas : None | int
        Analyze only the first n areas from the provided
        parcellation. 
        Usefull when the parcellation contains subcortical
        regions that must be ignored when computing the overlap
        with Yeo's cortical networks.

    Returns:
    --------
    correlations : ndarray with shape (rangeK, K_Max, 7networks).
        Contains the correlation coefficient
        between each phase-locking state and
        Yeo's 7 RSNs across the K range explored.

    pvalues : ndarray with shape (rangeK, K_Max, 7networks).
        Contains the p-values of the correlation
        coefficient between each phase-locking state
        and Yeo's 7 RSNs across the K range explored.
    """
    #validation of input data
    if isinstance(parcellation,str):
        if not parcellation.endswith(('.nii','.nii.gz')):
            raise ValueError("The parcellation must be either a .nii or .nii.gz file.")
    elif parcellation is None:
        raise ValueError("You must provide the path to the parcellation file.")
    else:
        raise TypeError("'parcellation' must be a string!")
        
    if n_areas is not None:
        if not isinstance(n_areas,int):
            raise TypeError("'n_areas' must be None or an integer!")

    #Step 1. Define the Yeo networks in the new parcellation

    #load our parcellation mask in MNI152 2mm space
    parc_user = nib.load(parcellation).get_fdata()

    if n_areas is None:
        n_areas = np.max(parc_user)
    else:
        parc_user[parc_user>n_areas] = 0 

    #load Yeo 7Networks parcellation mask in MNI152 2mm space
    yeo_path = os.path.dirname(__file__)
    parc_yeo = np.load(f'{yeo_path}/parc_MNI2mm.npz')['V_Yeo7']
    parc_yeo[parc_yeo>7] = 0 #delete cerebellum and subctx labels

    N_Yeo = np.max(parc_yeo) #number of Yeo networks
    yeo_in_user_parc = np.zeros((N_Yeo,n_areas))

    #create 7 vectors representing the 
    #7 Yeo RSNs in new parcellation scheme
    for n in range(n_areas):
        idx_n = parc_user==n+1
        for net in range(7):
            yeo_in_user_parc[net,n] = np.flatnonzero(parc_yeo[idx_n] == net+1).size / np.sum(idx_n)

    #Step 2. Compare with the LEiDA results
    kmax = 20
    krange = 19

    #create vector to store the correlation coefficients
    correlations = np.zeros((krange,kmax,N_Yeo))

    #create vector to store the coefficients' p-values
    pvalues = np.ones((krange,kmax,N_Yeo))

    for k in range(krange):
        #keep centroids for current K and
        #keep only the selected n areas
        centroids_ = centroids[centroids.k==k+2].iloc[:,2:n_areas+2].values
        centroids_[centroids_<0] = 0 #set negative values to 0

        n_centroids = centroids_.shape[0]

        for centroid_idx in range(n_centroids):
            #get current centroid
            centroid = centroids_[centroid_idx,:] 

            for yeo_net in range(N_Yeo):
                coef, pval = pearsonr(centroid,yeo_in_user_parc[yeo_net,:])

                correlations[k,centroid_idx,yeo_net] = coef
                pvalues[k,centroid_idx,yeo_net] = pval

    return correlations,pvalues

def state_overlap(correlations,pvalues,k=2,state=1,plot=True,darkstyle=False):
    """
    Return a dataframe with the correlation coefficients and 
    p-values of a particular phase-locking state with Yeo 7 
    resting-state networks.
    If plot is set to True, create a barplot showing the 
    correlation coefficient of each resting-state network.

    Params:
    -------
    correlations : ndarray with shape (rangeK, K_Max, 7networks).
        Contains the correlation coefficient between
        each phase-locking state and Yeo's 7 RSNs across
        the K range explored.

    pvalues : ndarray with shape (rangeK, K_Max, 7networks).
        Contains the p-values of the correlation
        coefficient between each phase-locking state
        and Yeo's 7 RSNs across the K range explored.

    k : int.
        Select the K-Means partition of interest.

    state : int.
        Select the phase-locking state of interest.

    plot : bool.
        Whether to create a barplot showing the correlation
        between the selected phase-locking state and the 7
        resting-state networks.

    darkstyle : bool.
        Whether to use a darkstyle for the barplot.

    Returns:
    --------
    state_data : pandas.dataframe with shape (7networks,3).
        Contains the correlation coefficient and p-value
        between the selected phase-locking state and each
        of the 7 resting-state networks from Yeo (2011).
    """
    nets = ['VIS','SMN','DAN','VAN','LIMB','CONTROL','DMN']
    colors = ['purple','royalblue','green','plum','khaki','orange','firebrick']

    state_data = pd.DataFrame(
        {
        'network':nets,
        'pcc':correlations[k-2,state-1,:7],
        'pval':pvalues[k-2,state-1,:7]
            }
        )

    if plot:
        idx = state_data['pcc']<0
        state_data['pval'][idx] = 1
        plt.ion()

        with plt.style.context("dark_background" if darkstyle else "default"):
            plt.figure(figsize=(7,3))
            sns.barplot(data=state_data,x='network',y='pcc',palette=colors)
            plt.xlabel('resting-state\nnetwork (Yeo 2011)',fontsize=16,labelpad=10)
            plt.ylabel('Pearson\ncorrelation',fontsize=16,labelpad=10)
            plt.ylim(top=1)
            plt.axhline(0,color='black' if not darkstyle else 'white')
            #add significance *
            for row_idx in range(state_data.shape[0]):
                if state_data.iloc[row_idx,-1]<0.05:
                    plt.text(x=row_idx,y=state_data.iloc[row_idx,1]+.01,s='*',fontweight='bold')
            plt.tight_layout()

    return state_data
    
def plot_yeo_pvalues(pvalues,k=2,state=1):
    """
    Create a scatter plot showing the p-values of
    correlation coefficients between the regions of
    the selected phase-locking state and the 7 RSNs
    from Yeo et al. (2011).
    
    Params:
    -------
    pvalues : ndarray with shape (rangeK, K_Max, 7networks).
        Contains the p-values of the correlation coefficient
        between each phase-locking state and Yeo's 7 RSNs across
        the K range explored.

    k : int.
        Select the 'k' partition of interest.

    state : int.
        Select the phase-locking state of interest.
    """
    nets = ['VIS','SMN','DAN','VAN','LIMB','CONTROL','DMN']
    colors = ['purple','royalblue','green','plum','khaki','orange','firebrick']
    data = pd.DataFrame({'network':nets,'pval':pvalues[k-2,state-1,:7]})
    plt.figure(figsize=(7,3))
    
    plt.scatter(np.arange(7),data.pval,c=colors)
    plt.xticks(np.arange(7),nets)
    plt.axhline(0.05,linestyle="dashed",c='firebrick',label=r'$\alpha^{1}$ = 0.05')
    plt.yscale('log')
    
    plt.xlabel('Yeo (2011) network',fontsize=16,labelpad=10)
    plt.ylabel('p-value',fontsize=16,labelpad=10)
    plt.ylim(top=1)
    plt.axhline(0,color='black')
    plt.legend(loc='best',frameon=False)
    plt.tight_layout()
    plt.show()

    return data
