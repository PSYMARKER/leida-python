from matplotlib.image import imread
import numpy as np
import pandas as pd
import os

#import matplotlib
#matplotlib.use('Qt5Agg')
#matplotlib.interactive(True)

import matplotlib.pyplot as plt
import seaborn as sns
import imageio

from nilearn.surface import vol_to_surf
from nilearn.maskers import NiftiLabelsMasker
from nilearn.datasets import fetch_surf_fsaverage

#from surfer import Brain
#%gui qt

def centroid2surf(centroid,discretize=False,parcellation=None):
    """
    Map a centroid/PL state to fsaverage cortical surface.

    The returned 'textures' can be used to generate plots
    either with nilearn's 'plot_surf_stat_map()' function,
    or with the '.add_data()' method of PySurfer's 'Brain'
    class (e.g.: brain.add_data(rh_texture,hemi='rh',thresh=None/.3,colormap=pal).

    Params:
    --------
    -centroid : numpy 1D array (N_Rois/parcels).
    -parcellation : str.
        Path to the .nii or .nii.gz file that contains
        our parcellation in MNI space. 
    -discretize : bool.
        Whether to keep the original (float) values of
        the centroid, or make a binarization.

    Returns:
    --------
    -texture : dict.
        Centroid/PL state mapped to cortical surface of both hemispheres.
        'lh' contains the map for left hemisphere, and 'rh' the map for the
        right hemisphere.
        
    """
    parcellation_mask = NiftiLabelsMasker(parcellation).fit()
    fs_surf = fetch_surf_fsaverage('fsaverage')

    n_rois = centroid.size
    if discretize:
        centroid_map = np.array([1 if node>0 else 0.1 for node in centroid]).reshape(1,n_rois)
    else:
        centroid_map = centroid.reshape(1,n_rois)

    stat_map = parcellation_mask.inverse_transform(centroid_map) #get stat map of current PL pattern to plot
    texture = {}
    texture['rh'] = vol_to_surf(stat_map,fs_surf['pial_right'])
    texture['lh'] = vol_to_surf(stat_map,fs_surf['pial_left'])

    return texture


def pattern_on_surfer(centroid,parcellation=None,surface='pial_semi_inflated',cortex='classic',hemi='split',views=['dorsal','lat','med'],discretize=True,bkg_color='white',cmap=None,only_rois=True):
    #from surfer import Brain 
    #%gui qt

    hemi_options = ['lh','rh','both','split']
    surf_options = ['inflated','pial','smoothwm','pial_semi_inflated','sphere','orig']
    cxt_options = ['classic','high_contrast','low_contrast','bone','ivory']

    if hemi not in hemi_options:
        raise ValueError(f"Selected 'hemi' is not valid. Available options are: {hemi_options}")

    if surface not in surf_options:
        raise ValueError(f"Selected 'surface' is not recognized. Available options are: {surf_options}")

    if cortex not in cxt_options:
        raise ValueError(f"'cortex' unrecognized. Valid options are {cxt_options}")

    if parcellation is None:
        raise ValueError(f"You must define the path to your 'parcellation' file!")

    texture = centroid2surf(centroid,parcellation=parcellation,discretize=discretize)
    if not cmap:
        pal = sns.diverging_palette(250, 15, s=75, l=40,n=9, center="dark",as_cmap=True)

    brain = Brain(
        'fsaverage',
        hemi,
        surface,
        #subjects_dir=f'{os.getcwd()}/example',
        background=bkg_color,
        views=views,
        cortex=cortex,
        alpha=.4 if only_rois else 1,
        show_toolbar=True,
        size=(850,500) if views=='lat' and hemi=='split' else 800
        )

    if hemi in ['both','split']:
        for hemi_ in ['lh','rh']:
            brain.add_data(
                texture[f'{hemi_}'],
                hemi=hemi_,
                thresh=.4 if only_rois else None,
                colormap=pal if not cmap else cmap,
                colorbar=False,
                time_label=None,
                alpha=.9,
                mid=None if discretize else 0.0 
                )

    else:
        brain.add_data(
            texture[f'{hemi}'],
            hemi=hemi,
            thresh=.4 if only_rois else None,
            colormap=pal if not cmap else cmap,
            colorbar=False,
            time_label=None,
            alpha=.9,
            mid=None if discretize else 0.0
            )

    if 'dorsal' in views:
        if hemi=='both':
            brain.show_view(roll=0,row=views.index('dorsal'),col=0)
        elif hemi=='split':
            for idx in range(2):
                brain.show_view(roll=0,row=views.index('dorsal'),col=idx)

    return brain


def states_transitions_gif(centroids,states_labels,parcellation=None,surf='pial_semi_inflated',hemi='split',views=['dorsal','lat','med'],discretize=False,darkstyle=False,cmap='jet',only_rois=False):
    plt.ioff()
    
    filenames = [] #create empty list to save the name of the created plot for each matrix
    for volume,state in enumerate(states_labels): #create plot of each volume.
        brain = pattern_on_surfer(
            centroids[state-1,:],
            parcellation=parcellation,
            surface=surf,
            cortex='classic',
            hemi=hemi,
            views=views,
            discretize=discretize,
            bkg_color='black' if darkstyle else 'white',
            cmap=cmap,
            only_rois=only_rois
            )
        
        filename_ = f'file{volume+1}.png' #define name to transiently save the figure
        filenames.append(filename_) #append the name of the current plot into the list that contains all the names
        brain.save_image(filename_)
        brain.close()
        #reopen and save generated imgs with matplotlib
        #to add title and state
        img_plt = plt.imread(filename_)
        with plt.style.context("dark_background" if darkstyle else "default"):
            plt.imshow(img_plt)
            plt.title(f'Volume: {volume+1} - PL state: {state}')
            sns.despine(bottom=True,left=True)
            plt.tick_params(axis='both',bottom=False,left=False,which='both',labelbottom=False,labelleft=False)
            plt.tight_layout()
            plt.savefig(filename_)
            plt.close()

    filename = 'cortical_dynamics'
    #Create the gif from the previously created plots
    with imageio.get_writer(f'{filename}.gif',mode='I',duration=1) as writer:
        for filename_ in filenames:
            image = imageio.imread(filename_)
            writer.append_data(image)
    
    #Eliminate the created single plots.
    for filename_ in set(filenames):
        os.remove(filename_)



def states_pyramid(states,view='lat',hemi='rh',darkstyle=False):

    K_min = 2
    K_max = 20

    #creating plot
    with plt.style.context('dark_background' if darkstyle else 'default'):
        fig,axs = plt.subplots(ncols=20,nrows=19)
        img = imread('ax_2.png')
        for idx in range(K_max-1):
            for state_idx in range(idx+2):
                axs[idx,state_idx].imshow(img)
                sns.despine(fig=fig,ax=axs[idx,state_idx],bottom=True,left=True)
                axs[idx,state_idx].tick_params(
                axis='both',
                which='both',
                bottom=False,
                top=False,
                left=False,
                labelbottom=False,
                labelleft=False)

                
            current_k = K_min+idx
            for add in np.arange(current_k,K_max):
                sns.despine(left=True,bottom=True,ax=axs[idx,add])
                axs[idx,add].tick_params(
                axis='both',
                which='both',
                bottom=False,
                top=False,
                left=False,
                labelbottom=False,
                labelleft=False)

            axs[idx,0].set_ylabel(f'K = {K_min+idx}',rotation=0,labelpad=30)

        plt.tight_layout(h_pad=0,w_pad=0,pad=0)
