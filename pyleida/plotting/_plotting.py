import pandas as pd
import numpy as np
import os 
import matplotlib.pyplot as plt
import seaborn as sns 
from nilearn.plotting import (
    view_connectome,
    view_markers,
    view_img_on_surf,
    plot_connectome,
    plot_surf_stat_map
)
from nilearn.maskers import NiftiLabelsMasker
from nilearn.surface import vol_to_surf
from nilearn.datasets import fetch_surf_fsaverage
import imageio


#Brain space plots
def brain_states_network(centroids,nodes_coords,state='all',node_size=15,node_color='black',linewidth=5,open=True):
    """
    Plot each provided cluster centroid (phase-locking state)
    as a connected network in a html file. Each ROI with a
    positive value is connected by an edge with all the other
    ROIs that have also positive values.
    Note: the order of brain regions in 'centroids' must
    coincide with the order of brain regions in 'nodes_coords'. 

    Params:
    -------
    centroids : ndarray with shape (N_centroids, N_rois) or (N_rois,).
        PL state/s to plot.

    nodes_coords : ndarray with shape (N_rois, 3). 
        Contains the coordinates (X, Y, Z) of each
        node of our parcellation in MNI space.

    state : 'all' or int.
        Select whether to plot 'all' the provided
        centroids or only a centroid of interest.

    node_size : int. 
        Select the size of the nodes.

    node_color : str. 
        Select the color of the nodes. If 'infer',
        then the nodes participating in the brain
        states are colored red and the rest blue.

    linewidth : int. 
        Select the size of the edges connecting
        the nodes.

    open : bool. 
        Whether to open the plots in web browser.

    Returns:
    --------
    plots : dict or single object. 
        A dictionary that contains each created plot,
        or a single plot. To open a particular plot, 
        use 'plots[x].open_in_browser()' and 'plots[x].save_as_html(args)'
        to save it.
    """
    #validation of input data
    if state!='all' and not isinstance(state,int):
        raise TypeError("'state' must be either 'all' or an integer specifying "
                        "the number of a particular PL state")
    
    if isinstance(state,int):
        centroids = centroids[state-1,:] #keep only the selected state vector

    N_centroids = centroids.shape[0] if centroids.ndim>1 else 1 #defining N of centroids
    N_rois = centroids.size if N_centroids==1 else centroids.shape[1] #defining N of ROIs.

    if N_rois != nodes_coords.shape[0]:
        raise Exception("The number of regions in 'centroids' and 'nodes_coords' must coincide.")

    if N_centroids>1:
        plots = {} #dictionary to save generated plots

    #creating connectiviy matrix of each centroid
    for centroid_idx in range(N_centroids):
        if N_centroids>1:
            centroid = centroids[centroid_idx,:]
        else:
            centroid = centroids.copy()

        network = centroid2network(centroid)

        #plotting
        g = view_connectome(
                network,
                nodes_coords,
                node_size=node_size,
                node_color=['mediumblue' if roi<0 else 'firebrick' for roi in centroid] if node_color=='infer' else 'black',
                linewidth=linewidth,
                colorbar=False,
                title=f'PL state {centroid_idx+1}' if state=='all' else f'PL state {state}'
                )
        
        #saving plot of current centroid in dictionary
        if N_centroids>1:
            plots[f'PL_state{centroid_idx+1}'] = g
        
        if open:
            g.open_in_browser()

    return plots if N_centroids>1 else g

def brain_states_nodes(centroids,nodes_coords,state='all',node_size=5,nodes_labels=None,open=True):
    """
    Create a 3D interactive figure embedded in a
    .html file showing the BOLD phase-locking (PL)
    states in anatomical MNI space. Each parcel/ROI
    is represented as a node. Nodes that are part of
    the PL pattern are coloured in red, and the rest
    of nodes are coloured in blue.

    Params:
    -------
    centroids : ndarray with shape (N_centroids, N_rois) or (N_rois,).
        PL state/s to plot. 

    nodes_coords : ndarray of shape (N_rois,3).
        Contains the coordinate (X,Y,Z) of each brain
        region in MNI space. The order must be the same
        as in 'centroids' columns.

    state : str | int.
        If 'all', then a figure of each centroid in
        'centroids' will be created. If 'int', then
        only the selected centroid will be plotted.
        If a single centroid is provided in 'centroids',
        then specify the PL state 'number' to define the
        plot title.

    node_size : int or float.
        Define the size of the nodes. Nodes that don't
        belong to the pattern are plotted smaller.

    nodes_labels : list (optional).
        Contains the name of each ROI. Must be in the
        same order as in 'nodes_coords'. Default is
        None, which doesn't show any node label.

    open : bool. 
        Whether to open the plots in web browser.

    Returns:
    --------
    plot/s : dict or single plot. 
        A dictionary that contains each created plot,
        or a single plot. To open a particular plot,
        use 'plots['PL_state_{x}'].open_in_browser()'
        and  'plots['PL_state_{x}'].save_as_html(args)'
        to save it.
    """

    #validation of input data
    if state!='all' and not isinstance(state,int):
        raise TypeError("'state' must be either 'all' or an integer specifying "
                        "the number of a particular PL state")
    
    if isinstance(state,int):
        centroids = centroids[state-1,:] #keep only the selected state vector

    N_centroids = centroids.shape[0] if centroids.ndim>1 else 1 #defining N of centroids
    N_rois = centroids.size if N_centroids==1 else centroids.shape[1] #defining N of ROIs.

    if N_rois != nodes_coords.shape[0]:
        raise Exception("The number of regions in 'centroids' and 'nodes_coords' must coincide.")

    if N_centroids>1:
        plots = {} #dictionary to save generated plots

    #plotting
    for centroid_idx in range(N_centroids):
        if N_centroids>1:
            centroid = centroids[centroid_idx,:]
        else:
            centroid = centroids.copy()

        #generating list of colors. Positive values get red, negative values get blue
        cols = ['mediumblue' if roi<0 else 'firebrick' for roi in centroid]
        #define node sizes
        sizes = [node_size if roi<0 else node_size*2 for roi in centroid]
        g = view_markers(
            nodes_coords,
            marker_size=sizes,
            marker_labels=nodes_labels if nodes_labels is None else [nodes_labels[roi_idx] if val>0 else '' for roi_idx,val in zip(range(centroid.size),centroid)],
            title=f'PL state {centroid_idx+1}' if state=='all' else f'PL state {state}',
            marker_color=cols
            ) #create plot object.
        
        #saving plot of current centroid in dictionary
        if N_centroids>1:
            plots[f'PL_state{centroid_idx+1}'] = g
        
        if open:
            g.open_in_browser()

    return plots if N_centroids>1 else g

def brain_states_on_surf(centroids,parcellation=None,discretize=True,cmap='auto',black_bg=False,open=True):
    """
    Create a 3D interactive figure embedded in a
    .html file showing the BOLD phase-locking (PL)
    states on cortical surface. By default, all the
    cortical regions that belong to a given PL state
    or pattern are coloured in red(s), while the rest
    of cortical regions are coloured in blue(s). You
    can change the colormap throught the 'cmap' argument.

    Params:
    ------
    centroids : ndarray with shape (N_centroids, N_rois) or (N_rois,).
        PL state/s to plot. 

    parcellation : str.
        Path to the parcellation file (.nii or .nii.gz).

    discretize : bool. Default = True.
        Whether to plot the raw values of the phase-locking
        state/centroid, or plot the brain regions than belong
        to the phase-locking state with the same intensity.

    cmap : str or matplotlib colormap, optional. Default = 'auto'.
        Colormap to use in the brain plot.
        If 'auto', then the brain regions that
        belong to the phase-locking state will
        be coloured in red, and the rest of regions
        in blue.

    black_bg : bool. 
        Whether to use a black background.

    open : bool. 
        Whether to open the plots in browser.
    """
    if isinstance(parcellation,str):
        if not parcellation.endswith(('.nii','.nii.gz')):
            raise ValueError("The parcellation must be either a .nii or .nii.gz file.")
    elif parcellation is None:
        raise ValueError("You must provide the path to the parcellation file.")
    else:
        raise TypeError("'parcellation' must be a string!")

    n_rois = centroids.shape[1] if centroids.ndim>1 else centroids.size
    n_centroids = centroids.shape[0] if centroids.ndim>1 else 1
    mask = NiftiLabelsMasker(parcellation).fit()

    if n_centroids>1:
        plots = {}

    if cmap=='auto':
        cmap = sns.diverging_palette(250, 15, s=75, l=40,n=9, center="dark",as_cmap=True)

    vol2surf_kwargs = {'interpolation':'linear','radius':1.0} 
    for c in range(n_centroids):
        if discretize: #for each centroid
            if n_centroids>1:
                centroid_map = np.array([1 if node>0.0 else 0.1 for node in centroids[c,:]]).reshape(1,n_rois)
            else:
                centroid_map = np.array([1 if node>0.0 else 0.1 for node in centroids]).reshape(1,n_rois)
        else:
            if n_centroids>1:
                centroid_map = centroids[c,:].reshape(1,n_rois)
            else:
                centroid_map = centroids.reshape(1,n_rois)

        stat_map = mask.inverse_transform(centroid_map) #get stat map of current PL pattern to plot

        g = view_img_on_surf(
            stat_map,
            surf_mesh='fsaverage',
            black_bg=black_bg,
            vmin=0 if discretize else None,
            vmax=1 if discretize else None,
            symmetric_cmap=False if discretize else True,
            cmap=cmap,
            colorbar=False if discretize else True,
            colorbar_height=0.25,
            threshold=None,
            vol_to_surf_kwargs=vol2surf_kwargs
            ) #plot current centroid
        
        if n_centroids>1:
            plots[f'PL_state{c+1}'] = g

        if open: 
            g.open_in_browser()

    return plots if n_centroids>1 else g

def brain_states_on_surf2(centroid,parcellation=None,surface='pial',hemi='right',view='lateral',only_mesh=True,mesh_alpha=0.05):
    """
    Plot a BOLD phase-locking state of interest 
    on cortical surface mesh. 

    Params:
    -------
    centroids : ndarray with shape (N_rois).
        Contain the centroid/s.

    parcellation : str.
        Path to the .nii or .nii.gz file containing the
        parcellation from which the signals were extracted.

    surface : str.
        Specify the surface type to plot the pattern on.
        Valid options are 'pial','infl', and 'white'.

    hemi : str.
        Select the hemisphere to plot. 
        Valid options are'right', 'left' or 'both'.

    view : str.
        View of the surface that is rendered. 
        Default='lateral'. Options = {'lateral', 'medial',
        'dorsal', 'ventral', 'anterior', 'posterior'}.
        If 'hemi'='both', then 'dorsal' and 'lateral' views
        are displayed.

    only_mesh : bool.
        Whether to show only the cortical mesh, or add
        background with sulcus information.

    mesh_alpha : float. Default = 0.05
        Specify the transparency of the mesh.

    Returns:
    --------
    g : SurfaceView. 
    """
    if isinstance(parcellation,str):
        if not parcellation.endswith(('.nii','.nii.gz')):
            raise ValueError("The parcellation must be either a .nii or .nii.gz file.")
    elif parcellation is None:
        raise ValueError("You must provide the path to the parcellation file.")
    else:
        raise TypeError("'parcellation' must be a string!")
    if hemi not in ['left','right','both']:
        raise ValueError("'hemi' must be either 'right', 'left', or 'both'.")
    elif hemi=='both':
        print("WARNING: 'view' is automatically set to 'dorsal' and 'lateral' when 'hemi'='both'.")
    if surface not in ['pial','white','infl']:
        raise ValueError("'surface' must be either 'pial','infl', or 'white'.")
    view_options = ['lateral', 'medial', 'dorsal', 'ventral', 'anterior', 'posterior'] 
    if view not in view_options:
        raise ValueError(f"Valid options for 'view' are {view_options} .")

    n_rois = centroid.size
    
    parcellation_mask = NiftiLabelsMasker(parcellation).fit()
    surf = fetch_surf_fsaverage('fsaverage')
    pal = sns.diverging_palette(250, 15, s=75, l=40,n=9, center="dark",as_cmap=True)

    centroid_map = np.array([1 if node>0.0 else 0.1 for node in centroid]).reshape(1,n_rois)
        
    #get stat map of current PL pattern to plot
    stat_map = parcellation_mask.inverse_transform(centroid_map)

    plt.ion()
    if hemi!='both':
        texture = vol_to_surf(stat_map,surf[f'pial_{hemi}'])

        fig_ = plot_surf_stat_map(
            surf[f'{surface}_{hemi}'],
            texture,
            threshold=0.2, 
            hemi=hemi,
            view=view,
            #title='Surface right hemisphere', 
            colorbar=False, 
            bg_map=None if only_mesh else surf[f'sulc_{hemi}'],
            alpha=mesh_alpha,
            cmap=pal,
            bg_on_data=False
            )

        if view=='dorsal':
            ax = fig_.axes
            ax[0].view_init(90,270)

    else:
        fig_, axes_ = plt.subplots(nrows=2,ncols=2,figsize=(11,11),subplot_kw={'projection': '3d'})

        ax_config = {
            0 : ['left','dorsal'],
            1 : ['right','dorsal'],
            2 : ['left','lateral'],
            3 : ['right','lateral']
            }

        axes_ = np.ravel(axes_)

        texture = {}
        texture['left'] = vol_to_surf(stat_map,surf['pial_left'])
        texture['right'] = vol_to_surf(stat_map,surf['pial_right'])

        for ax_idx,ax in enumerate(axes_):
            hemi = ax_config[ax_idx][0]
            view = ax_config[ax_idx][1]

            plot_surf_stat_map(
                surf[f'{surface}_{hemi}'],
                texture[hemi],
                threshold=0.2, 
                hemi=hemi,
                view=view,
                colorbar=False, 
                bg_map=None if only_mesh else surf[f'sulc_{hemi}'],
                alpha=mesh_alpha,
                cmap=pal,
                bg_on_data=False,
                axes=ax,
                figure=fig_,
                #engine='plotly',
                #kwargs={'symmetric_cmap':False}
                )

        axes_[0].view_init(90,270)
        axes_[1].view_init(90,270)
        plt.tight_layout(pad=0,h_pad=0,w_pad=0)

    return fig_

def states_k_glass(centroids,coords,darkstyle=False):
    """
    Create a glass brain (axial view) showing the
    network representation of each PL pattern for
    the selected 'k' partition.

    Params:
    -------
    centroids : ndarray with shape (N_centroids, N_rois).
        Contains the centroids (PL states) of a 
        specific 'k' partition.

    coords : ndarray with shape (N_rois, 3).
        ROIs coordinates in MNI space.
        
    darkstyle : bool.
        Whether to use a dark theme for the plots.
    """
    if not isinstance(centroids,np.ndarray) or (isinstance(centroids,np.ndarray) and centroids.ndim!=2):
        raise TypeError("'centroids' must be a 2D array!")
    
    if centroids.shape[1]!=coords.shape[0]:
        raise Exception("The number of brain regions in 'centroids' and 'coords' must be the same!")
    
    N_states = centroids.shape[0]

    #Decide number of columns and rows
    if N_states>10:
        if not N_states%2==0:
            n_columns = int((N_states+1)/2)
        else:
            n_columns = int(N_states/2)
    else:
        n_columns = N_states

    #creating plot
    plt.ion()
    
    fig,axs = plt.subplots(
        ncols=n_columns,
        nrows=1 if N_states<=10 else 2,
        figsize=(n_columns*2,2.5 if N_states<=10 else 5),
        sharey=False,
        sharex=False,
        subplot_kw=dict(box_aspect=1.3),
        constrained_layout=False
        )
    axs = np.ravel(axs)

    sizes_and_lws = {
        'n_columns':np.arange(2,11),
        'lw':np.linspace(.2,.8,9)[::-1],
        'node_size':np.linspace(10,20,9)[::-1]
    }

    for state_idx in range(N_states): #for each centroid/state in current k
        edges_lw = {'linewidth':sizes_and_lws['lw'][n_columns-2],'color':'firebrick'}
        centroid = centroids[state_idx,:]
        network = centroid2network(centroid)

        plot_connectome(
            network, 
            coords, 
            node_color='blue' if not np.any(network) else 'black' if not darkstyle else 'white', 
            node_size=sizes_and_lws['node_size'][n_columns-2], 
            #edge_cmap=<matplotlib.colors.LinearSegmentedColormap object>, 
            edge_vmin=None, 
            edge_vmax=None, 
            edge_threshold=None, 
            output_file=None, 
            display_mode='z', 
            figure=fig, 
            axes=axs[state_idx], 
            #title=f'FC pattern {state_idx+1}', 
            annotate=True, 
            black_bg=True if darkstyle else False, 
            alpha=0.3, 
            edge_kwargs=edges_lw, 
            node_kwargs=None, 
            colorbar=False
            )

        axs[state_idx].set_title(f'PL pattern {state_idx+1}',fontweight='bold',fontsize=8)
        
    #if k is odd, delete the lines and contents of the empty plot.
    if N_states>10:
        if not N_states%2==0:
            sns.despine(left=True,bottom=True,ax=axs[-1]) #delete axis
            axs[-1].tick_params( #delete thicks info
                        axis='both',          
                        which='both',      
                        bottom=False,      
                        top=False,         
                        left=False,      
                        labelbottom=False,
                        labelleft=False)

def brain_dynamics_gif(states_labels,centroids,coords,filename='dfc',darkstyle=False):
    """
    Create a .gif file showing an animated network
    representation of the  detected phase-locking
    pattern of each volume for a given subject.

    Params:
    -------
    states_labels : ndarray of shape (N_volumes). 
        Contain the predicted labels for each
        PL state following a specific K partition.

    centroids : ndarray of shape (N_centroids,N_ROIs).
        Contain the centroids for a given k
        partition.

    coords : ndarray of shape (N_ROIs,X-Y-Z).
        Contain the nodes coordinates of each
        ROI in MNI space.

    filename : str.
        Select the name of the created gif
        file.

    darkstyle : bool.
        Whether to create the plot using a
        dark theme.
    """
    #validate inputs
    if not isinstance(centroids,np.ndarray):
        raise TypeError("'centroids' must be a 2D array (N_centroids,N_ROIs)")
    if not isinstance(coords,np.ndarray):
        raise TypeError("'coords' must be a 2D array (N_ROIs,3)")
    if not isinstance(filename,str):
        raise TypeError("'filename' must be a string!")
    if not isinstance(darkstyle,bool):
        raise TypeError("'darkstyle' must be True or False!")
    if centroids.shape[1] != coords.shape[0]:
        raise Exception("The number of brain regions in 'centroids' "
                        "and 'coords' must be the same!")

    #plotting
    N_centroids,N_rois = centroids.shape
    networks = np.zeros((N_rois,N_rois,N_centroids))
    for centroid_idx in range(N_centroids): #for each centroid/state in current k
        centroid = centroids[centroid_idx,:]
        networks[:,:,centroid_idx] = centroid2network(centroid)

    filenames = [] #create empty list to save the name of the created plot for each matrix
    plt.ioff()
    with plt.style.context("dark_background" if darkstyle else "default"):
        for volume,state in enumerate(states_labels): #create plot of each volume.
            edges_lw = {'linewidth':.8,'color':'firebrick'}
            #plt.figure()
            fig,ax = plt.subplots(ncols=1,nrows=1)

            plot_connectome(
                networks[:,:,state], 
                coords, 
                node_color='blue' if not np.any(networks[:,:,state]) else 'black' if not darkstyle else 'white', 
                node_size=20, 
                #edge_cmap=<matplotlib.colors.LinearSegmentedColormap object>, 
                edge_vmin=None, 
                edge_vmax=None, 
                edge_threshold=None, 
                output_file=None, 
                display_mode='yz', 
                figure=fig, 
                axes=ax, 
                #title=f'Volume: {volume+1} - FC pattern {state+1}', 
                annotate=True, 
                black_bg=True if darkstyle else False, 
                alpha=0.3, 
                edge_kwargs=edges_lw, 
                node_kwargs=None, 
                colorbar=False
                )

            ax.set_title(f'Volume: {volume+1} - PL pattern {state+1}')
            filename_ = f'file{volume}.png' #define name to transiently save the figure
            filenames.append(filename_) #append the name of the current plot into the list that contains all the names
            plt.savefig(filename_) #save the plot as .png file
            plt.close()
    
    #Create the gif from the previously created plots
    with imageio.get_writer(f'{filename}.gif',mode='I') as writer:
        for filename_ in filenames:
            image = imageio.imread(filename_)
            writer.append_data(image)
    
    #Eliminate the created single plots.
    for filename_ in set(filenames):
        os.remove(filename_)


#Matrices plots
def matrices_gif(mats,cmap='jet',filename='dfc',vmin=-1,vmax=1,darkstyle=False):
    """
    Create a gif file showing the phase-coherence
    connectivity matrix for each time point of a
    given subject.
    
    Params:
    -------
    mats : ndarray of shape (N_rois, N_rois, N_volumes).
        Phase-coherence connectivity matrices
        of a particular subject.

    cmap : str. 
        Select the colormap of the heatmaps.

    filename : str. 
        Select the name of the created gif file.

    vmin : float. 
        Select the minimum value of the colormap.

    vmax : float. 
        Select the max value of the colormap.

    darkstyle : bool.
        Whether to use a dark theme for the plot.
    """
    if not isinstance(mats,np.ndarray):
        raise TypeError("'mats' must be a 3D array!")
    if mats.ndim!=3:
        raise Exception("'mats' must be a 3D array!")
    
    N_rois = mats.shape[0]
    filenames = [] #create empty list to save the name of the created plot for each matrix
    try:
        plt.ioff()
        with plt.style.context("dark_background" if darkstyle else "default"):
            linecolor = 'w' if darkstyle else 'k'
            for mat in range(mats.shape[-1]): #create plot of each matrix.
                plt.figure()
                sns.heatmap(
                    mats[:,:,mat],
                    square=True,
                    vmin=vmin,
                    vmax=vmax,
                    center=0,
                    cmap=cmap,
                    cbar_kws={"shrink": .5,"label": "Phase-coherence"}
                    )
                plt.xticks(
                    np.arange(20,N_rois,20),
                    np.arange(20,N_rois,20).tolist(),
                    rotation=0
                    )
                plt.yticks(
                    np.arange(20,N_rois,20),
                    np.arange(20,N_rois,20).tolist(),
                    )
                plt.xlabel('Brain region', fontsize=15,fontweight='regular')
                plt.ylabel('Brain region', fontsize=15,fontweight='regular') 
                plt.axhline(y=0, color=linecolor,linewidth=5)
                plt.axhline(y=mats[:,:,mat].shape[1], color=linecolor,linewidth=5)
                plt.axvline(x=0, color=linecolor,linewidth=5)
                plt.axvline(x=mats[:,:,mat].shape[0], color=linecolor,linewidth=5)
                plt.title(f'TR = {mat+1}',fontweight='regular',fontsize=18)
                plt.tight_layout()
                filename_ = f'file{mat}.png' #define name to transiently save the figure
                filenames.append(filename_) #append the name of the current plot into the list that contains all the names
                plt.savefig(filename_) #save the plot as .png file
                plt.close()
        
        #Create the gif from the previously created plots
        with imageio.get_writer(f'{filename}.gif',mode='I') as writer:
            for filename_ in filenames:
                image = imageio.imread(filename_)
                writer.append_data(image)
        
        #Eliminate the created single plots.
        for filename_ in set(filenames):
            os.remove(filename_)

    except:
        raise Exception("An error occured when creating the .gif file.")
        
def plot_static_fc_matrices(signals,n_rows,n_columns,cmap='jet',darkstyle=False):
    """
    Plot each subject static functional connectivity
    matrix (Pearson correlation as metric) in the same
    figure.
    
    Params:
    -------
    signals : dict. 
        Contains 'subject_ids' as keys and 2D arrays
        with the time series (N_rois, N_signals) as
        values.

    n_rows : int. 
        Define the number of rows of the plot.

    n_columns : int. 
        Define the number of columns of the plot.

    cmap : str. 
        Colormap for the heatmaps.

    darkstyle : bool.
        Whether to use a darkstyle for the create plot.
    """
    N_subjects = len(signals.keys())

    if n_rows*n_columns<N_subjects:
        raise ValueError("n_rows x n_columns must be higher or equal to the number of matrices to plot!")

    with plt.style.context('dark_background' if darkstyle else 'default'):
        _, axs = plt.subplots(
            n_rows,
            n_columns,
            figsize=(n_columns*2, n_rows*2),
            edgecolor='black',
            subplot_kw=dict(box_aspect=1)
            )

        axs = axs.ravel()    
        
        i = 0
        while i<N_subjects:
            for su in signals.keys():
                sns.heatmap(
                    np.corrcoef(signals[su]),
                    ax=axs[i],
                    vmax=1,
                    vmin=-1,
                    center=0,
                    cmap=cmap,
                    square=True,
                    cbar_kws={"shrink": 0.3,'label':'Pearson\ncorrelation'}
                    )
                axs[i].set_title(su,fontsize=7)

                axs[i].tick_params(
                    axis='both',          # changes apply to the x-axis
                    which='both',      # both major and minor ticks are affected
                    bottom=False,      # ticks along the bottom edge are off
                    left=False,
                    top=False,
                    labelsize=5,         # ticks along the top edge are off
                    labelbottom=False,
                    labelleft=False
                    ) # labels along the bottom edge are off

                i += 1

        if N_subjects!=n_columns*n_rows:
            missing = np.arange(N_subjects,n_columns*n_rows,1)
            for idx in missing:
                sns.despine(left=True,bottom=True,ax=axs[idx]) #delete axis
                axs[idx].tick_params( #delete thicks info
                            axis='both',       
                            which='both',      
                            bottom=False,      
                            top=False,         
                            left=False,      
                            labelbottom=False,
                            labelleft=False)

    plt.tight_layout(pad=1)


#clustering plots
def states_in_bold(signals,y,alpha=0.7):
    """
    Create plot showing the time-series of BOLD signals, 
    highlighting the dominant phase-locking (PL) pattern
    of each time point or volume.

    Params:
    -------
    signals : ndarray with shape (N_ROIs,N_volumes).
        Contains the signals of the subject
        of interest.

    y : ndarray with shape (N_volumes).
        Contains the cluster assignement of
        each time point or volume in 'signals'.

    alpha : float.
        Transparency of the background.
    """
    colors = {
        0:'royalblue',
        1:'grey',
        2:'tomato',
        3:'orange',
        4:'cyan',
        5:'violet',
        6:'yellow',
        7:'purple',
        8:'firebrick',
        9:'teal',
        10:'orchid',
        11:'red',
        12:'green',
        13:'steelblue',
        14:'indigo',
        15:'gold',
        16:'sienna',
        17:'coral',
        18:'olive',
        19:'salmon'
        }

    if signals.shape[1]!=y.size:
        raise ValueError(f"The number of time points of the provided signals ({signals.shape[1]}) "
                        f"must coincide with the number of provided labels ({y.size}")
    
    y_colors = pd.Series(y).map(colors).values

    plt.figure(figsize=(13,3.25))
    plt.plot(signals.T,c="dimgrey")
    plt.plot(np.mean(signals.T,axis=1),c="black",linewidth=3)

    N_volumes = signals.shape[1]
    for time_point,state in zip(range(N_volumes),y_colors):
        plt.axvspan(time_point-0.5,time_point+0.5,alpha=alpha,color=state)

    plt.xlabel("Time points (volumes)",fontsize=18)
    plt.ylabel("BOLD signals",fontsize=18)
    plt.xlim(0-0.5,(N_volumes-1)+0.5)

    plt.tight_layout()
    plt.show()

def states_in_bold_gif(filename,signals,y,alpha=0.7,duration=0.1,keep_previous=True,show_labels=True):
    """
    Create .gif file showing the BOLD time series and
    the dominant phase-locking state of each time point
    or volume.

    Params:
    -------
    filename : str.
        Define the name of the .gif file.

    signals : ndarray with shape (N_rois,N_volumes).
        Contains the BOLD time series to plot.

    y : ndarray with shape (N_volumes,).
        Contains the label of the dominant PL
        state of each time point or volumne.

    alpha : float.
        Transparency of lines that highlight the
        dominant PL state.

    duration : float.
        Duration of each frame.

    keep_previous : bool.
        Whether to preserve the previous lines
        that highlight the dominant PL states.

    show_labels : bool.
        Whether to show text indicating the number
        of the dominant PL state.
    """

    filenames = []
    colors = {
        0:'royalblue',
        1:'grey',
        2:'tomato',
        3:'orange',
        4:'cyan',
        5:'violet',
        6:'yellow',
        7:'purple',
        8:'firebrick',
        9:'teal',
        10:'orchid',
        11:'red',
        12:'green',
        13:'steelblue',
        14:'indigo',
        15:'gold',
        16:'sienna',
        17:'coral',
        18:'olive',
        19:'salmon'
        }

    if signals.shape[1]!=y.size:
        raise Exception(f"The number of time points of the provided signals ({signals.shape[1]}) "
                        f"must coincide with the number of provided labels ({y.size}")

    y_colors = pd.Series(y).map(colors).values
    N_volumes = signals.shape[1]

    #plt.figure()
    plt.ioff()
    plt.figure(figsize=(13,3.25))
    plt.plot(signals.T,c="dimgrey")
    plt.plot(np.mean(signals.T,axis=1),c="black",linewidth=3)
    plt.title('')
    plt.xlabel("Time points (volumes)",fontsize=18)
    plt.ylabel("BOLD signals",fontsize=18)
    plt.xlim(0-0.5,(N_volumes-1)+0.5)
    plt.tight_layout()
    filename_ = f'file0.png'
    filenames.append(filename_)
    plt.savefig(filename_)
    if not keep_previous:
        plt.close()

    for time_point,state in zip(range(N_volumes),y_colors):
        if not keep_previous:
            plt.figure(figsize=(13,3.25))
            plt.plot(signals.T,c="dimgrey")
            plt.plot(np.mean(signals.T,axis=1),c="black",linewidth=3)
            plt.xlabel("Time points (volumes)",fontsize=16)
            plt.ylabel("BOLD signals",fontsize=16)
            plt.xlim(0-0.5,(N_volumes-1)+0.5)
        if show_labels and not keep_previous:
            plt.axvspan(time_point-0.5,time_point+0.5,alpha=alpha,color=state,label=f'PL state {y[time_point]+1}')
            plt.legend(loc=2,prop={'size': 15})
        if show_labels and keep_previous:
            plt.axvspan(time_point-0.5,time_point+0.5,alpha=alpha,color=state)
            title_obj = plt.title(f'State {y[time_point]+1}',fontsize=22,fontweight='regular')
            plt.setp(title_obj, color=state)
            plt.tight_layout()

        else:
            plt.axvspan(time_point-0.5,time_point+0.5,alpha=alpha,color=state)
        if not keep_previous:
            plt.tight_layout()
        filename_ = f'file{time_point+1}.png' #define name to transiently save the figure
        filenames.append(filename_) #append the name of the current plot into the list that contains all the names
        plt.savefig(filename_) #save the plot as .png file
        if not keep_previous:
            plt.close()
    if keep_previous:
        plt.close()

    #Create the gif from the previously created plots
    with imageio.get_writer(f'{filename}.gif',mode='I',duration=duration) as writer:
        for filename_ in filenames:
            image = imageio.imread(filename_)
            writer.append_data(image)
    
    #Eliminate the created single plots.
    for filename_ in set(filenames):
        os.remove(filename_)

#Utils
def centroid2network(centroid):
    """
    Create a network representation of a centroid,
    such that all the regions that have positive
    values are connected between each other.

    Params:
    -------
    centroid : ndarray with shape (N_ROIs).
        Centroid of interest.

    Returns:
    ---------
    network_matrix : ndarray with shape (N_ROIs,N_ROIs).
        Network representation of the
        provided 'centroid'.
    """
    N_rois = centroid.size
    network_matrix = np.zeros((N_rois,N_rois))
    for node_idx,node_value in enumerate(centroid):
        if node_value>0:
            for node2_idx,node2_value in enumerate(centroid):
                if node2_value>0:
                    network_matrix[node_idx,node2_idx] = 1
                    network_matrix[node2_idx,node_idx] = 1
    return network_matrix

def _save_html(path,plot,k,state=None,plot_type=None):
    """
    Save brain plot/s as .html files.

    Params:
    --------
    path : str.
        Specify the folder in which
        the 'brain_plots' folder will
        be created.

    plot : dict or single plot.

    k : int.
        Specify the k-means partition.

    state : int.
        Specify a specific PL state number,
        if single plot is provided.

    plot_type : str.
        Type of plot (to use for the name of
        the saved files).
        E.g.: 'network', 'nodes', 'surface'.
    """
    try: 
        path = f'{path}/brain_plots'
        if not os.path.exists(path): 
            os.makedirs(path)
        if isinstance(plot,dict):           
            for fig in plot.keys():
                filename = f"{path}/K{k}_{fig}_{plot_type}.html"
                plot[fig].save_as_html(filename)
                print(f"The plot was saved at: {filename}")
        else:
            filename = f"{path}/K{k}_PL_state_{state}_{plot_type}.html"
            plot.save_as_html(filename)
            print(f"The plot was saved at: {filename}")
    except:
        print("The figures could't be saved on local folder.")

#Dynamical systems theory metrics plots

def plot_pyramid(metric_data,stats,K_min=2,K_max=20,class_column='condition',metric_name=None,despine=True):
    """
    Create pyramid showing the metric of interest
    for each group, cluster, and K. Significant
    states are coloured 'blue' if the associated
    p-value is lower than 0.05 but higher than 0.05/k,
    and 'red' if the p-value is lower than 0.05/k.

    Params:
    -------
    metric_data : dict.
        Contains the computed occupancies or
        dwell times for each k partition.

    stats : pandas.dataframe.
        Contains the pooled stats across values
        of K for a given metric (occupancies, dwell times).

    K_min,K_max : int.
        Min and max K partitions explored.

    class_column : str.
        Specify the name of the column that
        contains the classes labels.

    metric_name : str | None. Default: None.
        Specify the name of the metric to
        plot (only used for title.)

    despine : bool.
        Whether to despine top and right edges
        of the subplots.
    """
    alpha3 = np.sum(np.arange(K_min,K_max+1,1))
    
    #generate list with colors specifying the
    # color of the barplot of a particular state.
    color_list = []
    stats_df = stats.copy()
    for pval,bonf in zip(stats['p-value'],stats['alpha_Bonferroni']):
        color_list.append('green' if 0.05/alpha3<pval<bonf 
                        else 'firebrick' if 0.05>pval>bonf 
                        else 'royalblue' if pval<0.05/alpha3 
                        else 'black')

    stats_df['color'] = color_list

    #creating plot
    plt.ion()
    _,axs = plt.subplots(
        ncols=K_max,
        nrows=K_max-K_min+1,
        figsize=(15,10),
        sharex=True,
        subplot_kw=dict(box_aspect=1)
        )

    cond1 = np.unique(stats.group_1).item()
    cond2 = np.unique(stats.group_2).item()

    title = f"{cond1} vs {cond2}"
    if metric_name is not None:
        title = f"{metric_name.capitalize().replace('_',' ')}: {title}"
    _.suptitle(title,fontsize=20,fontweight='bold')

    for idx in range(K_max-K_min+1): #idx of rows/k's
        df = metric_data[f'k_{K_min+idx}']
        class_column_idx = df.columns.get_loc(class_column) #get the location (idx) of the class column

        for feature_idx,feature in enumerate(df.columns[class_column_idx+1:].values):
            sns.barplot(
                data=df,
                x=class_column,
                y=feature,
                ax=axs[idx,feature_idx],
                linewidth=0,
                color=stats_df[(stats_df.k==K_min+idx)&(stats.variable.str.contains(feature))].color.values[0],
                errcolor=".2", 
                edgecolor='white'
                )
            axs[idx,feature_idx].set_box_aspect(1)
            axs[idx,feature_idx].set_xlabel('')
            axs[idx,feature_idx].set_ylabel('')
            axs[idx,0].set_ylabel(f'K = {K_min+idx}',rotation=0,labelpad=30)
            axs[idx,feature_idx].spines[['bottom','top','right','left']].set_color('black')
            if despine:
                sns.despine(ax=axs[idx,feature_idx])
            axs[idx,feature_idx].tick_params(axis='y',labelsize=4)
            axs[idx,feature_idx].tick_params(axis='x',labelsize=7,labelrotation=45)

        #delete all the unpopulated subplots    
        current_k = K_min+idx 
        for add in np.arange(current_k,K_max): #iterate throught the empty subplots
            sns.despine(left=True,bottom=True,ax=axs[idx,add]) #delete axis
            axs[idx,add].tick_params( #delete thicks info
                        axis='both',         
                        which='both',      
                        bottom=False,
                        top=False,
                        left=False,      
                        labelbottom=False,
                        labelleft=False)

    plt.tight_layout(pad=.4,w_pad=.6)
    plt.show()

def _explore_state(centroid,rois_labels,occupancy,dwell_times,coords,state_number=None,darkstyle=False):
    """
    Create a figure showing a phase-locking state of
    interest in different formats:
    a barplot, a network representation in brain space,
    a matrix representation, and two boxplots with the
    occupancies and dwell times for each group/condition.

    Params:
    -------
    centroid : ndarray with shape (N_rois,).
        Vector representing a specific PL state.

    rois_labels : list.
        Contains the ROIs/parcels labels.

    occupancy, dwell_times : pd.DataFrame (2 columns).
        First column must be called condition, and
        the values must specify the group/condition
        of each observation. Second column (with any name)
        must contain the occupancy/dwell time of each subject
        for the PL state of interest.

    coords : ndarray with shape (N_rois, 3).
        Coordinates (in MNI space) of each ROI/parcel.

    state_number : int. Optional.
        Specify the number of the PL state of interest.

    darkstyle : bool.  
        Whether to use a dark background for the plot.
    """
    #validation of input data
    if not isinstance(centroid,np.ndarray):
        raise TypeError("'centroid' must be a ndarray with shape (N_rois,)!")
    if not centroid.size==len(rois_labels)==coords.shape[0]:
        raise ValueError("The number of regions in 'centroid','rois_labels', and 'coords' must be the same!")
    for metric in [occupancy,dwell_times]:
        if metric.columns[0]!='condition':
            raise Exception("The 1st column in 'occupancy' and 'dwell_times' must be called 'condition'.")

    ax_positions = [['left','upper center', 'upper right'],
                    ['left','lower center', 'lower right']]

    plt.ion()

    _, axd = plt.subplot_mosaic(
        ax_positions,
        figsize=(8, 8), 
        constrained_layout=False
        )

    #title of complete figure
    _.suptitle(
        'PL state' if state_number is None 
        else f'PL state {state_number}',
        fontsize=15,
        fontweight='bold'
        )

    #creating barplot with centroid values.
    centroid_ = pd.DataFrame(
        {'roi':rois_labels,
        'value':centroid}
        ).sort_values('value',ascending=False).reset_index()

    sns.barplot(
        x=centroid_.value,
        y=centroid_.roi,
        palette=['firebrick' if i>0 else 'royalblue' for i in centroid_.value],
        ax=axd['left']
        )

    if np.max(np.abs(centroid_.value))>0.15:
        axd['left'].set_xlim(-1.05,1.05)
    else:
        axd['left'].set_xlim(-0.11,0.11)

    axd['left'].set_title('fMRI phase\nprojection '+r'into $V_{c}$',fontsize=10)
    axd['left'].set_xlabel(' ')
    axd['left'].set_ylabel('Brain regions',fontsize=16,labelpad=20)
    axd['left'].tick_params(labelsize=5,axis='y')
    axd['left'].axvline(0,color='black' if not darkstyle else 'white')

    #creating brainplot
    network = centroid2network(centroid)
    edges_lw = {'linewidth':1,'color':'firebrick'}

    #nodes that don't belong to the
    #PL state are not shown
    if np.where(centroid>0)[0].size==0:
        sizes = 20
    else:
        sizes = [20 if roi>0 else 0 for roi in centroid]

    plot_connectome(
        network, 
        coords, 
        node_color='blue',
        node_size=sizes, 
        #edge_cmap=<matplotlib.colors.LinearSegmentedColormap object>, 
        edge_vmin=None, 
        edge_vmax=None, 
        edge_threshold=None, 
        output_file=None, 
        display_mode='z', 
        figure=_, 
        axes=axd['upper center'], 
        annotate=True, 
        black_bg=True if darkstyle else False, 
        alpha=0.1, 
        edge_kwargs=edges_lw, 
        node_kwargs=None, 
        colorbar=False
        )

    #axd['upper center'].set_title(f'PL state')

	#creating centroid in matrix format

    #scale centroid by its maximum value and transpose the matrix
    centroid_scaled = centroid/np.max(np.abs(centroid))
    matrix = np.outer(centroid_scaled,centroid_scaled.T)
    sns.heatmap(
        matrix,
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        cmap='jet',
        ax=axd['upper right'],
        cbar=False
        #cbar_kws={'shrink':0.25}
        )

    axd['upper right'].set_title(
        r'$V_{c}$ * $V^{T}_{c}$'
        )

    axd['upper right'].set_xticks(
        np.arange(20,matrix.shape[0],20),
        np.arange(20,matrix.shape[0],20).tolist(),
        rotation=0
        )
    axd['upper right'].set_yticks(
        np.arange(20,matrix.shape[0],20),
        np.arange(20,matrix.shape[0],20).tolist()
        )

    axd['upper right'].tick_params(
        axis='both',
        which='both',
        bottom=False,
        left=False,
        top=False,
        labelsize=7,
        labelbottom=True,
        labelleft=True
        )

    axd['upper right'].set_ylabel('Brain region',fontsize=10)
    axd['upper right'].set_xlabel('Brain region',fontsize=10)

    #creating boxplot with occupancies
    occupancy.columns = ['condition','value']

    str_length = [len(cond) for cond in np.unique(occupancy.condition)]
    N_conds = np.unique(occupancy.condition).size

    sns.boxplot(
        x=occupancy.condition,
        y=occupancy.value,
        ax=axd['lower center'],
        color='black' if not darkstyle else 'white',
        width=.4,
        fliersize=.4,
        #linewidth=.8,
        #boxprops = dict(linewidth=3,color='black'),
        medianprops = dict(linewidth=1.5, color='black' if darkstyle else 'white')
        )
    axd['lower center'].set_ylabel('Occupancy',fontsize=15)
    axd['lower center'].set_xlabel('')
    axd['lower center'].tick_params(
        labelsize=10,
        axis='x',
        labelrotation=0 if (N_conds<=2 and np.max(str_length)<5) else 45
        )
    sns.despine(ax=axd['lower center'])

    #creating boxplot with dwell times
    dwell_times.columns = ['condition','value']
    N_conds = np.unique(dwell_times.condition).size
    sns.boxplot(
        x=dwell_times.condition,
        y=dwell_times.value,
        ax=axd['lower right'],
        color='black' if not darkstyle else 'white',
        width=.4,
        fliersize=.4,
        #linewidth=.8,
        #boxprops = dict(linewidth=3, color='black'),
        medianprops = dict(linewidth=1.5, color='black' if darkstyle else 'white')
        )
    axd['lower right'].set_ylabel('Dwell time',fontsize=15)
    axd['lower right'].set_xlabel('')
    axd['lower right'].tick_params(
        labelsize=10,
        axis='x',
        labelrotation=0 if (N_conds<=2 and np.max(str_length)<5) else 45
        )
    sns.despine(ax=axd['lower right'])

    plt.tight_layout(w_pad=0.7,h_pad=10)
    plt.show()