import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns 
from itertools import groupby
from pyleida.signal_tools import (
    hilbert_phase,
    phase_coherence,
    kuramoto_op
)
 
#general functions

def compute_dynamics_metrics(clusters_labels,TR=None,save_results=False,path=None):
    """
    Execute the dynamics analysis using Dynamical Systems
    theory tools for each k explored: computes the fractional
    occupancies,dwell times, and transitions probabilities for 
    each subject of each group/condition.

    Params:
    -------
    clusters_labels : pd.dataframe with shape (n_eigenvectors, 2 + range_of_K).
        Contains the predicted labels of each eigenvector
        across the K range explored. 1st column contains
        the subjects ids, and 2nd column contains the 
        class/conditions labels. The remaining columns contain
        the predictions/labelling for each k explored.  
        
    TR : np.float or int.
        Specify the Repetition Time of the fMRI data.

    save_results : bool. 
        Whether to save results. If True, the results are
        saved in a folder called 'dynamics_metrics'.

    path : str.
        Specify the path in which the 'dynamics_metrics'
        folder will be created if 'save_results' was set
        to True.

    Returns:
    --------
    metrics : dict. 
        Contains 'dwell_times' (pd.DataFrame),
        'fract_occupancies' (pd.DataFrame),
        'transitions' (pd.DataFrame).
    """

    #create folder (if specified) to save the results
    if save_results:
        try:
            results_path = f'{path}/dynamics_metrics'
            if not os.path.exists(results_path): 
                os.makedirs(results_path)
            print(f"-Folder created to save results: '{results_path}'")
        except:
            raise Exception("PROCESS ABORTED: the folder to save the results could't be created.")
    
    #check the names of 1st and 2nd col, and change them to 'subject_id' and 'condition', if different.
    if 'subject_id' not in clusters_labels.columns: 
        clusters_labels.rename(columns={clusters_labels.columns[0]:'subject_id'},inplace=True)
    if 'condition' not in clusters_labels.columns: 
        clusters_labels.rename(columns={clusters_labels.columns[1]:'condition'},inplace=True)
    
    meta = clusters_labels[['subject_id','condition']] #keep only metadata
    Ks = [col for col in clusters_labels.columns if col.startswith('k_')]
    ys = clusters_labels[Ks].values #keep only clusters assignements for each k.

    dwell_times,occupancies,transitions = {},{},{}
    for k_idx,k in enumerate(Ks):
        _ = [int(s) for s in k.replace('_',' ').split() if s.isdigit()]  
        print(f'k = {_[0]}')
        if save_results: 
            k_path = f'{results_path}/{k}'
            if not os.path.exists(k_path):
                os.mkdir(k_path)    
        else:
            k_path = None

        #Compute the metrics
        occupancies[k] = fractional_occupancy_group(
            meta,
            ys[:,k_idx],
            save_results=save_results,
            path=k_path
            )
        dwell_times[k] = dwell_times_group(
            meta,
            ys[:,k_idx],
            TR=TR,
            save_results=save_results,
            path=k_path
            )
        transitions[k] = transition_probabilities_group(
            meta,
            ys[:,k_idx],
            save_results=save_results,
            path=k_path
            )

    print('\n*The metrics were succesfully computed.')

    return {
        'dwell_times':dwell_times,
        'occupancies':occupancies,
        'transitions':transitions,
        }


# Functions to compute metrics derived from phase-locking: occupancies, dwell times, and transitions

def fractional_occupancy(cluster_assignement):
    """
    Compute the fractional occupancy of each phase-locking state 
    across time in a subject scan (or the complete dataset to 
    determine the occupancies of each state), i.e, the temporal
    percentage of epochs assigned to a given cluster centroid Vc. 
    Formally, it is defined as the number of time points in which a
    PL state is active during the scan, divided by the total number
    of time points in the scan.

    Params:
    -------
    cluster_assignement : ndarray with shape (N_time_points,).
        Contains the clusters labels for each time point.

    Returns:
    ---------
    occupancies : pd.DataFrame with shape (N_clusters, 3).
        Contains the fractional occupancy of each cluster.
    """
    N_time_points = cluster_assignement.shape[0] #number of time points
    occupancies = pd.DataFrame(pd.Series(cluster_assignement).value_counts().reset_index())
    occupancies.columns = ['cluster','n_volumes']
    #clusters_count['occupancy'] = [(i/T)*100 for i in clusters_count.Count]
    occupancies['occupancy'] = [i/N_time_points for i in occupancies.n_volumes]

    return occupancies.sort_values(by='cluster')

def fractional_occupancy_group(metadata,labels,save_results=False,path=None):
    """
    Compute the fractional occupancy for each subject
    in 'metadata', and optionally and save the results.

    Params:
    -------
    metadata : pd.dataframe with shape (N_subjects x N_time_points, 2).
        Contains the metadata. 1st column contains
        the subjects ids, and 2nd column contains
        the class/conditions/sessions labels.

    labels : numpy 1D array. 
        Contains the clusters labels of each time
        point for each subject and condition contained
        in 'metadata'.

    save_results : bool. 
        Whether to save the computed fractional
        occupancies in a .csv file, and the plot
        in a .png file.

    path : str. 
        Define the path in which we want to save
        the results (necessary if 'save_results'
        was set to True).

    Returns:
    --------
    results : pd.dataframe. 
        Contains 'subject_id' in 1st column, 'condition'
        in 2nd column, and a column for each cluster (i.e.,
        phase-locking state) with the values indicating the
        fractional occupancy of each state.
    """
    assert metadata.shape[0] == labels.shape[0], \
        "The number of rows in 'metadata' must be the same as the number of provided 'labels'."
    if save_results and path is None: 
        raise ValueError('You must provide a path to save the results')

    results = []
    #N_clusters = len(np.unique(labels)) #get the number of clusters (brain states)
    for cond in np.unique(metadata.condition):
        subs_ids = metadata[metadata.condition==cond].subject_id.values #get the subject id's that belongs to the current condition.
        for sub in np.unique(subs_ids): #for each subject
            idx = np.logical_and(metadata.subject_id==sub,metadata.condition==cond) #get index of the current subject clusters labels
            occ = fractional_occupancy(labels[idx]) #compute the occupancies for current subject
            occ_ = {
                **{'subject_id':sub,'condition':cond},
                **{f'PL_state_{k+1}':v for k,v in zip(occ.cluster,occ.occupancy)}
                }
            results.append(occ_)
    
    results = pd.DataFrame(results).fillna(0.0) #convert results to dataframe and fill NaNs with 0.0

    #reorder columns by PL_state (necessary because, when a subject do not traverse
    # a particular state, that state will be located at the end of the dataframe columns)
    N_states = len(results.columns)-2
    states_columns = [f'PL_state_{i+1}' for i in range(N_states)]
    results = results[['subject_id','condition']+states_columns]

    #save results
    if save_results:
        try: 
            results.to_csv(f'{path}/occupancies.csv',sep='\t',index=False)
        except:
            print("Warning: 'occupancies.csv' was not saved in local folder.")

    return results

def transition_probabilities(labels,k,norm=True,plot=False):
    """
    Compute the transitions between patterns across time
    for a single subject. The number of states ('k')
    defines the matrix shape, ensuring that, if a given
    subject don't traverse a state of the detected states
    for the whole group, the matrix will be constructed
    correctly, respecting the original number of states.
    If a subject don't traverse a particular state,
    then the corresponding row will contain all zeros.
    
    Params:
    ------
    labels : ndarray with shape (N_time_points,). 
        Contains the KMeans predicted label
        for each time point.

    norm : bool.
        Whether to normalize the values of the
        transitions matrix.

    plot : bool. 
        Whether to plot the transitions matrix.
    
    Returns:
    --------
    transitions : ndarray with shape (N_states,N_states).
    """

    N_states = k
    transitions = np.zeros((N_states, N_states)) #create empty matrix 
    N_volumes = labels.size 

    #compute the transitions
    for volume_idx in range(N_volumes-1):
        _from = labels[volume_idx]
        _to = _from + labels[volume_idx+1] - labels[volume_idx]
        transitions[_from,_to]+=1

    if norm:
        np.seterr(invalid='ignore') # hide potential warning when dividing 0/0 = NaN
        transitions = np.divide(
            transitions.astype(np.float_),
            np.sum(transitions,axis=1).reshape(transitions.shape[1],1)
            ) #normalize the values to probabilities

    #replace nan (if any) with 0. Can happen when a subject
    # doesn't traverse a state at all.
    transitions = np.nan_to_num(transitions,copy=True)
        
    if plot:
        plt.figure()
        sns.heatmap(
            transitions,
            annot=True,
            cmap='viridis', 
            square=True,
            linecolor='black',
            linewidths=0.5,
            xticklabels=[f'State {i+1}' for i in range(N_states)],
            yticklabels=[f'State {i+1}' for i in range(N_states)],
            cbar_kws={"shrink": 0.5}
            )
        plt.yticks(rotation='horizontal')
        plt.xlabel('To',fontsize=15,fontweight='regular')
        plt.ylabel('From',fontsize=15,fontweight='regular')
        plt.tight_layout()

    return transitions

def transition_probabilities_group(metadata,labels,save_results=False,path=None):
    """
    Compute the transition probabilities between
    phase-locking states for each subject in 'metadata'.

    Params:
    -------
    metadata : pd.dataframe (N_subjects x N_time_points, 2).
        Contains the metadata. 1st column contains
        the subjects ids, and 2nd column contains
        the class/conditions/sessions labels.

    labels : ndarray with shape (N_samples,). 
        Contains the clusters labels of each time
        point for each subject and condition contained
        in 'metadata'.

    save_results: bool. 
        Whether to save the computed transition
        probabilities in a .csv file.

    path : str. 
        Only provided if save_results=True. Define
        the path in which we want to save the results.

    Returns:
    --------
    results: pd.dataframe. 
        Contains 'subject_id' in 1st col, 'condition' in 2nd col,
        and a column for each transition [e.g. From_1_to_2,From_1_to_3 ...].
    """
    assert metadata.shape[0] == labels.shape[0], \
        "The number of rows in 'metadata' must be the same as in 'labels'"
    if save_results and path is None: 
        raise Exception('You must provide a path to save the results')

    results = []
    subjects_list,cond_list = [],[]
    N_clusters = len(np.unique(labels))
    for cond in np.unique(metadata.condition):
        subs_ids = metadata[metadata.condition==cond].subject_id.values #get an array with the subjects ids that belongs to a given condition.
        for subject in np.unique(subs_ids):
            idx = np.logical_and(metadata.subject_id==subject,metadata.condition==cond)
            tr = transition_probabilities(labels[idx],k=N_clusters,norm=True,plot=False)
            results.append(np.ravel(tr)) #vectorize the transitions matrix.
            subjects_list.append(subject)
            cond_list.append(cond)

    results = pd.DataFrame(np.vstack(results))
    columns_names = []
    for i in range(N_clusters):
        for j in range(N_clusters):
            columns_names.append(f'From_{i+1}_to_{j+1}')
    
    results.columns = columns_names
    results.insert(0,'subject_id',subjects_list)
    results.insert(1,'condition',cond_list)

    if save_results: 
        try:
            results.to_csv(f'{path}/transitions_probabilities.csv',sep='\t',index=False)
        except:
            print("The results could't be saved in .csv file. Please check the provided path.")

    return results

def group_transition_matrix(transitions_df,metric='mean',plot=True,cmap='inferno',darkstyle=False):
    """
    Compute and plot the mean or median transition matrix
    of each group (or a single group).

    Params:
    -------
    transitions_df : pd.DataFrame.
        Contains the transitions probabilities of each
        subject and group/condition. Output of the
        'transition_probabilities_group' function.

    metric : str.
       Select whether to compute the 'mean' or 'median'
       transition matrix for each group.

    plot : bool.
        Whether to plot the matrices.

    cmap : str.
        Select the cmap to use in the heatmap/s.

    darkstyle : bool.
        Whether to use a darkstyle for the plot.

    Returns:
    --------
    mats : dict. 
        Contains the groups names as keys, and a
        numpy 2D array (N_states,N_states) as values.
    """
    if not isinstance(metric,str):
        raise TypeError("'metric' must be a string!")
    else:
        if metric not in ['mean','median']:
            raise ValueError("Valid options for 'metric' are 'mean' or 'median'.")

    N_clusters = np.sqrt(len(transitions_df.columns[2:])).astype(int) #get the number of clusters/states.
    mats = {}

    conds = np.unique(transitions_df.condition)

    for cond in conds:
        if metric=='mean': 
            current_group_matrix = transitions_df[transitions_df.condition==cond].iloc[:,2:].mean().values.reshape(N_clusters,N_clusters)
        elif metric=='median':
            current_group_matrix = transitions_df[transitions_df.condition==cond].iloc[:,2:].median().values.reshape(N_clusters,N_clusters)
        mats[cond] = current_group_matrix

        if plot:
            #plotting
            with plt.style.context('dark_background' if darkstyle else 'default'):
                plt.figure(figsize=(5.5,4) if N_clusters<10 else (8,6) if 10<N_clusters<15 else (10,8))
                sns.heatmap(
                    current_group_matrix,
                    cmap=cmap,
                    linecolor='black' if not darkstyle else 'white',
                    linewidths=1,
                    annot=True,
                    square=True,
                    cbar_kws={"shrink": .5,"label":"Transition\nprobability"},
                    vmin=0,
                    vmax=1,
                    center=0.5,
                    fmt='.2f'
                    )
                plt.xlabel('To',fontsize=15,fontweight='regular',labelpad=20)
                plt.ylabel('From',fontsize=15,fontweight='regular',labelpad=20)
                plt.yticks(np.arange(0.5,N_clusters+.5,1),[f'State {i+1}' for i in range(N_clusters)],fontweight='regular',rotation='horizontal')
                if N_clusters<=5:
                    plt.xticks(np.arange(0.5,N_clusters+.5,1),[f'State {i+1}' for i in range(N_clusters)],fontweight='regular')
                else:
                    plt.xticks(np.arange(0.5,N_clusters+.5,1),[f'State {i+1}' for i in range(N_clusters)],fontweight='regular',rotation=30,horizontalalignment="right")
                plt.title(cond,fontweight='regular',fontsize=18)
                plt.tight_layout()
                plt.show()

    return mats
    
def dwell_times(labels,TR=None,plot=False):
    """
    Computes the dwell time (mean duration) of each
    phase-locking state. The dwell time represents
    the mean number of consecutive epochs spent in
    a particular state throughout the duration of a scan.

    Params:
    --------
    labels : numpy 1D array.
        Labels of PL states traversed across a scan.

    TR : np.float | int | None
        Specify the Repetition Time of the fMRI data.
        If None, the dwell times are expressed in
        volume unit, instead of seconds.

    plot : bool.
        Whether to create a barplot showing
        the computed dwell time of each phase-locking
        state.
    """
    
    dwell = {}
    for cluster in np.unique(labels): #for each cluster (aka brain state)
        dwell[cluster] = []
        dwell[cluster].append([len(list(g[1])) for g in groupby(labels) if g[0]==cluster]) #get a sequence with the times it appears across time.
    
    #transforming list to array
    for cluster in dwell.keys(): 
        dwell[cluster] = np.concatenate([i for i in dwell[cluster]])
    
    #compute the average lifetime (TRs or seconds) of each state    
    if TR is None:
        mean_dwell = pd.DataFrame({'Cluster':[i for i in dwell.keys()],'Mean_lifetime':[np.mean(dwell[i]) for i in dwell.keys()]})
    else:
        mean_dwell = pd.DataFrame({'Cluster':[i for i in dwell.keys()],'Mean_lifetime':[np.mean(dwell[i])*TR for i in dwell.keys()]})
    
    if plot:
        plt.figure()
        plt.grid()
        sns.barplot(x=[f'State {i+1}' for i in range(len(mean_dwell))],y=mean_dwell.Mean_lifetime)
        plt.ylabel(f"Average lifetime ({'TRs' if TR is None else 's'})",fontsize=15,fontweight='regular') 
        plt.xlabel('State',fontsize=15,fontweight='regular') 
        plt.tight_layout()
        plt.show()
                                          
    return dwell,mean_dwell

def dwell_times_group(metadata,labels,TR=None,save_results=False,path=None):
    """
    Compute the dwell times for each subject and
    optionally save the results.

    Params:
    -------
    metadata : pd.dataframe with shape (N_subjects x N_time_points, 2).
        Contains the metadata. 1st column contains
        the subjects ids, and 2nd column contains the
        class/conditions/sessions labels.

    labels : numpy 1D array. 
        Contains the clusters labels of each time point
        for each subject and condition contained in 'metadata'.

    TR : np.float or int.
        Specify the Repetition Time of the fMRI data.

    save_results : bool. 
        Whether to save the computed dwell times in a .csv file,
        and the plot in a .png file.

    path : str. 
        Only provided if save_results=True. Define the path
        in which we want to save the results.

    Returns:
    --------
    results : pd.dataframe. 
        Contains 'subject_id' in 1st col, 'condition' in 2nd col,
        and a column for each cluster [brain state] with the
        values indicating the mean consecutive appareances of
        each state.
    """
    if save_results and path is None: 
        raise Exception('You must provide a path to save the results')
    assert metadata.shape[0] == labels.shape[0], \
        "The number of rows in 'metadata' must be the same as in 'labels'"
    
    results = []
    #N_clusters = len(np.unique(labels))
    for cond in np.unique(metadata.condition):
        subs = metadata[metadata.condition==cond].subject_id.values
        for s in np.unique(subs):
            idx = np.logical_and(metadata.subject_id==s,metadata.condition==cond)
            dw = dwell_times(labels[idx],TR=TR,plot=False)[-1]
            dw_ = {
                **{'subject_id':s,'condition':cond},
                **{f'PL_state_{k+1}':v for k,v in zip(dw.Cluster,dw.Mean_lifetime)}
                }
            results.append(dw_)
    
    results = pd.DataFrame(results).fillna(0.0)

    #reorder columns by PL_state (necessary because, when a subject do not traverse
    # a particular state, that state will be located at the end of the dataframe columns)
    N_states = len(results.columns)-2
    states_columns = [f'PL_state_{i+1}' for i in range(N_states)]
    results = results[['subject_id','condition']+states_columns]

    if save_results:
        try: 
            results.to_csv(f'{path}/dwell_times.csv',sep='\t',index=False)
        except:
            print("Warning: 'dwell_times.csv' was not saved in local folder.")

    return results

def promiscuity(centroids,as_proportion=True,plot=True,cmap='viridis'):
    """
    Calculates promiscuity of nodes across
    phase-locking modes/states.
    Promiscuity calculates the number of
    states each node is a member of.

    Params:
    -------
    centroids : pd.dataframe with shape (N_centroids,N_rois).
        Centroids of a specific K partition.
        Columns must have the ROI's names.

    as_proportion : bool.
        Whether to express promiscuity as
        proportion or as raw count.

    plot : bool.
        Whether to create a barplot
        showing the computed promiscuity.

    cmap : str.
        Colormap to use in the barplot.

    Returns:
    ---------
    promiscuity : pd.dataframe.
        Contains the computed promiscuity 
        of each node across modes.
    """
    if not isinstance(centroids,pd.DataFrame):
        raise TypeError("'centroids' must be a pandas dataframe!")
    if not isinstance(as_proportion,bool):
        raise TypeError("'as_proportion' must be a boolean!")
    if not isinstance(plot,bool):
        raise TypeError("'plot' must be a boolean!")
    
    N_states = centroids.shape[0]

    dct = {}

    for col in centroids.columns:
        dct[col] = np.sum([1 for i in centroids.loc[:,col] if i>0])

    df = pd.DataFrame(dct,index=[0]).T.reset_index()
    df = df.sort_values(by=0,ascending=False)
    df.columns = ['ROI','promiscuity']

    if as_proportion:
        df['promiscuity'] = (df.promiscuity / N_states) * 100
    else:
        df['promiscuity'] = [int(i) for i in df.promiscuity]

    if plot:
        plt.ion()
        plt.figure(figsize=(6,11))
        sns.barplot(data=df,x='promiscuity',y='ROI',palette=cmap)
        plt.yticks(fontsize=5)
        if not as_proportion:
            n_max = np.max(df.promiscuity)
            plt.xticks(
                np.arange(0,n_max+1),
                [int(i) for i in np.arange(0,n_max+1)]
                )
        plt.ylabel('')
        plt.xlabel('Proportion (%)' if as_proportion else 'Count',fontsize=15)
        plt.tight_layout()

    return df

def pl_metastability(tseries,centroids):
    """
    Metastability derived from phase-locking
    is calculated as the mean of the variance
    of instantaneous phase-locking (iPL) over
    time in each community. The mean value of
    this measure across communities, denoted
    as global metastability, represents the
    overall variance in the phase-locking across
    communities.

    Params:
    --------
    tseries : numpy ndarray with shape (N_rois,N_volumes).
        BOLD time series of a particular subject.

    centroids : numpy ndarray with shape (N_centroids,N_rois).
        Computed centroids for a given K-means
        partition.

    Returns:
    --------
    meta : numpy ndarray with shape (N_volumes,N_centroids).
        Computed metastability for each
        community at each time point.
    """    
    # Get the indices of each community/state
    communities_indices = get_communities_indices(centroids.values)

    iPL = phase_coherence(hilbert_phase(tseries))

    N_volumes = iPL.shape[-1]
    N_centroids = centroids.shape[0]

    var_meta = np.zeros((N_volumes,N_centroids))


    for t in range(N_volumes):
        iPL_t = iPL[:,:,t] #get iPL at time t
        for state in range(N_centroids):
            state_idx = communities_indices[f'centroid_{state+1}'] #get state rois indices
            state_matrix = iPL_t[state_idx][:,state_idx] #create matrix with only state rois.
            var_meta[t,state] = np.var(np.ravel(state_matrix)) #compute the variance of iPL values

    return var_meta

def get_communities_indices(centroids_array):
    """
    Get indices of each community/state
    """
    N_centroids = centroids_array.shape[0]
    indices = {}
    
    for c in range(N_centroids):
        if np.all(centroids_array[c,:]<0):
            indices[f'centroid_{c+1}'] = np.array([i for i in range(centroids_array.shape[1])])
        else:
            idx = np.where(centroids_array[c,:]>0)
            indices[f'centroid_{c+1}'] = idx

    return indices

# Functions to compute metrics derived from phase-synchrony: 
# synchronization and metastability

def compute_kuramoto(centroids,phases):
    """
    Compute the Kuramoto order parameter for each
    community of brain regions at each time t from
    a set of phases of a particular subject.

    Params:
    -------
    centroids : pd.DataFrame with shape (N_centroids, N_regions).
        Computed centroids for a specific K partition.

    phases : ndarray with shape (N_regions, N_volumes).
        Phases of a set of brain regions.

    Returns:
    --------
    synchrony : ndarray with shape (N_volumes, N_centroids).
        Kuramoto order parameter for each community
        at each time t.
    """    
    # Get the indices of each community/state
    communities_indices = get_communities_indices(centroids.values)

    N_volumes = phases.shape[1]
    N_centroids = centroids.shape[0]

    # Compute the kuramoto order parameter of each state at each time point
    kuramoto = np.zeros((N_volumes,N_centroids))

    for t in range(N_volumes):
        for state in range(N_centroids):
            kuramoto[t,state] = kuramoto_op(
                phases[communities_indices[f'centroid_{state+1}'],t]
                )

    return kuramoto