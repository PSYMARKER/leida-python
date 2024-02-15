"""Class to retrieve and explore the LEiDA results."""

import numpy as np
import pandas as pd
import os
from itertools import combinations
import matplotlib.pyplot as plt
import seaborn as sns

from .data_utils import (
    load_rois_labels,
    load_rois_coordinates, 
    load_tseries,
    load_classes
)
from .data_utils.validation import (
    _check_k_input,
    _check_metric,
    _check_state
)
from .plotting import (
    brain_states_network,
    brain_states_nodes,
    stats_pyramid,
    states_pyramid,
    states_k_glass,
    states_in_bold,
    brain_states_on_surf,
    brain_states_on_surf2,
    _explore_state,
    _save_html
)
from .clustering import (
    barplot_states,
    barplot_eig,
    centroid2matrix,
    plot_clusters3D,
    plot_voronoi,
    plot_clustering_scores
)
from .clustering import rsnets_overlap as rsnets
from .dynamics_metrics import (
    group_transition_matrix,
    compute_kuramoto,
    pl_metastability
)
from .stats import scatter_pvalues
from .signal_tools import hilbert_phase,phase_coherence


class DataLoader:
    """
    Class to retrieve from local folder all
    the outputs and results of the execution
    of the LEiDA pipeline through the 'Leida'
    class provided in the package. The DataLoader
    provides the same methods of the Leida class,
    thus allowing to load and deepen easily in the
    results of the previously executed analysis.

    Params:
    -------
    data_path : str.
        Path to the folder that contains the
        BOLD time series, metadata, ROIs labels, 
        and ROIs coordinates.

    results_path : str.
        Path to the folder that contains the
        LEiDA results (default: 'LEiDA_results').

    Attributes:
    -----------
    eigenvectors : pandas.dataframe.
        Contains the computed eigenvectors
        for each time point of each subject.

    predictions : pandas.dataframe.
        Contains the predicted cluster label of
        each eigenvector for each 'k' partition
        explored.

    rois_labels : list.
        The label/name of each ROI/parcel.

    rois_coordinates_ : ndarray of shape (n_rois,3).
        The MNI coordinates of each ROI/parcel.
    """
    def __init__(self,data_path='data',results_path='LEiDA_results'):
        #validation of input data
        for _path in (data_path,results_path):
            if not isinstance(_path,str):
                raise TypeError("'data_path' and 'results_path' must be strings!")
        #check if the provided paths exists
        if not os.path.exists(data_path):
            raise ValueError("The provided 'data_path' could't be founded.")
        if not os.path.exists(results_path):
            raise ValueError("The provided 'results_path' could't be founded.")
        
        #Paths in which the data and results are stored
        self._data_path_ = os.path.abspath(data_path)
        self._results_path_ = os.path.abspath(results_path)
        self._clustering_ = self._results_path_ + '/clustering'
        self._dynamics_ = self._results_path_ + '/dynamics_metrics'
        self._models_ = self._clustering_ + '/models'

        #check if the 'clustering' and 'dynamics_metrics' folders exists in 'results_path'
        if not os.path.exists(self._clustering_) or not os.path.exists(self._dynamics_):
            raise Exception(f"No results were found in the specified path.")
        
        #load eigenvectors and kmeans predictions dataframes.
        try:
            self.eigenvectors = pd.read_csv(f'{results_path}/eigenvectors.csv',sep='\t')
            nrois = self.eigenvectors.shape[1]-2
        except:
            raise Exception("The eigenvectors dataframe could't be loaded.")
        try:
            self.predictions = pd.read_csv(f'{self._clustering_}/predictions.csv',sep='\t')
        except:
            raise Exception("The dataframe with the KMeans models predictions couldn't be loaded.")

        #load rois labels
        self.rois_labels = load_rois_labels(self._data_path_)
        nlabels = len(self.rois_labels)

        #load rois coordinates
        self.rois_coordinates_ = load_rois_coordinates(self._data_path_)
        if self.rois_coordinates_ is None:
            print("The ROIs coordinates couldn't be loaded from the provided 'data_path'. "
                "Brain plots that show nodes in brain space will not be executed in consequence.")
        else:
            ncoords = self.rois_coordinates_.shape[0]

        #check that the N of ROIs coincide across
        #labels, coordinates, and eigenvectors
        if self.rois_coordinates_ is None:
            if not nrois==nlabels:
                raise Exception(f"The number of ROIs in 'eigenvectors.csv' ({nrois}) "
                                f"and 'rois_labels.txt' {nlabels} must coincide!")
        else:
            if not nrois==nlabels==ncoords:
                raise Exception(f"The number of ROIs in 'eigenvectors.csv' ({nrois}), "
                                f"'rois_labels.txt' ({nlabels}), and 'rois_coordinates.csv' "
                                f"({ncoords}) must coincide!")

        #getting kmin and kmax values
        ks = [int(i.split('_')[-1]) for i in self.predictions.columns.values if 'k' in i]
        self._K_min_ = np.min(ks)
        self._K_max_ = np.max(ks)

        self._classes_lst_ = np.unique(self.eigenvectors.condition).tolist()
        self._N_classes_ = len(self._classes_lst_)

    def time_series(self):
        """
        Return a dictionary having the 'subject_ids'
        as keys, and 2D arrays (N_ROIs,N_volumes) with
        BOLD time series as values.

        Returns:
        --------
        time_series : dict.
        """
        return load_tseries(self._data_path_)

    def load_model(self,k=2):
        """
        Load fitted model for a specific 'k' partition.
        Given that each model is an instance of the KMeansLeida
        class, once loaded you can access all the object methods
        and attributes.

        Params:
        -------
        k : int.
            Select the partition of interest.
        
        Returns:
        -------
        model : KMeansLeida instance.
            The fitted model that was used to predict
            the cluster labels of each observation.
        """
        _check_k_input(self._K_min_,self._K_max_,k)
        try:
            model = pd.read_pickle(f'{self._models_}/model_k_{k}.pkl')
            return model
        except:
            raise Exception("Can't find results for the selected k.")

    def load_centroids(self,k=2):
        """
        Return the computed clusters centroids
        for a specific 'k' partition.

        Params:
        -------
        k : int.
            Select the partition of interest.
        
        Returns:
        -------
        centroids : pd.DataFrame with shape (n_centroids,n_rois).
            Contains the computed centroids for
            the selected k partition.
        """
        _check_k_input(self._K_min_,self._K_max_,k)
        try:
            centroids = pd.DataFrame(self.load_model(k=k).cluster_centers_,columns=self.rois_labels)
        except:
            print("Can't find results for the selected k.")

        return centroids

    def centroids_distances(self,k=2):
        """
        Transform eigenvectors to a cluster-distance space. 
        Returns the distance between each eigenvector and the
        cluster centroids of the selected 'k' partition.

        Params:
        -------
        k : int.
            Select the partition of interest.
        
        Returns:
        --------
        distances : pd.DataFrame.
            Contains the distace between each eigenvector and
            each cluster centroid for the select 'k' partition.
            1st column contains 'subject_id', 2nd column the
            'condition', and the rest of columns the distances
            to the centroids.
        """
        model = self.load_model(k=k)
        distances = pd.DataFrame(model.transform(self.eigenvectors.iloc[:,2:].values))
        distances.columns = [f'centroid_{centroid+1}' for centroid in range(k)]
        distances = pd.concat((self.eigenvectors[['subject_id','condition']],distances),axis=1)
        return distances

    def stats(self,k=2,metric='occupancies'):
        """
        Retrieve the results from the statistical analysis of 
        a 'metric' of interest ('occupancies' or 'dwell_times')
        for a specific 'k' partition.

        Params:
        -------
        k : int.
            Select the 'k' partition of interest.

        metric : str.
            Select the metric to retrieve results.

        Returns:
        --------
        stats : pandas.dataframe.
            Results of the statistical analysis
            of each PL state.
        """
        _check_k_input(self._K_min_,self._K_max_,k)
        _check_metric(metric)

        try:
            df_stats = pd.read_csv(f'{self._dynamics_}/k_{k}/{metric}_stats.csv',sep='\t')
        except:
            print("Can't find results for the selected k or metric.")
        return df_stats

    def dwell_times(self,k=2):
        """
        Return the computed dwell times of each
        phase-locking state for a specific 'k'
        partition.

        Params:
        -------
        k : int.
            Specify the K-Means partition of interest.

        Returns:
        --------
        dwell_times : pd.DataFrame.
            Contains the computed dwell times of each
            PL state for each subject for the selected
            'k' partition.
        """
        _check_k_input(self._K_min_,self._K_max_,k)
        try:
            dt = pd.read_csv(f'{self._dynamics_}/k_{k}/dwell_times.csv',sep='\t')
        except:
            print("Can't find results for the selected k.")
        return dt
    
    def occupancies(self,k=2):
        """
        Return the computed fractional occupancy of each
        phase-locking state for a specific 'k' partition.

        Params:
        -------
        k : int.
            Specify the K-Means partition of
            interest.

        Returns:
        --------
        occupancies : pd.DataFrame.
            Contains the computed fractional occupancy
            of each PL state for each subject for the
            selected 'k' partition.
        """
        _check_k_input(self._K_min_,self._K_max_,k)
        try:
            occ = pd.read_csv(f'{self._dynamics_}/k_{k}/occupancies.csv',sep='\t')
        except:
            print("Can't find results for the selected k.")
        return occ

    def transitions(self,k=2):
        """
        Return the computed transition probabilities between
        phase-locking states for a specific 'k' partition.

        Params:
        -------
        k : int.
            Specify the K-Means partition of interest.

        Returns:
        --------
        transitions : pd.DataFrame.
            Contains the computed transition probabilities
            between PL states for each subject for the selected
            'k' partition.
        """
        _check_k_input(self._K_min_,self._K_max_,k)
        try:
            tr = pd.read_csv(f'{self._dynamics_}/k_{k}/transitions_probabilities.csv',sep='\t')
        except:
            print("Can't find results for the selected k.")
        return tr
    
    def significant_states(self,metric='occupancies'):
        """
        Return a dataframe containing only the statistics
        of the phase-locking states that are significantly
        different between groups.

        Params:
        -------
        metric : str.
            Metric of interest (Options: 'occupancies',
            'dwell_times').

        Returns:
        --------
        stats : pandas.dataframe.
        """
        _check_metric(metric)

        try:
            stats = self._pool_stats(metric=metric)
        except:
            raise Exception("The stats could't be loaded.")
        has_results = True if stats[stats.reject_null==True].shape[0]>=1 else False #check if some result was significant
        if has_results:
            return stats[stats.reject_null==True]
        else:
            print("No significant results were detected.")
            return None
        
    def state_rois(self,k=2,state=1):
        """
        Get a list with the names of the ROIs/parcels that
        participates in a specific phase-locking (PL) state.

        Params:
        -------
        k : int.
            Select the partition.

        state : int.
            Select the PL pattern or state of
            interest.

        Returns:
        --------
        rois : list.
            Contains the names of the ROIs that
            are part of the selected PL state.
        """
        _check_k_input(self._K_min_,self._K_max_,k)
        _check_state(k,state)

        data_k = self.load_centroids(k=k).iloc[state-1,:].reset_index()
        data_k.columns = ['rois','value']
        rois = list(data_k[data_k.value>0]['rois'])
        return rois

    def group_static_fc(self,group=None,plot=True,cmap='jet',darkstyle=False):
        """
        Compute the mean static functional
        connectivity matrix of a particular
        group/condition.

        Params:
        --------
        group : str.
            Specify the group of interest.

        plot : bool.
            Whether to create a heatmap showing
            the connectivity matrix.

        cmap : str.
            If plot=True, then select the colormap
            to use for the heatmap. Default = 'jet'.

        darkstyle : bool.
            Whether to use a dark background
            for plotting.

        Returns:
        --------
        static_fc : ndarray with shape (N_ROIs, N_ROIs).
            The computed static functional connectivity
            matrix.
        """
        if not isinstance(group,str):
            raise TypeError("'group' must be a string.")

        classes = load_classes(self._data_path_)

        #create list with conditions labels
        conditions = []
        for val in classes.values():
            for item in val:
                conditions.append(item)

        if len(conditions) != len(self.time_series().keys()):
            raise Exception("This method is only available in "
                "cases where each subject has only one condition label.")
        if group not in conditions:
            raise ValueError("'group' must be present in your data. "
                f"Possible options are: {[i for i in np.unique(conditions)]}.")       

        signals = load_tseries(self._data_path_)
        subjects_ids = [sub for sub,condition in zip(signals.keys(),conditions) if condition==group]
        N_subjects = len(subjects_ids)
        N_rois = len(self.rois_labels)
        pooled_static_fc = np.empty((N_rois,N_rois,N_subjects))
        
        for idx,sub in enumerate(subjects_ids):
            pooled_static_fc[:,:,idx] = np.corrcoef(signals[sub])
        
        static_fc = np.mean(pooled_static_fc,axis=-1)

        if plot:
            plt.ion()
            with plt.style.context('dark_background' if darkstyle else 'default'):
                plt.figure()
                sns.heatmap(
                    static_fc,
                    vmin=-1,
                    vmax=1,
                    center=0,
                    square=True,
                    cmap=cmap,
                    cbar_kws={'label': 'Pearson\ncorrelation','shrink': 0.5}
                )

                plt.xlabel('Brain region',fontsize=16,labelpad=20)
                plt.ylabel('Brain region',fontsize=16,labelpad=20)
                plt.title(group)
                plt.xticks(
                        np.arange(20,N_rois,20),
                        np.arange(20,N_rois,20).tolist(),
                        rotation=0
                        )
                plt.yticks(
                        np.arange(20,N_rois,20),
                        np.arange(20,N_rois,20).tolist()
                        )
                plt.tick_params(
                    axis='both',         
                    which='both',     
                    bottom=False,
                    left=False
                    )
                plt.tight_layout()
                #plt.show()

        return static_fc

    def group_transitions(self,k=2,metric='mean',cmap='inferno',darkstyle=False):
        """
        Compute and plot the mean or median transition
        probabilities matrix of each group/condition.

        Params:
        -------
        k : int.
            The k-means partition of interest.

        metric : str.
            Whether to plot the 'mean' or 'median'
            matrices.

        cmap : str.
            Colormap to use in the created heatmaps.

        darkstyle : bool.
            Whether to use a dark background for
            plotting.
        """
        _check_k_input(self._K_min_,self._K_max_,k)

        data = pd.read_csv(f'{self._dynamics_}/k_{k}/transitions_probabilities.csv',sep='\t')

        mats = group_transition_matrix(
            data,
            metric=metric,
            cmap=cmap,
            darkstyle=darkstyle
            )

        return mats

    def overlap_withyeo(self,parcellation=None,n_areas=None,k=2,state=None,darkstyle=False):
        """
        Compute the overlap between the 7 resting-state
        networks defined in Yeo et al. (2011) and the brain
        cortical regions/parcels of the phase-locking state
        of interest. The correlations are shown in a barplot,
        and a dataframe with the correlations and p-values is
        returned.

        Params:
        --------
        parcellation : str.
            Specify path to your parcellation .nii file.
            Note: the parcellation must be of 2mm resolution.

        n_areas : None | int.
            Analyze only the first n areas from the provided
            parcellation. 
            Usefull when the parcellation contains subcortical
            regions that must be ignored when computing the overlap
            with Yeo's cortical networks.

        k : int.
            Select the partition.

        state : int.
            Select the PL pattern or state of
            interest.

        darkstyle : bool.
            Whether to use a dark theme for the plot.

        Returns:
        --------
        overlap : pandas.dataframe with shape (7networks,3).
            Contains the correlation coefficient (and p-value)
            between the selected phase-locking state and each
            of the 7 resting-state networks from Yeo (2011).
        """
        _check_k_input(self._K_min_,self._K_max_,k)
        _check_state(k,state)

        #Get centroids of all k partitions
        #in a single dataframe        
        centr = {}

        for k_ in range(self._K_min_,self._K_max_+1):
            cent = self.load_centroids(k_)
            cent.insert(0,'state',[i+1 for i in range(cent.shape[0])])
            cent.insert(0,'k',k_)
            centr[f'k{k_}'] = cent
        
        centr = pd.concat(centr,ignore_index=True)

        #Compute overlap of all PL states
        corr,pvals = rsnets.compute_overlap(
            centr,
            parcellation=parcellation,
            n_areas=n_areas
            )

        #Get the information of
        #the selected PL state
        overlap = rsnets.state_overlap(
            corr,
            pvals,
            k=k,
            state=state,
            plot=True,
            darkstyle=darkstyle
            )

        return overlap

    def phase_synchrony_metrics(self,k=2):
        """
        For each subject, compute the Kuramoto order 
        parameter for each community at each time t.
        In addition, computes the synchronization
        and metastability of each community across
        time.
        Synchronization is calculated as the time-average
        of the Kuramoto order parameter in each community
        or centroid.
        Metastability is calculated for each community
        as the standard deviation of the Kuramoto order
        parameter.

        Note: See Hancock et al. (2022), NeuroImage.

        Params:
        -------
        k : int.
            Select the K partition of interest.

        Returns:
        --------
        kuramoto : pd.DataFrame.
            Contains the computed Kuramoto order parameter
            of each subject, time point and community.

        synchronization : pd.DataFrame with shape (N_subjects,N_communities +2).
            Contains the computed synchronization of each
            community, defined as the time-average of the
            Kuramoto order parameter.

        metastability : pd.DataFrame with shape (N_subjects,N_communities +2).
            Contains the computed metastability of each
            community for each subject, defined as the
            standard deviation of the Kuramoto order
            parameter across time.
        """
        _check_k_input(self._K_min_,self._K_max_,k)

        time_series = self.time_series()
        classes = load_classes(self._data_path_)
        subject_ids = list(time_series.keys()) #list of subject ids
        cent = self.load_centroids(k)

        kura_lst = []

        for sub_id in subject_ids: #for each subject
            #get current subject signals
            tseries = time_series[sub_id]
            n_volumes = tseries.shape[1]-2

            print(f"SUBJECT ID: {sub_id} ({n_volumes} volumes)")
            
            #Get the signals phases
            phases = hilbert_phase(tseries)

            #Compute synchrony of each community
            kura = compute_kuramoto(cent,phases[:,1:-1])

            df = pd.DataFrame(kura,columns=[f'PL_state_{i+1}' for i in range(cent.shape[0])])

            if len(classes[sub_id])==1:
                df.insert(0,'condition',[classes[sub_id][0] for i in range(n_volumes)])
            else:
                df.insert(0,'condition',classes[sub_id][1:-1])

            df.insert(0,'subject_id',[sub_id for i in range(n_volumes)])
            kura_lst.append(df)

        # Preparing results
        kuramoto = pd.concat(kura_lst) #full dataset of each community

        # Compute synchronization of each
        # community as their time-average
        synchronization = kuramoto.groupby(['subject_id','condition']).mean().reset_index()

        # Compute global synchronization
        global_sync = []

        for i in range(synchronization.shape[0]):
            global_sync.append(np.mean(synchronization.iloc[i,2:].values))

        synchronization['global'] = global_sync

        # Compute metastability of each community
        # as their std across time.
        metastability = kuramoto.groupby(['subject_id','condition']).std().reset_index()

        # Compute global metastability
        global_meta = []

        for i in range(metastability.shape[0]):
            global_meta.append(np.mean(metastability.iloc[i,2:].values))

        metastability['global'] = global_meta

        return kuramoto,synchronization,metastability

    def phase_locking_metastability(self,k):
        """
        Compute metastability from phase-locking (iPL) tensors.
        This measure is calculated as the mean of the variance of
        instantaneous phase-locking over time in each community.
        The mean value of this measure across communities, denoted
        as global metastability, represents the overall variance in
        the phase-locking across communities.

        Note: See Hancock et al. (2023), Plos One.

        Params:
        ------
        k : int.
            Select the K partition of
            interest.

        Returns:
        --------
        metastability : pd.DataFrame.
            Contains the computed metastability
            for each subject and condition.
        """
        _check_k_input(self._K_min_,self._K_max_,k)

        time_series = self.time_series()
        classes = load_classes(self._data_path_)
        subject_ids = list(time_series.keys()) #list of subject ids
        cent = self.load_centroids(k)

        meta_lst = []

        for sub_id in subject_ids: #for each subject
            # Get current subject signals
            tseries = time_series[sub_id]
            n_volumes = tseries.shape[1]-2

            print(f"SUBJECT ID: {sub_id} ({n_volumes} volumes)")
            
            # Compute metastability at each time point
            meta = pl_metastability(tseries,cent)

            df = pd.DataFrame(meta,columns=[f'PL_state_{i+1}' for i in range(cent.shape[0])])

            # Add metadata
            if len(classes[sub_id])==1:
                df.insert(0,'condition',[classes[sub_id][0] for i in range(n_volumes)])
            else:
                df.insert(0,'condition',classes[sub_id][1:-1])

            df.insert(0,'subject_id',[sub_id for i in range(n_volumes)])
            meta_lst.append(df)

        # Preparing results
        metastability = pd.concat(meta_lst) #full dataset of each community

        metastability = metastability.groupby(['subject_id','condition']).mean().reset_index()

        # Compute global metastability
        global_meta = []

        for i in range(metastability.shape[0]):
            global_meta.append(np.mean(metastability.iloc[i,2:].values))

        metastability['global'] = global_meta

        return metastability

    def regional_contribution(self,k=2,state=1,roi=None):
        """
        Compute the regional contribution of a 
        region of interest (ROI) in a particular
        state across time.
        Regional contribution is calculated as the
        mean value of instantaneous phase-locking
        over time for the region/s of interest (ROI).
        This metric can be used to explore for
        connectivity differences between groups
        or conditions.

        Note: See Hancock et al.(2023), PlosOne.

        Params:
        -------
        k : int.
            Partition of interest.

        state : int.
            Select the PL state/pattern of
            interest.

        roi : list | string.
            Name/s of the ROI/s for which the
            regional contribution will be
            computed. To compute the metric for
            more than one ROI, then you have to
            provide a list. If 'all', then
            compute the contribution of each
            ROI of the selected state.

        Returns:
        --------
        contribution : pd.DataFrame.
            Computed regional contribution
            for the selected ROI/s.
        """
        # Validation of input parameters
        _check_k_input(self._K_min_,self._K_max_,k)
        _check_state(k,state)

        if roi is None:
            raise ValueError("'roi' cannot be None!")
        else:
            if isinstance(roi,str):
                if roi=='all':
                    rois = self.state_rois(k,state)
                else:
                    rois = [roi]
            else:
                rois = roi.copy()

            for roi in rois:
                if roi not in self.state_rois(k,state):
                    raise ValueError(f"The specified ROI ({roi}) was not founded for the selected state!")
        
        # Load dict with time series of all subjects
        time_series = self.time_series()

        # Load dict with metadata of all subjects
        classes = load_classes(self._data_path_)

        subject_ids = list(time_series.keys()) #list of subject ids

        # Load selected centroid
        centr = self.load_centroids(k).iloc[state-1,:].values 

        # Get state/community indices
        if np.all(centr<0):
            community_indices = np.array([i for i in range(centr.size)])
        else:
            community_indices = np.where(centr>0)[0]

        # Create dict to save results
        full_db = {}

        for roi in rois:
            full_db[roi] = []

        # Compute regional contribution
        for sub_id in subject_ids: #for each subject
            # Get current subject signals
            tseries = time_series[sub_id]
            n_volumes = tseries.shape[1]-2

            print(f"SUBJECT ID: {sub_id} ({n_volumes} volumes)")

            # Get current subject iPL 3D tensor
            iPL = phase_coherence(hilbert_phase(tseries))

            for roi in rois:
                # Get current roi index
                roi_idx = self.rois_labels.index(roi)

                # Get roi iPL vector for each time point
                iPL_roi = iPL[roi_idx,:,:]
                iPL_roi = iPL_roi[community_indices,:]

                # Save in dataframe
                iPL_roi_df = pd.DataFrame(iPL_roi.T,columns=self.state_rois(k,state))

                # Drop column of ROI (because contains all ones)
                iPL_roi_df.drop(columns=roi,inplace=True)

                # Add metadata
                if len(classes[sub_id])==1:
                    iPL_roi_df.insert(0,'condition',[classes[sub_id][0] for i in range(n_volumes)])
                else:
                    iPL_roi_df.insert(0,'condition',classes[sub_id][1:-1])

                iPL_roi_df.insert(0,'subject_id',[sub_id for i in range(n_volumes)])

                # Append df of current subject
                # and roi in dictionary
                full_db[roi].append(iPL_roi_df)

        # Compute mean across vector for
        # each time point, roi, and subject
        final = {}
        
        for roi in rois:
            subs,conds,meta = [],[],[]
            for sub_id in subject_ids:
                df_roi = pd.concat(full_db[roi]) # Concatenate data of all subjects
                df_sub = df_roi[df_roi.subject_id==sub_id]
                conditions = np.unique(df_sub.condition)
                for cond in conditions:
                    df_sub_ = df_sub[df_sub.condition==cond].iloc[:,2:].values
                    subs.append(sub_id)
                    conds.append(cond)
                    meta.append(np.mean(df_sub_))

            final[roi] = pd.DataFrame({'subject_id':subs,'condition':conds,roi:meta})

        # Merge results of all rois in a single dataframe
        if len(rois)==1:
            contribution = final[roi]
        else:
            dfs = list(final.keys())
            contribution = final[dfs[0]]

            for df in dfs[1:]:
                contribution = pd.merge(contribution,final[df],how='inner',on=['subject_id','condition'])

        return contribution

    def _pool_stats(self,metric="occupancies"):
        """
        Pool the stats for each k for the selected
        metric in a single dataframe.

        Params:
        --------
        metric : str.
            Specify the metric of interest
            ('occupancies','dwell_times','transitions').
        """
        data_path = self._dynamics_
        #get list with folder names
        k_folders = [folder for folder in os.listdir(data_path) 
                    if os.path.isdir(os.path.join(data_path,folder)) 
                    and folder.startswith('k_')]

        stats_all = []
        for folder in k_folders:
            stats_all.append(
                pd.read_csv(f'{data_path}/{folder}/{metric}_stats.csv',sep='\t')
                )
        stats_all = pd.concat((stats_all),ignore_index=True)

        stats_all = stats_all.sort_values(by='k').reset_index(drop=True)

        return stats_all

    def _pool_dynamics_metric(self,metric="occupancies"):
        """
        Pool the computed values for each k for the
        selected metric in a dictionary.

        Params:
        --------
        metric : str.
            Specify the metric of interest.
            ('occupancies','dwell_times','transitions').
        """
        data_path = self._dynamics_
        #get list with folder names
        k_folders = [folder for folder in os.listdir(data_path)
                    if os.path.isdir(os.path.join(data_path,folder)) 
                    and folder.startswith('k_')]

        pooled_metric = {}
        for folder in k_folders:
            pooled_metric[folder] = pd.read_csv(f'{data_path}/{folder}/{metric}.csv',sep='\t')

        return pooled_metric

    #plotting methods

    def plot_states_nodes(self,k=2,state='all',node_size=15,show_labels=True,open=True,save=False):
        """
        Create a 3D interactive figure embedded in a
        .html file showing the BOLD phase-locking (PL)
        states in anatomical MNI space. Each parcel/ROI
        is represented as a node. Nodes that are part of
        the PL pattern are coloured in red, and the rest
        of nodes are coloured in blue.

        Params:
        -------
        k : int.
            Select the partition of interest.

        state : int or str.
            Use an integer to plot a single
            PL state of interest, or 'all'
            to plot all the PL states of the
            selected K partition.

        node_size: int or float.
            Define the size of the nodes. Nodes
            that don't belong to the pattern are
            plotted smaller.

        show_labels : bool.
            Whether to show each ROI label.

        open : bool. 
            Whether to open the plots in web
            browser. If False, you can open the
            figures using the '.open_in_browser()'
            method of the returned object/s.

        save : bool.
            Whether to save each plot in a
            .html file. If True, the files
            are saved in 'LEiDA_results/brain_plots'.

        Returns:
        --------
        plot/s : dict or single figure.
            If state='all', return a dictionary
            that contains the constructed plots.
            They can be opened or saved using
            '.open()' and '.save_as_html(path)',
            respectively. If state=int, then return
            a single figure.
        """
        if self.rois_coordinates_ is None:
            raise Exception("You can't create this plot because the "
                            "ROI's coordinates could't be loaded.")

        centroids = self.load_centroids(k=k).values
        plots = brain_states_nodes(
            centroids,
            self.rois_coordinates_,
            node_size=node_size,
            state=state,
            nodes_labels=None if not show_labels else self.rois_labels,
            open=open
            )
        
        if save:
            _save_html(self._results_path_,plots,k,state,plot_type='nodes')

        return plots
    
    def plot_states_network(self,k=2,state='all',node_size=8,node_color='infer',linewidth=3,open=True,save=False):
        """
        Create a 3D interactive figure embedded in a
        .html file showing the BOLD phase-locking (PL)
        states as a connected network. All the ROIs/parcels
        that belong to the selected phase-locking state are
        connected between each other.

        Params:
        -------
        k : int.
            Select the partition of interest.

        state : int or str.
            Use an integer to plot a single
            PL state of interest, or 'all'
            to plot all the PL states of the
            selected K partition.

        node_size : int. 
            Select the size of the nodes.

        node_color : str. 
            Select the color of the nodes. If
            'infer', then the nodes participating
            in the PL states are colored red and
            the rest blue. If 'black', then all the
            nodes are colored in the same way.

        linewidth : int. 
            Select the size of the edges
            connecting the nodes.

        open : bool. 
            Whether to open the plots in web
            browser. If False, you can open the
            figures using the '.open_in_browser()'
            method of the returned object/s.

        save : bool.
            Whether to save each plot in a
            .html file. If True, the files
            are saved in 'LEiDA_results/brain_plots'.

        Returns:
        --------
        plot/s : dict or single figure.
            If state='all', return a dictionary
            that contains the constructed plots.
            They can be opened or saved using
            '.open()' and '.save_as_html(path)',
            respectively. If state=int, then return
            a single figure.
        """
        if self.rois_coordinates_ is None:
            raise Exception("You can't create this plot because the "
                            "ROI's coordinates could't be loaded.")

        _check_k_input(self._K_min_,self._K_max_,k)
        if not isinstance(state,(int,str)):
            raise TypeError("'state' must be either 'all' or an integer "
                            "specifying the number of a particular PL state.")
        elif isinstance(state,str):
            if state!='all':
                raise ValueError("If a string is provided, 'state' must be 'all'!")
        else:
            _check_state(k,state)

        centroids = self.load_centroids(k=k).values #load centroids for the selected k

        #plotting
        plot = brain_states_network(
            centroids,
            self.rois_coordinates_,
            state=state,
            node_size=node_size,
            node_color=node_color,
            linewidth=linewidth,
            open=open
            )

        #saving figures
        if save:
            _save_html(self._results_path_,plot,k,state,plot_type='network')

        return plot

    def plot_stats_pyramid(self,metric='occupancies',conditions=None,despine=True):
        """
        Create a pyramid of barplots showing the 'metric'
        of interest for each group, cluster (PL state), and
        K partition. Each barplot (which represents a particular
        PL state) is coloured according to its associated p-value:
        -black: the p-value is higher than 0.05.
        -red: the p-value is lower than 0.05 but higher than 0.05 / k.
        -green: the p-value is lower than 0.05/k but higher than 0.05 / Σ(k).
        -blue: the p-value is lower than 0.05 / Σ(k).

        Params:
        -------
        metric : str.
            Select the dynamical systems theory metric
            of interest (either 'occupancies' or 'dwell_times').

        conditions : None | list. Optional.
            (Usefull only when your data contains more
            than two conditions). You can provide a list
            specifying only two conditions of interest to
            plot. Otherwise create a plot for each pair of
            conditions. 

        despine : bool. Default = True.
            Whether to despine top and right axes of the
            subplots.
        """
        _check_metric(metric)

        if conditions is not None:
            if not isinstance(conditions,list) or len(conditions)!=2:
                raise Exception("If provided, 'conditions' must be a list with two items.")
            for cond in conditions:
                if cond not in self._classes_lst_:
                    raise Exception(f"'{cond}' was not founded in the data. "
                                    f" Valid options are: {self._classes_lst_}")
        else:
            conditions = self._classes_lst_.copy()

        pooled_stats = self._pool_stats(metric=metric)
        K_min = np.min(pooled_stats.k)
        K_max = np.max(pooled_stats.k)

        for cond in combinations(conditions,2):
            pooled_stats_ = pooled_stats[
                    (pooled_stats.group_1.isin(cond))
                    &
                    (pooled_stats.group_2.isin(cond))
                    ].reset_index(drop=True)

            dyn_metric = self._pool_dynamics_metric(metric=metric) 
            dyn_metric = {k:v[v.condition.isin(conditions)] for k,v in dyn_metric.items()}

            stats_pyramid(
                dyn_metric,
                pooled_stats_,
                K_min=K_min,
                K_max=K_max,
                despine=despine
                )

    def plot_clusters3D(self,k=2,clusters_colors=None,grid=True,alpha=.7,dot_size=3,edgecolor=None,darkstyle=False):
        """
        Visualize the identified clusters (BOLD phase-locking
        states) in a 3D scatter plot, which constitutes a
        low-dimensional representation of the 'state space'. 
        Method : take the eigenvectors and extract the first
        three principal components to reduce the dimensionality
        of the data to a 3D space. Each dot in the plot thus
        represents a single eigenvector, and is coloured according
        to the cluster it belongs to.
        
        Params:
        -------
        k : int.
            Specify the partition to plot.

        clusters_colors : list (optional). 
            Provide a list with the desired color
            of each cluster. If not provided, then
            a predefined set of colors will be used.

        grid : bool. 
            Whether to show grid or not.

        alpha : float. 
            Set transparency of dots.

        dot_size : float. 
            Select the dot size.

        edge_color : None | str.
            Specify an edge color to use
            on dots.

        darkstyle : bool.
            Whether to use a dark theme for
            the plot.
        """
        X = self.eigenvectors.iloc[:,2:].values #keep array containing only the eigenvectors
        y = self.predictions[f'k_{k}'].values #keep 1D array with the labels of each eigenvector

        with plt.style.context('dark_background' if darkstyle else 'default'):
            plot_clusters3D(
                X,
                y,
                clusters_colors=clusters_colors,
                grid=grid,
                alpha=alpha,
                dot_size=dot_size,
                edgecolor=edgecolor
                )

    def barplot_centroids(self,k=2,state='all'):
        """
        Create either subplots with barplots showing the
        values of each cluster centroid for the selected 'k'
        partition, or a single barplot showing the values
        of a specific phase-locking state.

        Params:
        ------
        k : int.
            Select the partition of interest.

        state : str or int.
            Specify if plot all the states for
            the selected 'k', or a single state
            of interest.
        """
        centroids = np.array(self.load_centroids(k=k),dtype=np.float32) #get centroids of the selected 'k'.
        if state=='all':
            barplot_states(centroids,self.rois_labels)
        else:
            _check_state(k,state)
            barplot_eig(centroids[state-1,:],self.rois_labels)
            plt.title(f'PL pattern {state}',fontsize=18)
            plt.tight_layout()

    def plot_pvalues(self,metric='occupancies',conditions=None,darkstyle=False,fill_areas=True):
        """
        Create a scatter plot showing the p-values
        obtained by the statistical analysis of a given
        'metric' across the explored 'k' range.

        Params:
        -------
        metric : str. 
            Specify the metric of interest
            ('occupancies','dwell_times','transitions').

        conditions : None | list. Optional
            (Usefull only when your data contains more
            than two conditions). You can provide a list
            specifying only two conditions of interest to
            plot. Otherwise create a plot for each pair of
            conditions. 

        darkstyle : bool.
            Whether to use a dark theme for
            the plots.

        fill_areas : bool.
            Select whether to fill the significance
            areas with color.
        """
        _check_metric(metric)

        if not isinstance(fill_areas,bool) or not isinstance(darkstyle,bool):
            raise TypeError("'fill_areas' and 'darkstyle' must be True or False!")

        if conditions is not None:
            if not isinstance(conditions,list) or len(conditions)!=2:
                raise Exception("If provided, 'conditions' must be a list with two items.")
            for cond in conditions:
                if cond not in self._classes_lst_:
                    raise Exception(f"'{cond}' was not founded in the data. "
                                    f" Valid options are: {self._classes_lst_}")
        else:
            conditions = self._classes_lst_.copy()

        pooled_stats = self._pool_stats(metric=metric)

        for cond in combinations(conditions,2):
            pooled_stats_ = pooled_stats[
                    (pooled_stats.group_1.isin(cond))
                    &
                    (pooled_stats.group_2.isin(cond))
                    ].reset_index(drop=True)

            scatter_pvalues(pooled_stats_,metric=metric,darkstyle=darkstyle,fill_areas=fill_areas)

    def plot_voronoi_cells(self,k=2):
        """
        Plot the clusters centroids in a 2D Voronoi
        cells space. Performs a PCA to reduce the
        dimensionality of the original centroid space
        to a 2D space.

        Params:
        --------
        k : int.
            Select the clustering solution to plot.

        Note: see Vohryzek, Deco et al. (2020) p.4
        """
        _check_k_input(self._K_min_,self._K_max_,k)
        centroids = self.load_centroids(k=k).values 
        plot_voronoi(centroids)

    def plot_clustering_performance(self):
        """
        Create a 2x2 panel with lineplots showing
        the clustering evaluation metrics for each
        k partition explored (Dunn score, distortion,
        silhouette score, and Davis-Bouldin score).
        """
        performance = pd.read_csv(f'{self._clustering_}/clustering_performance.csv',sep='\t')
        plot_clustering_scores(performance)

    def plot_states_network_glass(self,k=2,darkstyle=False):
        """
        Create a glass brain (axial view) showing the
        network representation of each phase-locking
        (PL) state for the selected 'k' partition.

        Params:
        -------
        k : int.
            Select the partition of interest.

        darkstyle : bool.
            Whether to use a dark theme for
            the plots.
        """
        _check_k_input(self._K_min_,self._K_max_,k)
        pl_states = self.load_centroids(k=k).values

        with plt.style.context("dark_background" if darkstyle else "default"):
            states_k_glass(pl_states,self.rois_coordinates_,darkstyle=darkstyle)

    def plot_states_in_bold(self,subject_id,k=2,alpha=.5,darkstyle=False):
        """
        Create plot showing the time-series of BOLD signals, 
        highlighting the dominant phase-locking (PL) state
        at each time point or volume.

        Params:
        -------
        subject_id : str.
            Specify the 'id' of the subject
            of interest.

        k : int.
            Select the k partition.

        alpha : float.
            Transparency of the colors that
            show the dominant PL pattern of
            each time point.

        darkstyle : bool.
            Whether to create the plot using
            a darkstyle.
        """
        _check_k_input(self._K_min_,self._K_max_,k)

        signals = load_tseries(self._data_path_)
        if subject_id not in signals.keys():
            raise ValueError("The provided 'subject_id' was not founded!")
            
        signals = signals[subject_id][:,1:-1] #get subject signals (and exclude 1st and last volumes)
        y = self.predictions[self.predictions.subject_id==subject_id][f'k_{k}'].values #get predictions for selected k.

        with plt.style.context('dark_background' if darkstyle else 'default'):
            states_in_bold(signals,y,alpha=alpha)

    def plot_states_on_surf(self,k=2,state='all',parcellation=None,discretize=True,cmap='auto',darkstyle=False,open=True,save=False):
        """
        Create a 3D interactive figure embedded in a
        .html file showing the BOLD phase-locking (PL)
        states on cortical surface. By default, all the
        cortical regions that belong to a given PL state
        or pattern are coloured in red(s), while the rest
        of cortical regions are coloured in blue(s). You
        can change the colormap throught the 'cmap' argument.

        Params:
        -------
        k : int.
            Partition of interest.

        state : str or int.
            Whether to plot 'all' the PL states of
            the selected partition or a single state
            of interest.

        parcellation : str.
            Path to the .nii file containing the
            parcellation from which the time series
            were extracted.

        discretize : bool. Default = True.
            Whether to plot the raw values of the
            phase-locking state/centroid, or plot
            all the brain regions that belong to
            the phase-locking state with the same
            intensity.

        cmap : str or matplotlib colormap, optional. Default = 'auto'.
            Colormap to use in the brain plot.
            If 'auto', then the brain regions that
            belong to the phase-locking state will
            be coloured in red(s), and the rest of
            regions in blue(s).

        darkstyle : bool.
            Whether to use a black background.

        open : bool. 
            Whether to open the plots in web
            browser. If False, you can open the
            figures using the '.open_in_browser()'
            method of the returned object/s.

        save : bool.
            Whether to save each plot in a
            .html file. If True, the files
            are saved in 'LEiDA_results/brain_plots'.

        Returns:
        --------
        g : SurfaceView or dictionarity of SurfaceViews. 
        """
        _check_k_input(self._K_min_,self._K_max_,k)

        if not isinstance(state,(int,str)):
            raise TypeError("'state' must be either 'all' or an integer specifying the number of a particular PL state")
        if isinstance(state,str):
            if state!='all':
                raise ValueError("If a string is provided, 'state' must be 'all'!")
        else:
            _check_state(k,state)

        centroids = self.load_centroids(k=k).values 
        if state!='all':
            centroids = centroids[state-1,:]
        
        g = brain_states_on_surf(
            centroids,
            parcellation=parcellation,
            black_bg=darkstyle,
            open=open,
            discretize=discretize,
            cmap=cmap
            )

        if save:
            _save_html(self._results_path_,g,k,state,plot_type='surface')

        return g

    def plot_states_on_surf2(self,k=2,state=1,parcellation=None,surface='pial',hemi='right',view='lateral',darkstyle=False,save=False):
        """
        Plot a BOLD phase-locking state of interest 
        on cortical surface mesh. 

        Params:
        -------
        k : int.
            Partition of interest.

        state : int.
            Select the PL state/pattern of
            interest.

        parcellation : str.
            Path to the .nii file containing
            the parcellation from which the
            signals were extracted.

        surface : str.
            Specify the surface type to plot
            the pattern on. Valid options are
            'pial','infl', and 'white'.

        hemi : str.
            Select the hemisphere to plot.
            Valid options are 'right', 'left',
            or 'both'.

        view : str
            View of the surface that is rendered. 
            Default='lateral'. Options = {'lateral',
            'medial', 'dorsal', 'ventral', 'anterior',
            'posterior'}. If 'hemi'='both', then 'dorsal'
            and 'lateral' views are displayed.

        darkstyle : bool.
            Whether to use a black background.

        save : bool.
            Whether to save the created figure in
            local folder. If True, the files are
            saved in 'LEiDA_results/brain_plots',
            and the plot will not be displayed.

        Returns:
        --------
        g : matplotlib figure. 
        """
        _check_k_input(self._K_min_,self._K_max_,k)
        _check_state(k,state)

        centroids = self.load_centroids(k=k).values 
        centroid = centroids[state-1,:]

        #plotting
        print("\n-Creating plot. This may take "
            "some minutes. Please wait...")
            
        with plt.style.context('dark_background' if darkstyle else 'default'):
            g = brain_states_on_surf2(
                centroid,
                parcellation=parcellation,
                hemi=hemi,
                surface=surface,
                view=view,
                open=False if save else True
                )
        
        if save:
            try:
                path = f'{self._results_path_}/brain_plots'
                if not os.path.exists(path): 
                    os.makedirs(path)
                filename = f"{path}/K{k}_PL_state_{state}_{surface}surf_{hemi}hemi_{view if hemi!='both' else 'multiview'}.png"
                g.savefig(filename,dpi=300)
                plt.clf()
                plt.close('all')
                del g
                print(f"The plot was save at: {filename}")
            except:
                raise Exception("An error occured when saving the plot.")
        else:
            return g

    def explore_state(self,k=2,state=1,darkstyle=False):
        """
        Create a figure showing a phase-locking state of
        interest in different formats:
        a barplot, a network representation in brain space,
        a matrix representation, and two boxplots with the
        occupancies and dwell times for each group/condition.

        Params:
        ------
        k : int.
            Select the partition of interest.

        state : int.
            Select the PL state of interest.

        darkstyle : bool.
            Whether to use a dark background.
        """
        _check_k_input(self._K_min_,self._K_max_,k)
        _check_state(k,state)

        centroid = self.load_centroids(k=k).values
        centroid = centroid[state-1,:]

        occ = self.occupancies(k=k)[['condition',f'PL_state_{state}']]
        dt = self.dwell_times(k=k)[['condition',f'PL_state_{state}']]

        with plt.style.context('dark_background' if darkstyle else 'default'):
            _explore_state(
                centroid,
                self.rois_labels,
                occ,
                dt,
                self.rois_coordinates_,
                state_number=state,
                darkstyle=darkstyle
                )

    def plot_states_matrices(self,k=2,cmap='jet',darkstyle=False):
        """
        Take the controids resulting from the k-means
        clustering (i.e., the phase-locking states) and
        reconstruct the connectivity patterns in matrix
        format.

        Params:
        -------
        k : int.
            Specify the K partition of interest.
        
        cmap : str. Default = 'jet'.
            Select the colormap to use.

        darkstyle : bool.
            Whether to use a black background.
        """
        _check_k_input(self._K_min_,self._K_max_,k)

        if not isinstance(cmap,str):
            raise TypeError("'cmap' must be a string!")
        if not isinstance(darkstyle,bool):
            raise TypeError("'darkstyle' must be True or False!")

        _ = centroid2matrix(
            self.load_centroids(k).values,
            plot=True,
            cmap=cmap,
            darkstyle=darkstyle
            )

    def plot_states_pyramid(self,parcellation=None,surface='pial',hemi='right',view='lateral',darkstyle=False):
        """
        Create a pyramid showing the PL states
        of each K partition in transparent 
        surface mesh. The created figure is
        automatically saved as a .png file in
        'LEiDA_results/clustering'.
        Note: subcortical regions are not
        displayed.

        Params:
        -------
        parcellation : str.
            Path to the .nii file containing
            the parcellation from which the
            signals were extracted.

        surface : str.
            Specify the surface type to plot
            the pattern on. Valid options are
            'pial','infl', and 'white'.

        hemi : str. (Options = 'right', 'left','both').
            Select the hemisphere to plot when 
            'lateral' view is selected.
            If 'dorsal' view is selected, then
            'view' must be 'both'.

        view : str.
            View of the surface that is rendered. 
            Default='lateral'. 
            Options = {'lateral', 'dorsal'}.

        darkstyle : bool.
            Whether to use a black background.

        save : bool.
            Whether to save the created figure in
            local folder. If True, the file is
            saved in 'LEiDA_results/clustering'.
        
        """
        states_pyramid(self,parcellation,surface,hemi,view,darkstyle)