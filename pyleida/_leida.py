"""Class to execute the LEiDA pipeline."""

import numpy as np 
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import Bunch
from itertools import combinations

import warnings
warnings.filterwarnings("ignore")

from .dynamics_metrics import (
    group_transition_matrix,
    compute_dynamics_metrics,
) 
from .data_utils import (
    load_tseries,
    load_classes,
    load_rois_labels,
    load_rois_coordinates
)
from .clustering import (
    identify_states,
    plot_clusters3D,
    centroid2matrix,
    plot_voronoi,
    barplot_states,
    barplot_eig,
    plot_clustering_scores
)
from .clustering import rsnets_overlap as rsnets
from .plotting import (
    brain_states_nodes,
    brain_states_network,
    states_in_bold,
    plot_pyramid,
    states_k_glass,
    brain_states_on_surf,
    brain_states_on_surf2,
    _explore_state,
    _save_html
)
from .signal_tools import (
    hilbert_phase,
    phase_coherence,
    get_eigenvectors
)
from .data_utils.validation import (
    _check_k_input,
    _check_metric,
    _check_state
)
from .stats import _compute_stats,scatter_pvalues

class Leida:
    """
    Class to execute the LEiDA pipeline
    and explore the results.

    Params:
    -------
    data_path : str.
        Path to the folder that contains the time
        series, ROIs labels and coordinates, and
        the group/condition of each subject.

    Attributes:
    -----------
    eigenvectors : pandas.dataframe.
        Contains the computed eigenvectors.

    predictions : pandas.dataframe.
        Contains the predicted cluster label of
        each eigenvector for each 'k' partition
        explored.

    rois_labels : list.
        The label/name of each ROI/parcel.

    rois_coordinates : ndarray of shape (n_rois,3).
        The MNI coordinates of each ROI/parcel.

    time_series : dict.
        Contains the BOLD signals of each subject.
        The keys are the subject id's, and the values
        are numpy 2D arrays (N_ROIs, N_volumes) with
        the time series of each brain region/parcel.

    classes : dict.
        Contains the condition/group label/s
        of each subject.
        Keys are subject id's.

    """
    def __init__(self,data_path):
        if not isinstance(data_path,str):
            raise TypeError("'data_path' must be a string!")
        else:
            if not os.path.exists(data_path):
                raise ValueError(f"The specified path could't be founded.")

        self.time_series = load_tseries(data_path)
        self.classes = load_classes(data_path)
        self.rois_labels = load_rois_labels(data_path)
        self.rois_coordinates = load_rois_coordinates(data_path)

        self._validate_constructor_params() #check if the data has been loaded sucessfully.

    def fit_predict(self,TR=None,paired_tests=False,n_perm=5_000,save_results=True,random_state=None):
        """
        Execute the LEiDA pipeline: 
            1) Compute the instantaneous phase of each signal, the
            phase-coherence matrices, and extract the eigenvectors.

            2) Fit a K-means model for each k partition, identify
            the phase-locking (PL) states (centroids), and assign
            a cluster label to each observation (eigenvectors).

            3) Compute the dwell times, occupancies, and transitions
            probabilities for each 'k' partition.

            4) Perform a statistical analysis of occupancies and
            dwell times for each 'k' partition.

        Params:
        -------
        TR : None | np.float | int.
            Specify the Repetition Time of the fMRI data.
            If 'None', then the dwell times will express
            the mean lifetime of each PL state in TR units,
            instead of seconds.

        paired_tests : bool. Default: False.
            Specify if groups are independent or related/paired,
            to run the correct statistical tests.

        n_perm : int.
            Select the number of permutations that will be
            applied when running the statistical analysis of
            dwell times and occupancies for each k.

        save_results : bool.
            Whether to create folders and files to save the
            results on local folder. If True, then a folder
            called 'LEiDA_results' containing all the results
            will be created. Note: These results can be easely
            retrieved later using the 'DataLoader' class.

        random_state : None | int.
            Determines random number generation for centroid
            initialization. Use an int to make the randomness
            deterministic.
        """
        self._K_min_ = 2
        self._K_max_ = 20
        
        #validate provided 'TR'
        if TR is not None and not isinstance(TR,(int,float)):
            raise TypeError("'TR' must be 'None', and integer or a floating number!")

        #validate paired_tests input
        if not isinstance(paired_tests,bool):
            raise TypeError("'paired_tests' must be either True or False.")
        
        #check that a valid number of permutations was provided
        if not isinstance(n_perm,int):
            raise TypeError("'n_perm' must be an integer!")
        else:
            if n_perm<100:
                raise ValueError("The number of permutations cannot be lower than 100.")

        #validate 'save_results' input
        if not isinstance(save_results,bool):
            raise TypeError("'save_results' must be a boolean value (True or False)!")

        #Run the analysis
        self._results_path_ = 'LEiDA_results'

        self.eigenvectors,self._clustering_,self._dynamics_ = self._execute_all(
            TR=TR,
            random_state=random_state,
            paired_tests=paired_tests,
            n_perm=n_perm,
            save_results=save_results,
            )

        self.predictions = self._clustering_.predictions
        self._classes_lst_ = np.unique(self.eigenvectors.condition).tolist()
        self._N_classes_ = len(self._classes_lst_)
        self._is_fitted = True

    def _execute_all(self,TR=None,random_state=None,paired_tests=False,n_perm=5_000,save_results=True):
        """
        Perform all the steps of the LEiDA (for each subject
        and group/condition/session):

        1) Compute neccesary data:
            a) Computes the instantaneous phase of each
            signal at each time point.
            b) Computes the phase-coherence or PL matrices
            from the previously computed signals phases.
            c) Extracts the leading eigenvector from each
            phase-coherence matrix at time t.

        2) Compute dynamics metrics for each K.
        3) Performs statistical analysis of occupancies and
        dwell times of each PL pattern and for each K partition.

        Params:
        -------
        TR : None | np.float or int. Default = None.
            Specify the Repetition Time of the fMRI data.
            If None, the Dwell times express the mean
            lifetime duration of each PL state in volumes.

        random_state : None or int.
            Determines random number generation for centroid
            initialization. Use an int to make the randomness
            deterministic.

        paired_tests : bool. Default: False
            Specify if groups are independent or related/paired,
            to run the correct statistical tests.

        n_perm : int. Default = 5000.
            Select number of permutations to apply when running
            the statistical tests.

        save_results : bool. Default = True.
            Whether to save the results on local disk.
        
        Returns: 
        --------
        df_eigens : pd.DataFrame.
            Contains the computed eigenvectors.
            1st column contains subject ids, and
            2nd column the group/condition.

        clustering : dict/bunch.
            Contains the k-means predictions for each
            K partition, the scores of clustering performance,
            and the fitted models.

        dynamics : dict/bunch.
            Contains the computed occupancies, dwell times,
            transitions probabilities and statistical analysis
            results for each K.
        """
        subject_ids = list(self.time_series.keys()) #list of subject ids
        N_subjects = len(subject_ids) #number of provided subjects

        #creating folder to save results
        if save_results:
            if os.path.exists(self._results_path_):
                raise Warning("EXECUTION ABORTED: The folder 'LEiDA_results' already "
                            "exists. If you have results from earlier executions of "
                            "the analysis, consider changing the folder's name or moving "
                            "the folder to another location.")
            else:
                try:
                    print(f"\n-Creating folder to save results: './{self._results_path_}'")
                    os.makedirs(self._results_path_)
                except:
                    raise Exception("The folder to save the results could't be created.")

        #creating variables to save results    
        eigens = []
        sub_list, class_list = [], []

        #Starting process
        print("\n-STARTING THE PROCESS:\n"
            "========================\n"
            f"-Number of subjects: {N_subjects}")
            
        print("\n 1) EXTRACTING THE EIGENVECTORS.\n")

        for sub_idx,sub_id in enumerate(subject_ids): #for each subject
            #get current subject signals
            tseries = self.time_series[sub_id]
            N_volumes = tseries.shape[1]-2

            print(f"SUBJECT ID: {sub_id} ({tseries.shape[1]} volumes)")
            
            #Extract the eigenvectors from each phase-coherence matrix at time t.
            eigens.append(
                get_eigenvectors(phase_coherence(hilbert_phase(tseries)))
                )

            #Append metadata to lists (to complete the eigenvectors dataset)
            for volume in range(N_volumes):
                sub_list.append(sub_id)
                if len(self.classes[sub_id])>1:
                    class_list.append(self.classes[sub_id][volume+1])
                else:
                    class_list.append(self.classes[sub_id][0])
        
        # creating dataframe with the extracted eigenvectors
        # and metadata (subjects ids and conditions)
        df_eigens = pd.DataFrame(np.vstack(eigens),columns=self.rois_labels)
        df_eigens.insert(0,'subject_id',sub_list)
        df_eigens.insert(1,'condition',class_list)
        
        #saving results
        if save_results:
            try:
                df_eigens.to_csv(f'{self._results_path_}/eigenvectors.csv',sep='\t',index=False)
            except:
                print("Warning: An error ocurred when saving the 'eigenvectors.csv' file to local folder.")
        
        #clustering
        print("\n 2) RUNNING K-MEANS CLUSTERING ON EIGENVECTORS.")
        predictions,clustering_performance,models = identify_states(
            df_eigens,
            K_min=self._K_min_,
            K_max=self._K_max_,
            random_state=random_state,
            save_results=save_results,
            path=self._results_path_ if save_results else None
            )


        #computing dynamical systems theory metrics for each K
        print("\n 3) COMPUTING THE DYNAMICAL SYSTEMS THEORY METRICS FOR EACH K.")
        dynamics_data = compute_dynamics_metrics(
            predictions,
            TR=TR,
            save_results=save_results,
            path=self._results_path_ if save_results else None
            )

        #Statistical analysis of occupancies and dwell times for each k
        print("\n 4) EXECUTING THE STATISTICAL ANALYSIS OF "
            "OCCUPANCIES AND DWELL TIMES FOR EACH K.")
        stats = _compute_stats(
            dynamics_data,
            paired_tests=paired_tests,
            n_perm=n_perm,
            save_results=save_results,
            path=self._results_path_ if save_results else None
            )

        #Plotting statistical analyses results
        # (scatter plots and barplots' pyramid)
        print("\n-Creating figures with the statistical analyses results for dwell times "
            "and fractional occupancies. This may take some time. Please wait...")

        classes = np.unique(df_eigens.condition)

        for metric in ['occupancies','dwell_times']:
            pooled_stats = pd.concat((stats[metric]),ignore_index=True)

            for conditions in combinations(classes,2):
                pooled_stats_ = pooled_stats[
                    (pooled_stats.group_1.isin(conditions))
                    &
                    (pooled_stats.group_2.isin(conditions))
                    ].reset_index(drop=True)

                #plot pyramid
                dyn_data = {k:v[v.condition.isin(conditions)] for k,v in dynamics_data[metric].items()}
                plot_pyramid(
                    dyn_data,
                    pooled_stats_,
                    K_min=self._K_min_,
                    K_max=self._K_max_,
                    metric_name=metric,
                    despine=True
                    )
                if save_results:
                    plt.savefig(f'{self._results_path_}/dynamics_metrics/{conditions[0]}_vs_{conditions[-1]}_{metric}_barplot_pyramid.png',dpi=300)

                #plot p-values scatter plot
                scatter_pvalues(pooled_stats_,metric=metric,fill_areas=True)
                if save_results:
                    plt.savefig(f'{self._results_path_}/dynamics_metrics/{conditions[0]}_vs_{conditions[-1]}_{metric}_scatter_pvalues.png',dpi=300)

        #Preparing output
        clustering = Bunch(
            predictions =  predictions,
            performance = clustering_performance,
            models = models
            )

        dynamics = Bunch(
                dwell_times = dynamics_data['dwell_times'],
                occupancies = dynamics_data['occupancies'],
                transitions = dynamics_data['transitions'],
                stats = stats
            )

        print("\n** THE ANALYSIS HAS FINISHED SUCCESFULLY!")
        if save_results:
            print(f"-All the results were save in './{self._results_path_}'")

        print("\n-You can explore the results in detail by using "
            "the methods and attributes of the Leida class.")
        
        #output
        return df_eigens,clustering,dynamics

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
        self._check_is_fitted()
        _check_k_input(self._K_min_,self._K_max_,k)
        model = self._clustering_.models[f'k_{k}']
        return model

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
        self._check_is_fitted()
        _check_k_input(self._K_min_,self._K_max_,k)
        centroids = pd.DataFrame(self.load_model(k=k).cluster_centers_,columns=self.rois_labels)
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
        _check_k_input(self._K_min_,self._K_max_,k)

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
        self._check_is_fitted()
        _check_k_input(self._K_min_,self._K_max_,k)
        _check_metric(metric)

        df_stats = self._dynamics_.stats[metric][f'k_{k}']
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
        self._check_is_fitted()
        _check_k_input(self._K_min_,self._K_max_,k)

        return self._dynamics_.dwell_times[f'k_{k}']

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
        self._check_is_fitted()
        _check_k_input(self._K_min_,self._K_max_,k)

        return self._dynamics_.transitions[f'k_{k}']

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
        self._check_is_fitted()
        _check_k_input(self._K_min_,self._K_max_,k)

        return self._dynamics_.occupancies[f'k_{k}']

    def significant_states(self,metric="occupancies"):
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
        self._check_is_fitted()
        _check_metric(metric)

        stats = self._pool_stats(metric=metric)
        has_results = True if stats[stats.reject_null==True].shape[0]>=1 else False #check if some result was significant
        if has_results:
            return stats[stats.reject_null==True]
        else:
            print("No significant results were founded.")
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
        self._check_is_fitted()
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

        #create list with conditions labels
        conditions = []
        for val in self.classes.values():
            for item in val:
                conditions.append(item)

        if len(conditions) != len(self.time_series.keys()):
            raise Exception("This method is only available in "
                "cases where each subject has only one condition label.")
        if group not in conditions:
            raise ValueError("'group' must be present in your data. "
                f"Possible options are: {[i for i in np.unique(conditions)]}.")       

        subjects_ids = [sub for sub,condition in zip(self.time_series.keys(),conditions) if condition==group]
        N_subjects = len(subjects_ids)
        N_rois = len(self.rois_labels)
        pooled_static_fc = np.empty((N_rois,N_rois,N_subjects))
        
        for idx,sub in enumerate(subjects_ids):
            pooled_static_fc[:,:,idx] = np.corrcoef(self.time_series[sub])
        
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
        self._check_is_fitted()
        _check_k_input(self._K_min_,self._K_max_,k)

        mats = group_transition_matrix(
            self._dynamics_.transitions[f'k_{k}'],
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
        self._check_is_fitted()
        _check_k_input(self._K_min_,self._K_max_,k)
        _check_state(k,state)

        centr = {}

        for k_ in range(self._K_min_,self._K_max_+1):
            cent = self.load_centroids(k_)
            cent.insert(0,'state',[i+1 for i in range(cent.shape[0])])
            cent.insert(0,'k',k_)
            centr[f'k{k_}'] = cent
        
        centr = pd.concat(centr,ignore_index=True)

        corr,pvals = rsnets.compute_overlap(
            centr,
            parcellation=parcellation,
            n_areas=n_areas
            )

        overlap = rsnets.state_overlap(
            corr,
            pvals,
            k=k,
            state=state,
            plot=True,
            darkstyle=darkstyle
            )

        return overlap

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
        self._check_is_fitted()

        data = {}
        for k in np.arange(self._K_min_,self._K_max_+1):
            data[k] = self.stats(k=int(k),metric=metric)
        data = pd.concat((data),ignore_index=True)
        return data

    #Parameters validation

    def _validate_constructor_params(self):
        """Validate input data provided in 'path'."""
        
        #1.validate time series.
        #check whether all the subjects have
        #the same number of brain regions
        nrois = [self.time_series[sub].shape[0] for sub in self.time_series.keys()]
        if np.unique(nrois).size>1:
            raise Exception("The number of brain regions must be "
                            "the same for all the subjects.")

        #check that the number of ROIs labels
        #coincide with the number of ROIs in time series
        if np.unique(nrois).item()!=len(self.rois_labels):
            raise Exception(f"The number of brain regions in time series ({np.unique(nrois).item()}) must "
                            f"coincide with the number of ROIs labels ({len(self.rois_labels)}).")

        #2.validate metadata (labels info).
        #check if the number of subjects in 'time_series' 
        #and 'classes' dictionaries coincide.
        nsub_signals = len(self.time_series.keys())
        nsub_classes = len(self.classes.keys())
        if nsub_signals != nsub_classes:
            raise Exception(f"The number of subjects in 'time_series' (n={nsub_signals}) "
                            f"and'metadata' (n={nsub_classes}) must be equal.")
        #check if each subject in 'time_series' has his corresponding metadata.
        elif bool(set(self.time_series.keys())-set(self.classes.keys())):
            raise Exception(f"The metadata of the following subject/s was not founded:\n "
                            f"{set(self.time_series.keys())-set(self.classes.keys())}")
        #check if the number of labels match with the number of volumes.
        else:
            n_labels = [len(self.classes[sub]) for sub in self.classes.keys()] #get n of labels per subject
            if np.max(n_labels)!=1:
                labels_info = list()
                for sub in self.time_series.keys():
                    labels_info.append(
                        {'subject_id':sub,
                        'n_volumes':self.time_series[sub].shape[1],
                        'n_labels':len(self.classes[sub])}
                    )
                labels_info = pd.DataFrame(labels_info)
                
                mismatchs_ids = labels_info[~(labels_info['n_volumes'] == labels_info['n_labels'])].subject_id.tolist()

                if bool(mismatchs_ids):
                    print(labels_info[labels_info.subject_id.isin(mismatchs_ids)])
                    raise Exception("Some subject/s have a different number of "
                                    "volumes and labels!")
        
            #check if more than 1 condition is present in 'metadata'
            conditions = []
            for val in self.classes.values():
                for item in val:
                    conditions.append(item)
            if np.unique(conditions).size==1:
                raise Exception("The number of conditions must be at least 2!")

        #3.validate rois coordinates
        if self.rois_coordinates is None:
            print("The ROIs coordinates couldn't be loaded from the provided 'data_path'. "
                "Brain plots that show nodes in brain space will not be executed in consequence.")
        else:
            #check if the number of roi labels match 
            #with the number of MNI coordinates.
            if len(self.rois_labels) != self.rois_coordinates.shape[0]:
                raise Exception(f"The number of 'ROIs labels' ({len(self.rois_labels)} were provided) "
                    f"must coincide with the number of provided coordinates ({self.rois_coordinates.shape[0]} were provided)")

        print("All the data has been sucesfully loaded.")

    def _check_is_fitted(self):
        """
        Check if the k-means models had been already fitted.
        """
        if not hasattr(self,"_is_fitted"):
            raise Exception("You have to fit the models first by using the 'fit_predict' method.")

    #Plotting methods

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
        self._check_is_fitted()
        _check_k_input(self._K_min_,self._K_max_,k)

        X = np.array(self.eigenvectors.iloc[:,2:], dtype=np.float32) #keep array containing only the eigenvectors
        y = self._clustering_.predictions[f'k_{k}'].values #keep 1D array with the labels of each eigenvector

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
        self._check_is_fitted()
        _check_k_input(self._K_min_,self._K_max_,k)

        centroids = self.load_centroids(k=k).values 
        plot_voronoi(centroids)

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
        self._check_is_fitted()
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

    def plot_states_nodes(self,k=2,state=1,node_size=15,show_labels=True,open=True,save=False):
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
        self._check_is_fitted()
        if self.rois_coordinates is None:
            raise Exception("You can't create this plot because the "
                            "ROI's coordinates could't be loaded.")

        _check_k_input(self._K_min_,self._K_max_,k)
        centroids = self.load_centroids(k=k).values
        plots = brain_states_nodes(
            centroids,
            self.rois_coordinates,
            node_size=node_size,
            state=state,
            nodes_labels=None if not show_labels else self.rois_labels,
            open=open
            )

        if save:
            _save_html(self._results_path_,plots,k,state,plot_type='nodes')

        return plots

    def plot_states_network(self,k=2,state=1,node_size=8,node_color='infer',linewidth=3,open=True,save=False):
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
        if self.rois_coordinates is None:
            raise Exception("You can't create this plot because the "
                            "ROI's coordinates could't be loaded.")
        
        self._check_is_fitted()
        _check_k_input(self._K_min_,self._K_max_,k)

        if not isinstance(state,(int,str)):
            raise TypeError("'state' must be either 'all' or an integer "
                            "specifying the number of a particular PL state.")
        if isinstance(state,str):
            if state!='all':
                raise ValueError("If a string is provided, 'state' must be 'all'!")
        else:
            _check_state(k,state)

        centroids = self.load_centroids(k=k).values #load centroids for the selected k

        #plotting
        plot = brain_states_network(
            centroids,
            self.rois_coordinates,
            state=state,
            node_size=node_size,
            node_color=node_color,
            linewidth=linewidth,
            open=open,
            )

        #saving figures
        if save:
            _save_html(self._results_path_,plot,k,state,plot_type='network')
            
        return plot

    def plot_states_in_bold(self,subject_id,k=2,alpha=.5,darkstyle=False):
        """
        Create plot showing the time-series of BOLD signals, 
        highlighting the dominant phase-locking (PL) state
        of each time point or volume.

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
        self._check_is_fitted()
        _check_k_input(self._K_min_,self._K_max_,k)
        tseries = self.time_series[subject_id][:,1:-1] #get subject signals (excluding the 1st and last volumes)
        predictions = self._clustering_.predictions
        y = predictions[predictions.subject_id==subject_id][f'k_{k}'].values #get predictions for selected k.

        with plt.style.context("dark_background" if darkstyle else "default"):
            states_in_bold(tseries,y,alpha=alpha)

    def plot_states_pyramid(self,metric='occupancies',conditions=None,despine=True):
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
        self._check_is_fitted
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

        for cond in combinations(conditions,2):
            pooled_stats_ = pooled_stats[
                    (pooled_stats.group_1.isin(cond))
                    &
                    (pooled_stats.group_2.isin(cond))
                    ].reset_index(drop=True)

            dyn_metric = self._dynamics_[metric]
            dyn_metric = {k:v[v.condition.isin(conditions)] for k,v in dyn_metric.items()}

            plot_pyramid(
                dyn_metric,
                pooled_stats_,
                K_min=self._K_min_,
                K_max=self._K_max_,
                despine=despine
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
        self._check_is_fitted()
        _check_k_input(self._K_min_,self._K_max_,k)

        centroids = np.array(self.load_centroids(k=k),dtype=np.float32) #get centroids of the selected 'k'.
        if state=='all':
            barplot_states(centroids,self.rois_labels)
        else:
            _check_state(k,state)
            barplot_eig(centroids[state-1,:],self.rois_labels)
            plt.title(f'PL pattern {state}',fontsize=18,pad=15)
            plt.tight_layout()

    def plot_clustering_performance(self):
        """
        Create a 2x2 panel with lineplots showing
        the clustering evaluation metrics for each
        k partition explored (Dunn score, distortion,
        silhouette score, and Davis-Bouldin score).
        """
        self._check_is_fitted()
        plot_clustering_scores(self._clustering_.performance)

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
        if self.rois_coordinates is None:
            raise Exception("You can't create this plot because the ROI's coordinates could't be loaded.")

        self._check_is_fitted()
        _check_k_input(self._K_min_,self._K_max_,k)

        pl_states = self.load_centroids(k=k).values

        with plt.style.context("dark_background" if darkstyle else "default"):
            states_k_glass(pl_states,self.rois_coordinates,darkstyle=darkstyle)

    def plot_states_on_surf(self,k=2,state='all',parcellation=None,discretize=True,cmap='auto',darkstyle=False,open=False,save=False):
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
        self._check_is_fitted()
        _check_k_input(self._K_min_,self._K_max_,k)

        if not isinstance(state,(int,str)):
            raise TypeError("'state' must be either 'all' or an integer "
                            "specifying the number of a particular PL state")
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

        darkstyle : bool
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
        self._check_is_fitted()
        _check_k_input(self._K_min_,self._K_max_,k)
        _check_state(k,state)

        centroids = self.load_centroids(k=k).values 
        centroid = centroids[state-1,:]

        if save: 
            path = f'{self._results_path_}/brain_plots'
            if not os.path.exists(path): 
                os.makedirs(path)
        #plotting
        print("\n-Creating plot. This may take "
            "some minutes. Please wait...")

        with plt.style.context('dark_background' if darkstyle else 'default'):
            g = brain_states_on_surf2(
                centroid,
                parcellation=parcellation,
                hemi=hemi,
                surface=surface,
                view=view
                )
        
        if save:
            try:
                filename = f"{path}/K{k}_PL_state_{state}_{surface}surf_{hemi}hemi_{view if hemi!='both' else 'multiview'}{'_dark' if darkstyle else ''}.png"
                g.savefig(filename,dpi=300)
                plt.close()
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
        self._check_is_fitted()
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
                self.rois_coordinates,
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
        self._check_is_fitted()
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
