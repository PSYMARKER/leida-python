import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    adjusted_mutual_info_score,
    adjusted_rand_score
)
from sklearn.model_selection import (
    train_test_split,
    KFold,
    StratifiedKFold
)
from sklearn.decomposition import PCA
from scipy.spatial import Voronoi,voronoi_plot_2d
import pickle
from ..data_utils.validation import _check_isint

class KMeansLeida():
    """
    K-Means algorithm with cosine or euclidean
    distance-based optimization.

    Params:
    -------
    k : int.
        The number of clusters to form as well as
        the number of centroids to generate.

    metric : str.
        Whether to use 'cosine' or 'euclidean' distance
        to optimize the centroids.

    n_init : int.
        Number of time the k-means algorithm will be
        run with different centroid seeds. The final
        results will be the best output of n_init
        consecutive runs in terms of inertia.

    n_iter : int.
        Maximum number of iterations of the k-means
        algorithm for a single run.

    Attributes:
    -----------
    cluster_centers_ : ndarray of shape (n_centroids,n_ROIs).
        Coordinates of cluster centers.

    labels_ : ndarray of shape (n_samples,).
        Labels of each point.

    inertia_ : float.
        Sum of squared distances of samples to their closest
        cluster center (aka distortion).
    """
    def __init__(self,k=2, metric='cosine',n_init=10,n_iter=1_000):
        #validation of input data
        if not isinstance(metric,str):
            raise TypeError("'metric' must be a string!")
        if metric not in ['euclidean','cosine']:
            raise ValueError("'metric' must be either 'cosine' or 'euclidean'")

        _check_isint({
            'k':k,
            'n_init':n_init,
            'n_iter':n_iter
            })

        if k<2:
            raise ValueError("'k' must be > 1")

        self._k_ = k
        self._metric_ = metric
        self._n_iter_ = n_iter
        self._n_init_ = n_init

    def fit(self,y,random_state=None):
        """
        Compute k-means clustering.

        Params:
        --------
        y : ndarray of shape (n_samples,n_features).
            Training instances to cluster.

        random_state : int or None.
            Determines random number generation
            for centroid initialization. Use an
            int to make the randomness deterministic.

        Returns:
        --------
        self : object.
            Fitted estimator.
        """
        for init_idx in range(self._n_init_):
            #print(f'\tCurrent initialization: {init_idx+1}')
            if random_state is not None:
                np.random.seed(random_state)
            idx = np.random.choice(len(y), self._k_, replace=False)  

            #Step 1. Randomly assigning first centroids positions
            centroids = y[idx, :]
             
            #Step 2. Finding the distance between centroids and all the data labels
            distances = cdist(y, centroids ,self._metric_)
             
            #Step 3. Predict each observation label based on the minimum Distance
            labels = np.array([np.argmin(i) for i in distances]) #Step 3
             
            #Step 4.Repeating the above steps for a defined number of iterations
            labels_all = []
            for iter in range(self._n_iter_): 
                centroids = []
                for centroid_idx in range(self._k_):
                    #Updating centroids by computing the mean of the observations in each cluster
                    temp_cent = y[labels==centroid_idx].mean(axis=0) 
                    centroids.append(temp_cent)
         
                centroids = np.vstack(centroids) #Updated centroids 
                 
                distances = cdist(y, centroids ,self._metric_)
                labels = np.array([np.argmin(i) for i in distances])
                if iter!=0:
                    if np.array_equal(labels,labels_all[-1]):
                        #print(f'\tConverged at iteration n° {iter+1}')
                        break
                labels_all.append(labels)

            #compute distortion on final centroids of current centroids initialization
            distortion = self._distortion(y,centroids)

            #update centroids,distortion and labels keeping always the best updated
            if init_idx == 0:
                best_distortion = distortion
                optimum_centroids = centroids
                optimum_labels = labels.copy()
            if distortion < best_distortion:
                best_distortion = distortion.copy()
                optimum_centroids = centroids.copy()
                optimum_labels = labels.copy()
             
        self.cluster_centers_ = optimum_centroids
        self.labels_ = optimum_labels
        self.inertia_ = best_distortion

        self._remap_labels()
        self._is_fitted = True

    def _check_is_fitted(self):
        """
        Check if the k-means model had been already fitted.
        """
        if not hasattr(self,"_is_fitted"):
            raise Exception("You have to fit the model first by using the 'fit' method.")

    def transform(self,y,closest=False):
        """
        Computes distances between each observation
        or data point and each centroid. Differs from
        sklearn transform method in that here we can select
        if retrieve all the distances or only the distances
        to closest centroid. In the new space, each dimension
        is the distance to the cluster centers.

        Params:
        --------
        y : ndarray of shape (n_samples,n_features).
            Data to transform. 
        
        Returns:
        --------
        distances : ndarray of shape (n_samples,n_centroids).
            y transformed in the new space.
        """
        self._check_is_fitted()

        distances = cdist(y, self.cluster_centers_ ,self._metric_)
        if not closest:
            return distances
        else:
            return distances.min(1)

    def predict(self,y):
        """
        Assign cluster label to each data point in 'y'.
        Predict the closest cluster each sample in y belongs to.

        Params:
        -------
        y : ndarray of shape (n_samples,n_features).
            Data to predict.
        
        Returns:
        --------
        labels : ndarray of shape (n_samples,).
            Index of the cluster each sample belongs to.
        """
        self._check_is_fitted()
        
        labels = np.array([np.argmin(i) for i in self.transform(y)])
        return labels

    def _distortion(self,y,centroids):
        """
        Sum of squared distances of samples to their
        closest cluster center.

        Params:
        ------
        y : ndarray of shape (n_samples,n_features).
            Samples clustered.

        centroids : ndarray of shape (n_centroids,n_features).
            Computed centroids/prototypes.

        Returns:
        --------
        sum_sqr_dist : float.
            Computed distortion.
        """
        #distances = self.transform(y,closest=True)
        distances = cdist(y, centroids ,self._metric_).min(1)
        sqr_dist = distances**2
        sum_sqr_dist = np.sum(sqr_dist)
        
        return sum_sqr_dist

    def _remap_labels(self):
        """"
        After fitting the k-means model and making the
        labelling of each sample, this function computes
        the frequency of each cluster label across all samples,
        and relabels the original labels so that centroid 1 is
        always the cluster with higher number of observations.
        """
        label,freq = np.unique(self.labels_,return_counts=True)
        sorted_idxs = np.argsort(freq)[::-1]
        dct = {k:v for k,v in zip(sorted_idxs,label)}
        new_labels = np.array([dct[n] for n in self.labels_])
        self.labels_ = new_labels
        self.cluster_centers_ = self.cluster_centers_[sorted_idxs,:]


def identify_states(eigens_dataset,K_min=2,K_max=20,n_init=15,random_state=None,plot=True,save_results=False,path=None):
    """
    Perform k-means clustering for each value of 'k' to identify the
    different (discrete) number of clusters (i.e.,the phase-locking
    states). At eack 'k', a k-means model is fitted and a cluster is
    assign to each eigenvector. These 'predictions' are located in a
    new column of the provided 'eigens_dataset'.
    In addition, each 'k' is evaluated by means of the Dunn score or
    Dunn index, distortion (aka sum squared errors), silhouete score,
    and Davies Bouldin score.
    
    Params:
    -------
    eigens : ndarray with shape (n_eigevectors, n_ROIs). 
        Contains the extracted eigenvectors from the dFC matrices. 

    K_max, K_min : int. 
        Max and min number of clusters to fit.

    n_init : int.
        Number of times the clustering algorithm
        will be run with different centroid seeds.

    random_state : None or int.
        Determines random number generation for
        centroid initialization.
        Use an int to make the randomness deterministic.

    plot : bool. 
        Whether to create plots showing the number
        of k in the X axis, and the Dunn mean score,
        Silhoutte score, distortion, and Davis Bouldin
        score for each k in the Y axis.

    save_results : bool. 
        Whether to save results in '{path}/clustering'.

    path : str.
        Where to create the 'clustering' folder if
        'save_results' was set to True.

    Returns:
    --------
    eigens_predictions : pd.dataframe.
        Contains the metadata (subjects_ids and conditions),
        and a column for each K partition is added containing
        the predicted label for each observation (row).

    clustering_performance : pd.dataframe.
        Contains the Dunn score, Silhouette score and distorion
        values for each K explored.

    models : dict.
        Contain the fitted k-means models.
    """
    if save_results:
        try:
            results_path = f'{path}/clustering'
            if not os.path.exists(results_path):
                os.makedirs(results_path)
            print(f"-Creating folder to save results: './{results_path}'")
        except:
            print("Warning: the folder to save the results could't be created.")

    #Execute the k-means models fitting process
    eigens_predictions = eigens_dataset[['subject_id','condition']] #copy provided data (to avoid overwriting)
    X = np.array(eigens_dataset.iloc[:,2:],dtype=np.float32) #keep array with the eigenvectors (remove 'subject_ids' and 'condition' columns)

    N_samples = X.shape[0]
    if N_samples>30_000:
        random_samples_idx = np.random.choice(np.arange(N_samples),size=30_000,replace=False)

    clustering_performance = []

    models = {} #dict to save each fitted model

    for k in range(K_min,K_max+1):
        print(f'k = {k}')
        kmeans = KMeansLeida(k=k,n_init=n_init,n_iter=1_000)
        kmeans.fit(X,random_state=random_state)
        models[f'k_{k}'] = kmeans
        clustering_performance.append({
            'k':k,
            'Dunn_score':dunn_fast(X,kmeans.labels_) if N_samples<30_000 
                        else dunn_fast(X[random_samples_idx,:],kmeans.labels_[random_samples_idx]),
            'distortion':kmeans.inertia_,
            'silhouette_score':silhouette_score(X=X, labels=kmeans.labels_, metric='cosine') if N_samples<30_000
                        else silhouette_score(X=X[random_samples_idx,:], labels=kmeans.labels_[random_samples_idx], metric='cosine'),
            'Davies-Bouldin_score':davies_bouldin_score(X, kmeans.labels_)
            })
        
        #predict each volume label
        eigens_predictions.loc[:,f'k_{k}'] = kmeans.labels_ #adding column with clusters assignment

        #save model to disk
        if save_results:
            models_folder = f'{results_path}/models'
            if not os.path.exists(models_folder):
                os.mkdir(models_folder)
            model_filename = f'model_k_{k}.pkl'
            pickle.dump(kmeans, open(f'{models_folder}/{model_filename}', 'wb'))

    clustering_performance = pd.DataFrame(clustering_performance)

    if plot:
        plt.ioff()
        plot_clustering_scores(clustering_performance)
        if save_results:
            try:
                plt.savefig(f'{results_path}/clustering_performance.png',dpi=300)
            except:
                print('The figures were not saved on local folder.')
        plt.close()
    #save results
    if save_results:
        try:
            clustering_performance.to_csv(f'{results_path}/clustering_performance.csv',sep='\t',index=False)
            eigens_predictions.to_csv(f'{results_path}/predictions.csv',sep='\t',index=False)
        except:
            print('The .csv files were not saved on disk. '
                'You can still save them by yourself using the returned variables.')

    print('\n*The clustering process has succesfully ended.')
        
    return eigens_predictions,clustering_performance,models

def centroid2matrix(centroids,plot=True,cmap='jet',darkstyle=False):
    """
    Take the controids resulting from the k-means
    clustering (i.e., the phase-locking states) and
    reconstruct the connectivity pattern in matrix
    format.
    
    Note: see Cabral et al. 2017 [p.4 and p.6].
    
    Params:
    -------
    centroids : ndarray with shape (n_centroids,n_rois).
        Clusters centroids of a specific K partition.

    plot : bool.
        Whether to create plot showing
        the matrices.

    cmap : str. Default = 'jet'.
        Select the colormap to use.

    darkstyle : bool.
        Whether to use a black background.
    
    Returns:
    --------
    states : ndarray with shape (n_rois,n_rois,n_centroids).
        PL states in matrix format.
    """
    if not isinstance(centroids,np.ndarray) or centroids.ndim!=2:
        raise TypeError("'centroids' must be a 2D array (n_centroids,n_rois)!")

    N_states, N_regions = centroids.shape
    states = np.empty((N_regions,N_regions,N_states))
    
    for state_idx in range(N_states):
        #scale centroid by its maximum value and transpose the matrix
        centroid = centroids[state_idx,:]/np.max(np.abs(centroids[state_idx,:]))
        states[:,:,state_idx] = np.outer(centroid,centroid.T)

    if plot:
        plt.ion()
        with plt.style.context('dark_background' if darkstyle else 'default'):
            _, axs = plt.subplots(
                nrows=1,
                ncols=N_states,
                figsize=(N_states*2, 2 if N_states<=10 else 1),
                edgecolor='black',
                subplot_kw=dict(box_aspect=1)
                )

            axs = axs.ravel()
            ticks_size = 6 if N_states<=10 else 5 if 10<N_states<=15 else 4

            for state_idx in range(N_states):
                sns.heatmap(
                    states[:,:,state_idx],
                    ax=axs[state_idx],
                    vmax=1,
                    vmin=-1,
                    center=0,
                    cmap=cmap,
                    square=True,
                    cbar=False
                    )
                axs[state_idx].set_title(f'PL state {state_idx+1}',fontsize=7 if N_states<15 else 5)
                #axs[state_idx].set_xlabel(f'ROIs',fontsize=6.5)
                #axs[state_idx].set_ylabel(f'ROIs',fontsize=6.5)
                axs[state_idx].set_xticks(
                    np.arange(20,N_regions,20),
                    np.arange(20,N_regions,20).tolist(),
                    rotation=0
                    )
                axs[state_idx].set_yticks(
                    np.arange(20,N_regions,20),
                    np.arange(20,N_regions,20).tolist(),
                    )
                
                axs[state_idx].tick_params(
                    axis='both',
                    which='both',
                    bottom=False,
                    left=False,
                    top=False,
                    #labelsize=5 if N_states<15 else 3.5,
                    labelsize=ticks_size,
                    pad=0
                    #labelbottom=False,
                    #labelleft=False
                    )

        plt.tight_layout(pad=1,w_pad=1)

    return states

def patterns_stability(X,y=None,n_clusters=None,folds=5,metric='ari',plot=True,darkstyle=False):
    """
    Run a stratified KFold cross-validation to explore
    the stability of assigned clusters across folds.
    The provided data in 'X' and is splitted in a train
    set and a test set. Then, a cross-validation is
    performed by splitting the train set into n number
    of 'folds'. At each iteration, a given fold is used to
    fit a KMeans model, which is used to assign a cluster
    label to each observation of the test set. Finally,
    the similarity between clusters assignments of the test
    set across folds is evaluated by using either the ARI
    (adjusted Rand index) score ('ari') or the AMI (adjusted
    mutual information) score ('ami').

    Params:
    -------
    X : ndarray with shape (n_volumes,n_rois).
        Contains the eigenvectors.

    y : ndarray with shape (n_volumes,). Optional.
        Clusters assignment of each observation in 'X'.
        If 'None', you must provide 'n_clusters'.

    n_clusters : int. Optional.
        Select the numbers of clusters to fit the
        K-Means models. If 'y' is not None, then
        the number of clusters is inferred from
        the provided array.

    folds : int.
        Select the number of folds for the
        cross-validation.

    metric : str.
        Metric to compute the similarity between
        clusters assignments across folds.

    plot : bool.
        Whether to create a heatmap showing the
        scores obtained in the cross-validation.

    darkstyle : bool.
        Whether to use a dark background for plotting.

    Returns:
    --------
    scores : ndarray with shape (n_folds,n_folds).
        Array containing the computed values of the
        selected metric for each pair of folds.
    """
    if not isinstance(metric,str) or metric not in ['ari','ami']:
        raise Exception("'metric' must be either 'ari' or 'ami'")

    if not isinstance(X,np.ndarray):
        raise TypeError("'X' must be a 2D array!")

    if y is None and n_clusters is None:
        raise Exception("You must specify either predicted labels in 'y' "
                        "or a number of clusters ('n_clusters').")
    elif y is not None and n_clusters is not None:
        print("Warning: when 'y' is provided, the 'n_clusters' "
            "is inferred from 'y'.")

    if y is not None:
        if not isinstance(y,np.ndarray):
            raise TypeError("'y' must be an array!")
        n_clusters = np.unique(y).size
        X_tr,X_ts,y_tr,_ = train_test_split(X,y,stratify=y,train_size=.5)
    else:
        X_tr,X_ts = train_test_split(X,train_size=.5)

    #compute centroids on different folds
    kfold = StratifiedKFold(n_splits=folds) if y is not None else KFold(n_splits=folds)
    predictions_ = np.empty((X_ts.shape[0],folds))

    if y is not None:
        for i,(tr_idx,_) in enumerate(kfold.split(X_tr,y_tr)):
            print(f'Current fold: {i+1}')
            km = KMeansLeida(k=n_clusters)
            km.fit(X_tr[tr_idx])
            predictions_[:,i] = km.predict(X_ts)
    else:
        for i,(tr_idx,_) in enumerate(kfold.split(X_tr)):
            print(f'Current fold: {i+1}')
            km = KMeansLeida(k=n_clusters)
            km.fit(X_tr[tr_idx])
            predictions_[:,i] = km.predict(X_ts)

    #compute scores
    scores = np.empty((folds,folds))

    for fold_idx in range(folds):
        for fold_idx2 in range(folds):
            if metric=='ari':
                scores[fold_idx,fold_idx2] = adjusted_rand_score(
                    predictions_[:,fold_idx],
                    predictions_[:,fold_idx2]
                    )
            else:
                scores[fold_idx,fold_idx2] = adjusted_mutual_info_score(
                    predictions_[:,fold_idx],
                    predictions_[:,fold_idx2]
                    )

    font_sizes = {'folds':np.arange(2,21,1),'fsize':np.linspace(6,15,19)[::-1]}

    if plot:
        plt.ion()
        with plt.style.context('dark_background' if darkstyle else 'default'):
            matrix_ = np.triu(scores)
            plt.figure()
            sns.heatmap(
                scores[1:,:-1],
                vmin=0,
                center=.5,
                vmax=1,
                cmap='viridis',
                square=True,
                linecolor='black' if darkstyle else 'white',
                linewidths=.4,
                annot=True,
                annot_kws={'size':font_sizes['fsize'][folds-2]},
                mask=matrix_[1:,:-1],
                yticklabels=[f'{i+2}' for i in range(folds-1)],
                xticklabels=[f'{i+1}' for i in range(folds-1)],
                fmt='.2f',
                cbar_kws={'label': 'ARI score' if metric=='ari' else 'AMI score','shrink': 0.5}
                )
            plt.xlabel('Fold',fontsize=16)
            plt.ylabel('Fold',fontsize=16)
            plt.yticks(rotation=0)
            plt.tight_layout()

    return scores

# Plotting functions

def plot_clustering_scores(data):
    """
    Create a 2x2 panel with lineplots showing
    the clustering evaluation metrics for each
    explored k value (Dunn score, distortion,
    silhouette score, and Davis-Bouldin score).

    Params:
    -------
    data : pandas.dataframe.
        Contain k in the 1st col, with the
        rest of columns containing the values
        of the clustering evaluation metrics.
    """
    _,axs = plt.subplots(nrows=2,ncols=2,figsize=(11,6))
    axs = np.ravel(axs)
    for fig_idx,metric in enumerate(data.columns[1:].values):
        sns.lineplot(x='k',y=metric,data=data,ax=axs[fig_idx],linewidth=3)
        axs[fig_idx].set_xticks([i+2 for i in range(np.max(data.k)-1)])
        axs[fig_idx].set_xlabel('Number of clusters',fontsize=15)
        axs[fig_idx].set_ylabel(metric.replace('_',' '),fontsize=15)
        axs[fig_idx].grid(False)

    plt.tight_layout()

def barplot_eig(eig,features_list):
    """ 
    Create barplot showing the values of the
    eigenvector for each brain region. This
    eigenvector could be either an eigenvector
    of a particular time point, or a cluster
    centroid.

    Params:
    -------
    eig : ndarray with shape (N_ROIs,). 
        Contains the eigenvector to plot.

    features_list : list. 
        Contains the names of the ROIs. 
        Must be in the same order as in 'eig'.
    """
    # generating list of colors. 
    # Positive values get red, negative values get blue
    cols = ['mediumblue' if i<0 else 'firebrick' for i in eig]

    plt.ion()
    plt.figure(figsize=(4,15))
    sns.barplot(y=features_list,x=eig,orient='h',palette=cols)
    plt.axvline(0,color='black')
    if np.max(np.abs(eig))>0.15:
        plt.xlim(-1.05,1.05)
    else:
        plt.xlim(-0.11,0.11)
    plt.xticks(fontweight='regular')
    plt.yticks(fontweight='regular',fontsize=8)
    plt.ylabel('Brain areas',rotation='vertical',fontsize=24,labelpad=40)
    plt.tight_layout()

def barplot_states(centroids,rois_labels):
    """
    Create subplots with barplots showing
    the values of each cluster centroid.

    Params:
    ------
    centroids : ndarray with shape (n_centroids,n_rois).
        Contains the centroids.

    rois_labels : list. 
        Contains the names of each ROI.
    """
    if not isinstance(centroids,np.ndarray):
        raise TypeError("'centroids' must be a numpy 2D array!")
    if centroids.shape[1]!=len(rois_labels):
        raise Exception("The number of brain regions in 'centroids' and 'rois_labels' must be the same!")
    
    N_centroids = centroids.shape[0]

    plt.ion()
    _,axs = plt.subplots(
        ncols=N_centroids,
        nrows=1,
        sharey=True,
        figsize=(N_centroids if N_centroids>4 else 4,8)
        )
    axs = np.ravel(axs)

    for fig_idx in range(N_centroids):
        # generating list of colors.
        # Positive values get red, negative values get blue
        cols = ['mediumblue' if i<0 else 'firebrick' for i in centroids[fig_idx,:]]

        sns.barplot(
            y=rois_labels,
            x=centroids[fig_idx,:],
            orient='h',
            palette=cols,
            ax=axs[fig_idx]
            )
        axs[fig_idx].axvline(0,color='black')
        if np.max(np.abs(centroids[fig_idx,:]))>0.15:
            axs[fig_idx].set_xlim(-1.05,1.05)
        else:
            axs[fig_idx].set_xlim(-0.11,0.11)
        #axs[fig_idx].set_xlim(-0.11,0.11) 
        axs[fig_idx].tick_params(axis='y',labelsize=5)
        axs[fig_idx].tick_params(axis='x',labelsize=6 if N_centroids>5 else 3)
        axs[fig_idx].set_title(
            f'PL pattern {fig_idx+1}',
            fontweight='regular',
            fontsize=4,
            pad=10
            )
    axs[0].set_ylabel('Brain areas',rotation='vertical',fontsize=24,labelpad=25)
    plt.tight_layout()

def plot_clusters3D(eigens,labels,clusters_colors=None,grid=True,alpha=.7,dot_size=3,edgecolor=None):
    """
    Visualize the identified clusters (brain states) in a
    3D scatter plot, which constitutes a low-dimensional
    representation of the 'state space'. 
    Method : take the eigenvectors and extract the first
    three principal components to reduce the dimensionality
    of the data to a 3D space. Each dot in the plot thus
    represents a single eigenvector, and is coloured according
    to the cluster it belongs to.
    
    Params:
    -------
    eigens : ndarray with shape (n_samples, n_rois).
        Contains the eigenvectors of each
        subject for each time point.

    labels : ndarray with shape (n_samples,).
        Contains the predicted cluster assignment
        for each eigenvector in 'eigens'.

    clusters_colors : list [optional].
        Provide a list with the desired color
        of each cluster. If not provided, then
        a predefined set of colors will be used.

    grid : bool.
        Whether to show grid or not.

    alpha : float.
        Set transparency of dots.

    dot_size : float or int. 
        Set dot size.

    edge_color : None | str.
        Specify an edge color to use on dots.

    Returns:
    --------
    fig : generated plot.
    """
    if clusters_colors is not None:
        if np.unique(labels).size!=len(clusters_colors):
            raise ValueError('The number of clusters and the number of colours provided must match')
    else: 
        clusters_colors = [
            'royalblue',
            'grey',
            'tomato',
            'orange',
            'cyan',
            'violet',
            'yellow',
            'purple',
            'firebrick',
            'teal',
            'orchid',
            'red',
            'green',
            'steelblue',
            'indigo',
            'gold',
            'sienna',
            'coral',
            'olive',
            'salmon'
        ]

    pca = PCA(n_components=3)
    pcs = pca.fit_transform(eigens)
    x_pcs = pd.concat((pd.DataFrame(pcs),pd.Series(labels)),axis=1)
    x_pcs.columns = np.array(['PC_1','PC_2','PC_3','Cluster'])

    #plotting
    fig = plt.figure(figsize=(8,8))
    ax = plt.axes(projection ="3d")
    for i,color in zip(np.unique(labels),clusters_colors):
        ax.scatter3D(
            x_pcs[x_pcs.Cluster==i]['PC_3'],
            x_pcs[x_pcs.Cluster==i]['PC_2'],
            x_pcs[x_pcs.Cluster==i]['PC_1'], 
            c=color,
            edgecolors=edgecolor,
            s=dot_size,
            alpha=alpha
            )

    ax.w_xaxis.pane.fill = False
    ax.w_yaxis.pane.fill = False
    ax.w_zaxis.pane.fill = False

    ax.set_xlabel('3rd PC', fontsize=15, fontweight='regular',labelpad=20)
    ax.set_ylabel('2nd PC', fontsize=15, fontweight='regular',labelpad=20)
    ax.set_zlabel('1st PC', fontsize=15, fontweight='regular',labelpad=20)
    if not grid: 
        ax.grid(False)
    plt.tight_layout()
    plt.show()

    return fig

def plot_voronoi(centroids):
    """
    Plot the clusters centroids in a 2D Voronoi
    cells space. Performs a PCA to reduce the
    dimensionality of the original centroid space
    to a 2D space.

    Params:
    --------
    centroids : ndarray with shape (n_centroids, n_regions).
        Centroids of a particular K partition.

    """
    pcs = PCA(n_components=2).fit_transform(centroids)
    vor = Voronoi(pcs)

    plt.ion()
    voronoi_plot_2d(vor,point_size=10,show_vertices=False)
    plt.title(f'K = {centroids.shape[0]}')
    plt.tight_layout()

def plot_clusters_boundaries(y,n_clusters=2,alpha=0.05):
    """
    Plot cluster centroids decision boundaries in
    2D space after applying PCA.

    Params:
    -------
    y : ndarray with shape (n_samples, n_rois).
        Contains the eigenvectors.

    n_clusters : int.
        Specify the k number of clusters.

    alpha : float.
        Specify the transparency of the
        dots (observations).
    """
    ỹ = PCA(n_components=2).fit_transform(y) #embedding
    kmeans = KMeansLeida(k=n_clusters, n_init=4)
    kmeans.fit(ỹ)

    # Step size of the mesh. Decrease to increase the quality of the VQ.
    h = 0.0007  # point in the mesh [x_min, x_max]x[y_min, y_max].

    # Plot the decision boundary. For that, we will assign a color to each
    x_min, x_max = ỹ[:, 1].min(), ỹ[:, 1].max()
    y_min, y_max = ỹ[:, 0].min(), ỹ[:, 0].max()
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Obtain labels for each point in mesh. Use last trained model.
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)

    plt.ion()
    plt.figure(figsize=(8,8))
    #plt.clf()
    plt.imshow(
        Z,
        interpolation="nearest",
        extent=(xx.min(), xx.max(), yy.min(), yy.max()),
        cmap=plt.cm.Paired,
        aspect="auto",
        origin="lower",
        alpha=.9
    )

    # Plot dots (eigenvectors/samples)
    plt.plot(
        ỹ[:, 1],
        ỹ[:, 0],
        'o',
        markersize=4,
        markerfacecolor='black',
        markeredgecolor='grey',
        markeredgewidth=.5,
        alpha=alpha
        )

    # Plot the centroids as a white X
    centroids = kmeans.cluster_centers_
    plt.scatter(
        centroids[:, 1],
        centroids[:, 0],
        marker="X",
        s=400,
        linewidths=3,
        color="black",
        zorder=10,
        edgecolors='w'
    )

    plt.title(
        f"K-means clustering (K={n_clusters}) on the eigenvectors (PCA-reduced data)"
    )
    plt.xlim(x_min, x_max)
    plt.xlabel("Centroids are marked with white cross.\nObservations are marked with black dots")
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.tight_layout()

# Compute Dunn Score functions 

def dunn_fast(points, labels):
    """ 
    Compute the Dunn index.
    
    Params:
    ----------
    points : ndarray with shape (N_samples,N_features).
        Observations/samples.
        
    labels : ndarray with shape (N_samples).
        Labels of each observation in 'points'.
    """
    distances = cosine_distances(points)
    ks = np.sort(np.unique(labels))
    
    deltas = np.ones([len(ks), len(ks)])*1_000_000
    big_deltas = np.zeros([len(ks), 1])
    
    l_range = list(range(0, len(ks)))
    
    for k in l_range:
        for l in (l_range[0:k]+l_range[k+1:]):
            deltas[k, l] = _delta_fast((labels == ks[k]), (labels == ks[l]), distances)
        
        big_deltas[k] = _big_delta_fast((labels == ks[k]), distances)

    di = np.min(deltas)/np.max(big_deltas)
    return di

def _delta_fast(ck, cl, distances):
    values = distances[np.where(ck)][:, np.where(cl)]
    values = values[np.nonzero(values)]

    return np.min(values)
    
def _big_delta_fast(ci, distances):
    values = distances[np.where(ci)][:, np.where(ci)]
    #values = values[np.nonzero(values)]
            
    return np.max(values)
