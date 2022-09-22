import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind,levene,ks_2samp
from scipy.stats import permutation_test
from itertools import combinations

def ks_distance(txt1,txt2,plot=False):
    """
    Compute the 2-sample Kolmogorov-Smirnov test of
    goodness-of-fit (aka distance) between distributions
    of two (or groups of) Time x Time matrices. This
    technique can be used, e.g., to compare matrices
    between groups/conditions or to determine the best
    fit between a simulated FC matrix and a real/empirical
    FC matrix.

    Params:
    -------
    txt1 : ndarray with shape (N_time_points, N_time_points) or (N_time_points, N_time_points, N_subjects). 

    txt2 : ndarray with shape (N_time_points, N_time_points) or (N_time_points, N_time_points, N_subjects). 

    Returns:
    --------
    distance : float
        Computed distances between matrices.

    pval : float
        P-value of the statistical test.
    """
    if txt1.ndim>2:
        txt1_,txt2_ = [],[]
        for sub in range(txt1.shape[-1]):
            txt1_.extend(txt1[:,:,sub][np.triu_indices_from(txt1[:,:,sub],k=1)])
            txt2_.extend(txt2[:,:,sub][np.triu_indices_from(txt2[:,:,sub],k=1)])
        txt1_, txt2_ = np.array(txt1_),np.array(txt2_)

    else:
        txt1_ = txt1[np.triu_indices_from(txt1,k=1)]
        txt2_ = txt2[np.triu_indices_from(txt2,k=1)]
    distance,pval = ks_2samp(txt1_,txt2_)

    if plot: 
        plt.figure()
        sns.distplot(txt1_)
        sns.distplot(txt2_)
        plt.tight_layout()
        plt.show()

    return distance, pval

def permtest_ind(data,class_column=None,n_perm=5_000,alternative='two-sided'):
    """
    Compute a permutation test on two independent
    groups for each variable or feature in 'data',
    and additionally computes a Bonferroni-corrected
    alpha.

    Params:
    -------
    data : pd.dataframe.
        Contains the dataset with the metric values
        we want to compare. 
        E.g.: the fractional occupancies.

    class_colum : str.
        Specify the name of the column that contains
        the class/group/session/condition information.

    n_perm : int. Default 5000.
        Select the number of permutations to perform.

    alternative : str. Default='two-sided'.
        Select the test type. Options are 'less', 'two-sided',
        'greater'.

    Returns:
    --------
    results : pd.dataframe.
        Contains the results of the permutation tests.
    """
    #Validation of input data
    if class_column is None:
        raise ValueError("You must specify the 'class_column'.")
    elif class_column not in data.columns:
        raise ValueError(f"The 'class_column' '{class_column}' not founded in 'data'!")
    if not isinstance(n_perm,int):
        raise TypeError("'n_perm' must be an integer!")
    
    features = [col for col in data if col!=class_column] #list with variables to be tested
    n_tests = len(features) #number of tests that will be executed
    groups = np.unique(data[class_column]) #get the groups names.
    results = [] #list to save results

    for col in features:
        x1 = data[data[class_column]==groups[0]][col].values #data of first group.
        x2 = data[data[class_column]==groups[1]][col].values #data of the other group.

        #running Levene's test
        _,p_levene = levene(x1,x2,center='mean')

        #running the permutation test
        test = ttest_ind(
            x1,
            x2,
            alternative=alternative,
            permutations=n_perm,
            equal_var=True if p_levene>0.05 else False
        )
        
        #computing effect size
        eff = hedges_g(x1,x2)

        results.append({
            #'k': k,
            'variable':col, 
            'group_1':groups[0],
            'group_2':groups[-1],
            'statistic':test.statistic,
            'p-value':test.pvalue,
            'test':'t-test' if p_levene>0.05 else 'welch',
            'effect_size':eff
            })

    results = pd.DataFrame(results)

    #Bonferroni correction for multiple testing comparison
    results['alpha_Bonferroni'] = 0.05/n_tests
    results['reject_null'] = [True if p<(0.05/n_tests) else False for p in results['p-value'].values]

    return results

def hedges_g(x1,x2,paired=False):
    """
    Computes Hedges's g (effect size).
    To understand the magnitude of the detected intergroup
    differences  independently of the sample size, the effect
    size is estimated using Hedge's statistic (Hedges, 1981).
    The use of this measure is based on its appropriateness
    to easure the effect size for the difference between means
    and to account for the size of the sample from each group.

    Params:
    -------
    x1 : ndarray with shape (N_samples).
        Observations of 1st condition/group.

    x2 : ndarray with shape (N_samples).
        Observations of 2nd condition/group.

    paired : bool.
        Whether conditions/groups are paired
        or independent.

    Returns:
    --------
    g : float.
        Hedge's effect size.
    """
    #sample sizes
    n1 = x1.size
    n2 = x2.size
    
    #degrees of freedom
    dof = n1+n2-2
    
    #variances
    var1 = np.var(x1)
    var2 = np.var(x2)
    
    #difference in means
    m1 = np.mean(x1)
    m2 = np.mean(x2)
    diff_mean = np.abs(m1-m2)
    
    #pooled standard deviation
    #s1 = np.std(x1)
    #s2 = np.std(x2)
    
    #Hedges's g
    if not paired:
        s_pooled = np.sqrt(
            (((n1-1)*var1)+((n2-1)*var2))/dof
            )
        g = diff_mean/s_pooled

    else:
        g = diff_mean / np.sqrt((var1+var2) / 2)

    return g

def permtest_rel(data,class_column=None,n_perm=5_000,alternative='two-sided'):
    """
    Compute a permutation test on two related-paired
    groups for each variable or feature in 'data', and
    additionally computes a Bonferroni-corrected alpha.

    Params:
    -------
    data : pd.dataframe.
        Contains the dataset with the metric values
        we want to compare. E.g.: the fractional occupancies.

    class_colum : str.
        Specify the name of the column that contains
        the class/group/session/condition information.

    n_perm : int. Default 5000.
        Select the number of permutations to perform.

    alternative : str. Default='two-sided'.
        Select the test type. Options are 'less', 'two-sided', 'greater'.

    Returns:
    --------
    results : pd.dataframe.
        Contains the results of the permutation test.
    """
    #Validation of input data
    if class_column is None:
        raise ValueError("You must specify the 'class_column'.")
    elif class_column not in data.columns:
        raise ValueError(f"The 'class_column' '{class_column}' is not present in 'data'!")
    if not isinstance(n_perm,int):
        raise TypeError("'n_perm' must be an integer!")
    
    features = [col for col in data if col!=class_column] #list with variables to be tested
    n_tests = len(features) #number of tests that will be executed
    groups = np.unique(data[class_column]) #get the groups names.
    results = [] #list to save results
    rng = np.random.default_rng()

    for col in features:
        x1 = data[data[class_column]==groups[0]][col].values #data of first group.
        x2 = data[data[class_column]==groups[1]][col].values #data of the other group.

        #running the permutation test
        test = permutation_test(
            (x1, x2), 
            _statistic, 
            n_resamples=n_perm, 
            vectorized=True, 
            alternative=alternative,
            permutation_type='samples',
            random_state=rng
            )

        #computing effect size
        eff = hedges_g(x1,x2,paired=True)

        results.append({
            #'k': k,
            'variable':col, 
            'group_1':groups[0],
            'group_2':groups[-1],
            'statistic':test.statistic,
            'p-value':test.pvalue,
            'effect_size':eff
            })

    results = pd.DataFrame(results)

    #Bonferroni correction for multiple testing comparison
    results['alpha_Bonferroni'] = 0.05/n_tests
    results['reject_null'] = [True if p<(0.05/n_tests) else False for p in results['p-value'].values]

    return results

def _statistic(x, y, axis):
    """Necessary function to run the 'permtest_ind' and 'permtest_rel' functions."""
    return np.mean(x, axis=axis) - np.mean(y, axis=axis)

def _compute_stats(dynamics_data,paired_tests=False,n_perm=5_000,alternative='two-sided',save_results=True,path=None):
    """
    Performs the statistical analysis of dwell times and
    occupancies for each cluster (i.e., phase-locking state)
    of each K partition.

    Params:
    --------
    dynamics_data : dict.
        Output of 'compute_dynamics_metrics' function.
        Contains the values of a given dynamical systems
        theory metric for each K partition.

    paired_tests : bool. Default: False
        Specify if groups are independent or related/paired,
        to run the correct test.

    n_perm : int.
        Number of permutations.

    alternative : str. Default: 'two-sided'.
        Specify the hypothesis to test.
        Options: 'two-sided','greater','less'. 

    save_results : bool.
        Whether to save results on local folder.

    path : str.
        Specify the path in which the results will
        be saved if 'save_results' was set to True.
    """
    #validation of input arguments parameters
    alternative_options = ['two-sided','greater','less']

    if alternative not in alternative_options:
        raise ValueError(f"Valid 'alternative' options are {alternative_options}.")

    stats_all = {}

    ks = list(dynamics_data['occupancies'].keys())

    for metric in ['dwell_times','occupancies']:
        stats_all[metric] = {}
        for k in ks:
            data = dynamics_data[metric][k]
            data = data.iloc[:,1:] #drop the column with subject_ids

            conditions = np.unique(data.condition) #get conditions
            N_conditions = conditions.size
            results_stacked = []

            #for each combination of conditions
            for conds in combinations(conditions,2): 
                data_ = data[data.condition.isin(conds)] #keep data from current 2 conditions

                #compute statistics
                if paired_tests:
                    stats_results = permtest_rel(
                        data_,
                        class_column='condition',
                        alternative=alternative,
                        n_perm=n_perm
                        )
                else:
                    stats_results = permtest_ind(
                        data_,
                        class_column='condition',
                        alternative=alternative,
                        n_perm=n_perm
                        )

                results_stacked.append(stats_results)

            if N_conditions>2:
                metric_results = pd.concat(results_stacked,axis=0).reset_index(drop=True)
            else:
                metric_results = pd.DataFrame(stats_results)

            metric_results.insert(0,'k',int([i for i in k.replace('_',' ').split() if i.isdigit()][0]))
            stats_all[metric][k] = metric_results

            if save_results:
                try:
                    data_path = f'{path}/dynamics_metrics' #path in which the dynamical systems theory metrics for each k were saved.
                    metric_results.to_csv(f'{data_path}/{k}/{metric}_stats.csv',sep='\t',index=False)
                except:
                    print("Warning: there was a problem when saving the results to local folder.")
    
    print('*The statistical analysis has finished.')
    return stats_all

#plotting
def scatter_pvalues(pooled_stats,metric='occupancies',fill_areas=True,darkstyle=False):
    """
    Create a scatter plot showing the computed
    p-values for each cluster in each clustering
    partition ('k'). In addition, the plot shows
    the significance thresholds defined by the
    standard alpha value (0.05) and the Bonferroni-corrected
    alpha for each k (0.05/k).

    Params:
    -------
    pooled_stats : pandas.dataframe.
        Contain the computed statistics for each
        cluster and for each k.

    metric : str. Optional.
        Specify the provided metric. Just used for
        plot title. If None, no title is shown.

    fill_areas : bool.
        Whether to fill with color the areas
        defined by the two significance thresholds.
    """
    pooled_stats.rename(columns={'alpha_Bonferroni':'bonf','p-value': 'p'},inplace=True)

    K_min = min(np.unique(pooled_stats.k))
    K_max = max(np.unique(pooled_stats.k))
    alpha3 = np.sum(np.arange(K_min,K_max+1,1))

    #Create list assigning a color to each pvalue.
    color_list = []
    for pval,bonf in zip(pooled_stats.p,pooled_stats.bonf):
        color_list.append('mediumseagreen' if 0.05/alpha3<pval<bonf 
                        else 'firebrick' if 0.05>pval>bonf
                        else 'royalblue' if pval<0.05/alpha3
                        else ('black' if not darkstyle else 'white')
                        )

    #adding data to bonferroni alphas so we can plot from K-min-1 to K_max+1
    bonf_data = pooled_stats[['k','bonf']]
    bonf_data.loc[len(bonf_data.index)] = [K_min-1,0.05/(K_min-1)]
    bonf_data.loc[len(bonf_data.index)] = [K_max+1,0.05/(K_max+1)]
    bonf_data = bonf_data.sort_values(by='k').reset_index(drop=True) 

    #plotting
    with plt.style.context('dark_background' if darkstyle else 'default'):
        plt.ion()
        plt.figure(figsize=(11,6))
        plt.scatter(pooled_stats.k,pooled_stats.p,c=color_list,s=2)
        plt.axhline(0.05,linestyle="dashed",c='firebrick',label=r'$\alpha^{1}$ = 0.05')

        plt.plot(
            bonf_data.k,
            bonf_data.bonf,
            linestyle="dashed",
            c='mediumseagreen',
            label=r'$\alpha^{2}$ = $\alpha^{1}$ / k'
            )

        plt.axhline(
            0.05/alpha3,
            linestyle="dashed",
            c='royalblue',
            label=r'$\alpha^{3}$ = $\alpha^{1}$ / Σ (k)'
            ) #alpha 3 = Σk / 0.05

        plt.yscale('log')
        plt.xlabel('Number of PL states in\neach clustering solution (K)',fontsize=15)
        plt.ylabel('Two-sided p-value',fontsize=15)
        plt.xticks(np.arange(K_min,K_max+1),[k for k in np.arange(K_min,K_max+1)])
        plt.xlim(left=K_min-1,right=K_max+1)
        plt.ylim(bottom=0.00000001)
        if metric is not None:
            title = f"{str(metric).capitalize().replace('_',' ')}: {np.unique(pooled_stats.group_1)[0]} vs {np.unique(pooled_stats.group_2)[0]}"
            plt.title(title,fontsize=20,pad=20)
        plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')

        if fill_areas:
            plt.fill_between(
                np.arange(K_min-1,K_max+2), 
                np.flip(np.unique(bonf_data.bonf)), 
                0.05/alpha3,
                alpha=0.15,
                color='mediumseagreen'
                )
            plt.fill_between(
                np.arange(K_min-1,K_max+2),
                np.flip(np.unique(bonf_data.bonf)),
                0.05,
                alpha=0.15,
                color='firebrick'
                )
            plt.fill_between(
                np.arange(K_min-1,K_max+2),
                -1,
                0.05/alpha3,
                alpha=0.15,
                color='royalblue'
                )

        plt.tight_layout()