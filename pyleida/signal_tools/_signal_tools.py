import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import hilbert
from scipy.sparse.linalg import eigs
from nilearn.signal import clean
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr

# Signals utils
def fourier_frec(signal_ts,T):
	sr = 1./T
	if signal_ts.ndim == 1:
		signal_f = np.abs(np.fft.fft(signal_ts))
		f = np.arange(0, sr / 2, 1. * sr / signal_ts.size)
		return f, signal_f[:f.size]
	elif signal_ts.ndim == 2:
		signal_f = np.abs(np.fft.fft(signal_ts,axis=1))
		f = np.arange(0, sr / 2, 1. * sr / signal_ts.shape[1])
		return f, signal_f[:, :f.size]
	else:
		raise Exception('signal_ts has more than 2 dimensions')

def hilbert_phase(signals):
    """
    Compute the Hilbert transform to get the
    instantaneous phase of the BOLD time series
    of each brain region/parcel.

    Params:
    -------
    signals : ndarray of shape (N_rois, N_time_points).
        BOLD time series of a particular subject.

    Return:
    -------
    phase : ndarray of shape (N_rois, N_time_points).
    """
    phase = hilbert(signals, axis=1)
    N_rois = phase.shape[0]
    for roi in range(N_rois): 
        phase[roi, :] = np.angle(phase[roi, :]).real
    phase = phase.real
    return phase

def ang_shortest_diff(a,b):
	"""
    Compute the shortest difference between angles.
	
    Params:
    -------
	a : angle

	b : angle
	
    Returns:
    --------
    c : the shorthest difference.
	"""
	if np.abs(a-b)> np.pi:
		c = 2*np.pi-np.abs(a-b)
	else:
		c = np.abs(a-b)
	return c

def clean_signals(signals,detrend=True,standardize='zscore',filter_type=None,low_pass=None,high_pass=None,TR=None):
    """
    Perform a cleaning process of the signals.
    Low-pass filtering improves specificity.
    High-pass filtering should be kept small,
    to keep some sensitivity.

    Note: uses nilearn's 'signal.clean()' function.

    Params:
    -------
    signals : dict 
        Contains the BOLD signals to process.
        Keys must be subjects ids and values
        arrays with shape (N_rois, N_volumes).

    detrend : bool.
        Whether to perform a detrending of the
        signals.

    standardize : str or bool {'zscore','psc',False}.
        Method to standardize the signals. If 'zscore',
        the signals are shifted to zero mean and unit
        variance.
        If 'psc', the signals are shifted to zero mean
        and scaled to percent signal change.

    filter_type : str or bool {'butterworth','cosine',False}
        Method to filter the signals, if desired. If
        False, do not perform filtering.

    low_pass : None or float.
        Low cutoff frequency in Hertz. If specified,
        signals above this frequency will be filtered
        out. 
        If None, no low-pass filtering will be performed.
        Default=None

    high_pass : None or float.
        High cutoff frequency in Hertz. If specified,
        signals below this frequency will be filtered
        out. Default=None.

    TR : None, int, or float.
        Specify the Time Repetition of the fMRI scans.

    Returns:
    --------
    clean_signals : dict.
        Contains the cleaned signals.
    """
    if not isinstance(signals,dict):
        raise ValueError("'signals' must be a dictionary!")
    
    cleaned_signals = {}
    
    for sub in signals.keys():
        cleaned_signals[sub] = clean(
            signals[sub].T,
            detrend=detrend, 
            standardize=standardize,
            filter=False if filter_type is None else filter_type, 
            low_pass=low_pass, 
            high_pass=high_pass, 
            t_r=TR
            ).T

    return cleaned_signals


# Matrix utils
def phase_coherence(signals_phases):
    """
    Compute the phase-coherence (or phase-locked) 
    connectivity matrices for a given subject.
    
    Because cos(0) = 1, if two areas n and p have
    temporarily aligned BOLD signals (i.e. they have
    similar phases), then dFC(n, p, t) will be close
    to 1. Instead, in periods where the BOLD signals
    are orthogonal (for instance, one increasing at 45°
    and the other decreasing at 45°) dFC(n, p, t) will
    be close to 0.
    
    Params:
    --------
    signals_phases : ndarray with shape (N_rois,time_points).
        Instantaneous phase of each brain
        region/parcel for each time point/volume.
    
    Return:
    -------
    dFC : ndarray with shape (N_rois,N_rois,time_points). 
        Time-resolved dynamic FC matrix where
        N_rois is the number of brain areas and
        time_points is the total number of recording
        frames. 
    """
    
    if not isinstance(signals_phases,np.ndarray) or (isinstance(signals_phases,np.ndarray) and signals_phases.ndim!=2):
        raise Exception("'signals_phases' must be a 2D array.")
    
    N = signals_phases.shape[0] #number of voxels/parcels
    T = signals_phases.shape[1]-2 #number of time points/volumes

    dFC = np.zeros((N,N,T)) #matrix to save the phase-coherence between regions n and p at time t.
    signals_phases = signals_phases[:,1:-1] #delete the fist and last time point of the time series of each ROI signal.
    
    for time_point in range(T): #for each time point:
        for roi_1 in range(N): #for each region of interest
            for roi_2 in range(N): #relate with other region of interest
                dFC[roi_1,roi_2,time_point] = np.cos(
                    ang_shortest_diff(
                        signals_phases[roi_1,time_point],
                        signals_phases[roi_2,time_point]
                        )
                    )
    return dFC

def get_eigenvectors(dFC,n=1):
    """
    For a given subject, extract the leading
    eigenvector of each phase-coherence connectivity
    matrix at time t.
    
    Params:
    -------
    dFC : ndarray with shape (N_rois,N_rois,N_volumes). 
        Contains the phase-coherence matrices
        for each time point t.

    n : int. 
        The number of desired eigenvalues and
        eigenvectors.
    
    Returns:
    --------
    LEi : ndarray with shape (N_time_points, N_ROIs)
        Extracted leading eigenvectors.
    """
    if not isinstance(dFC,np.ndarray) or (isinstance(dFC,np.ndarray) and dFC.ndim!=3):
        raise Exception("'dFC' must be a 3D array!")
    
    T, N = dFC.shape[-1], dFC.shape[0] #number of time points and number of regions
    
    LEi = np.empty((T,n*N))
    for t in range(T):
        avals, avects = eigs(dFC[:,:,t], n, which='LM')
        ponderation = avals.real / np.sum(avals.real)
        for x in range(avects.shape[1]):
            # convention, negative orientation
            if np.mean(avects[:, x] > 0) > .5:
                avects[:, x] *= -1
            elif np.mean(avects[:, x] > 0) == .5 and np.sum(avects[avects[:, x] > 0, x]) > -1. * sum(avects[avects[:, x] < 0, x]):
                avects[:, x] *= -1

        LEi[t] = np.hstack([p * avects.real[:, x].real for x, p in enumerate(ponderation)])

    return LEi

def txt_matrix(dFC,similarity_metric='pearson',plot=True):
    """
    To study the evolution of the dFC over time,
    we compute a time-versus-time matrix representing 
    the functional connectivity dynamics (FCD), where
    each entry, FCD(tx, ty), corresponds to a measure 
    of resemblance between the dFC at times tx and ty.
    
    Params:
    -------
    dFC : either ndarray of shape (N_rois,N_rois,time_points)
    containing the phase-coherence matrix for each time point;
    or ndarray with shape (N_time_points, N_ROIs) thats contains
    the eigenvectors of each dFC matrix.

    similarity_metric : str. 
        Whether to use 'pearson' or cosine similarity ('cosine')
        to determine the similarity between time points.
    
    Returns:
    --------
    time_x_time_mat : ndarray of shape (N_time_points, N_time_points).
        Time vs time matrix.
    """
    if not isinstance(similarity_metric,str):
        raise TypeError("'similarity_metric' must be a string.")
    
    if similarity_metric not in ['pearson','cosine']:
        raise ValueError("You must provide a valid 'similarity_metric' (pearson or cosine).")

    if not isinstance(dFC,np.ndarray):
        raise TypeError("'dFC' must be either a 2D array or a 3D array.")

    #Cheking whether the provided dFC data
    # is a connectivity matrix or eigenvectors
    # to define the number of time points T.
    N_time_points = dFC.shape[-1] if dFC.ndim>2 else dFC.shape[0]

    #create a empty 2D array with time_points x time_points    
    time_x_time_mat = np.zeros((N_time_points,N_time_points)) 
    
    #Computing the time x time matrix
    for t in range(N_time_points):
        for t2 in range(N_time_points):
            if dFC.ndim==3:
                if similarity_metric=='pearson': 
                    time_x_time_mat[t,t2] = pearsonr(
                        dFC[:,:,t][np.triu_indices_from(dFC[:,:,t],k=1)],
                        dFC[:,:,t2][np.triu_indices_from(dFC[:,:,t2],k=1)]
                        )[0] #compute similarity with Pearson correlation coefficient
                else: 
                    time_x_time_mat[t,t2] = 1 - cosine(
                        dFC[:,:,t][np.triu_indices_from(dFC[:,:,t],k=1)],
                        dFC[:,:,t2][np.triu_indices_from(dFC[:,:,t2],k=1)]
                        )
            else:
                if similarity_metric=='pearson':
                    #compute similarity with Pearson correlation coefficient 
                    time_x_time_mat[t,t2] = pearsonr(dFC[t,:],dFC[t2,:])[0] 
                else:
                    #compute similarity with cosine 
                    time_x_time_mat[t,t2] = 1 - cosine(dFC[t,:],dFC[t2,:]) 
                    
    if plot:
        plt.ion()
        plt.figure()
        sns.heatmap(
            time_x_time_mat,
            cmap='jet',
            vmin=-1,
            vmax=1,
            center=0,
            square=True,
            cbar_kws={
                "shrink": .5,
                "label":'Pearson\ncorrelation' if similarity_metric=='pearson' else 'Cosine\nsimilarity'
                }
            )
        plt.xticks(
                np.arange(20,N_time_points,20),
                np.arange(20,N_time_points,20).tolist(),
                rotation=0
            )
        plt.yticks(
                np.arange(20,N_time_points,20),
                np.arange(20,N_time_points,20).tolist(),
                rotation=0
            )
        plt.tick_params(
            axis='both',         
            which='both',     
            bottom=False,
            left=False
            )
        plt.xlabel('Time',fontweight='regular',fontsize=18)
        plt.ylabel('Time',fontweight='regular',fontsize=18)
        plt.title('Functional connectivity\ndynamics',fontweight='regular')
        plt.tight_layout()
        
    return time_x_time_mat
