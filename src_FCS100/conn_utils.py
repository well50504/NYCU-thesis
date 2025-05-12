# -*- coding: utf-8 -*-
"""
@author: Muhammad Salman Kabir
@purpose: Functions to do band pass filteration and computing connectivity matrix
          and connectivity vector
@regarding: Functional connectivity anaylsis    
"""
## Importing necassary libraries
import numpy as np
import scipy.signal as ss
import pyinform

def pli_connectivity(sensors,data):
    """
    Computing PLI connectivity
    
    Parameters
    ----------
    sensors : INT (means channels)
        DESCRIPTION. No of sensors used for capturing EEG
    data : Array of float 
        DESCRIPTION. EEG Data

    Returns
    -------
    connectivity_matrix : Matrix of float
        DESCRIPTION. PLI connectivity matrix
    connectivity_vector : Vector of flaot 
        DESCRIPTION. PLI connectivity vector

    """
    print("PLI in process.....")
    # Predefining connectivity matrix
    connectivity_matrix = np.zeros([sensors,sensors],dtype=float)
    
    # Computing hilbert transform
    data_points = data.shape[-1]
    data_hilbert = ss.hilbert(data)
    phase = np.angle(data_hilbert)
    # phase = np.arctan2(np.imag(data_hilbert), np.real(data))
    
    # Computing connectivity matrix
    for i in range(sensors):
        for k in range(sensors):
            connectivity_matrix[i,k] = np.abs(np.sum(np.sign(np.sin(phase[i,:]-phase[k,:]))))/data_points
    
    # Computing connectivity vector
    connectivity_vector = connectivity_matrix[np.triu_indices(connectivity_matrix.shape[0],k=1)] 
    
    # returning connectivity matrix and vector
    print("PLI done!")
    return connectivity_matrix, connectivity_vector



def coh_connectivity(sensors,data,f_min,f_max,fs):
    """
    Computing Coherence
    
    Parameters
    ----------
    sensors : INT
        DESCRIPTION. No of sensors used for capturing EEG
    data : Array of float 
        DESCRIPTION. EEG Data
    f_min : float
        DESCRIPTION. Low pass frequency of band pass filter given in hertz
    f_max : TYPE: float
        DESCRIPTION. High pass frequency of band pass filter given in hertz
    fs : TYPE: float
        DESCRIPTION. Sampling frequency of data given in hertz
    
    Returns
    -------
    connectivity_matrix : Matrix of float
        DESCRIPTION. COH connectivity matrix
    connectivity_vector : Vector of float 
        DESCRIPTION. COH connectivity vector

    """
    print("COH in process.....")
    
    # Predefinig connectivity matrix
    connectivity_matrix = np.zeros([sensors,sensors],dtype=float)
    
    # Computing coherence 
    for i in range(sensors):
        for k in range(sensors):
            f, Cxy = ss.coherence(data[i,:],data[k,:],fs = fs, nperseg=fs/2)
            connectivity_matrix[i,k] = np.mean(Cxy[np.where((f>=f_min) & (f<=f_max))])
    
    # Computing connectivity vector
    connectivity_vector = connectivity_matrix[np.triu_indices(connectivity_matrix.shape[0],k=1)] 
    
    # returning connectivity matrix and/or vector
    print("COH done!")
    return connectivity_matrix, connectivity_vector



def ordinal_patterns(ts, m, d):
    """

    Parameters
    ----------
    ts : 1D array
        DESCRIPTION. time series data.
    m : INT
        DESCRIPTION. embedding dimension (window length).
    d : INT
        DESCRIPTION. time delay (step size).
         
    Returns
    -------
    patterns : 1D array of int
        DESCRIPTION. ordinal patterns.
    """
    n = len(ts) - (m - 1) * d
    patterns = np.zeros(n, dtype=int)
    for i in range(n):
        window = ts[i : i + m*d : d]
        order = np.argsort(window)
        value = 0
        for elem in order:
            value = value * m + elem
        patterns[i] = value
    return patterns



def ste_connectivity(sensors, data, m=3, d_values=None):
    """
    Computing STE connectivity
    
    Parameters
    ----------
    sensors : INT (means channels)
        DESCRIPTION. No of sensors used for capturing EEG
    data : Array of float 
        DESCRIPTION. EEG Data
    m : INT
        DESCRIPTION. Embedding dimension
    d_values : List of INT
        DESCRIPTION. List of delay values

    Returns
    -------
    connectivity_matrix : Matrix of float
        DESCRIPTION. STE connectivity matrix
    connectivity_vector : Vector of flaot 
        DESCRIPTION. STE connectivity vector

    """
    print("STE in process.....")
    
    if d_values is None:
        d_values = list(range(1, 31, 2))
    
    connectivity_matrix = np.zeros((sensors, sensors), dtype=float)
    

    channel_patterns = {}
    for i in range(sensors):
        ts = data[i, :]
        for d in d_values:
            if len(ts) < (m - 1) * d + 1:
                channel_patterns[(i, d)] = None
            else:
                channel_patterns[(i, d)] = ordinal_patterns(ts, m, d)
    

    for i in range(sensors):
        for j in range(sensors):
            if i == j:
                continue
            TE_d = []  
            for d in d_values:
                patterns_i = channel_patterns.get((i, d), None)
                patterns_j = channel_patterns.get((j, d), None)
                if patterns_i is None or patterns_j is None:
                    continue
                try:
                    TE = pyinform.transfer_entropy(patterns_j.tolist(), 
                                                   patterns_i.tolist(), 
                                                   k=m, local=False)
                except Exception as e:
                    TE = 0.0
                    print(f"Error calculating TE for channels {j}->{i} at d={d}: {e}")
                TE_d.append(TE)
            max_TE = max(TE_d) if TE_d else 0.0
            connectivity_matrix[i, j] = max_TE
            
    connectivity_vector = connectivity_matrix[np.triu_indices(sensors, k=1)]
    
    print("STE done!")
    return connectivity_matrix, connectivity_vector

