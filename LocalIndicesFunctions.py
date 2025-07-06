#!/usr/bin/env python
# coding: utf-8

# In[1]:

import numpy as np
from statsmodels.tsa.stattools import acf

import os
import sys
import time
import xarray as xr
from tqdm import tqdm

import scipy.stats as sc
import statsmodels.api as sm
import multiprocessing as mp
import sklearn.metrics.pairwise as skmp
import pickle
import logging
logging.captureWarnings(True)

#====================================================================================
#====================================================================================
                      #COMPUTE EXCEEDS
#====================================================================================
#====================================================================================

def compute_exceeds(X, ql, n_jobs, theiler_len, p_value_exp=None, exp_test_method='anderson', **kwargs):
    
    """ Computes the exceedances of the neighbouring states """
    
    #1. Preprocess the data.
    
    #Read kwargs
    metric = kwargs.get("metric")
    if metric is None: metric = "euclidean"
    #Check and read data 
    if X.ndim != 2:	print("data is not in correct shape.")
    n_samples, n_features = X.shape  
    #Initialize distances 
    dist = np.zeros((n_samples, n_samples))

    
    #2. Distance matrix construction.
    dist[:,:] = skmp.pairwise_distances(X, X, metric=metric, n_jobs=n_jobs) #dist[i,j] matrix
    
    
    #3. Theiler window exclusion
    
    for i in range(1, 1+theiler_len):       
        np.fill_diagonal(dist[:-i, i:], sys.float_info.max)  
        np.fill_diagonal(dist[i:, :-i], sys.float_info.max)
    np.fill_diagonal(dist, sys.float_info.max)
    
    #4. Calculate Logarithmic Returns
    
    dist_log = -np.log(dist)
    
    
    #5. Compute The Threshold Quantile 
    
    q = np.quantile(dist_log, ql, axis=1, interpolation='midpoint')
    R = np.exp(-q.reshape(-1,1))
    
    
    #6. Find The Exceedences
    
    exceeds_bool = dist_log > q.reshape(-1,1) #Finds the states satisfying the condition g > s 
    n_neigh = np.sum(exceeds_bool, axis=1).min() #count all the number of neighbours per state
    #Corrects the matrix `exceeds_bool` in case some neighbours fall on the edge of the circle defined by the quantile.
    exceeds_bool = _correct_n_neigh(exceeds_bool, dist_log, q, n_neigh) 
    exceeds_idx  = np.argwhere(exceeds_bool).reshape(n_samples,n_neigh,2)[:,:,1]
    row_idx = np.arange(n_samples)[:,None]
    exceeds = dist_log[row_idx,exceeds_idx] - q[:, None] #This is our final 'U' function...
    
    return dist, exceeds, exceeds_idx, exceeds_bool, R  

def _correct_n_neigh(exceeds_bool, dist, q, n_neigh):
    """ Corrects the matrix `exceeds_bool` in-case some neighbours fall on the edge of the circle defined by q."""
    
    exceeds_bool_geq = dist >= q.reshape(-1,1)
    idx_edge = np.where(np.sum(exceeds_bool, axis=1)!=n_neigh)[0]
    for i in idx_edge:
        idx_add = np.where(exceeds_bool[i,:]!=exceeds_bool_geq[i,:])[0][0]
        exceeds_bool[i,idx_add] = True
    
    return exceeds_bool

def _exp_test(exceeds, p_value, exp_test):
    """ Test if `exceeds` follow an exponential distribution.

    Parameters:
    - exceeds (1d array): exceedances array
    - p_value (float)   : significance level to test
    - exp_test (string) : 'anderson' (default) or 'chi2'

    Returns:
    - res_stat (float): The residual between the calculated statistic
    and the reference value used to check if H0 needs to be rejected or not.
    Positive value: do not reject H0 (i.e., we cannot reject the null hypothesis
    for which the data are exponentially distributed).
    """
    
    if exp_test=='anderson':
        if p_value==0.15:
            ind_p_value_anderson = 0
        elif p_value==0.1:
            ind_p_value_anderson = 1
        elif p_value==0.05:
            ind_p_value_anderson = 2
        elif p_value==0.025:
            ind_p_value_anderson = 3
        elif p_value==0.01:
            ind_p_value_anderson = 4
        else:
            raise ValueError(
                'p_value must be one of the following values: ',
                '0.15'' 0.10, 0.05, 0.025, 0.01')

        ## perform anderson test
        anderson_stat, anderson_crit_val, anderson_sig_lev =             sc.anderson(exceeds, dist='expon')
        ref = anderson_crit_val[ind_p_value_anderson]
        # reject H0 if anderson_stat > ref
        # i.e., if res_stat < 0
        res_stat = 100*(ref - anderson_stat) / ref

    elif exp_test=='chi2':
        pplot = sm.ProbPlot(exceeds, sc.expon)
        xq = pplot.theoretical_quantiles
        yq = pplot.sample_quantiles
        p_fit = np.polyfit(xq, yq, 1)
        yfit = p_fit[0] * xq + p_fit[1]

        ## perform Chi-Square Goodness of Fit Test
        _, p_chi2 = sc.chisquare(f_obs=yq, f_exp=yfit)
        # reject H0 if p_value > p_chi2
        # i.e., if res_stat < 0
        res_stat = 100*(p_chi2 - p_value) / p_value
    return res_stat



#====================================================================================
#====================================================================================
                      #COMPUTE LOCAL DIMENSION
#====================================================================================
#====================================================================================

def LocalDimension(exceeds):
    """Computes the local dimension d of the trajectory given exceeds calculated from the previous func"""
    
    d = 1 / np.mean(exceeds, axis=1)

    return d


#====================================================================================
#====================================================================================
                      #COMPUTE EXTREMAL INDEX
#====================================================================================
#====================================================================================

def ExtremalIndex(idx, ql):
    """Computes the extremal index using the Süveges maximum likelihood estimator"""
    
    q = 1 - ql
    Ti = idx[:,1:] - idx[:,:-1]
    Si = Ti - 1
    Nc = np.sum(Si > 0, axis=1)
    K  = np.sum(q * Si, axis=1)
    N  = Ti.shape[1]
    theta = (K + N + Nc - np.sqrt((K + N + Nc)**2 - 8 * Nc * K)) / (2 * K)
    return theta

def ExtremalIndexSingle(idx, ql):
    """Computes the extremal index for a single state using the Süveges maximum likelihood estimator."""
    q = 1 - ql
    Ti = idx[1:] - idx[:-1]
    Si = Ti - 1
    Nc = np.sum(Si > 0)
    K = np.sum(q * Si)
    N = len(Ti)
    theta = (K + N + Nc - np.sqrt((K + N + Nc)**2 - 8 * Nc * K)) / (2 * K)
    return theta

#====================================================================================
#====================================================================================
                      #COMPUTE PREDICTABILITY INDEX
#====================================================================================
#====================================================================================

def Predictability(dist, exceeds_bool, time_lag, ql, theiler_len, l=1):
    """Computes the lagged co-recurrence of the binary matrix 'neigh'. Requiring trajectories to stick to the reference trajectory""" 

    exceeds_bool_now = exceeds_bool.copy()
    
    n_samples = exceeds_bool.shape[0]   #number of states
    n_neigh = np.sum(exceeds_bool[0,:]) #number of neighbours
    alphat_dict = {} #set up dictionary

    for lag in time_lag:
        # find forward recurrences shifting by lag exceeds_bool_now
        exceeds_bool_lag = np.zeros([n_samples-lag, n_samples], dtype=bool)
        exceeds_bool_lag[:, lag:] = exceeds_bool_now[:-lag, :-lag]

        # find forward-reference-state recurrences
        exceeds_bool_lag_ref = exceeds_bool_now[lag:, :]
        
        # compute the intersection of forward recurrences and
        # forward-reference-state recurrences and use a flag of -1 where there
        # is no intersection
        exceeds_bool_intersect = np.logical_and(exceeds_bool_lag_ref, exceeds_bool_lag)
        
        if l == 0:
            alphat_dict[lag] = np.sum(exceeds_bool_intersect, axis=1) / n_neigh
        else:
            dist_sum_in = np.nansum(np.where(exceeds_bool_intersect, dist[lag:, :], np.nan) ** l, axis=1) # Neighbors sticking
            dist_sum_all = np.nansum(np.where(exceeds_bool_lag, dist[lag:, :], np.nan) ** l, axis=1)      # All forward neighbors
            alphat_dict[lag] = dist_sum_in / dist_sum_all

    return alphat_dict


#====================================================================================
#====================================================================================
                      #COMPUTE THEILER LEN
#====================================================================================
#====================================================================================

def Theiler_Len(X, Nlags):
    """Uses autocorrelation to find largest system theiler length needed"""
    
    n_samples, n_features = X.shape
    
    if n_features == 3: #The system is Lorenz
        
        acorr1     = acf(X[:,0], nlags=Nlags)
        ind_acorr1 = np.where(acorr1<=0)[0][0]

        acorr2     = acf(X[:,1], nlags=Nlags)
        ind_acorr2 = np.where(acorr2<=0)[0][0]

        acorr3     = acf(X[:,2], nlags=Nlags)
        ind_acorr3 = np.where(acorr3<=0)[0][0]

        #Find the largest Theiler Len
        if ind_acorr1 > ind_acorr2 and ind_acorr1 > ind_acorr3:
            Theiler_Len = ind_acorr1
        elif ind_acorr2 > ind_acorr1 and ind_acorr2 > ind_acorr1:
            Theiler_Len = ind_acorr2
        elif ind_acorr3 > ind_acorr1 and ind_acorr3 > ind_acorr2:
            Theiler_Len = ind_acorr3
        else:
            Theiler_Len = ind_acorr3 #tie so doesnt matter 
    else:
        acorr1      = acf(X[:,0], nlags=Nlags)
        ind_acorr1  = np.where(acorr1<=0)[0][0]
        Theiler_Len = ind_acorr1
    
    return Theiler_Len