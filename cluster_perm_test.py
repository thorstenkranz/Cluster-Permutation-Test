#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Thorsten Kranz <thorstenkranz@gmail.com>
# License: Simplified BSD
# Feedback welcome!!!

import copy
import numpy as np
from scipy.stats import f_oneway, percentileofscore

"""Cluster permutation algorithm as described in 
Maris/Oostenveld (2007), "Nonparametric statistical testing of EEG- and MEG-data"
Journal of Neuroscience Methods, Vol. 164, No. 1. (15 August 2007), pp. 177-190. 
doi:10.1016/j.jneumeth.2007.03.024
"""

class ClusterSearch1d:
    """Search for clusters in 1d-data"""

    def __init__(self,array_list,stat_fun=None,threshold=1.67,num_surrogates=1000):
        """Initialization

           :Parameters:
             array_list: list 
               List of 2d-arrays containing the data, dim 1: timepoints, dim 2: elements of groups
             stat_fun : function
               function called to calculate statistics, must accept 1d-arrays as arguments (default: scipy.stats.f_oneway)
        """
        #TODO: Do some checks
        self._al = array_list
        if stat_fun==None:
            self._sf = self.f_oneway
        else:
            self._sf = stat_fun
        self._threshold = threshold
        self._ns = num_surrogates
        #self._not_anova = not_anova

    #make read-only properties
    @property
    def array_list(self):
        return self._al
    
    @property
    def stat_fun(self):
        return self._sf
    
    @property
    def threshold(self):
        return self._threshold
    
    @property
    def num_surrogates(self):
        return self._ns

    #More interesting methods...
    def f_oneway(self,*args):
        """Call scipy.stats.f_oneway, but return only f-value"""
        return f_oneway(*args)[0]

    #THE method...
    def search(self):
        """For a list of 2d-arrays of data, e.g. power values, calculate some
        statistics for each timepoint (dim 1) over groups.  Do a cluster
        analysis with permutation test like in Maris, Oostenveld (2007)
        for calculating corrected p-values
        """
        #Create Shorthands
        al = self._al
        sf = self._sf

        #print len(al), [ar.shape for ar in al]
        ns_trs = [ar.shape[1] for ar in al] # Number of trials for each group
        crit_f = self._threshold
        l=al[0].shape[0]
        #Step 1: Calculate Anova (or other stat_fun) for original data
        #-------------------------------------------------------------
        fs = np.zeros((l),"d")
        for i in range(l):
            anova_ars = [ar[i,:] for ar in al]
            fs[i] = sf(*anova_ars)
        clusters = self.find_clusters(fs,crit_f,"greater") 
        #Step 2: If we have some clusters, repeat process for num_surrogates surrogates
        #------------------------------------------------------------------------------
        if len(clusters)>0:
            cluster_stats = [np.sum(fs[c[0]:c[1]]) for c in clusters]
            cluster_ps = np.ones((len(clusters)),"d")
            cluster_stats_hist = np.zeros((self._ns)) #For making histogram (not visually) and finding percentile
            for i_s in range(self._ns):
                ar_shuffle = np.concatenate(al,axis=1)
                #Mache Liste mit Indices fuer alle Trials, permutiere, zerlege in Einzellisten der Laengen ns_trs
                indices_lists = np.split(np.random.permutation(sum(ns_trs)),np.cumsum(ns_trs)[:-1])
                #print ar_shuffle.shape, ar_shuffle
                ar_shuffle_list = [ar_shuffle[:,indices] for indices in indices_lists]
                #print "ar_shuffle_list shapes", [ar.shape for ar in ar_shuffle_list]
                fs_surr = np.zeros((l))
                for i in range(l):
                    anova_ars_perm = [ar[i,:] for ar in ar_shuffle_list]
                    fs_surr[i] = sf(*anova_ars_perm)
                clusters_perm = self.find_clusters(fs_surr,crit_f,"greater")
                #print "clusters_perm", clusters_perm
                if len(clusters_perm)>0:
                    cluster_stats_perm = [np.sum(fs_surr[c[0]:c[1]]) for c in clusters_perm]
                    cluster_stats_hist[i_s] = max(cluster_stats_perm)
                else:
                    cluster_stats_hist[i_s] = 0
            #for each cluster in original data, calculate p-value as percentile of its 
            #cluster statistics within all cluster statistics in surrogate data
            cluster_ps[:] = [percentileofscore(cluster_stats_hist,cluster_stats[i_cl]) for i_cl in range(len(clusters))]
            cluster_ps[:] = (100.0 - cluster_ps[:]) / 100.0 # From percent to fraction
            return fs, np.array(clusters)[cluster_ps<0.05], cluster_ps[cluster_ps<0.05], np.array(clusters), cluster_ps
        else:
            return fs,np.array([]),np.array([]),np.array([]),np.array([])
    
    def find_clusters(self,ar,thres,cmp_type="greater"):
        """For a given 1d-array (test statistic), find all clusters which
        are above/below a certain threshold. Returns a list of 2-tuples.
        """
        #clusters =  []
        if not cmp_type in ["lower","greater","abs_greater"]:
            raise ValueError("cmp_type must be in [\"lower\",\"greater\",\"abs_greater\"]")
        ar = np.concatenate([np.array([thres]),ar,np.array([thres])])
        if cmp_type=="lower":
            ar_in = (ar<thres).astype(np.int)
        elif cmp_type=="greater":
            ar_in = (ar>thres).astype(np.int)
        else: #cmp_type=="abs_greater":
            ar_in = (abs(ar)>thres).astype(np.int)
        ar_switch = np.diff(ar_in)
        inpoints = np.arange(ar.shape[0])[ar_switch>0]
        outpoints = np.arange(ar.shape[0])[ar_switch<0]

        #print inpoints, outpoints
        in_out = np.concatenate([inpoints.reshape(-1,1),outpoints.reshape(-1,1)],axis=1)
        clusters = [(c[0],c[1]) for c in in_out]
        return clusters

if __name__ == "__main__":
    noiselevel = 40 
    condition1 = np.random.random((500,50))*noiselevel
    normfactor = np.hanning(20).sum()
    for i in range(50):
        condition1[:,i] = np.convolve(condition1[:,i],np.hanning(20),mode="same")/normfactor
    condition2 = np.random.random((500,43))*noiselevel
    for i in range(43):
        condition2[:,i] = np.convolve(condition2[:,i],np.hanning(20),mode="same")/normfactor
    pseudoekp = np.hanning(200).reshape(-1,1).repeat(50,axis=1)
    condition1[200:400,:]+=pseudoekp[:]
    condition2[200:400,:]-=pseudoekp[:,:43]
    cs = ClusterSearch1d([condition1,condition2],num_surrogates=100)
    fs, signif_cluster_times, signif_cluster_probs, cluster_times, cluster_probs = cs.search()

    #Plotting for a better understanding
    import pylab as p
    p.subplot(211)
    p.plot(condition1.mean(axis=1),label="Condition 1")
    p.plot(condition2.mean(axis=1),label="Condition 2")
    p.ylabel("signal [a.u.]")
    p.subplot(212)
    for i_c in range(len(cluster_times)):
        start, end = cluster_times[i_c]
        p.axvspan(start,end,color="b",alpha=0.3)
        p.text(start,2,"%.2f"%cluster_probs[i_c],fontsize=8)
    for i_c in range(len(signif_cluster_times)):
        start, end = signif_cluster_times[i_c]
        p.axvspan(start,end,0.95,1.0,color="b",alpha=1.0)
    p.plot(fs)
    p.xlabel("timepoints")
    p.ylabel("f-values")
    p.show()
                                                                                                                                                   
