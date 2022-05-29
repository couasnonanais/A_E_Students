
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 11:57:30 2019

@author: jose.antolinez@deltares.nl
"""

import os
import numpy as np
import pandas as pd
# import modin.pandas as pd
idx = pd.IndexSlice
import time
import sys

from scipy.stats import genextreme as gev
from scipy.stats import gumbel_r as gumbel 
from scipy.stats import genpareto as gepd
from scipy.stats import expon
from sklearn.utils import resample
import statsmodels.distributions.empirical_distribution as edf
from scipy.stats import poisson
import scipy.stats as scst
import psutil
import ray

num_cpus = psutil.cpu_count(logical=False)
virt_mem = psutil.virtual_memory().available
# ray.init(num_cpus=num_cpus-2,memory=(223.29-50)*(1024**3), object_store_memory=(18.63+50)*(1024)**3)
ray.init(num_cpus=num_cpus,memory=int(virt_mem*2/3), object_store_memory=int(virt_mem*1/3))
# ray.init(num_cpus=num_cpus,memory=int(virt_mem*3/4), object_store_memory=int(virt_mem*1/4))
print('cpus = %d'%num_cpus)

# usage
# python p2_gtsmpost_extreme '01_HIST' 'wl' 'seas' 6
# describes the extreme 01_HIST water level monthly
# python p2_gtsmpost_extreme '01_HIST' 'ss' 'full' 6
# describes the extreme 01_HIST residual full period number of jobs to parallelize 6 

#%% paths
class Paths: 
    def __init__(self,pth):
        self.base = pth         
        self.pregtsm = os.path.join(self.base,'data')
        self.out = os.path.join(self.base,'outEVA')
        self.plots = os.path.join(self.out,'plots')

    def _makedirs(self):
        for path in vars(self).values():
            if not os.path.exists(path):
                os.mkdir(path)

class Object(object):
    pass

@ray.remote
def extreme_pd(paths,wlc,seasc,ii,ntl,namedi,fitd,evam,rv,qopt,q,indep,mindur,res,resunit,nr,per,pgev,ind2):    
    if ii%100 ==0: print(ii,'-->', flush=True)
    fname = '%s_%s%05d.pxz'%('gtsm','station',ii)
    gtsmpd= pd.read_pickle(os.path.join(paths.pregtsm,fname),compression='xz')
    gtsmpd = gtsmpd.loc[ind2,:]/1000.  

    if wlc=='ss':
        tidepd = pd.read_pickle(os.path.join(paths.tide,fname),compression='xz')
        gtsmpd -= tidepd.loc[gtsmpd.index]/1000.
        gtsmpd[gtsmpd>99999]=np.nan
        gtsmpd = gtsmpd.dropna()
    else:
        if resunit.lower() in ['minutes', 'minute', 'min', 'm']:
            fact = int(60/res)
        elif resunit.lower() in ['hours', 'hour', 'hr', 'h']:
            fact = 1

        gtsmpd-=gtsmpd.rolling(365*24*fact,center=True).mean().fillna(method='backfill').fillna(method='pad') #detrend 10min data with centered mean
        # gtsmpd-=gtsmpd.rolling(365*24*fact).mean().fillna(method='backfill') #detrend 10min data with backward mean
    ny = gtsmpd.index.year.unique().shape[0]
    if seasc in ['seasonal','annual','decadal','periods']:
        for jj in ntl:
            ny = 0
            if seasc == 'seasonal':
                ind2 = gtsmpd.index[gtsmpd.index.month==jj]
            elif seasc == 'annual':
                ind2 = gtsmpd.index[gtsmpd.index.year==jj]
            elif seasc == 'decadal':
                ind2 = gtsmpd.index[(np.floor(gtsmpd.index.year/10)*10).astype(int)==jj] 
            elif seasc == 'periods':
                ind2 = gtsmpd.index[(gtsmpd.index.year>=int(jj.split('-')[0]))&(gtsmpd.index.year<=int(jj.split('-')[1]))]
            ny = np.max((ny,ind2.year.unique().shape[0]))

    if seasc in ['seasonal','annual','decadal','periods']:
        extremesdf = pd.DataFrame(index = [ii],columns=pd.MultiIndex.from_product([np.hstack((pgev,rv.astype(str))),['mean']+['%s' % ip for ip in per],ntl],names=['metrics','ci']+namedi))
        # extremesdf = pd.DataFrame(index = [ii],columns=pd.MultiIndex.from_product([np.hstack((pgev,rv.astype(str))),['mean']+['%s' % ip for ip in per]+['meanB'],ntl],names=['metrics','ci']+namedi))
    else:        
        extremesdf = pd.DataFrame(index = [ii],columns=pd.MultiIndex.from_product([np.hstack((pgev,rv.astype(str))),['mean']+['%s' % ip for ip in per]],names=['metrics','ci']))
        # extremesdf = pd.DataFrame(index = [ii],columns=pd.MultiIndex.from_product([np.hstack((pgev,rv.astype(str))),['mean']+['%s' % ip for ip in per]+['meanB']],names=['metrics','ci']))

    if seasc in ['seasonal','annual','decadal','periods']:
        for jj in ntl:
            if seasc == 'seasonal':
                ind2 = gtsmpd.index[gtsmpd.index.month==jj]
            elif seasc == 'annual':
                ind2 = gtsmpd.index[gtsmpd.index.year==jj]
            elif seasc == 'decadal':
                ind2 = gtsmpd.index[(np.floor(gtsmpd.index.year/10)*10).astype(int)==jj] 
            elif seasc == 'periods':
                ind2 = gtsmpd.index[(gtsmpd.index.year>=int(jj.split('-')[0]))&(gtsmpd.index.year<=int(jj.split('-')[1]))]
            if ind2.empty or len(ind2)<6*24*30:
                continue
            nyjj = ind2.year.unique().shape[0]

            # else:
            #     ind3 = ind2[0::24*7]
            #     ind3 = ind2

            if evam == 'BlockMaxima':
                probY,gevmodel = block_maxima(gtsmpd.loc[ind2],rv,fitd,nr,per)
                extremesEdf = pd.DataFrame(index = [ii],columns=pd.MultiIndex.from_product([['rv','date','emp','mod_mean']+['mod_%s' % ip for ip in per],ntl,np.arange(ny)],names=['metrics',namedi[0],'samples']))

            elif evam == 'POT':
                probY,gevmodel = peak_over_threshold(gtsmpd.loc[ind2],rv,fitd,qopt,q,indep,mindur,res,resunit,nr,per)
                ny = probY.shape[0] #number of events
                extremesEdf = pd.DataFrame(index = [ii],columns=pd.MultiIndex.from_product([['rv','date','duration','emp','mod_mean']+['mod_%s' % ip for ip in per],ntl,np.arange(ny)],names=['metrics',namedi[0],'samples']))

            for cii in gevmodel.columns:
                extremesdf.loc[ii,idx[gevmodel.index,cii,jj]] = gevmodel.loc[extremesdf.columns.unique(level=0),cii].values.squeeze()
                # extremesdf.loc[ii,idx[gevmodel.index,cii,jj]] = gevmodel[cii].values.squeeze()
                # this line is causing that the column names are wrong,
                # gevmodel has index
                # lambda,shape,location,scale,0.5,1.0,2.0,5.0,10.0,25.0,50.0,75.0,100.0,thresh,qthres
                # but extremesdf stores in this order
                # lambda, thresh, qthres, shape, location, scale, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 75.0, 100.0   
                # the right line is this
                # extremesdf.loc[ii,idx[:,cii,jj]] = gevmodel.loc[extremesdf.columns.unique(level=0),cii].values.squeeze()

            for ik in probY.columns:
                extremesEdf.loc[ii,idx[ik,jj,np.arange(0,nyjj)]] = probY[ik].values


    else:
        if evam == 'BlockMaxima':
            probY,gevmodel = block_maxima(gtsmpd,rv,fitd,nr,per)
            extremesEdf = pd.DataFrame(index = [ii],columns=pd.MultiIndex.from_product([['rv','date','emp','mod_mean']+['mod_%s' % ip for ip in per],np.arange(ny)],names=['metrics','samples']))

        elif evam == 'POT':
            probY,gevmodel = peak_over_threshold(gtsmpd,rv,fitd,qopt,q,indep,mindur,res,resunit,nr,per)
            ny = probY.shape[0] #number of events
            extremesEdf = pd.DataFrame(index = [ii],columns=pd.MultiIndex.from_product([['rv','date','duration','emp','mod_mean']+['mod_%s' % ip for ip in per],np.arange(ny)],names=['metrics','samples']))

        for cii in gevmodel.columns:
            extremesdf.loc[ii,idx[gevmodel.index,cii]] = gevmodel.loc[extremesdf.columns.unique(level=0),cii].values.squeeze()
            # extremesdf.loc[ii,idx[gevmodel.index,cii]] = gevmodel[cii].values.squeeze()
            # this line is causing that the column names are in the wrong order,
            # gevmodel has index
            # lambda,shape,location,scale,0.5,1.0,2.0,5.0,10.0,25.0,50.0,75.0,100.0,thresh,qthres
            # but extremesdf stores in this order
            # lambda, thresh, qthres, shape, location, scale, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 75.0, 100.0   
            # the right line is this
            # extremesdf.loc[ii,idx[:,cii]] = gevmodel.loc[extremesdf.columns.unique(level=0),cii].values.squeeze()


        for ik in probY.columns:
            extremesEdf.loc[ii,idx[ik,:]] = probY[ik].values

    return extremesdf,extremesEdf

#%% EV functions
def block_maxima_bootstrap(params,fitd,ns,nr,per,peaks):
    ### block_maxima_bootstrap ###
    # Bootstrapping of extremes value distribution parameters
    # params: distribution mean parameters
    # fitd: distribution used for the extreme value analysis
    # ns: number of original samples when fitting the extreme value model
    # nr: set of bootstrap samples of size ns
    # per: percentiles to return parameter estimations
    shape = np.zeros((nr,))
    location = np.zeros_like(shape)
    scale = np.zeros_like(shape)
    if fitd.lower() == 'gev':
        for rpi in range(nr):           
            # vals = gev.ppf(np.random.rand(ns),params['shape'],loc=params['location'],scale=params['scale'])
            boot = resample(peaks, replace=True, n_samples=len(peaks))
            shape[rpi],location[rpi],scale[rpi] = gev.fit(boot,0)

    elif fitd.lower() == 'gumbel':
        for rpi in range(nr):           
            # vals = gumbel.ppf(np.random.rand(ns),loc=params['location'],scale=params['scale'])
            boot = resample(peaks, replace=True, n_samples=len(peaks))
            location[rpi],scale[rpi] = gumbel.fit(boot)

    # Adjust the mean of the bootstrap estimates using the point estimates
    shape = shape-shape.mean()+params['shape']
    location = location-location.mean()+params['location']
    scale = scale-scale.mean()+params['scale']
    # return {'shape':np.hstack((np.percentile(shape,per),np.mean(shape))),'location':np.hstack((np.percentile(location,per),np.mean(location))),'scale':np.hstack((np.percentile(scale,per),np.mean(scale)))}
    return {'shape':shape,'location':location,'scale':scale}
            
def block_maxima(ds,rv,fitd,nr,per):
    ### block_maxima ###
    # Analysis of extremes following block maxima
    # ds: pandas dataframe
    # rv: return values to compute the modelled return periods
    # fitd: distribution function to use for the extreme value analysis
    # nr: set of bootstrap samples of size ns
    # per: percentiles to return parameter estimations
    
    # Empirical model
    #1- get annual maxima values
    # dftotY = ds.dropna().resample('A').max()
    dftotY = ds.dropna().resample('A').max().dropna()
    stdche = dftotY.values.std()
    if np.isnan(stdche) or stdche==0:
        return pd.DataFrame(),pd.DataFrame() # stops the computations if the data is wrong at this node

    #2- sort data smallest to largest
    dftotYs = dftotY.sort_values(by='waterlevel');dftotYs.index.name='date'
    dftotYs = dftotYs.reset_index()
#    dftotYs = dftotY.reset_index().drop(columns=['index'])
#    dftotYs[:] = np.sort(dftotYs.values,axis=0)
    #3- count total obervations
    n = dftotYs.shape[0]
    # calculate probability - note you may need to adjust this value based upon the time period of your data
    probY=pd.DataFrame(index=dftotYs.index,columns=['probex','rv','emp','mod_mean']+['mod_%s' % ip for ip in per])
    probY['probex'] = ((n - (dftotYs.index.values+1) + 1) / (n + 1))
    probY['rv'] = (1 / probY['probex'])     
    probY['date'] = dftotYs.iloc[:,0]
    probY['emp'] = dftotYs.iloc[:,1]
    
    # fit GEV model
#    rv = np.array([2,5,10,25,50,75,100])
#    dpars = ['shape','location','scale']
    dpars = ['shape','location','scale']
    dparsrv = np.hstack((dpars,rv.astype(str)))
    pgev = np.array(dparsrv)

            
    # gevmodel = pd.DataFrame(columns=['mean']+['%s' % ip for ip in per]+['meanB'],index=pgev)
    gevmodel = pd.DataFrame(columns=['mean']+['%s' % ip for ip in per],index=pgev)
    gevmodelc = pd.DataFrame(columns=['mean']+['%s' % ip for ip in per],index=np.arange(rv.min(),rv.max()+1))
    #    gevmodelq = pd.DataFrame(columns=dftotY.columns,index=probY['rv'])
    gvmd = np.nan*np.ones((3,))
    if fitd.lower() == 'gev':
        gvmd[:] = gev.fit(dftotYs.iloc[:,1].values,0)
        gvmd2 = gev.ppf(1.-1./gevmodelc.index.values,gvmd[0],loc=gvmd[1],scale=gvmd[2])
        probY['mod_mean'] = gev.ppf(1.-probY['probex'].values,gvmd[0],loc=gvmd[1],scale=gvmd[2])
        
    elif fitd.lower() == 'gumbel':
        gvmd[1],gvmd[2] = gumbel.fit(dftotYs.iloc[:,1].values);gvmd[0]=0 # force shape=0
        gvmd2 = gumbel.ppf(1.-1./gevmodelc.index.values,loc=gvmd[1],scale=gvmd[2])
        probY['mod_mean'] = gumbel.ppf(1.-probY['probex'].values,loc=gvmd[1],scale=gvmd[2])

    gevmodel.loc[dpars,idx['mean']] = gvmd.squeeze()
    gevmodelc.loc[:,idx['mean']] = gvmd2
    gevmodel.loc[rv.astype(str),idx['mean']]=gevmodelc.loc[rv,idx['mean']].values

    # # bootstrapping
    # bootparamsci = block_maxima_bootstrap(gevmodel.loc[:,idx['mean']],fitd,probY.shape[0],nr,per,probY['emp'].values)
    # for pari in dpars:
    #     # gevmodel.loc[pari,idx['meanB']] = bootparamsci[pari][len(per)]
    #     for ii,peri in enumerate(per):
    #         gevmodel.loc[pari,idx['%s'%peri]] = bootparamsci[pari][ii]

    # for peri in per:
    #     if fitd.lower() == 'gev':
    #         gevmodelc.loc[:,idx['%s'%peri]] = gev.ppf(1.-1./gevmodelc.index.values,gevmodel.loc['shape',idx['%s'%peri]],loc=gevmodel.loc['location',idx['%s'%peri]],scale=gevmodel.loc['scale',idx['%s'%peri]])
    #         gevmodel.loc[rv.astype(str),idx['%s'%peri]]=gevmodelc.loc[rv,idx['%s'%peri]].values
    #         probY['mod_%s'%peri] = gev.ppf(1.-probY['probex'].values,gevmodel.loc['shape',idx['%s'%peri]],loc=gevmodel.loc['location',idx['%s'%peri]],scale=gevmodel.loc['scale',idx['%s'%peri]])
            
    #     elif fitd.lower() == 'gumbel':
    #         gevmodelc.loc[:,idx['%s'%peri]] = gumbel.ppf(1.-1./gevmodelc.index.values,loc=gevmodel.loc['location',idx['%s'%peri]],scale=gevmodel.loc['scale',idx['%s'%peri]])
    #         gevmodel.loc[rv.astype(str),idx['%s'%peri]]=gevmodelc.loc[rv,idx['%s'%peri]].values
    #         probY['mod_%s'%peri] = gumbel.ppf(1.-probY['probex'].values,loc=gevmodel.loc['location',idx['%s'%peri]],scale=gevmodel.loc['scale',idx['%s'%peri]])
    
    # bootstrapping
    bootparamsci = block_maxima_bootstrap(gevmodel.loc[:,idx['mean']],fitd,probY.shape[0],nr,per,dftotYs.iloc[:,1].values)
    for ii,peri in enumerate(per):
        for pari in dpars:
            gevmodel.loc[pari,idx['%s'%peri]] = np.percentile(bootparamsci[pari],peri)

    rvals=np.nan*np.ones((gevmodelc.index.values.size,nr))
    rvalsval=np.nan*np.ones((probY['probex'].values.size,bootparamsci['shape'].size))
    if fitd.lower() == 'gev':
        for ii in range(nr):
            rvals[:,ii] = gev.ppf(1.-1./gevmodelc.index.values,bootparamsci['shape'][ii],loc=bootparamsci['location'][ii],scale=bootparamsci['scale'][ii])
            rvalsval[:,ii] = gev.ppf(1.-probY['probex'].values,bootparamsci['shape'][ii],loc=bootparamsci['location'][ii],scale=bootparamsci['scale'][ii])
    elif fitd.lower() == 'gumbel':
        for ii in range(nr):
            rvals[:,ii] = gumbel.ppf(1.-1./gevmodelc.index.values,loc=bootparamsci['location'][ii],scale=bootparamsci['scale'][ii])
            rvalsval[:,ii] = gumbel.ppf(1.-probY['probex'].values,loc=bootparamsci['location'][ii],scale=bootparamsci['scale'][ii])

    for peri in per:
        gevmodelc.loc[:,idx['%s'%peri]] = np.percentile(rvals,peri,axis=1)
        gevmodel.loc[rv.astype(str),idx['%s'%peri]]=gevmodelc.loc[rv,idx['%s'%peri]].values
        probY['mod_%s'%peri] = np.percentile(rvalsval,peri,axis=1)

            
#plt.plot(gevmodelc.loc[:,idx['mean']],'-k');plt.plot(gevmodelc.loc[:,idx['75']],'r--');plt.plot(gevmodelc.loc[:,idx['25']],'r--')
#plt.plot(probY['rv'],probY['emp'],'ob');plt.plot(probY['rv'],probY['mod_mean'],'-k');plt.plot(probY['rv'],probY['mod_25'],'r--');plt.plot(probY['rv'],probY['mod_75'],'r--')


#    plt.plot(gevmodel.loc[rv.astype(str)].index.astype(int),gevmodel.loc[rv.astype(str)].values,'ob')
#    plt.plot(probY['rv'].values,probY['val'].values,'or')
    return probY.drop(columns='probex'),gevmodel
#%% POT functions
def pot(var,indep,mindur,res,resunit,qopt,q):
#    dt = pd.infer_freq(var.index) # way to get the frequency of the data
#    indep = 4*24# independency test in hours if data is hourly
#    mindur = 6h*60min minimun duration in units of var above the threshold for the event to be considered a storm
#    res = 10 min resolution of the data in minutes
#    resunit = 'min' resolution of the data units
#    q = 0.98 # quantile for estimating the threshold
    if qopt[0:4] == 'quan':
        varth = var.quantile(q) # threshold
    elif qopt[0:4] == 'thre':
        varth = q # the threshold is passed from outside
#    imax = argrelextrema(var.values, np.greater, order=indep) # this function works with number of elements so it has to be apply before the threshold
#    vare = var.iloc[imax];vare = vare[vare>varth] # looks for the peaks above the threshold
    
    # obtain storms (going up and down the threshold)    
#    res=1 #resolution of the data
    vare = var[var>=varth]
    istorm = np.insert(np.where((vare.index[1:]-vare.index[:-1])>pd.Timedelta(res,resunit))[0],0,-1)

    indx=[];vmax=[];dur=[];
    for i,j in zip(istorm[:-1]+1,istorm[1:]):
        indx.append(vare[i:j+1].idxmax())
        vmax.append(vare[i:j+1].max())
        dur.append(vare[i:j+1].count()*res)
    indx = pd.Index(indx)
    if len(indx)==0:
        print('no INDX', flush=True)
        return np.nan,np.nan,np.nan # stops the computations if the data is wrong at this node
    # check for independence
    indx2=[indx[0]];vmax2=[vmax[0]];dur2=[dur[0]]
    for i,v,d in zip(indx[1:],vmax[1:],dur[1:]):
        if (i-indx2[-1])>pd.Timedelta(indep,'h'):
            indx2.append(i);vmax2.append(v);dur2.append(d)
        else:
            imx = np.argmax([vmax2[-1],v])
            indx2[-1] = [indx2[-1],i][imx]
            vmax2[-1] = [vmax2[-1],v][imx]
            dur2[-1] = dur2[-1]+d
    vare = pd.DataFrame(index=indx2,data=np.vstack((vmax2,dur2)).T,columns=['max','duration'])
    vare = vare[vare['duration']>=mindur]
    events = vare['max'].groupby(vare.index.year).agg('count')
    #fig = plt.figure()
    #fig.set_size_inches(16,12)
    #plt.plot(var);plt.plot(vare,'o')
    return vare,events,varth

def peak_over_threshold(ds,rv,fitd,qopt,q,indep,mindur,res,resunit,nr,per):
    ### block_maxima ###
    # Analysis of extremes following block maxima
    # ds: pandas dataframe
    # rv: return values to compute the modelled return periods
    # fitd: distribution function to use for the extreme value analysis
    # qopt: type of threshold indicated, value or quantile based
    # indep: independence between events [see definition of pot for the units]
    # nr: set of bootstrap samples of size ns
    # per: percentiles to return parameter estimations
    #
    
    # Empirical model
    #1- find storms
    potmax,evts,thresh = pot(ds['waterlevel'],indep,mindur,res,resunit,qopt,q) # 10min data
    if np.isnan(thresh):
        return pd.DataFrame(),pd.DataFrame() # stops the computations if the data is wrong at this node

    if qopt=='thre':
        ecdf = edf.ECDF(ds.values)
        qth = ecdf(thresh)
    elif qopt=='quan':
        qth = q
        
#    maxm = potmax['max'].groupby(potmax.index.year).agg('mean')# mean annual maximum
#    dura = potmax['duration'].groupby(potmax.index.year).agg('mean')# mean annual duration

#    pmord=pd.DataFrame(columns=['vals','probex','rv','date'])
    #2- sort data smallest to largest    
    pmord = potmax.sort_values(by='max');pmord.index.name = 'date'
    pmord = pmord.reset_index()
    pmord = pmord.rename(columns={'max':'emp'})
#    pmord['vals'] = np.sort(potmax['max'].values,axis=0)
    #3- count total obervations
    n = len(pmord.index)

    pmord['probex'] = ((n - (pmord.index.values+1) + 1) / (n + 1))          
    pmord['rv'] = 1./evts.mean()*(1./pmord['probex'])   #1/lambda*1/F lambda is Poisson parameter the average number of events per year  

    #4- Fit gpd
    rv = np.array([0.5,1,2,5,10,25,50,75,100])
    dpars = ['lambda','shape','location','scale']
    dparsrv = np.hstack((dpars,rv.astype(str)))
    pgpd = np.array(dparsrv)
       
    # potmodel = pd.DataFrame(columns=['mean']+['%s' % ip for ip in per]+['meanB'],index=pgpd)
    potmodel = pd.DataFrame(columns=['mean']+['%s' % ip for ip in per],index=pgpd)
    potmodelc = pd.DataFrame(columns=['mean']+['%s' % ip for ip in per],index=np.hstack(([rv[0]],np.arange(rv[1],rv.max()+1))))
    
#    ppot = np.array(['qthres','thresh','lambda','shape','location','scale'])
#    potmodel = pd.DataFrame(columns=col,index=np.hstack((ppot,rv.astype(str))))
    if fitd == 'gpd':
        potmodel.loc[['shape','location','scale'],'mean'] = gepd.fit(pmord['emp'].values,0,loc=thresh)
    elif fitd == 'exp':
#        potmodel.loc[['shape','location','scale'],'mean'] = gepd.fit(pmord['max'].values,0,loc=thresh)                
        potmodel.loc[['location','scale'],'mean'] = expon.fit(pmord['emp'].values,loc=thresh)                
        potmodel.loc['shape','mean'] = 0

    # potmodel.loc['lambda','mean'] = evts.mean()
    potmodel.loc['lambda','mean'],potmodel.loc['lambda',idx['%s'%per[0]]],potmodel.loc['lambda',idx['%s'%per[1]]] = mean_confidence_interval(evts,np.abs(max(per)-min(per))/100.)
    # potmodel.loc['lambda','mean'] = lamme
    # potmodel.loc['lambda',idx['%s'%per[0]]] = lamlb
    # potmodel.loc['lambda',idx['%s'%per[1]]] = lamub
    # print(potmodel.loc['lambda',:])
    potmodel.loc['thresh','mean'] = thresh
    potmodel.loc['qthres','mean'] = qth


    potmodelc.loc[:,'mean'] = gepd.ppf(1.-1./(potmodel.loc['lambda','mean']*potmodelc.index.values),potmodel.loc['shape','mean'],loc=potmodel.loc['location','mean'],scale=potmodel.loc['scale','mean'])
#    potmodelc.loc[:,'mean'].plot(figsize=(12,9));plt.plot(pmord['rv'],pmord['max'],'o');plt.semilogx()
    potmodel.loc[rv.astype(str),'mean']=potmodelc.loc[rv,'mean'].values
    pmord['mod_mean'] = gepd.ppf(1.-pmord['probex'].values,potmodel.loc['shape','mean'],loc=potmodel.loc['location','mean'],scale=potmodel.loc['scale','mean'])
    
    # # bootstrapping
    # bootparamsci = pot_bootstrap(potmodel.loc[:,idx['mean']],fitd,pmord.shape[0],nr,per,pmord['emp'].values)
    # for pari in dpars[1:]:
    #     # potmodel.loc[pari,idx['meanB']] = bootparamsci[pari][len(per)]
    #     for ii,peri in enumerate(per):
    #         potmodel.loc[pari,idx['%s'%peri]] = bootparamsci[pari][ii]
    
    # for peri in per:
    #     potmodelc.loc[:,'%s'%peri] = gepd.ppf(1.-1./(potmodel.loc['lambda','mean']*potmodelc.index.values),potmodel.loc['shape','%s'%peri],loc=potmodel.loc['location','%s'%peri],scale=potmodel.loc['scale','%s'%peri])
    #     potmodel.loc[rv.astype(str),'%s'%peri]=potmodelc.loc[rv,'%s'%peri].values
    #     pmord['mod_%s'%peri] = gepd.ppf(1.-pmord['probex'].values,potmodel.loc['shape','%s'%peri],loc=potmodel.loc['location','%s'%peri],scale=potmodel.loc['scale','%s'%peri])

    # bootstrapping
    bootparamsci = pot_bootstrap(potmodel.loc[:,idx['mean']],fitd,pmord.shape[0],nr,per,pmord['emp'].values)
    for ii,peri in enumerate(per):
        for pari in dpars[1:]:
            potmodel.loc[pari,idx['%s'%peri]] = np.percentile(bootparamsci[pari],peri)

    rvals=np.nan*np.ones((potmodelc.index.values.size,nr))
    rvalsval=np.nan*np.ones((pmord['probex'].values.size,bootparamsci['shape'].size))
    if fitd.lower() == 'gpd':
        for ii in range(nr):
            rvals[:,ii] = gepd.ppf(1.-1./(potmodel.loc['lambda','mean']*potmodelc.index.values),bootparamsci['shape'][ii],loc=bootparamsci['location'][ii],scale=bootparamsci['scale'][ii])
            rvalsval[:,ii] = gepd.ppf(1.-pmord['probex'].values,bootparamsci['shape'][ii],loc=bootparamsci['location'][ii],scale=bootparamsci['scale'][ii])
    elif fitd.lower() == 'exp':
        for ii in range(nr):
            rvals[:,ii] = expon.ppf(1.-1./(potmodel.loc['lambda','mean']*potmodelc.index.values),loc=bootparamsci['location'][ii],scale=bootparamsci['scale'][ii])
            rvalsval[:,ii] = expon.ppf(1.-pmord['probex'].values,loc=bootparamsci['location'][ii],scale=bootparamsci['scale'][ii])

    for peri in per:
        potmodelc.loc[:,idx['%s'%peri]] = np.percentile(rvals,peri,axis=1)
        potmodel.loc[rv.astype(str),idx['%s'%peri]]=potmodelc.loc[rv,idx['%s'%peri]].values
        pmord['mod_%s'%peri] = np.percentile(rvalsval,peri,axis=1)



    # plt.figure(figsize=(12,9));plt.plot(potmodelc.loc[:,idx['mean']],'-k');plt.plot(potmodelc.loc[:,idx['95']],'r--');plt.plot(potmodelc.loc[:,idx['5']],'r--');plt.semilogx()
    # plt.plot(pmord['rv'],pmord['emp'],'ob');plt.plot(pmord['rv'],pmord['mod_mean'],'-k');plt.plot(pmord['rv'],pmord['mod_5'],'r--');plt.plot(pmord['rv'],pmord['mod_95'],'r--')


#    plt.plot(gevmodel.loc[rv.astype(str)].index.astype(int),gevmodel.loc[rv.astype(str)].values,'ob')
#    plt.plot(probY['rv'].values,probY['val'].values,'or')
    return pmord.drop(columns='probex'),potmodel

def mean_confidence_interval(data, confidence):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scst.sem(a)
    h = se * scst.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

def pot_bootstrap(params,fitd,ns,nr,per,peaks):
    ### pot_bootstrap ###
    # Bootstrapping of extremes value distribution parameters
    # params: distribution mean parameters
    # fitd: distribution used for the extreme value analysis
    # ns: number of original samples when fitting the extreme value model
    # nr: set of bootstrap samples of size ns
    # per: percentiles to return parameter estimations
    # poisson estimation is not yet implemented, we do delta method for this parameter defined in mean_confidence_interval
    shape = np.zeros((nr,))
    location = np.zeros_like(shape)
    scale = np.zeros_like(shape)
    lmbda = np.zeros_like(shape)
    
    for rpi in range(nr):           
        boot = resample(peaks, replace=True, n_samples=len(peaks))
        # vals = gepd.ppf(np.random.rand(ns),params['shape'],loc=params['location'],scale=params['scale'])
        # vals2 = poisson.ppf(np.random.rand(ns),params['lambda'])
        if fitd == 'gpd':
            shape[rpi],location[rpi],scale[rpi] = gepd.fit(boot,loc=params['thresh'])
        elif fitd == 'exp':
#            shape[rpi],location[rpi],scale[rpi] = gepd.fit(vals,0,loc=params['thresh'])
            location[rpi],scale[rpi] = expon.fit(boot,loc=params['thresh']);shape[rpi]=0            
        # lmbda[rpi] = vals2.mean() 
    # return {'shape':np.hstack((np.percentile(shape,per),np.mean(shape))),'location':np.hstack((np.percentile(location,per),np.mean(location))),'scale':np.hstack((np.percentile(scale,per),np.mean(scale))),'lambda':np.hstack((np.percentile(lmbda,per),np.mean(lmbda)))}
    # Adjust the mean of the bootstrap estimates using the point estimates
    shape = shape-shape.mean()+params['shape']
    location = location-location.mean()+params['location']
    scale = scale-scale.mean()+params['scale']
    # return {'shape':np.hstack((np.percentile(shape,per),np.mean(shape))),'location':np.hstack((np.percentile(location,per),np.mean(location))),'scale':np.hstack((np.percentile(scale,per),np.mean(scale)))}
    return {'shape':shape,'location':location,'scale':scale}


#%% main script
if __name__ == '__main__':

    if os.name=='nt':
        projectdir = r'p:\11204672-gtsm-cmip6'
        codecdir = r'p:\11200665-c3s-codec'
        projectdir = r'N:\My Documents\unix-h6\GTSmip'
    elif os.name == 'posix':
        projectdir = r'/projects/0/GTSM'
        codecdir = r'/projects/0/GTSM'
        # projectdir = r'/p/11204672-gtsm-cmip6'
        # codecdir = r'/p/11200665-c3s-codec'
        
    #run = '01_HIST'
    #run = '00_TIDES'
#    run = '02_RCP85'
    run = sys.argv[1]
    print(run)
    timeperiods={'00_TIDES':[1950,2050],'01_HIST':[1950,2014],'02_RCP85':[2015,2050],
                 '03_ERA5':[1979,2018],'04_SST_HIST':[1951,2014],'05_SST_RCP85':[2016,2050]}


    workdir = os.path.join(projectdir,'05_ANALYSIS',run)
    paths = Paths(workdir);paths._makedirs()
    paths.tide = os.path.join(projectdir,'05_ANALYSIS','00_TIDES','data')
#    pxyn = pd.read_csv(os.path.join(projectdir,'05_ANALYSIS','gtsmip.xyn'),index_col=0)
    # pxyn = pd.read_csv(os.path.join(projectdir,'05_ANALYSIS',run,'data',run+'_id_lon_lat.xyn'),index_col=0)
    pxyn = pd.read_csv(os.path.join(projectdir,'05_ANALYSIS',run,'data',run+'_id_lon_lat_bedlev.xyzn'),index_col=0)
    pxyn.index=pxyn.index.astype(int)
    #%% Compute Metrics and save in an excel file
    start_time = time.time()
    

        


    pgev = np.array(['shape','location','scale'])
    nr = 599 # number of bootstrap repetitions Wilcox, R. R. (2010). Fundamentals of modern statistical methods: Substantially improving power and accuracy. Springer.
    # per = [25,75] # percentiles for the CI estimation of the distribution parameters
    per = [5,95] # percentiles for the CI estimation of the distribution parameters CI= Confidence Interval
    ngs = pxyn.shape[0]
    interval = range(ngs)
    # interval = range(1000)
    
    wlc = str(sys.argv[2])
    print(wlc)
    seasc = str(sys.argv[3])
    print(seasc)
    periodc = str(sys.argv[4])
    if '[' in periodc:
        periodc = np.array(periodc.strip('[]').split(',')).astype(int)

    fitd = str(sys.argv[5])
    print(fitd)           
    if fitd in ['gumbel','gev']:
        evam = 'BlockMaxima'
        rv = np.array([2,5,10,25,50,75,100])
    elif fitd in ['exp','gpd']:
        evam = 'POT'
        pgev = np.hstack((['lambda','thresh','qthres'],pgev)) # pgev = ['lambda','thresh','qthres','shape','location','scale']
        rv = np.array([0.5,1,2,5,10,25,50,75,100])

    qopt='quan'
    q = 0.99 # this is only valid for POT
    indep = 24*3 #3 days independence
    mindur = 10*1 #1minute 1hours 
    res = 10 # resolution of the data [min here]
    resunit = 'min'
    #print(int(sys.argv[1]),int(sys.argv[2]))
    #interval = range(int(sys.argv[1]),int(sys.argv[2])+1)

    # initialize
    fname = '%s_%s%05d.pxz'%('gtsm','station',101)
    gtsmpd= pd.read_pickle(os.path.join(paths.pregtsm,fname),compression='xz')
    if run in ['00_TIDES','01_HIST','02_RCP85','03_ERA5']:
        gtsmpd = gtsmpd[gtsmpd.index.year>timeperiods[run][0]] # we remove the first year of computation if we are not in SST runs that have not been run already
    # ny = gtsmpd.index.year.unique().shape[0]

    if len(periodc)>1:
        flagrecind2 = False
        ind2 = gtsmpd.index[(gtsmpd.index.year>=periodc[0])&(gtsmpd.index.year<=periodc[-1])]
        if ind2.empty or len(ind2)<6*24*30:
            sys.exit()
        else:
            gtsmpd = gtsmpd.loc[ind2,:] 
    else:
        flagrecind2 = True
             
    
    periodc = [gtsmpd.index[0].year,gtsmpd.index[-1].year]
    print(periodc, flush=True)
    if flagrecind2:
        ind2 = gtsmpd.index[(gtsmpd.index.year>=periodc[0])&(gtsmpd.index.year<=periodc[-1])]
        # this is the case when requested full period we have to obtain the index
    print(ind2[0],ind2[-1], flush=True)

    ntl=[];namedi=[]
    if seasc=='seasonal':
        ntime = 12
        ntl = np.arange(1,12+1)
        namedi = ['months']
    elif seasc=='annual':
        ntime = gtsmpd.index.year.unique().shape[0]
        ntl = gtsmpd.index.year.unique()
        namedi = ['years']
    elif seasc=='decadal':
        ntime = np.floor(gtsmpd.index.year.unique().shape[0]/10)
        ntl = np.unique(np.floor(gtsmpd.index.year.unique()/10)*10).astype(int)
        namedi = ['decades']
    elif seasc=='periods':
        if run in ['00_TIDES','01_HIST','04_SST_HIST']:
            ntl = ['1951-1980', '1985-2014']
        elif run in ['03_ERA5']:
            ntl = ['1985-2014']
        elif run in ['02_RCP85','05_SST_RCP85']:
            ntl = ['2021-2050']
        # ntl = ['1951-1980', '1985-2014', '2021-2050']
        ntime = len(ntl)
        namedi = ['periods']        
        
    # if seasc in ['seasonal','annual','decadal']:
    #     extremesdf = pd.DataFrame(index = pxyn.index,columns=pd.MultiIndex.from_product([np.hstack((pgev,rv.astype(str))),['mean']+['%s' % ip for ip in per],ntl],names=['metrics','ci']+namedi))
    #     extremesEdf = pd.DataFrame(index = pxyn.index,columns=pd.MultiIndex.from_product([['rv','emp','mod_mean']+['mod_%s' % ip for ip in per],ntl,np.arange(ny)],names=['metrics',namedi[0],'samples']))
    # else:        
    #     extremesdf = pd.DataFrame(index = pxyn.index,columns=pd.MultiIndex.from_product([np.hstack((pgev,rv.astype(str))),['mean']+['%s' % ip for ip in per]],names=['metrics','ci']))
    #     extremesEdf = pd.DataFrame(index = pxyn.index,columns=pd.MultiIndex.from_product([['rv','emp','mod_mean']+['mod_%s' % ip for ip in per],np.arange(ny)],names=['metrics','samples']))

    sii = int(sys.argv[6])
    see = int(sys.argv[7])
    stt = int(sys.argv[8])
    # stt = 5000 # size of the remote call 

    # for ii in interval[0::stt]:
    for ii in np.arange(sii,see+1,stt):
        print(ii,flush=True)
        fnamep = '%s_%s_%s_%s_%s_%s_%4d-%4d_%d.p'%('metrics',evam,fitd,run,wlc,seasc,periodc[0],periodc[1],ii)
        fnamepE = '%s_%s_%s_%s_%s_%s_%4d-%4d_%d.p'%('valida',evam,fitd,run,wlc,seasc,periodc[0],periodc[1],ii)
        if (os.path.isfile(os.path.join(paths.out,fnamep)))&(os.path.isfile(os.path.join(paths.out,fnamepE))):
            if (os.stat(os.path.join(paths.out,fnamep)).st_size>500)&(os.stat(os.path.join(paths.out,fnamepE)).st_size>500):
                continue

        ee = np.min([ii+stt,interval[-1]+1])
        listdf=ray.get([extreme_pd.remote(paths,wlc,seasc,jj,ntl,namedi,fitd,evam,rv,qopt,q,indep,mindur,res,resunit,nr,per,pgev,ind2) for jj in np.arange(ii,ee)])
        listdf = [[ i for i, j in listdf],[ j for i, j in listdf]]
        # if ii == sii:
        #     extremesEdf = pd.concat(listdf[1])
        #     extremesdf = pd.concat(listdf[0])
        # else:
        #     print(ii)
        #     listdf1 = pd.concat(listdf[1])
        #     listdf0 = pd.concat(listdf[0])
        #     extremesEdf = pd.concat((extremesEdf,listdf1))
        #     extremesdf = pd.concat((extremesdf,listdf0))

        extremesEdf = pd.concat([ld for ld in listdf[1] if ~ld.empty])
        extremesdf = pd.concat([ld for ld in listdf[0] if ~ld.empty])

        extremesdf.to_pickle(os.path.join(paths.out,fnamep))
        extremesEdf.to_pickle(os.path.join(paths.out,fnamepE))

    # extremesdf=ray.get([extreme_pd.remote(paths,wlc,seasc,per,ii) for ii in interval])
    # extremesdf = [[ i for i, j in extremesdf],[ j for i, j in extremesdf]]
    # extremesEdf = pd.concat(extremesdf[1])
    # extremesdf = pd.concat(extremesdf[0])

    # for ii in interval:
        # print(ii,'/',ngs-1)
        # fname = '%s_%s%05d.pxz'%('gtsm','station',ii)
        # gtsmpd= pd.read_pickle(os.path.join(paths.pregtsm,fname),compression='xz')
        # gtsmpd = gtsmpd[gtsmpd.index.year>timeperiods[run][0]] # we remove the first year of computation
        # if wlc=='ss':
        #     tidepd = pd.read_pickle(os.path.join(paths.tide,fname),compression='xz')
        #     gtsmpd -= tidepd.loc[gtsmpd.index]

        # if seasc in ['seasonal','annual','decadal']:
        #     for jj in ntl:
        #         if seasc == 'seasonal':
        #             ind2 = gtsmpd.index[gtsmpd.index.month==jj]
        #         elif seasc == 'annual':
        #             ind2 = gtsmpd.index[gtsmpd.index.year==jj]
        #         elif seasc == 'decadal':
        #             ind2 = gtsmpd.index[np.floor((gtsmpd.index.year/10)*10).astype(int)==jj] 
                    
        #         if ind2.empty or len(ind2)<6*24*30:
        #             continue
        #         # else:
        #         #     ind3 = ind2[0::24*7]
        #         #     ind3 = ind2
        #         probY,gevmodel = block_maxima(gtsmpd.loc[ind2],rv,fitd,nr,per)
        #         for cii in gevmodel.columns:
        #             extremesdf.loc[ii,idx[gevmodel.index,cii,jj]] = gevmodel[cii].values.squeeze()
        #         for ik in probY.columns:
        #             extremesEdf.loc[ii,idx[ik,jj,:]] = probY[ik].values
        # else:
        #     probY,gevmodel = block_maxima(gtsmpd,rv,fitd,nr,per)
        #     for cii in gevmodel.columns:
        #         extremesdf.loc[ii,idx[gevmodel.index,cii]] = gevmodel[cii].values.squeeze()
        #     for ik in probY.columns:
        #         extremesEdf.loc[ii,idx[ik,:]] = probY[ik].values

    # fnamep = '%s_%s_%s_%s_%s_%s.p'%('metrics',evam,fitd,run,wlc,seasc)
    # extremesdf.to_pickle(os.path.join(paths.out,fnamep))

    # fnamee = '%s_%s_%s_%s.xlsx'%('metrics',evam,fitd,run)
    # if os.path.isfile(os.path.join(paths.out,fnamee)):
    #     with pd.ExcelWriter(os.path.join(paths.out,fnamee),mode='a') as writer:  # doctest: +SKIP
    #         extremesdf.to_excel(writer,sheet_name='%s_%s'%(wlc,seasc))
    # else:
    #     with pd.ExcelWriter(os.path.join(paths.out,fnamee),mode='w') as writer:  # doctest: +SKIP
    #         extremesdf.to_excel(writer,sheet_name='%s_%s'%(wlc,seasc))

    # fnamepE = '%s_%s_%s_%s_%s_%s.p'%('valida',evam,fitd,run,wlc,seasc)
    # extremesEdf.to_pickle(os.path.join(paths.out,fnamepE))
    
    comp_time = time.time() - start_time
    print("FINISHED in %.2f minutes" % (comp_time/60))
