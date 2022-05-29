# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 18:32:54 2021

@author: acn980
"""

from scipy.stats import genpareto
import numpy as np


scale = 0.07844764121312833
location = 0.3208195364304547
shape = -0.0008422459823042047

lambda_val = 10.466666666666667

qthreshold = 0.99
thr_value = 0.320143

#Method one - using your book
T=50.0
x_T = thr_value + (scale/shape)*(((lambda_val*T)**shape)-1)


#Method two - using the quantile
q_T = 1-(1/(lambda_val*T))
x_T2 = genpareto.ppf(q_T, c=shape, loc=location, scale=scale)


#gepd.ppf(1.-1./(potmodel.loc['lambda','mean']*potmodelc.index.values),potmodel.loc['shape','mean'],loc=potmodel.loc['location','mean'],scale=potmodel.loc['scale','mean'])
x_T3 = genpareto.ppf(1.-1./(lambda_val*T),shape,loc=location,scale=scale)


