# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 10:21:58 2020

@author: acn980
"""
import os
import pandas as pd
#%%
fn_file = r'E:\surfdrive\Documents\Master2019\Thomas\data\matlab_csv'
file_name = 'skew_WACC.csv'
dateparse = lambda x: pd.datetime.strptime(x, '%d-%m-%Y %H:%M:%S')
data = pd.read_csv(os.path.join(fn_file, file_name), parse_dates = True, date_parser=dateparse, index_col = 'Date', usecols = ['Date','Skew '])


