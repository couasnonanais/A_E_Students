# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 10:21:58 2020

@author: acn980
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
#%%
fn_file = r'E:\surfdrive\Documents\Master2019\Thomas\data\matlab_csv'
file_name = 'skew_WACC_Cleaned.csv'
dateparse = lambda x: datetime.strptime(x, '%d-%m-%Y %H:%M:%S')
data = pd.read_csv(os.path.join(fn_file, file_name), parse_dates = True, date_parser=dateparse, index_col = 'Date', usecols = ['Date','Skew '])

#%% To convert a date string to a datetime object
fn_file = r'E:\surfdrive\Documents\Master2019\Thomas\data\matlab_csv'
file_name = 'skew_WACC_Cleaned.csv'
data = pd.read_csv(os.path.join(fn_file, file_name), parse_dates = False, index_col = 'Date', usecols = ['Date','Skew '])
data.reset_index(inplace = True)
data['Date'] = [datetime.strptime(x, '%d-%m-%Y %H:%M:%S') for x in data['Date']]
data.set_index('Date', inplace = True)

#%% To combine day, month, year columns to a datetime object
# see also: https://stackoverflow.com/questions/19350806/how-to-convert-columns-into-one-datetime-column-in-pandas 
df = pd.DataFrame()
df['year'] = [2008,2008,2009,2010,2015]
df['month'] = [12,12,11,1,4]
df['day'] = [30,15,1,28,7]
df['date'] = pd.to_datetime(df[['year', 'month', 'day']]) # see also the function description: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.to_datetime.html
