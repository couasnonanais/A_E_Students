# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 08:40:05 2020

@author: acn980
"""

import os, sys, glob
import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
# import matplotlib.colors as mcolors
# from matplotlib import ticker
# import matplotlib as mpl

#%%
#This is for the storm surge netcdf
fn_folder = r'E:\surfdrive\Documents\VU\GTSR\tide_surge_dmax\New\tide_surge_daily_max.nc'
data = xr.open_dataset(fn_folder)
pts_HCMC = [3614,3619,3623,3624,453]

for pt_HCMC in pts_HCMC:
    print(pt_HCMC)
    hcmc = data.sel(stations = pt_HCMC)
    pt_data = hcmc['tide_surge'].to_series()

    fn_out_file = str(pt_HCMC)
    fn_out_folder = r'E:\surfdrive\Documents\Master2020\A&E\GTSM_HCMC'    
    pt_data.to_csv(os.path.join(fn_out_folder, 'swl_'+fn_out_file+'.csv'), index_label = 'date', header = 'daily_max_swl')
    
    pt_data.plot()

#%%
fn = r'E:\surfdrive\Documents\Master2020\Marike'
allfiles = glob.glob(os.path.join(fn, '*.nc'))

lat_HCMC = 10.4930
lon_HCMC = 106.3745
whole_ts = pd.DataFrame()
for file in allfiles:
    print(file)
    data = xr.open_dataset(file)
    ts = data['pr'].sel(rlat=lat_HCMC, rlon=lon_HCMC, method = 'nearest').to_series()
    #Transform to mm/day
    ts = pd.DataFrame(ts * 24 * 60 * 60)
    
    whole_ts = pd.concat([whole_ts, ts], axis = 0) 

fn_out =  r'E:\surfdrive\Documents\Master2020\Marike\CORDEX_pr_HCMC.csv'
whole_ts.to_csv(fn_out, index = True, index_label = 'date')
    
#%%    

all_data = pd.read_csv(fn_out) #Reading the data stored
all_data.set_index('date', inplace = True, drop = True)

#Store only possible dates
correct_data = pd.DataFrame(index = pd.date_range(all_data.index[0], all_data.index[-1]), columns = ['pr'])

for i in all_data.index:
    print(i)
    try:
        correct_data.loc[pd.date_range(i, i), 'pr'] = all_data.loc[i, 'pr']
    except:
        continue
    
#Sum precipitation per month
sum_month_total = correct_data.groupby(correct_data.index.month).sum()

#Resample monthly
sum_month = correct_data.resample('M').sum()
sum_month.plot()

#Amount of days with more than 50 mm/day
threshold = 80
extreme = correct_data.where(correct_data['pr']>= threshold)

sum_month_year = extreme['pr'].groupby([extreme.index.year, extreme.index.month]).agg('count')

#%%
fn = r'E:\surfdrive\Documents\Master2020\Marike\r50_observed_label.csv'
data = pd.read_csv(fn)

data_new = pd.DataFrame(index=np.arange(data['year'].min(), data['year'].max()+1,1),
                        columns = np.arange(data['month'].min(), data['month'].max()+1,1))

for row in data_new.index:
    for col in data_new.columns:
        print(row,col)
        data_new.loc[row,col] = np.float(data[((data.loc[:,'year']==row) & (data.loc[:,'month']== col))]['pr'])
 
season1 = np.arange(1,5,1)
season2 = np.arange(4,9,1)

sel1 = data_new.loc[:, season1]
sel1['avg_season'] = sel1.mean(axis = 1)
sel1['sum_season'] = sel1.loc[:,season1].sum(axis = 1)

sel1.mean()





