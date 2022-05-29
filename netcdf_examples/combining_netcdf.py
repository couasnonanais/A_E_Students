# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 11:26:25 2022

@author: acn980
"""

import os, sys, glob
import xarray as xr
import numpy as np

fn_root = r'C:\Users\acn980\DATA\CODEC\US'
os.chdir(fn_root)
# all_files = os.listdir(fn_root)
# for file in all_files:
#     os.rename(file, file+'.nc')
#     print(file+'.nc')

#Did not work because of memory issues
#ds = xr.open_mfdataset(os.path.join(fn_root,'*.nc'))

ds4 = xr.concat([xr.open_dataset(f).set_coords(['latitude','longitude']).assign({"station_id": (int(f.split('gtsm_station')[-1].split('.nc')[0]))}) for f in glob.glob(os.path.join(fn_root,'*.nc'))], "stations")
ds4 = ds4.rename_dims({"index":"time"})
ds4.to_netcdf(os.path.join(r'C:\Users\acn980\DATA\CODEC\ts_waterlevel_US.nc'))

#%%
fn = r'C:\Users\acn980\DATA\CODEC\US\gtsm_station10208.nc'
fn_id = int(fn.split('gtsm_station')[-1].split('.nc')[0])
ds1 = xr.open_dataset(fn).set_coords(['latitude','longitude']).assign({"station_id": (fn_id)})

fn = r'C:\Users\acn980\DATA\CODEC\US\gtsm_station10211.nc'
fn_id = int(fn.split('gtsm_station')[-1].split('.nc')[0])
ds2 = xr.open_dataset(fn).set_coords(['latitude','longitude']).assign({"station_id": (fn_id)})

ds3 = xr.concat([ds1,ds2], "stations")
