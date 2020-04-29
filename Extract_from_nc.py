# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 08:40:05 2020

@author: acn980
"""

import os, sys
import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import ticker
import matplotlib as mpl

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