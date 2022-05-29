# -*- coding: utf-8 -*-
"""
Created on Mon May 25 13:53:21 2020

@author: acn980
"""

import os
import pandas as pd
import numpy as np

#%%
fn_file = r'E:\surfdrive\Documents\Master2020\Jip\curves_HCMC.csv'
data = pd.read_csv(fn_file, index_col = 0, dtype = np.float32)
data = data.reset_index().round(decimals = 2).set_index('index').dropna()
data.index.name='Depth'

inter_depth = np.arange(0.00,5.01,0.01)
interp_data = pd.DataFrame(data=None, index=np.arange(0.00,5.01,0.01))
interp_data.index.name='Depth'

interp_data_final = interp_data.join(data, on = 'Depth', how = 'outer') #, left_index = True, right_index = True) #, left_on=interp_data.columns, right_on = data.columns)
interp_data_final.interpolate()

fn_out = r'' #Your file location
interp_data_final.to_csv(fn_out, index_label = 'Depth')
