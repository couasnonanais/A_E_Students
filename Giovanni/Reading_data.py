# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 16:37:53 2020

@author: acn980
"""

import os
import pandas as pd
from datetime import datetime
import datetime
import matplotlib.pyplot as plt

foldername = r'E:\github\A_E_Students\Giovanni\DATA'
filename = "EM_DAT.csv"

full_path = os.path.join(foldername, filename)
data = pd.read_csv(full_path, skiprows=6)

data.dropna(axis=0, how = 'any', subset=['Start Day'], inplace = True)
data.dropna(axis=0, how = 'any', subset=['End Day'], inplace = True)

data['start_date'] = pd.to_datetime(dict(year=data.loc[:,'Start Year'], month=data.loc[:,'Start Month'], day=data.loc[:,'Start Day']))
data['end_date'] = pd.to_datetime(dict(year=data.loc[:,'End Year'], month=data.loc[:,'End Month'], day=data.loc[:,'End Day']))

data['duration'] = data['end_date'] - data['start_date']

#data.loc[:,'Disaster Type'].unique() #To check the unique values

#%%

#Do you see this change?