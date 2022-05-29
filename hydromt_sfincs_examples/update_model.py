#%%
import xarray as xr
import numpy as np
import pandas as pd
import os, sys
from os.path import join, basename
import matplotlib.pyplot as plt
import hydromt
from hydromt_sfincs import SfincsModel
from hydromt_sfincs.utils import parse_datetime
from datetime import timedelta
from hydromt.log import setuplog

#%%
#We open the file with the events
events = xr.open_dataarray(r'./boundary_conditions/fluvial_design_events.nc').reset_coords(drop=True)

#These times are the start and end of your simulation

t_peak = '20200101 000000'
t0 = parse_datetime(t_peak)

#We need to convert the time index (relative to the peak to a datetime index)
sel_rp = 100
q0 = events.sel(rps=sel_rp).reset_coords(drop=True).to_series().unstack(0)   #We convert the xr.dataarray to a pd.dataframe
q0.index = t0 + np.array([timedelta(days=int(dt)) for dt in q0.index.values]) #We convert the index to a date
#Looking at the index of q0, you see now that the event starts on 2019/12/21 and ends on 2020/01/10
print('Peak date is:', t_peak)
print('Start date is:', q0.index[0])
print('End date is:',  q0.index[-1])

#Note that (tstop - tstart) correspond to the window size picked in the eva (wdw_size)
tstart = q0.index[0].strftime("%Y%m%d %H%M%S") 
tstop = q0.index[-1].strftime("%Y%m%d %H%M%S") 
#These times are suggested only: you can also select other start and end time as long as there is data!

#Load the basemodel
mdir = r"./models/initial_model" #Initial model
logger = setuplog('update', join(mdir, "hydromt.log"), log_level=10)

#Defining the new model
new_root = f'./models/updated_model'
mod1 = SfincsModel(join(mdir), mode='r', deltares_data=True, logger=logger)
mod1.read()

#We extract the locations of the points
gdf_q = mod1.forcing['dis'].vector.to_gdf()

#To update a discharge BC
mod1.set_forcing_1d(ts=q0.copy(), xy=gdf_q, name='discharge') 
mod1.forcing['dis'] = mod1.forcing['dis'].fillna(0.0)

#To update a coastal BC
#gdf_h = mod1.forcing['bzs'].vector.to_gdf() #This is for a waterlevel location
# mod1.set_forcing_1d(ts=h0.copy(), xy=gdf_h, name='waterlevel')
# mod1.forcing['bzs'] = mod1.forcing['bzs'].fillna(0.0)

#We update the start and end time 
mod1.setup_config(**{
'tref': tstart, 
'tstart': tstart, 
'tstop': tstop})

# update model and save
mod1.set_root(new_root)
mod1.write_forcing()
mod1.write_config(rel_path=f'../{basename(mdir)}')
