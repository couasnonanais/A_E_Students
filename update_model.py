import xarray as xr
import numpy as np
import pandas as pd
import os, sys
from os.path import join, basename
import matplotlib.pyplot as plt
import hydromt
from hydromt_sfincs import SfincsModel
from hydromt_sfincs.utils import parse_datetime, write_timeseries, write_inp
from datetime import timedelta
from hydromt.log import setuplog

#%%

#Read events
events = dict(
    coastal = xr.open_dataarray(r'./boundary_conditions/dullaart_events.nc').reset_coords(drop=True),
    fluvial = xr.open_dataarray(r'./boundary_conditions/glofas_discharge_events.nc').reset_coords(drop=True),
)

#These times should correspond to the time of your design event
tstart = '20200101 000000'
tstop = '20200115 000000'  

#Prepare basemodel
mdir = r"./models/00_start" #Initial model
logger = setuplog('update', join(mdir, "hydromt.log"), log_level=10)


#Defining the new model
new_root = f'./models/{case}'
mod1 = SfincsModel(join(mdir), mode='r', deltares_data=True, data_libs=data_libs_fn, logger=logger)
mod1.read()

#We extract the locations of the points
gdf_q = mod1.forcing['dis'].vector.to_gdf()
gdf_h = mod1.forcing['bzs'].vector.to_gdf()

#To update a discharge BC
mod1.set_forcing_1d(ts=q0.copy(), xy=gdf_q, name='discharge') 
mod1.forcing['dis'] = mod1.forcing['dis'].fillna(0.0)

#To update a coastal BC
mod1.set_forcing_1d(ts=h0.copy(), xy=gdf_h, name='waterlevel')
mod1.forcing['bzs'] = mod1.forcing['bzs'].fillna(0.0)

mod1.setup_config(**{
'tref': tstart, 
'tstart': tstart, 
'tstop': tstop})

# update model and save
mod1.set_root(new_root)
mod1.write_forcing()
mod1.write_config(rel_path=f'../{basename(mdir)}')
