#%%
import hydromt
from hydromt import DataCatalog
from hydromt_sfincs import SfincsModel
import datetime
import xarray as xr
import sys, os
import numpy as np
import matplotlib.pyplot as plt
sys.path.append('./eva_script')

from eva_script import eva #This contains all the functions  you need
#See the description of all the functions here: https://github.com/Deltares/hydromt/blob/eva/hydromt/stats/eva.py 

#%%
#To export your BC timeseries from your SfincsModel
sfincs_root = '...' #Path to your model
mod = hydromt.SfincsModel(sfincs_root, mode='r') #Reading the model

#We export the timeseries
da_coast = mod.forcing['..'].copy() #Pick the boundary condition you want (e.g. bzs for coastal). To see all forcings, try mod.forcing
da_coast.to_netcdf('./boundary_conditions/ts_discharge.nc') #Path to where you want to save the BC data

#%% CREATING DESIGN EVENTS: In this case for discharge
figure_plotting = True #To show the figures and enter in the if functions

#Open your data
da_q = xr.open_dataarray('./boundary_conditions/ts_discharge.nc', chunks={"time": -1}) #Replace this with your dataset
ds_bm = eva.eva_block_maxima(da_q, min_dist=14).load() #We extract the maxima over a certain period: by default a year

if figure_plotting == True:
    for i in np.arange(1, len(da_q.index.values)+1):
        fig, ax = plt.subplots(1,1, figsize=(8,3))
        da_q.sel(index=i).plot()
        ds_bm['peaks'].sel(index=i).plot(marker='.', color='red', label='peaks')
        ax.set_ylabel('Discharge [m3/s]')
        ax.set_title(f'Discharge time series - River {i}')

    for i in np.arange(1, len(da_q.index.values)+1):
        da0 = ds_bm.sel(index=i)
        _, ax = plt.subplots(1, 1, figsize=(8, 5))
        _ = eva.plot_return_values(
            da0['peaks'].values,
            da0['parameters'].values,
            distribution=da0['distribution'].item(),
            extremes_rate=da0['extremes_rate'].item(),
            ax=ax
        )
        ax.set_ylabel('Discharge [m3/s]')
        ax.set_xlabel('Return period [year]')
        ax.set_title(f'Extreme value analysis - River {[i]} ')
#%% We extract the shape of the events for a given time window
da_hydrograph0 = eva.get_peak_hydrographs(
    da_q.chunk({'time':-1}), 
    ds_bm["peaks"].chunk({'time':-1}), 
    wdw_size=21,
    n_peaks=20,
)
da_hydrograph = da_hydrograph0.mean("peak")
da_hydrograph['time'].attrs.update(unit='days')

if figure_plotting == True:
    for i in np.arange(1, len(da_q.index.values)+1):
        fig, ax = plt.subplots(1,1, figsize=(8,5))
        da_hydrograph0.sel(index=i).plot.line(x='time', color='gray', lw=0.5, add_legend=False)
        da_hydrograph.sel(index=i).plot.line(x='time', color='red', lw=1.5, )
        ax.set_ylabel('normalized discharge [-]')
        ax.set_xlabel('time to peak [days]')
        ax.set_title(f'Normalized peak hydrograph - River {[i]}')
        #plt.savefig(f'..', dpi=300, bbox_inches='tight') #To save your figure

#%% We do the Extreme Value Analysis anc combine with the shape of the hydrograph

#We calculate the mean discharge and save this >> depending on the variable you might want this or not!
m = da_q['time'].dt.month
da_q_rp0 = da_q.isel(time=np.logical_or(m>=10, m<=9)).mean('time').expand_dims('rps')  #Note that the mean is calculated of hydrological years. You can change this depending on your case study
da_q_rp0['rps'] = xr.IndexVariable('rps', [0])
da_q_rp0 = da_q_rp0.reset_coords(drop=True).compute()

da_q_rps = xr.concat([da_q_rp0/da_hydrograph.mean('time'), ds_bm['return_values']], dim='rps')

#We multiple the magnitude of the events (da_q_rps) with the shape of the hydrograph (from 0 to 1)  (da_hydrograph)
da_events = da_q_rps * da_hydrograph 
da_events.name = 'events'

#We save the results
da_events.to_netcdf('./boundary_conditions/fluvial_design_events.nc')