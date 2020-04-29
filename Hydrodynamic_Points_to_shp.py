# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 11:40:14 2020

@author: acn980
"""

import os, sys, glob
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import gdal 
from shapely.geometry import Point


import matplotlib as mpl
import numpy as np
#%% Setting the files and folder correctly
home = False
save = False

if home ==True:
    fn_trunk = 'D:/surfdrive/Documents'
else:
    fn_trunk = 'E:/surfdrive/Documents'
    
fn_files = 'Paper/Paper5/Hydrodynamic_runs/TELEMAC3d'
fn = os.path.join(fn_trunk,fn_files)

#Changing the working directory to the location you specify
os.chdir(fn)
#%%
all_files = glob.glob(os.path.join(fn,'*.csv'))

#Import crs we need
prj_file = gpd.read_file(os.path.join(fn_trunk, 'Paper/Paper5/Hydrodynamic_runs/Rivers_HCMC_VN2K/ThuyHeDung_HCM_VN2K_region.shp'))

for file in all_files:
#file = all_files[16]
    
    print(file)
    fn_name = os.path.split(file)[-1].split('.csv')[0]    

    data = pd.read_csv(file)
    data['geometry'] = data.apply(lambda x: Point((float(x.x), float(x.y))), axis=1)
    pts_shp = gpd.GeoDataFrame(data, geometry='geometry')
    
    #Set crs
    pts_shp.crs = prj_file.crs
    
    #convert_crs
    pts_shp.to_crs(epsg=4326)
    
    #Export points
    pts_shp.to_file(os.path.join(fn,fn_name+'.shp'), driver='ESRI Shapefile')

#%% Calculating the difference between the two
min_shp = gpd.read_file(os.path.join(fn_trunk, fn_files, '1_R060_H080_maxdepth.shp'))
max_shp = gpd.read_file(os.path.join(fn_trunk, fn_files, '25_R300_H400_maxdepth.shp'))

diff = pd.DataFrame(data = None, columns = ['value', 'geometry'])
diff['value'] = max_shp['value'] - min_shp['value']
diff['geometry'] = min_shp['geometry']
diff_shp = gpd.GeoDataFrame(diff, geometry='geometry')
diff_shp.crs = min_shp.crs
diff_shp.to_file(os.path.join(fn_trunk,'Paper/Paper5/Hydrodynamic_runs','diff_max_min.shp'), driver='ESRI Shapefile')

#%%    
#    f = plt.figure()
#    ax = plt.axes()
#    cmap = plt.cm.Blues #plt.cm.seismic 
#    bounds = np.linspace(0, 35 ,36)
#    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
#    sc = ax.scatter(data.x, data.y, c=[data.value], cmap=cmap, norm = norm, alpha = 1, s=1) #vmin=0., vmax=90, norm=norm, 
#    f.colorbar(sc, ax = ax)
#    plt.show()
#    
#    shp_pts = gpd.GeoDataFrame()
#    
#    fn_name = os.path.split(file)[-1].split('.xyz')[0]
#    
#    fn_out = os.path.join(fn, fn_name+'.csv')
#    data.to_csv(fn_out, index = False)