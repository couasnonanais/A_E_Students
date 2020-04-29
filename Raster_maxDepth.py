# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 19:36:55 2020

@author: acn980
"""
import os, sys, glob
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import gdal 

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

os.chdir(os.path.join(fn_trunk,fn_files))
#%% Export all the files in .csv
all_files = glob.glob(os.path.join(fn,'*.xyz'))

for file in all_files:
    print(file)
#file = all_files[16]
    data = pd.read_table(file, skiprows = 13, delim_whitespace=True, names = ['x','y','value'])
    
#    f = plt.figure()
#    ax = plt.axes()
#    cmap = plt.cm.Blues #plt.cm.seismic 
#    bounds = np.linspace(0, 35 ,36)
#    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
#    sc = ax.scatter(data.x, data.y, c=[data.value], cmap=cmap, norm = norm, alpha = 1, s=1) #vmin=0., vmax=90, norm=norm, 
#    f.colorbar(sc, ax = ax)
#    plt.show()
    
    fn_name = os.path.split(file)[-1].split('.xyz')[0]
    
    fn_out = os.path.join(fn, fn_name+'.csv')
    data.to_csv(fn_out, index = False)


#%% We create the .vrt file
all_files = glob.glob(os.path.join(fn,'*.xyz'))
for file in all_files:
    print(file)
    fn_name = os.path.split(file)[-1].split('.xyz')[0]
    fin = open(os.path.join(fn,"csv_to_vrt.vrt"), "rt")
    fout = open(os.path.join(fn,fn_name+".vrt"), "wt")
    
    for line in fin:
    	fout.write(line.replace('$', fn_name))
    	
    fin.close()
    fout.close()

#%% We create the tiff file
all_files = glob.glob(os.path.join(fn,'*.shp'))
for file in all_files:
    print(file) 
    fn_name = os.path.split(file)[-1].split('.shp')[0]    
#    os.system("gdal_grid -zfield value -a invdist:power=10.0:smoothing=1.0 -outsize 400.0 400.0 -ot Float64 -of GTiff -l {} {}.shp {}.tiff --config GDAL_NUM_THREADS ALL_CPUS".format(fn_name,fn_name, fn_name))
    os.system("gdal_grid -zfield value -a linear:radius=1 -outsize 400.0 400.0 -ot Float64 -of GTiff -l {} {}.shp {}.tiff --config GDAL_NUM_THREADS ALL_CPUS".format(fn_name,fn_name, fn_name+'_linear'))

#%% OLD
#'invdist:power=2.0:radius1=100.0:radius2=100.0:max_points=500:min_points=10'                        
#gdal_grid -a invdist:power=2.0:smoothing=1.0 -outsize 100 100 -of GTiff -ot Float64 -l 9_R120_H320_maxdepth csv_to_vrt.vrt trial.tiff
              

## We create the mask file?
#mask_raster = 'mask'
#os.system("gdal_grid -a count -outsize 400 400 -of GTiff -ot Float64 -l {} {}.vrt {}.tiff".format(fn_name,fn_name,'count_cells'))

#    output = gdal.Grid(os.path.join(fn, fn_name+'.tif'),os.path.join(fn,fn_name+".vrt"), 
#                       algorithm = 'linear:radius=100')  
#    gdal.GridOptions()
#    gdal.Grid(fn_name+'.tiff',fn_name+'.vrt', width = 400, height = 400, 
#              algorithm='linear:radius=100')