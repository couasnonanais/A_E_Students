# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 19:36:55 2020

@author: acn980
"""
import os, sys, glob
import gdal 
import geopandas as gpd

#%% Setting the files and folder correctly
fn_trunk = 'E:/surfdrive/Documents'    
fn_files = 'Paper/Paper5/Hydrodynamic_runs/TELEMAC3d/max_telemac'
fn = os.path.join(fn_trunk,fn_files)

os.chdir(os.path.join(fn_trunk,fn_files))
#%% We create the tiff file
all_files = glob.glob(os.path.join(fn,'*.shp')) #All the shp files we have

#Go check gdal_grid for more info if you need it: gdal.org/programs/gdal_grid.html
for file in all_files:
    print(file) 
    fn_name = os.path.split(file)[-1].split('.shp')[0]    
#    os.system("gdal_grid -zfield value -a invdist:power=10.0:smoothing=1.0 -outsize 400.0 400.0 -ot Float64 -of GTiff -l {} {}.shp {}.tiff --config GDAL_NUM_THREADS ALL_CPUS".format(fn_name,fn_name, fn_name))
#    os.system("gdal_grid -zfield H -a linear:radius=1 -outsize 8000.0 8000.0 -ot Float64 -of GTiff -l {} {}.shp {}.tiff --config GDAL_NUM_THREADS ALL_CPUS".format(fn_name,fn_name, fn_name+'_linear'))
    os.system("gdal_grid -zfield H -a linear:radius=0 -outsize 8000.0 8000.0 -ot Float64 -of GTiff -l {} {}.shp {}.tiff --config GDAL_NUM_THREADS ALL_CPUS".format(fn_name,fn_name, fn_name+'_linear'))

#%% We clip with the the tiff file - option A USING GDAL

#To read a shapefile in geopandas
#fn_mask = os.path.join(fn_trunk, 'Paper/Paper5/1_linearound.shp')
#mask_shp = gpd.read_file(fn_mask)

fn_mask = os.path.join(fn_trunk, 'Paper/Paper5/1_linearound.shp')
fn_mask = fn_mask.split('.shp')[0]  

#We will loop through the tiff files
all_files = glob.glob(os.path.join(fn,'*_linear.tiff'))
for file in all_files:
    print(file)
    fn_name = os.path.split(file)[-1].split('.tiff')[0]  
    os.system("gdalwarp -cutline {}.shp {}.tiff {}.tiff --config GDALWARP_IGNORE_BAD_CUTLINE YES".format(fn_mask,fn_name, fn_name+'_masked'))
    
#%% Clipping - option B - USING RASTERIO: 
    #Have a look at this link: https://stackoverflow.com/questions/41462999/how-do-i-use-rasterio-python-to-mask-a-raster-using-a-shapefile-to-set-the-rast

    
    