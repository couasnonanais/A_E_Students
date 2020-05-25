# -*- coding: utf-8 -*-
"""
Created on Wed May 20 09:04:16 2020

@author: acn980
"""

#Clip inundation maps

#%% Import packages
import os, sys, glob
import gdal 
import geopandas as gpd

#%% Setting the files and folder correctly
fn_trunk = r'E:\github\A_E_Students\Contour Jip'   
fn = os.path.join(fn_trunk,fn_files)

os.chdir(os.path.join(fn_trunk,fn_files))

#%% We clip with the the tiff file - option A USING GDAL

#To read a shapefile in geopandas
fn_mask = os.path.join(fn_trunk, 'LU_2010_clippedinun_dissolved.shp')
fn_mask = fn_mask.split('.shp')[0]  


#We will loop through the tiff files
all_files = glob.glob(os.path.join(fn_trunk,'Data_raw/Inundationmaps/1_R060_H080_maxdepth_linear.tiff'))

all_files = glob.glob(r'E:\surfdrive\Documents\Paper\Paper5\Hydrodynamic_runs\TELEMAC3d\1_R060_H080_maxdepth_linear.tiff')
os.system("gdalwarp -cutline {}.shp {}.tiff {}.tiff".format(fn_mask,fn_name, fn_name+'_clipped'))


for file in all_files:
    print(file)
    fn_name = os.path.split(file)[-1].split('.tiff')[0]  
#    os.system("gdalwarp -cutline {}.shp {}.tiff {}.tiff --config GDALWARP_IGNORE_BAD_CUTLINE YES".format(fn_mask,fn_name, fn_name+'_clipped'))
    