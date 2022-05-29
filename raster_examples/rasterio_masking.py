# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 11:48:34 2021

@author: acn980
"""

import rasterio
import matplotlib.pyplot as plt

fn_mask = r'C:\Users\acn980\Desktop\Lecture8\Joost\output.tif'
fn_input = r'C:\Users\acn980\Desktop\Lecture8\Joost\landuse_sauer_original.tif'

#Read raster data
with rasterio.open(fn_mask) as src: #Reading the mask map and storing it in an array
    mask = src.read()[0, :, :]
    transform = src.transform

with rasterio.open(fn_input) as src: #Reading the landuse map and storing it in an array
    landuse = src.read()[0, :, :]
    transform = src.transform
    crs = src.crs


final_landcover = np.where((mask==-9999), landuse, 9999) #Where the mask is -9999 then use landcover otherwise set the value to 9999

rst_opts = {
    'driver': 'GTiff', #Type of raster
    'height': landuse.shape[0],  #nb of cells of raster - vertical
    'width': landuse.shape[1],  #nb of cells of raster - horizontal
    'count': 1, #Defines the number of bands to write
    'dtype': np.float32, #data type - here float
    'crs': crs, #coordinate system 
    'transform': transform, #information normally stored in the .prj file in GIS projects
    'compress': "LZW" #to compress the data and make it less voluminous
}

# Save the final landcover as a geotiff
with rasterio.open(r'C:\Users\acn980\Desktop\Lecture8\Joost\landuse_river.tif', 'w', **rst_opts) as dst: #we use the .open() function to write the raster ('w' stands for write)
     dst.write(final_landcover, 1) #We write the 2D array damagemap as a tif raster with 1 band


