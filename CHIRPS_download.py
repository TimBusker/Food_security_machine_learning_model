# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 11:33:21 2022

@author: tbr910
"""

import os
import matplotlib.pyplot as plt
import pandas as pd
from netCDF4 import Dataset as netcdf_dataset
import numpy as np
import cartopy
from cartopy import config
import cartopy.crs as ccrs
import cartopy.feature as cf
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature
import cartopy.feature as cfeature
import cartopy.mpl.geoaxes
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.cm as cm
import geopandas as gpd
import regionmask
import xarray as xr
import urllib.request
import requests
import re
import glob
from bs4 import BeautifulSoup
import datetime as dtmod
import re
from rasterio.enums import Resampling
import scipy.stats as stats




path='/scistor/ivm/tbr910/ML/input_collector/CLIM_data/Rainfall/CHIRPS005'
os.chdir(path)




# ##################################################################### Download data ######################################################
# def download_data(start_year, end_year, rfe_url, file_path): 

        
#         # Define climatological period
#         clim_years = np.arange(start_year, end_year + 1, 1)

#         # Create empty array to store precip file names


#         for yr in clim_years:
#             server_filenames=[]
#             soup = BeautifulSoup(requests.get(rfe_url + '/%s'%(yr)).text, 'html.parser')
            
#             for a in soup.findAll(href = re.compile(".tif$")): #for 2021-12 use .tif, other years use .tif.gz
#                 server_filenames.append(a['href'])
#             print (server_filenames)
            
#             for server_filename in server_filenames: 
                
#                 fname= file_path + "/" + server_filename
#                 if os.path.isfile(fname) == False: # only download if file does not exist yet
                    
#                     url= rfe_url+"/"+str(yr)+'/'+server_filename
#                     print (url, fname)
#                     urllib.request.urlretrieve(url, fname)
                
#             print ("CHIRPS download in progress...... year:%s" %(yr))            


# # execute data download 
# start_year=2020
# end_year=2023
# rfe_url = 'https://data.chc.ucsb.edu/products/CHIRPS-2.0/africa_daily/tifs/p05'
# file_path = path

# download_data(start_year, end_year, rfe_url, file_path)

# ##################################################################### Unzip data ######################################################
# import gzip
# import shutil
# for i in os.listdir(path):
#     if i.endswith('.gz'):
#         print (i)
#         with gzip.open(i, 'rb') as f_in:
#             with open(i[:-3], 'wb') as f_out:
#                 shutil.copyfileobj(f_in, f_out)


# delete the .gz files 
# [os.remove(i) for i in os.listdir(path) if i.endswith('.gz')]

##################################################################### .nc conversion ######################################################
target_files=[i for i in os.listdir(path) if '2021.12' in i]
for i in target_files:
    if i.endswith('.tif'):
        print (i)
        xrds=xr.open_dataset(i)
        xrds.to_netcdf(i[:-4]+'.nc')

[os.remove(i) for i in os.listdir(path) if i.endswith('.tif')]

exit() 

##################################################################### HAD CROPPING ######################################################
os.chdir(path) ## unit mm/ time step 
HAD_path='/scistor/ivm/tbr910/ML/input_collector/CLIM_data/Rainfall/CHIRPS005_HAD'

for i in os.listdir():
    os.chdir(path)
    rainfall=xr.open_dataset(i) 
    rainfall=rainfall.rename({'x': 'longitude','y': 'latitude'}).drop('spatial_ref').squeeze('band').drop('band') # not necessary for TAMSAT
    rainfall=rainfall.assign_coords(time=pd.Timestamp(i[12:-4])).expand_dims('time')# not necessary for TAMSAT
    rainfall_HAD=rainfall.where((rainfall.latitude > -4.7) & (rainfall.latitude<18.4) & (rainfall.longitude > 32.5) &(rainfall.longitude < 51.44) , drop=True)
    os.chdir(CHIRPSV2_HAD)
    rainfall_HAD.to_netcdf("%s_HAD.nc" %(i[:-4]))  #for tamsat prate_tamsat -11:-7  for chirps --> "chirps-v2_%s_sub_HAD.nc" %(i)[14:-3]




# check for missing dates 
# start= pd.to_datetime('1981-01-01')
# end= pd.to_datetime('2023-05-31')
# date_range=pd.date_range(start, end, freq='D')

# list=[i for i in os.listdir() if i.endswith('.nc')]
# dates_present= [pd.to_datetime(i[12:22]) for i in list]
# missing_dates= [i for i in date_range if i not in dates_present]

# wrong_files= [i for i in dates_present if i != len(os.listdir()[0])]




##################################################################### MERGE TO ONE DATASET ######################################################
# os.chdir(CHIRPSV2_HAD)
# P_HAD= xr.open_mfdataset([i for i in os.listdir() if 'ALL' not in i],combine='nested', concat_dim='time') #chunks={'time':10} chunks={'time':1, 'longitude':73, 'latitude':79}
# P_HAD.to_netcdf("chirps-v2_ALL_YEARS_sub_HAD_ORIGINAL.nc")  #[-11:-7] for tamsat  prate_tamsat_sub_HAD_ALL_YEARS.nc























#%% Wet days calculation 
def wet_days(rainfall_input, threshold, period):
    #%%%%% flag dry and wet days 
    dry_days=rainfall_input.where((rainfall_input['tp'] >threshold) | rainfall_input.isnull(), 1) # isnull causes nan values to persist 
    dry_days=dry_days.where((dry_days['tp'] ==1) | dry_days.isnull(), 0)
    
    wet_days=rainfall_input.where((rainfall_input['tp'] >threshold) | rainfall_input.isnull(), 0)
    wet_days=wet_days.where((wet_days['tp'] ==0) | wet_days.isnull(), 1)
    #%%%%% monthly sums 
    wet_days_number=wet_days.resample(time=period).sum()
    #wet_days_number = wet_days_number.where((((wet_days_number['time'].dt.month >=3) & (wet_days_number['time'].dt.month <=5)) | ((wet_days_number['time'].dt.month >=10) & (wet_days_number['time'].dt.month <=12))), np.nan)#this makes dry season months nan's
    return(wet_days_number)
    

#%% dry days-spells calculation 
def max_dry_spells(rainfall_input, threshold, period):
    #%%% dry days 
    dry_days=rainfall_input.where((rainfall_input['tp'] >threshold) | rainfall_input.isnull(), 1) # isnull causes nan values to persist 
    dry_days=dry_days.where((dry_days['tp'] ==1) | dry_days.isnull(), 0)
    
    #%%% dry spells 
    ###### put beginning of season to 0! Leave all the other values representing the dry spell length --> creates inaccuracy of dry spell length with 0 day for dry spells that start at first day of the season. This is important as otherwise the fake dry days of the non-rainy season months (np.nan values) will create a very long dry spell in the rainy season. 
    dry_spell=dry_days.where((((dry_days['time'].dt.month !=3) | (dry_days['time'].dt.day !=1)) & ((dry_days['time'].dt.month !=10) | (dry_days['time'].dt.day !=1))), 0)
    ###### make a cumsum using only the dry days ##########
    # This function restarts cumsum after every 0, so at every wet day or beginning of the season. 
    cumulative = dry_spell['tp'].cumsum(dim='time')-dry_spell['tp'].cumsum(dim='time').where(dry_spell['tp'].values == 0).ffill(dim='time').fillna(0)
    dry_spell_length= cumulative.where(cumulative>=5, 0) ## only keep length of dry spells when >=5 days. Rest of the values to zero

    
    ####################################################################################### number of dry spells per season  ################################################################################
    # dry_spell_start=dry_spell_length.where(dry_spell_length==5, 0)
    # dry_spell_start=dry_spell_start.where(dry_spell_start!=5, 1) ## binary dry spell start  
    
    # dry_spell_number=dry_spell_start.resample(time='MS').sum() ## sum all dry spell starts 
    # dry_spell_number = dry_spell_number.where((((dry_spell_number['time'].dt.month >=3) & (dry_spell_number['time'].dt.month <=5)) | ((dry_spell_number['time'].dt.month >=10) & (dry_spell_number['time'].dt.month <=12))), np.nan)#this makes dry season months nan's
    
    # dry_spell_number=dry_spell_number.where(land_mask==1, np.nan) ## land mask
    # dry_spell_number=dry_spell_number.to_dataset()

    ####################################################################################### maximum (seasonal!!) dry spell length ################################################################################
    dry_spell_max= dry_spell_length.resample(time=period).max() ## max dry spell length per month, per season
    dry_spell_max=dry_spell_max.to_dataset() ## for processing later on 
    
    return (dry_spell_max)


