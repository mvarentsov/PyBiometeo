#!/usr/bin/env python
# coding: utf-8

# In[1]:


import xarray as xr
import numpy as np
import biometeo

from mrt_utils import *

from tqdm import tqdm


# In[ ]:


ground_alb  = 0.2
ground_emis = 0.95
ground_z0 = 0.1

ds_dir = r"G:\!Data\Moscow\CLM\Experiments\Short_runs_MSK\20210601\MSK_0.0025_WorldCover1_rucpOVMr2\OUT_v6teb_ICONfixed2ERA5_AEV5t2_alb2_rt25_noconv_rucpOVMr2_thDS\cr\\"

ds = xr.open_mfdataset(ds_dir + '*.nc')
ds.load()

ds['VEL_10M'] = np.sqrt (ds['U_10M']**2 + ds['V_10M']**2)
ds['GHI'] = ds['SWDIRS_RAD'] + ds['SWDIFDS_RAD']


# In[ ]:


if 'MRT_sun' not in ds.data_vars or 'MRT_shd' not in ds.data_vars:
    ds['MRT_sun'],ds['MRT_shd'] = calc_MRT4ds (ds, 'T_2M', 'GHI', 'SWDIFDS_RAD', 'THDS_RAD', 'T_G', ground_alb, ground_emis)
    ds['MRT_sun'].to_netcdf(ds_dir + 'MRT_sun.nc')
    ds['MRT_shd'].to_netcdf(ds_dir + 'MRT_shd.nc')


# In[ ]:


ds['PET_sun'] = ds['T_2M'] * np.nan
ds['PET_shd'] = ds['T_2M'] * np.nan

ds['UTCI_sun'] = ds['T_2M'] * np.nan
ds['UTCI_shd'] = ds['T_2M'] * np.nan

rlon_range = range (ds.rlon.shape[0])
rlat_range = range (ds.rlat.shape[0])

for i,time in tqdm (enumerate(ds.time), total = ds.time.shape[0]): 
    ds4time = ds.sel(time=time)
    
    ds4time['T_2Mc'] = ds4time['T_2M'] - 273.15

    ds4time['E_2M'] = 6.11 * np.exp(17.27 * ds4time['T_2Mc']  / (ds4time['T_2Mc'] + 237.3))
    ds4time['e_2M'] = ds4time['E_2M'] * ds4time['RELHUM_2M'] / 100
    
    ds4time['VEL_1.1m'] = ds4time['VEL_10M'] * np.log (1.1/ground_z0) / np.log(10/ground_z0)
    ds4time['VEL_1.1m'] = ds4time['VEL_1.1m'].where(ds4time['VEL_1.1m'] > 0.341,  0.341)
    ds4time['VEL_1.1m'] = ds4time['VEL_1.1m'].where(ds4time['VEL_1.1m'] < 11.568, 11.568)

    for i_rlon in rlon_range:
        for i_rlat in rlat_range:
            pet_res = biometeo.PET(Ta = float(ds4time['T_2Mc'][i_rlon,i_rlat]),
                                   VP = float(ds4time['e_2M'][i_rlon,i_rlat]),
                                   v  = float(ds4time['VEL_1.1m'][i_rlon,i_rlat]),
                                   Tmrt = float(ds4time['MRT_sun'][i_rlon,i_rlat] - 273.15)) 
            ds['PET_sun'][i,i_rlon,i_rlat] = pet_res['PET_v']
            
            pet_res = biometeo.PET(Ta = float(ds4time['T_2Mc'][i_rlon,i_rlat]),
                                   VP = float(ds4time['e_2M'][i_rlon,i_rlat]),
                                   v  = float(ds4time['VEL_1.1m'][i_rlon,i_rlat]),
                                   Tmrt = float(ds4time['MRT_shd'][i_rlon,i_rlat] - 273.15)) 
            ds['PET_shd'][i,i_rlon,i_rlat] = pet_res['PET_v']

            utci_res = biometeo.UTCI(Ta = float(ds4time['T_2Mc'][i_rlon,i_rlat]),
                                     VP = float(ds4time['e_2M'][i_rlon,i_rlat]),
                                     v  = float(ds4time['VEL_1.1m'][i_rlon,i_rlat]),
                                     Tmrt = float(ds4time['MRT_sun'][i_rlon,i_rlat] - 273.15)) 
            ds['UTCI_sun'][i,i_rlon,i_rlat] = utci_res
            
            utci_res = biometeo.UTCI(Ta = float(ds4time['T_2Mc'][i_rlon,i_rlat]),
                                     VP = float(ds4time['e_2M'][i_rlon,i_rlat]),
                                     v  = float(ds4time['VEL_1.1m'][i_rlon,i_rlat]),
                                     Tmrt = float(ds4time['MRT_shd'][i_rlon,i_rlat] - 273.15)) 
            ds['UTCI_sun'][i,i_rlon,i_rlat] = utci_res
            
ds['PET_sun'].to_netcdf(ds_dir + 'PET_sun.nc')
ds['PET_shd'].to_netcdf(ds_dir + 'PET_shd.nc')
ds['UTCI_sun'].to_netcdf(ds_dir + 'UTCI_sun.nc')
ds['UTCI_shd'].to_netcdf(ds_dir + 'UTCI_shd.nc')