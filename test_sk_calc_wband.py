#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 16 Nov 18:02

@author: claudia
@test skewness calculation for dataset Wband

"""


# importing necessary libraries
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import netCDF4 as nc4
from netCDF4 import Dataset
import glob
import pandas as pd
#import atmos
from pathlib import Path
import xarray as xr
from scipy.interpolate import CubicSpline
import custom_color_palette as ccp
from matplotlib import rcParams
import matplotlib
# importing necessary libraries
import os.path
from datetime import datetime, timedelta


def moments_calc(spec_col, v_doppler):
    """
    

    Parameters
    ----------
    spec_col : TYPE ndarray
        DESCRIPTION. doppler spectra column of a chirp
    v_doppler : TYPE ndarray
        DESCRIPTION. doppler velocity of that chirp

    Returns
    -------
    moments_col column of doppler moments

    """
    dim_height = np.shape(spec_col)[0]
    dim_doppler = np.shape(spec_col)[1]

    
    # calculation of reflectivity 
    ze_col = np.nansum(spec_col, axis=1)
    
    
    # calculation of mean doppler velocity
    vd_col = np.nansum(spec_col * v_doppler, axis=1)/ze_col
    
    
    # calculation of spectral width
    dim_height = spec_col.shape[0]
    num_term = np.zeros((dim_height, len(v_doppler)))
    for ind in range(dim_height):
        num_term[ind] = v_doppler - np.repeat(vd_col[ind], len(v_doppler))
    
    sw_col =  np.sqrt(np.nansum(spec_col * num_term**2, axis=1)/ze_col)
    
    
    # calculation of skewness and kurtosis
    sk_col = np.nansum(spec_col * num_term**3, axis=1) *  1/(ze_col*sw_col**3)
    ku_col = np.nansum(spec_col * num_term**4, axis=1) *  1/(ze_col*sw_col**4)
    
    return(ze_col, vd_col, sw_col, sk_col, ku_col)

def f_string_from_time_stamp(time_sel):
    '''function to derive string from a datetime input values
    input: time_sel (datetime)
    output: time_string (yyyymmdd_hh)

    '''
    # read file time string
    hour = str(time_sel.hour)
    day = str(time_sel.day)
    month = str(time_sel.month)
    year = str(time_sel.year)
    if len(hour) == 1:
        hour= '0'+hour
    time_string = year+month+day+'_'+hour
    return(time_string)

# set day and time at which to do the slice
yy = '2020'
mm = '02'
dd = '10'
time_sel = datetime(2020,1,21,00,0,0)
hh = '00'
# path on local mac
#path = '/Volumes/Extreme SSD/ship_motion_correction_merian/corrected_data/hourly_files_Wband/2020/02/10/'
path = '/Volumes/Extreme SSD/ship_motion_correction_merian/corrected_data/wband_daily_with_DOI/latest/with_DOI/'+yy+'/'+mm+'/'+dd+'/'
filename = path+dd+mm+yy+'_'+hh+'msm94_msm_ZEN_corrected.nc'

# path on ostro machine
#filename = '/work/cacquist/w_band_eurec4a_LWP_corr/'+yy+'/'+mm+'/'+dd+'/10022020_13msm94_msm_ZEN_corrected.nc'
#path_out = '/work/cacquist/w_band_eurec4a_LWP_corr/plots/'
path_out = '/Volumes/Extreme SSD/ship_motion_correction_merian/plots/spectrograms/'
radar_data = xr.open_dataset(filename)
radar_slice = radar_data.sel(time=time_sel, method='nearest')

sk_slice = radar_slice['sk'].values
time_string = f_string_from_time_stamp(time_sel)
spec = radar_slice.spec.values
v = radar_slice.Doppler_velocity.values
range_offsets = radar_slice.range_offsets.values
height = radar_slice.range.values
spec_db= 10*np.log10(spec)

v0 = v[range_offsets[0]:range_offsets[1]-1, :]
v1 = v[range_offsets[1]:range_offsets[2]-1, :]
v2 = v[range_offsets[2]:, :]
print(np.shape(spec))


# removing nans and selecting corresponding spectra matrices

i_good_v0 = ~np.isnan(v0)
i_good_v1 = ~np.isnan(v1)
i_good_v2 = ~np.isnan(v2)


print(np.shape(i_good_v1))
spec_0 = spec_db[0:range_offsets[1]-1, ~np.isnan(v0)]
spec_1 = spec_db[range_offsets[1]-1:range_offsets[2]-1,~np.isnan(v1)]
spec_2 = spec_db[range_offsets[2]-1:, ~np.isnan(v2)]

spec_00 = spec[0:range_offsets[1]-1, ~np.isnan(v0)]
spec_11 = spec[range_offsets[1]-1:range_offsets[2]-1,~np.isnan(v1)]
spec_22 = spec[range_offsets[2]-1:, ~np.isnan(v2)]


v0_plot = v0[~np.isnan(v0)]
v1_plot = v1[~np.isnan(v1)]
v2_plot = v2[~np.isnan(v2)]


# calculating doppler moments from original spectra
ze_col, vd_col, sw_col, sk_col, ku_col = moments_calc(spec_00, v0_plot)

#%%

# shifting spectra of a fixed quantity in doppler DS
DS = 10
# shift in the spectra of the found number of bins and saving in spec_shifted
spec_shifted = np.zeros((148,1024))
for ind in range(148):
    spec_shifted[ind, :] = np.roll(spec_00[ind,:], -DS, axis=0)
    
# recalculate moments for shifted spectra
ze_col_shift, vd_col_shift, sw_col_shift, sk_col_shift, ku_col_shift = moments_calc(spec_shifted, v0_plot)






matplotlib.rc('xtick', labelsize=26)  # sets dimension of ticks in the plots
matplotlib.rc('ytick', labelsize=26)  # sets dimension of ticks in the plots
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,10))


rcParams['font.sans-serif'] = ['Tahoma']
matplotlib.rcParams['savefig.dpi'] = 100
plt.gcf().subplots_adjust(bottom=0.15)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

mesh1 = ax.pcolormesh(v0_plot, height[0:range_offsets[1]-1], spec_0, cmap='jet', rasterized=True)
#mesh2 = ax.pcolormesh(v1_plot, height[range_offsets[1]-1:range_offsets[2]-1], spec_1, cmap='jet', rasterized=True)
#mesh3 = ax.pcolormesh(v2_plot, height[range_offsets[2]-1:], spec_2, cmap='jet', rasterized=True)
ax.set_xlim(-5., 1.)
ax.set_ylim(500., 2000.)
ax.set_xlabel('Doppler velocity [ms$^{-1}$]', fontsize=24)
ax.set_ylabel('Height [m]', fontsize=24)

cbar = fig.colorbar(mesh1, use_gridspec=True)
cbar.set_label(label='Power [dB]',  size=20)

fig.tight_layout()
fig.savefig(path_out+'/sk_test/'+time_string+'_height_spectrogram.png')



fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,10))
rcParams['font.sans-serif'] = ['Tahoma']
matplotlib.rcParams['savefig.dpi'] = 100
plt.gcf().subplots_adjust(bottom=0.15)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

mesh1 = ax.pcolormesh(v0_plot, height[0:range_offsets[1]-1], spec_00, cmap='jet', rasterized=True)
#mesh2 = ax.pcolormesh(v1_plot, height[range_offsets[1]-1:range_offsets[2]-1], spec_11, cmap='jet', rasterized=True)
#mesh3 = ax.pcolormesh(v2_plot, height[range_offsets[2]-1:], spec_22, cmap='jet', rasterized=True)
ax.axhline(height[74], color='blue', linestyle='--', linewidth=1, xmin=v1_plot[0], xmax=v1_plot[-1])
ax.axhline(height[60], color='red', linestyle='--', linewidth=1, xmin=v1_plot[0], xmax=v1_plot[-1])
ax.axhline(height[90], color='green', linestyle='--', linewidth=1, xmin=v1_plot[0], xmax=v1_plot[-1])

ax.set_xlim(-5., 1.)
ax.set_ylim(500., 2000.)
ax.set_xlabel('Doppler velocity [ms$^{-1}$]', fontsize=24)
ax.set_ylabel('Height [m]', fontsize=24)

cbar = fig.colorbar(mesh1, use_gridspec=True)
cbar.set_label(label='Power [mm$^{3}$mm$^{-6}$]',  size=20)

fig.tight_layout()
fig.savefig(path_out+'/sk_test/'+time_string+'_height_spectrogram_linear.png')




fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,10))
rcParams['font.sans-serif'] = ['Tahoma']
matplotlib.rcParams['savefig.dpi'] = 100
plt.gcf().subplots_adjust(bottom=0.15)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.plot(v0_plot, spec[74,:], label=str(height[74]), color='blue')
ax.plot(v0_plot, spec[60,:], label=str(height[60]), color='red')
ax.plot(v0_plot, spec[90,:], label=str(height[90]), color='green')

ax.plot(v0_plot, spec_shifted[74,:], label=str(height[74]), linestyle='--', color='blue')
ax.plot(v0_plot, spec_shifted[60,:], label=str(height[60]), linestyle='--', color='red')
ax.plot(v0_plot, spec_shifted[90,:], label=str(height[90]), linestyle='--', color='green')
fig.tight_layout()
fig.savefig(path_out+'/sk_test/'+time_string+'_spec_examples.png')

#%%
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,10))
rcParams['font.sans-serif'] = ['Tahoma']
matplotlib.rcParams['savefig.dpi'] = 100
plt.gcf().subplots_adjust(bottom=0.15)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.plot(sk_slice, height, label='sk post_processing', color='black')
ax.plot(sk_col, height[0:range_offsets[1]-1], label='sk claudia', color='red', linestyle='--')
ax.plot(sk_col_shift, height[0:range_offsets[1]-1], label='sk claudia shifted', color='green', linewidth=4, linestyle='dotted')


ax.legend(frameon=False)


fig.tight_layout()
fig.savefig(path_out+'/sk_test/'+time_string+'_skewness_col.png')
