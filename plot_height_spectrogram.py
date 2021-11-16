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
from functions_essd import f_closest


#%%

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
time_sel = datetime(2020,2,10,13,5,0)
# path on local mac
#path = '/Volumes/Extreme SSD/ship_motion_correction_merian/corrected_data/hourly_files_Wband/2020/02/10/'
path = '/Volumes/Extreme SSD/ship_motion_correction_merian/corrected_data/wband_daily_with_DOI/latest/with_DOI/2020/02/10/'
filename = path+'10022020_13msm94_msm_ZEN_corrected.nc'

# path on ostro machine
#filename = '/work/cacquist/w_band_eurec4a_LWP_corr/'+yy+'/'+mm+'/'+dd+'/10022020_13msm94_msm_ZEN_corrected.nc'
#path_out = '/work/cacquist/w_band_eurec4a_LWP_corr/plots/'
path_out = '/Volumes/Extreme SSD/ship_motion_correction_merian/plots/spectrograms/'
radar_data = xr.open_dataset(filename)
radar_slice = radar_data.sel(time=time_sel, method='nearest')
time_string = f_string_from_time_stamp(time_sel)
spec = radar_slice.spec.values
v = radar_slice.Doppler_velocity.values
range_offsets = radar_slice.range_offsets.values
height = radar_slice.range.values
spec_db= 10*np.log10(spec)


# reading doppler velocity and spectra for each chirp
v0 = v[range_offsets[0]:range_offsets[1]-1, :]
v1 = v[range_offsets[1]:range_offsets[2]-1, :]
v2 = v[range_offsets[2]:, :]

    
spec_0 = spec_db[range_offsets[0]:range_offsets[1]-1, :]
spec_1 = spec_db[range_offsets[1]-1:range_offsets[2]-1,:]
spec_2 = spec_db[range_offsets[2]-1:,:]


matplotlib.rc('xtick', labelsize=26)  # sets dimension of ticks in the plots
matplotlib.rc('ytick', labelsize=26)  # sets dimension of ticks in the plots
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,10))
rcParams['font.sans-serif'] = ['Tahoma']
matplotlib.rcParams['savefig.dpi'] = 100
plt.gcf().subplots_adjust(bottom=0.15)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

height_matrix = np.array([height[range_offsets[0]:range_offsets[1]-1]] * 512).T

mesh1 = ax.pcolormesh(v0, height_matrix, spec_0, cmap='jet', rasterized=True)
#mesh2 = ax.pcolormesh(v1, height[range_offsets[1]-1:range_offsets[2]-1], spec_1, cmap='jet', rasterized=True)
#mesh3 = ax.pcolormesh(v2, height[range_offsets[2]-1:], spec_2, cmap='jet', rasterized=True)
ax.set_xlim(-10., 1.)
ax.set_ylim(100., 500.)
ax.set_xlabel('Doppler velocity [ms$^{-1}$]', fontsize=24)
ax.set_ylabel('Height [m]', fontsize=24)

cbar = fig.colorbar(mesh1, use_gridspec=True)
cbar.set_label(label='Power [dB]',  size=20)

fig.tight_layout()
fig.savefig(path_out+time_string+'_height_spectrogram.png')


#%%
# plot of single spectras
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,10))
rcParams['font.sans-serif'] = ['Tahoma']
matplotlib.rcParams['savefig.dpi'] = 100
plt.gcf().subplots_adjust(bottom=0.15)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.plot(v0_plot, spec_0[-1,:], label='last spec of the spec_0[-1,:]', color='black', linestyle='dotted', linewidth= 7 )
plt.plot(v1_plot, spec_0[range_offsets[1]-1,i_good_v1], label='range_offsets[1]-1', color='blue')
plt.plot(v1_plot, spec_1[0,:], label='first spec upper chirp spec_1[0,:]', color='red')
plt.plot(v0_plot, spec_0[range_offsets[1]-2,:], label='range_offsets[1]-2', color='green')
plt.plot(v0_plot, spec_0[range_offsets[1]-3,:], label='range_offsets[1]-3', color='orange')

plt.legend(frameon=False)
fig.tight_layout()
fig.savefig(path_out+time_string+'_single_specs.png')


print(spec_0[range_offsets[1]-1,:])
