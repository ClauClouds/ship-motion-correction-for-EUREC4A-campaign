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
from matplotlib import rcParams
import matplotlib
# importing necessary libraries
import os.path
from datetime import datetime, timedelta
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
mm = '01'
dd = '28'
time_sel = datetime(2020,1,28,10,0,0)

#loading daily file full spectra and compressed file
filename_compr = '/net/ostro/cacquist/w_band_eurec4a_LWP_corr/'+yy+'/'+mm+'/'+dd+'/20200128_wband_radar_msm_eurec4a_intake.nc'
filename_full = '/net/ostro/cacquist/w_band_eurec4a_LWP_corr/'+yy+'/'+mm+'/'+dd+'/20200128_wband_radar_msm_eurec4a_full.nc'

path_out = '/net/ostro/cacquist/w_band_eurec4a_LWP_corr/plots/'
radar_data = xr.open_dataset(filename_compr)

date = yy+mm+dd

keyword = 'compr'
if keyword == 'full':
    radar_slice = radar_data.sel(time=time_sel, method='nearest')
    time_string = f_string_from_time_stamp(time_sel)
    spec = radar_slice.doppler_spectrum.values
    v = radar_slice.doppler_velocity.values
    range_offsets = radar_slice.range_offset.values
    height = radar_slice.height.values
    spec_db= 10*np.log10(spec)

    v0 = v[range_offsets[0], :]
    v1 = v[range_offsets[1], :]
    v2 = v[range_offsets[2], :]
    print(range_offsets)
    # removing nans and selecting corresponding spectra matrices

    i_good_v0 = ~np.isnan(v0)
    i_good_v1 = ~np.isnan(v1)
    i_good_v2 = ~np.isnan(v2)


    print(np.shape(i_good_v1))
    spec_0 = spec_db[0:range_offsets[1]-1, ~np.isnan(v0)]
    spec_1 = spec_db[range_offsets[1]-1:range_offsets[2]-1,~np.isnan(v1)]
    spec_2 = spec_db[range_offsets[2]-1:, ~np.isnan(v2)]

    v0_plot = v0[~np.isnan(v0)]
    v1_plot = v1[~np.isnan(v1)]
    v2_plot = v2[~np.isnan(v2)]


    matplotlib.rc('xtick', labelsize=26)  # sets dimension of ticks in the plots
    matplotlib.rc('ytick', labelsize=26)  # sets dimension of ticks in the plots
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,10))
    rcParams['font.sans-serif'] = ['Tahoma']
    matplotlib.rcParams['savefig.dpi'] = 100
    plt.gcf().subplots_adjust(bottom=0.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    mesh1 = ax.pcolormesh(v0_plot, height[0:range_offsets[1]-1], spec_0, cmap='jet', rasterized=True)
    mesh2 = ax.pcolormesh(v1_plot, height[range_offsets[1]-1:range_offsets[2]-1], spec_1, cmap='jet', rasterized=True)
    mesh3 = ax.pcolormesh(v2_plot, height[range_offsets[2]-1:], spec_2, cmap='jet', rasterized=True)
    ax.set_xlim(-5., 1.)
    ax.set_ylim(100., 2500.)
    ax.set_xlabel('Doppler velocity [ms$^{-1}$]', fontsize=24)
    ax.set_ylabel('Height [m]', fontsize=24)

    cbar = fig.colorbar(mesh1, use_gridspec=True)
    cbar.set_label(label='Power [dB]',  size=20)

    fig.tight_layout()
    fig.savefig(path_out+time_string+'_height_spectrogram.png')


from matplotlib.path import Path
import matplotlib.patches as patches
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

# plot reflectivity and mean doppler Doppler_velocity
time = pd.to_datetime(radar_data.time.values)
height = radar_data.height.values
ze = radar_data.radar_reflectivity.values
vm = radar_data.mean_doppler_velocity.values
startTime = time[0]
endTime= time[-1]
grid = True
fontSizeX = 22
Path_out = '/net/ostro/cacquist/w_band_eurec4a_LWP_corr/plots/'
fig, axs = plt.subplots(2, 1, figsize=(20,20), constrained_layout=True)
# setting dates formatter
[a.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M')) for a in axs[:].flatten()]
matplotlib.rc('xtick', labelsize=26)  # sets dimension of ticks in the plots
matplotlib.rc('ytick', labelsize=26)  # sets dimension of ticks in the plots
# build colorbar
mesh = axs[0].pcolormesh(time, height,  ze.T, vmin=-60., vmax=40., cmap='viridis', rasterized=True)
#axs[0].set_title('Reflectivity ', loc='left')
axs[0].spines["top"].set_visible(False)
axs[0].spines["right"].set_visible(False)
axs[0].spines["bottom"].set_linewidth(3)
axs[0].spines["left"].set_linewidth(3)
axs[0].set_xlim(startTime, endTime)
cbar = fig.colorbar(mesh, ax=axs[0], location='right', aspect=10, use_gridspec=grid)
cbar.set_label(label='Ze [dBZ]',  size=fontSizeX)
axs[0].set_ylabel('Height [m]', fontsize=fontSizeX)
axs[0].set_ylim(100., 2500.)
axs[0].xaxis.set_minor_locator(AutoMinorLocator(n=3))
axs[0].tick_params(which='minor', length=7, width=3)
axs[0].tick_params(which='major', length=7, width=3)

mesh = axs[1].pcolormesh(time, height, -vm.T, vmin=-4., vmax=4., cmap='seismic', rasterized=True)

#axs[0].set_title('Reflectivity ', loc='left')
axs[1].spines["top"].set_visible(False)
axs[1].spines["right"].set_visible(False)
axs[1].spines["bottom"].set_linewidth(3)
axs[1].spines["left"].set_linewidth(3)
axs[1].set_xlim(startTime, endTime)
cbar = fig.colorbar(mesh, ax=axs[1], location='right', aspect=10, use_gridspec=grid)
cbar.set_label(label='Vd [ms$^{-1}$]',  size=fontSizeX)
axs[1].set_ylabel('Height [m]', fontsize=fontSizeX)
axs[1].set_ylim(100., 2500.)
axs[1].xaxis.set_minor_locator(AutoMinorLocator(n=3))
axs[1].tick_params(which='minor', length=7, width=3)
axs[1].tick_params(which='major', length=7, width=3)

for ax, l in zip(axs.flatten(), [ '(a) Reflectivity - W-band','(b) Mean Doppler velocity - W-band ']):
    ax.text(-0.05, 1.05, l,  fontweight='black', fontsize=20, transform=ax.transAxes)
fig.savefig(Path_out+date+'_quicklooks_compr.png')
fig.savefig(Path_out+date+'_quicklooks_compr.pdf')
