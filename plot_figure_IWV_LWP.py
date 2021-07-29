#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 10:30:23 2021

@author: claudia
"""

# importing necessary libraries
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import rcParams
import matplotlib
import netCDF4 as nc4
from netCDF4 import Dataset
import glob
import os.path
import pandas as pd
import numpy as np
import xarray as xr
import scipy.integrate as integrate
from datetime import datetime
from datetime import timedelta
from pathlib import Path
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import matplotlib.ticker as ticker

# directory to be set where you save the data of the wetransfer
file_list_lwp = np.sort(glob.glob('/Volumes/Extreme SSD/ship_motion_correction_merian/corrected_data/wband_daily_with_DOI/latest/with_DOI/daily_intake/*.nc'))
#Volumes/Extreme SSD/ship_motion_correction_merian/corrected_data/wband_daily_with_DOI/latest/with_DOI
# set output path where you want to save the plot
path_plot = '/Volumes/Extreme SSD/ship_motion_correction_merian/plots_essd/'

# reading LWP data to list
LWP_list = []
LWP_clear_sky = []
for ind_file, filename in enumerate(file_list_lwp):
    data =  xr.open_dataset(filename)
    print('reading ', filename)
    LWP = data.liquid_water_path.values
    ze = data.radar_reflectivity.values
    
    for ind_val in range(len(LWP)):
        
        # check if the time step is clear sky (Ze > -60.)
        ze_column = ze[ind_val, :]
        

        if len(np.where(ze_column > -60.)[0]) == 0:
            LWP_clear_sky.append(LWP[ind_val])
        else:
            LWP_list.append(LWP[ind_val])

    print('file finished')
    print('**************')
    
#%%
# plotting histogram of LWP values 
hist_lwp, bin_edges_lwp = np.histogram(LWP_list, bins=500, range=[0.1,4000.], density=True)
hist_lwp_clear_sky, bin_edges_lwp_clear_sky = np.histogram(LWP_clear_sky, bins=500,  range=[0.1,4000.], density=True)


var_x = np.random.randint(1,101,500)
var_y = np.random.randint(1,101,500)
var_x_radiosonde = np.random.randint(1,101,20)
var_y_radiosonde = np.random.randint(1,101,20)


print(np.nanmedian(LWP_list))
print(np.nanmedian((LWP_clear_sky)))
print(np.nanstd(LWP_list))
print(np.nanstd((LWP_clear_sky)))

# plotting quicklooks of the values map and the picked time serie interval
labelsizeaxes    = 32
fontSizeTitle    = 32
fontSizeX        = 32
fontSizeY        = 32
cbarAspect       = 10
fontSizeCbar     = 32

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12,12))
matplotlib.rc('xtick', labelsize=32)  # sets dimension of ticks in the plots
matplotlib.rc('ytick', labelsize=32)  # sets dimension of ticks in the plots
ax.text(-0.05, 1.05, 'LWP', fontweight='black', fontsize=26, transform=ax.transAxes)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["bottom"].set_linewidth(2)
ax.spines["left"].set_linewidth(2)
ax.tick_params(which='minor', length=7, width=3)
ax.tick_params(which='major', length=7, width=3)
ax.set_xscale('log')
ax.set_yscale('log')
ax.plot(bin_edges_lwp[:-1],hist_lwp, label='cloudy', color='blue', lw=3)
ax.plot(bin_edges_lwp_clear_sky[:-1], hist_lwp_clear_sky, label='clear sky', color='red',lw=3)
ax.legend(frameon=False, fontsize=32)
ax.set_xlim(5., 1000.)
#ax.set_ylim(10.**(-6), 10.**(-1))

ax.xaxis.grid(True, which='major', linestyle='dotted', color='black')
ax.set_ylabel('Frequency []', fontsize=32)
ax.set_xlabel('Liquid water path [g m$^{-2}$]', fontsize=32)
fig.tight_layout()
fig.savefig(path_plot+'_figure_LWP.png')
#%%
fig, axs = plt.subplots(2,1, figsize=(24,14), constrained_layout=True)

# setting dates formatter 
#[a.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M')) for a in axs[:].flatten()]
matplotlib.rc('xtick', labelsize=32)  # sets dimension of ticks in the plots
matplotlib.rc('ytick', labelsize=32)  # sets dimension of ticks in the plots
grid            = True

#axs[0].plot(time_mrr, profile_mrr, color='white')
axs[0].spines["top"].set_visible(False)
axs[0].spines["right"].set_visible(False)
axs[0].get_xaxis().tick_bottom()
axs[0].get_yaxis().tick_left()
axs[0].set_xscale('log')
axs[0].set_yscale('log')
axs[0].spines["bottom"].set_linewidth(2)
axs[0].spines["left"].set_linewidth(2)

axs[0].plot(bin_edges_lwp[:-1],hist_lwp, label='cloudy', color='blue', lw=3)
axs[0].plot(bin_edges_lwp_clear_sky[:-1], hist_lwp_clear_sky, label='clear sky', color='red',lw=3)
axs[0].legend(frameon=False, fontsize=32)
axs[0].set_xlim(5., 4000.)
#axs[0].set_ylim(ymin_w, ymax_w)
axs[0].tick_params(which='minor', length=7, width=3)
axs[0].tick_params(which='major', length=7, width=3)
axs[0].xaxis.grid(True, which='major', linestyle='dotted', color='black')
#axs[0].xaxis.set_minor_locator(MultipleLocator(500))
axs[0].set_ylabel('Frequency []', fontsize=32)
axs[0].set_xlabel('Liquid water path [g m$^{-2}$]', fontsize=32)

#mrr_cs.Zea.plot(x='time', y='height', cmap=cmap_ze_mrr, vmin=-10., vmax=40.)
axs[1].spines["top"].set_visible(False)
axs[1].spines["right"].set_visible(False)
axs[1].get_xaxis().tick_bottom()
axs[1].get_yaxis().tick_left()
axs[1].spines["bottom"].set_linewidth(2)
axs[1].spines["left"].set_linewidth(2)
axs[1].tick_params(which='minor', length=7, width=3)
axs[1].tick_params(which='major', length=7, width=3)
# add scatter plot of data (blue) as above
axs[1].scatter(var_x, var_y, color='blue', s=70)
axs[1].tick_params(which='minor', length=7, width=3)
axs[1].tick_params(which='major', length=7, width=3)
axs[1].scatter(var_x_radiosonde, var_y_radiosonde, color='red', label='radiosondes', s=150, marker="X")
# add with a different color (red) as above, the scatter plot of the IWV from radiosondes vs their corresponding GNSS value
#axs[1].set_xlim(time_start, time_end)
#axs[1].set_ylim(ymin_mrr, ymax_mrr)
axs[1].set_ylabel('IWV GNSS data (gps) [kg m$^{-2}$]', fontsize=32)
axs[1].set_xlabel('IWV w-band radar [kg m$^{-2}$]', fontsize=32)
#axs[1].xaxis.grid(True, which='minor')
#axs[1].xaxis.set_minor_locator(MultipleLocator(5))
axs[1].legend(frameon=False, fontsize=32)

for ax, l in zip(axs.flatten(), ['a) LWP ', 'b) IWV ']):
    ax.text(-0.05, 1.05, l,  fontweight='black', fontsize=32, transform=ax.transAxes)
fig.savefig(path_plot+'_figure_IWV_LWP.png')


