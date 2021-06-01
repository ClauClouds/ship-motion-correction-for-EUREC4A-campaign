#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 14:40:20 2021
code to test if the doppler spectra  shifted are nans or not
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



def f_closest(array,value):
    '''
    # closest function
    #---------------------------------------------------------------------------------
    # date :  16.10.2017
    # author: Claudia Acquistapace
    # goal: return the index of the element of the input array that in closest to the value provided to the function
    '''
    import numpy as np
    idx = (np.abs(array-value)).argmin()
    return idx  
filetest = '/Users/claudia/Downloads/28012020_10_preprocessedClau_4Albert.nc'
data = xr.open_dataset(filetest)
mask = data.mask.values
spec = data.spec_shifted.values
time = pd.to_datetime(data.time.values)
vDoppler = data.VDoppler.values
height = data.height.values
timesel= datetime(2020,1,28,10,30,0)
i_time_sel = f_closest(time, timesel)

#plot quicklooks of the spectra at given heights where the signal is shifted and original, with printed values of corresponding mean Doppler velocities
HeightSel = [70., 300., 600., 800.]
colorSel = ['red','blue','green','purple']
labelsizeaxes   = 16
fontSizeTitle   = 16
fontSizeX       = 16
fontSizeY       = 16
cbarAspect      = 10
fontSizeCbar    = 16
# selecting time to be plotted :

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12,6))
rcParams['font.sans-serif'] = ['Tahoma']
matplotlib.rcParams['savefig.dpi'] = 100
plt.gcf().subplots_adjust(bottom=0.15)
fig.tight_layout()
ax = plt.subplot(1,1,1)
ax.spines["top"].set_visible(False)  
ax.spines["right"].set_visible(False)  
ax.get_xaxis().tick_bottom()  
ax.get_yaxis().tick_left() 
matplotlib.rc('xtick', labelsize=labelsizeaxes)  # sets dimension of ticks in the plots
matplotlib.rc('ytick', labelsize=labelsizeaxes)  # sets dimension of ticks in the plots
for indH in range(len(HeightSel)):
    ind_closest= f_closest(height, HeightSel[indH])
    ax.plot(-vDoppler, np.log10(spec[i_time_sel, ind_closest, :]), label=str(HeightSel[indH]), color=colorSel[indH])

ax.legend(frameon=False, prop={'size': 12})
#ax.set_title(' day : '+strTitleSignal, fontsize=fontSizeTitle, loc='left')
ax.set_xlabel("VDoppler [$ms^{-1}$]", fontsize=fontSizeX)
ax.set_ylabel("Power [dB]", fontsize=fontSizeY)
ax.set_xlim(-12., 0.)
#ax.set_ylim(10.**(-9), 10.**(-6))
ax.set_ylim(-8.0)
fig.tight_layout()
fig.savefig('/Users/claudia/Downloads/spec_test.png', format='png')    