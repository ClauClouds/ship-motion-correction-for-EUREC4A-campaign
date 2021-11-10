#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 20 16:32:27 2021

@author: claudia
"""

# importing necessary libraries
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import netCDF4 as nc4
from netCDF4 import Dataset
import netCDF4
import glob
import pandas as pd
from datetime import datetime
from datetime import timedelta
#import atmos
import xarray as xr
from functions_essd import f_calculateMomentsCol
from functions_essd import f_readAndMergeRadarDataDay_DopplerCorrection
from functions_essd import f_readAndMergeRadarDataDay
from functions_essd import generate_preprocess
from scipy.interpolate import CubicSpline
import custom_color_palette as ccp
from matplotlib import rcParams
import matplotlib
# importing necessary libraries
from matplotlib import rcParams
import matplotlib
import os.path
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import netCDF4 as nc4
from netCDF4 import Dataset
import glob
import pandas as pd
from datetime import datetime, timedelta
# import atmos
import xarray as xr
from functions_essd import f_calculateMomentsCol
from functions_essd import f_readAndMergeRadarDataDay_DopplerCorrection
from functions_essd import f_readAndMergeRadarDataDay
from functions_essd import generate_preprocess
from scipy.interpolate import CubicSpline
import custom_color_palette as ccp
from matplotlib import rcParams
import matplotlib
from functions_essd import f_closest
import matplotlib.ticker as ticker

def f_defineSingleColorPalette(colors, minVal, maxVal, step):
    """
    author: Claudia Acquistapace
    date : 13 Jan 2020
    goal: define color palette and connected variables associated to a given
    range for a variable
    input:
        colors: list of strings identifying the colors selected (HEX)
        minVal: min value for the color palette
        maxVal: max value to be assigned to the color paletter
        step  : increment for the color palette
    output:
        cmap: personalized color map
        ticks
        norm
        bounds
    """

    Intervals = ccp.range(minVal, maxVal, step)

    # sublist with colors that will be used to create a custom palette
    palette = [colors, Intervals]

    # we pass the parm_color inside a list to the creates_palette module
    cmap, ticks, norm, bounds = ccp.creates_palette([palette])

    return(cmap, ticks, norm, bounds)
def f_defineDoubleColorPalette(colorsLower, colorsUpper, minVal, maxVal, step, thrs):
    """
    author: Claudia Acquistapace
    date : 13 Jan 2020
    goal: define dual color palette (i.e. based on two sets of colors) and its parameters based on one ensemble of colors based on the input parameters
    input:
        lower_palette: list of strings identifying the colors selected (HEX) for the lower part of the palette i.e. Values <Thr
        upper_palette: list of strings identifying the colors selected (HEX) for the upper part of the palette i.e. Values >Thr
        minVal: min value for the color palette
        maxVal: max value to be assigned to the color paletter
        step  : increment for the color palette
        thrs  : threshold value to be used to as separation for the upper and lower color palette
    output:
        cmap: personalized color map
        ticks
        norm
        bounds
    """
    lower_palette = [colorsLower, ccp.range(minVal, thrs, step)] # grigio: 8c8fab
    upper_palette = [colorsUpper, ccp.range(thrs, maxVal, step)] #sk # last old color 987d7b

    # we pass the parm_color inside a list to the creates_palette module
    cmap, ticks, norm, bounds = ccp.creates_palette([lower_palette, upper_palette])

    return(cmap, ticks, norm, bounds)



Path_out = '/Volumes/Extreme SSD/ship_motion_correction_merian/plots_essd/'
path_data = '/Volumes/Extreme SSD/ship_motion_correction_merian/case_studies_paper/'
#mrr_file = '20200128_MRR_PRO_msm_eurec4a.nc'
#w_band_file = '20200128_wband_radar_msm_eurec4a_intake.nc'
mrr_file = '20200212_MRR_PRO_msm_eurec4a.nc'
w_band_file = '20200212_wband_radar_msm_eurec4a_intake.nc'
path_lcl = '/Volumes/Extreme SSD/ship_motion_correction_merian/ship_data/new/'
lcl_file = 'LCL_dataset.nc'


w_band = xr.open_dataset(path_data+w_band_file)


# selecting time intervals for the case study
time_start = datetime(2020,2,12,15,0,0)
time_end = datetime(2020,2,12,17,0,0)
#time_start = datetime(2020,1,28,9,45,0)
#time_end = datetime(2020,1,28,11,15,0)

# slicing the data for plotting the selected time interval
w_band_cs = w_band.sel(time=slice(time_start, time_end))

# removing gaps in time from w band radar data
datetimeRadar = pd.to_datetime(w_band_cs['time'].values)
time_diff = np.ediff1d(datetimeRadar)

#converting time differences to seconds
diff_list = [pd.Timedelta(t).total_seconds() for t in time_diff]

# find where diff_list > 4 seconds:
diff_arr = np.asarray(diff_list)
i_gaps = np.where(diff_arr > 4.)[0][:]

# defining new list of time stamps where we add the new missing times
new_time_arr = datetimeRadar.tolist()
len_added_times = []
# finding how many time stamps have to be introduced.
for i, i_gaps_val in enumerate(i_gaps):
    #print(i, i_gaps_val)
    time_stop = datetimeRadar[i_gaps_val]
    time_restart = datetimeRadar[i_gaps_val+1]

    # calculate number of time stamps to add
    deltaT = diff_arr[i_gaps_val]
    n_times_to_add = deltaT//3

    # calculate time stamps to add in the gaps
    time_to_add = [time_stop+i_n*(timedelta(seconds=3)) for i_n in np.arange(1,n_times_to_add+1)]

    # storing amount of inserted values
    len_added_times.append(len(time_to_add))

    #print('time stamps to add: ', time_to_add)
    # loop on time to add elements for inserting them in the list one by one
    for ind in range(len(time_to_add)):
        # read value to insert
        val_to_insert = time_to_add[ind]

        # find index where to insert
        if i == 0:
            ind_start = i_gaps_val+1
        else:
            ind_start = new_time_arr.index(time_stop)+1
        #print(i_gaps_val, ind_start)
        new_time_arr.insert(ind_start+ind, val_to_insert)


new_time_arr = pd.to_datetime(np.asarray(new_time_arr))
print('gaps found: ', len(i_gaps))
print('dim new time array ', len(new_time_arr))
print('dim old time array ', len(datetimeRadar))
print('******************')

# resampling radar data on new time array
w_band_cs = w_band_cs.reindex({"time":new_time_arr}, method=None)
print('resampling on new axis for time, done. ')


# reading stable table status
# reading ST flag file
ST_flag_file = '/Volumes/Extreme SSD/ship_motion_correction_merian/stable_table_processed_data/stabilization_platform_status_eurec4a.nc'
ST_flag = xr.open_dataset(ST_flag_file)





# reading variables to plot
Vd = w_band_cs['mean_doppler_velocity'].values
rangeRadar = w_band_cs['height'].values
timeLocal = pd.to_datetime(w_band_cs['time'].values)

LWP = w_band_cs['liquid_water_path'].values

Vd[Vd == -999.] = np.nan
#Vd = np.ma.masked_invalid(Vd)
Ze = w_band_cs['radar_reflectivity'].values
Ze[Ze == -999.]    = np.nan
ZeLog = 10.*np.log10(Ze)
#ZeLog = np.ma.masked_invalid(ZeLog)

Sw = w_band_cs['spectral_width'].values
Sw[Sw == -999.]    = np.nan
#Sw = np.ma.masked_invalid(Sw)

Sk = w_band_cs['skewness'].values
Sk[Sk == -999.]    = np.nan
#Sk = np.ma.masked_invalid(Sk)


#reading and slicing lcl data
lcl_data = xr.open_dataset(path_lcl+lcl_file)
lcl_cs = lcl_data.sel(time=slice(time_start, time_end))


# reading and slicing MRR data
mrr = xr.open_dataset(path_data+mrr_file)
mrr_cs = mrr.sel(time=slice(time_start, time_end))


#interpolating data on radar time stamps
mrr_interp = mrr_cs.interp(time=w_band_cs.time.values)
lcl_interp = lcl_cs.interp(time=w_band_cs.time.values)

lcl = lcl_interp.lcl.values
lcl_time = lcl_interp.time.values

# reading MRR variables
RR = mrr_interp.rain_rate.values
Ze_mrr = mrr_interp.Zea.values
range_mrr = mrr_interp.height.values
time_mrr = mrr_interp.time.values
fall_speed_mrr = -mrr_interp.fall_speed.values

#%%
# opening radar file for plotting doppler spectra related to the selected point
#spectra_file = xr.open_dataset('/Volumes/Extreme SSD/ship_motion_correction_merian/corrected_data/2020/01/28/28012020_10msm94_msm_ZEN_corrected.nc')
#time_profile = datetime(2020,1,28,10,24,0)

#spectra_slice = spectra_file.sel(time=slice(time_profile, time_end))
#spectra_slice = spectra_file.sel(time=time_profile, method='nearest')


# reading variables for plotting spectras
#min_vel = spectra_slice.MinVel.values
#range_offsets = spectra_slice.range_offsets.values
#time = spectra_slice.time.values
#chirp_vel = spectra_slice.chirp_vel.values
#DoppMax = spectra_slice.nqv.values
#dopp_len =  spectra_slice.dopp_len.values
#spec = spectra_slice.sze.values
# doing correction for one chirp
#for ixt in range(len(time)):


#    velshift1 = min_vel[ixt, 0:range_offsets[1]]
#    velshift2 = min_vel[ixt, range_offsets[1]:range_offsets[2]]
#    velshift3 = min_vel[ixt, range_offsets[2]:]

    # excluding

    # finding indeces where min_vel is different from the chirp vel ( that's where aliasing occurs)
#    ind1 = np.where((round(chirp_vel[0,1]*10**3) != np.around(velshift1*10**3)) * (~np.isnan(velshift1)))[0]
#    ind2 = np.where((round(chirp_vel[1,1]*10**3) != np.around(velshift2*10**3)) * (~np.isnan(velshift2)))[0]
#    ind3 = np.where((round(chirp_vel[2,1]*10**3) != np.around(velshift3*10**3)) * (~np.isnan(velshift3)))[0]

    # calculating doppler resolution
#    dv1 = 2*np.double(DoppMax[0])/ np.double(dopp_len[0])
#    dv2 = 2*np.double(DoppMax[1])/ np.double(dopp_len[1])
#    dv3 = 2*np.double(DoppMax[2])/ np.double(dopp_len[2])

#    for ind,val in enumerate(ind1):

        # calculating the shift to be applied
#        shift1 = int(np.round((velshift1[val] - chirp_vel[0,1])/dv1))

        # shift in the spectra of the found number of bins and saving in spec_shifted
#        spec[ixt, val, :] = np.roll(spec[ixt,val,:], -shift1)


#    for ind,val in enumerate(ind2):

        # calculating the shift to be applied
#        shift2 = int(np.round((velshift2[val] - chirp_vel[1,1])/dv2))

        # shift in the spectra of the found number of bins and saving in spec_shifted
#        spec[ixt, val, :] = np.roll(spec[ixt,val,:], -shift2)

#    for ind,val in enumerate(ind3):

        # calculating the shift to be applied
#        shift3 = int(np.round((velshift3[val] - chirp_vel[2,1])/dv3))

        # shift in the spectra of the found number of bins and saving in spec_shifted
#        spec[ixt, val, :] = np.roll(spec[ixt,val,:], -shift3)


#height =spectra_slice.range.values
#v1 = chirp_vel[0,:]
#v2 = chirp_vel[1,:]
#v3 = chirp_vel[2,:]
#i_good_chirp2 = ~np.isnan(v2)
#i_good_chirp3 = ~np.isnan(v3)

from functions_essd import f_closest
#spec_db= 10*np.log10(spec)
#matplotlib.rc('xtick', labelsize=26)  # sets dimension of ticks in the plots
#matplotlib.rc('ytick', labelsize=26)  # sets dimension of ticks in the plots
#fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,12))
#rcParams['font.sans-serif'] = ['Tahoma']
#matplotlib.rcParams['savefig.dpi'] = 100
#plt.gcf().subplots_adjust(bottom=0.15)
#ax.spines["top"].set_visible(False)
#ax.spines["right"].set_visible(False)
#ax.get_xaxis().tick_bottom()
#ax.get_yaxis().tick_left()
#mesh1 = ax.pcolormesh(v1, height[0:range_offsets[1]], spec_db[0, 0:range_offsets[1], :], cmap='jet', rasterized=True)
#mesh2 = ax.pcolormesh(v2[i_good_chirp2], height[range_offsets[1]:range_offsets[2]], spec_db[0, range_offsets[1]:range_offsets[2], i_good_chirp2].T, cmap='jet', rasterized=True)
#mesh3 = ax.pcolormesh(v3[i_good_chirp3], height[range_offsets[2]:], spec_db[0, range_offsets[2]:,i_good_chirp3].T, cmap='jet', rasterized=True)
#ax.set_xlim(-10., 1.)
#ax.set_ylim(100., 2500.)
#cbar = fig.colorbar(mesh1, use_gridspec=True)
#cbar.set_label(label='Power [dB]',  size=20)
#fig.tight_layout()
#fig.savefig(Path_out+'_height_spectrogram_1031.png')

#%%

labelsizeaxes   = 26
fontSizeTitle   = 26
fontSizeX       = 26
fontSizeY       = 26
cbarAspect      = 26
fontSizeCbar    = 26

# setting y range limits for the plots
ymin_w = 100.
ymax_w = 2200.
ymin_mrr = 50.
ymax_mrr = 2200.


# settings for w band radar reflectivity
mincm_ze = -40.
maxcm_ze = 30.
step_ze = 0.1
colors_ze =['#4f8c9d', '#1c4c5e', '#8ae1f9', '#8f0f1b', '#e0bfb4', '#754643', '#ef7e58', '#ff1c5d']
cmap_ze, ticks_ze, norm_ze, bounds_ze =  f_defineSingleColorPalette(colors_ze, mincm_ze, maxcm_ze, step_ze)

# settings for mean Doppler velocity
mincm_vd = -6.
maxcm_vd = 2.
step_vd = 0.1
thrs_vd = 0.
colorsLower_vd = ["#4553c2", "#2b19d9", "#d0d2f0", "#42455e", "#66abf9"]# grigio: 8c8fab
colorsUpper_vd = ["#fd2c3b", "#59413f", "#fdc7cc", "#8f323c", "#e66c4c", "#ae8788"] #MDV
cmap_vd, ticks_vd, norm_vd, bounds_vd =  f_defineDoubleColorPalette(colorsLower_vd, colorsUpper_vd, mincm_vd, maxcm_vd, step_vd, thrs_vd)
cbarstr = 'Vd [$ms^{-1}$]'

# settings for Spectral width
mincm_sw = 0.
maxcm_sw = 0.8
step_sw = 0.01
#colors = ["#72e5ef", "#460942", "#4dc172", "#cd71b5", "#274c56", "#91ec17", "#b00bd9", "#abc177"]
#colors_sw = [ "#e6bcbe", "#51443f", "#f17a74", "#e81659", "#cdd2df", "#124ac1"]
colors_sw =["#0cc0aa", "#4e2da6", "#74de58", "#bf11af", "#50942f", "#2a5fa0", "#e9b4f5", "#0b522e", "#95bbef", "#8a2f6b"]
cmap_sw, ticks_sw, norm_sw, bounds_sw =  f_defineSingleColorPalette(colors_sw, mincm_sw, maxcm_sw, step_sw)


# settings for skewness
mincm_sk = -2.
maxcm_sk = 2.
step_sk = 0.1
thrs_sk = 0.
colorsLower_sk = ["#4553c2", "#2b19d9", "#d0d2f0", "#42455e", "#66abf9"]# grigio: 8c8fab
colorsUpper_sk = ["#fd2c3b", "#59413f", "#fdc7cc", "#8f323c", "#e66c4c", "#ae8788"] #MDV
cmap_sk, ticks_sk, norm_sk, bounds_sk =  f_defineDoubleColorPalette(colorsLower_vd, colorsUpper_vd, mincm_vd, maxcm_vd, step_vd, thrs_vd)
cbarstr = 'Sk []'



#settings for MRR-pro reflectivity
mincm_ze_mrr = -10.
maxcm_ze_mrr = 40.
step_ze_mrr = 1.
#colors_ze_mrr =['#4f8c9d', '#1c4c5e', '#8ae1f9', '#8f0f1b', '#e0bfb4', '#754643', '#ef7e58', '#ff1c5d']
colors_ze_mrr = ["#56ebd3", "#9e4302", "#2af464", "#d6061a", "#1fa198"]
cmap_ze_mrr, ticks_ze_mrr, norm_ze_mrr, bounds_ze_mrr =  f_defineSingleColorPalette(colors_ze_mrr, mincm_ze_mrr, maxcm_ze_mrr, step_ze_mrr)


# settings for rain fall speed
mincm_vd_mrr = -12.
maxcm_vd_mrr = 0.
step_vd_mrr = 0.1
thrs_vd_mrr = 0.
#colorsLower_vd_mrr = ["#4553c2", "#2b19d9", "#d0d2f0", "#42455e", "#66abf9"]# grigio: 8c8fab
colorsLower_vd_mrr = ["#208eb7", "#cddb9b", "#6146ca", "#a0e85b", "#3f5aa8"]
colorsUpper_vd_mrr = ["#fd2c3b", "#59413f", "#fdc7cc", "#8f323c", "#e66c4c", "#ae8788"] #MDV

cmap_vd_mrr, ticks_vd_mrr, norm_vd_mrr, bounds_vd_mrr =  f_defineSingleColorPalette(colorsLower_vd_mrr, mincm_vd_mrr, maxcm_vd_mrr, step_vd_mrr)
#f_defineDoubleColorPalette(colorsLower_vd_mrr, colorsUpper_vd_mrr, mincm_vd_mrr, maxcm_vd_mrr, step_vd_mrr, thrs_vd_mrr)


# settings for rain rate
mincm_rr = 0.
maxcm_rr = 1.
step_rr = 0.01
colors_rr =['#4f8c9d', '#1c4c5e', '#8ae1f9', '#8f0f1b', '#e0bfb4', '#754643', '#ef7e58', '#ff1c5d']
cmap_rr, ticks_rr, norm_rr, bounds_rr =  f_defineSingleColorPalette(colors_rr, mincm_rr, maxcm_rr, step_rr)



profile_mrr = []
# extracting profile or MRR ze field
for ind_time in range(len(time_mrr)):
    ranges = range_mrr[np.where(~np.isnan(Ze_mrr[ind_time, :]))[0]]
    if len(ranges) > 0:
        profile_mrr.append(np.nanmax(ranges))
    else:
        profile_mrr.append(0.)

# calculating running mean on the profile values
N = 3
profile_mrr_rm = pd.Series(profile_mrr).rolling(window=N).mean().iloc[N-1:].values
plt.plot(timeLocal[N-1:], pd.Series(profile_mrr).rolling(window=N).mean().iloc[N-1:].values)

#%%
# Create rectangle x coordinates
startTime = datetime(2020,2,12,16,20,0)
endTime = startTime + timedelta(minutes = 15)



# generate color array
ST_flag_hour = ST_flag.sel(time=slice(startTime, endTime))
ST_interp_Wband = ST_flag_hour.interp(time=w_band_cs.time.values)

color_arr = np.repeat('black', len(ST_interp_Wband.time.values))
flag = ST_interp_Wband['flag_table_working'].values
color_arr[np.where(flag == 1)] = 'red'



# convert to matplotlib date representation
start = mdates.date2num(startTime)
end = mdates.date2num(endTime)
width = end - start

verts1 = [
   (start, 0.),  # left, bottom
   (start, 2200.),  # left, top
   (end, 2200.),  # right, top
   (end, 0.),  # right, bottom
   (0., 0.),  # ignored
]
verts2 = [
   (start, 0.),  # left, bottom
   (start, 1000.),  # left, top
   (end, 1000.),  # right, top
   (end, 0.),  # right, bottom
   (0., 0.),  # ignored
]

codes = [
    matplotlib.path.Path.MOVETO,
    matplotlib.path.Path.LINETO,
    matplotlib.path.Path.LINETO,
    matplotlib.path.Path.LINETO,
    matplotlib.path.Path.CLOSEPOLY,
]

path1 = matplotlib.path.Path(verts1, codes)
path2 = matplotlib.path.Path(verts2, codes)



# settings for w band radar reflectivity
mincm_ze = -5.
maxcm_ze = 20.
step_ze = 0.1
colors_ze =['#4f8c9d', '#1c4c5e', '#8ae1f9', '#8f0f1b', '#e0bfb4', '#754643', '#ef7e58', '#ff1c5d']
cmap_ze, ticks_ze, norm_ze, bounds_ze =  f_defineSingleColorPalette(colors_ze, mincm_ze, maxcm_ze, step_ze)

# settings for mean Doppler velocity
mincm_vd = -8.
maxcm_vd = 2.
step_vd = 0.1
thrs_vd = 0.
colorsLower_vd = ["#4553c2", "#2b19d9", "#d0d2f0", "#42455e", "#66abf9"]# grigio: 8c8fab
colorsUpper_vd = ["#fd2c3b", "#59413f", "#fdc7cc", "#8f323c", "#e66c4c", "#ae8788"] #MDV
cmap_vd, ticks_vd, norm_vd, bounds_vd =  f_defineDoubleColorPalette(colorsLower_vd, colorsUpper_vd, mincm_vd, maxcm_vd, step_vd, thrs_vd)
cbarstr = 'Vd [$ms^{-1}$]'

# settings for Spectral width
mincm_sw = 0.
maxcm_sw = 1.
step_sw = 0.01
#colors = ["#72e5ef", "#460942", "#4dc172", "#cd71b5", "#274c56", "#91ec17", "#b00bd9", "#abc177"]
#colors_sw = [ "#e6bcbe", "#51443f", "#f17a74", "#e81659", "#cdd2df", "#124ac1"]
colors_sw =["#0cc0aa", "#4e2da6", "#74de58", "#bf11af", "#50942f", "#2a5fa0", "#e9b4f5", "#0b522e", "#95bbef", "#8a2f6b"]
cmap_sw, ticks_sw, norm_sw, bounds_sw =  f_defineSingleColorPalette(colors_sw, mincm_sw, maxcm_sw, step_sw)

# settings for skewness
mincm_sk = -2.
maxcm_sk = 2.
step_sk = 0.1
thrs_sk = 0.
colorsLower_sk = ["#4553c2", "#2b19d9", "#d0d2f0", "#42455e", "#66abf9"]# grigio: 8c8fab
colorsUpper_sk = ["#fd2c3b", "#59413f", "#fdc7cc", "#8f323c", "#e66c4c", "#ae8788"] #MDV
cmap_sk, ticks_sk, norm_sk, bounds_sk =  f_defineDoubleColorPalette(colorsLower_sk, colorsUpper_sk, mincm_sk, maxcm_sk, step_sk, thrs_sk)
cbarstr = 'Sk []'

# settings for MRR variables
RR = mrr_cs.liquid_water_content.values
Ze = mrr_cs.Ze.values
range_mrr = mrr_cs.height.values
time_mrr = mrr_cs.time.values
fall_speed = mrr_cs.fall_speed.values
grid            = True

fontSizeX = 26
mincm_ze_mrr = 0.
maxcm_ze_mrr = 40.
step_ze_mrr = 0.1
colors_ze_mrr =['#4f8c9d', '#1c4c5e', '#8ae1f9', '#8f0f1b', '#e0bfb4', '#754643', '#ef7e58', '#ff1c5d']
cmap_ze_mrr, ticks_ze_mrr, norm_ze_mrr, bounds_ze_mrr =  f_defineSingleColorPalette(colors_ze_mrr, mincm_ze_mrr, maxcm_ze_mrr, step_ze_mrr)

# settings for mean Doppler velocity
mincm_vd_mrr = -8.
maxcm_vd_mrr = 0.
step_vd_mrr = 0.1
thrs_vd_mrr = 0.
colorsLower_vd_mrr = ["#4553c2", "#2b19d9", "#d0d2f0", "#42455e", "#66abf9"]# grigio: 8c8fab
colorsUpper_vd_mrr = ["#fd2c3b", "#59413f", "#fdc7cc", "#8f323c", "#e66c4c", "#ae8788"] #MDV
#cmap_vd, ticks_vd, norm_vd, bounds_vd =  f_defineDoubleColorPalette(colorsLower_vd, colorsUpper_vd, mincm_vd, maxcm_vd, step_vd, thrs_vd)
cmap_vd_mrr, ticks_vd_mrr, norm_vd_mrr, bounds_vd_mrr =f_defineSingleColorPalette(colorsLower_vd_mrr, mincm_vd_mrr, maxcm_vd_mrr, step_vd_mrr)

mincm_rr = 0.
maxcm_rr = 1.
step_rr = 0.01
colors_rr =['#4f8c9d', '#1c4c5e', '#8ae1f9', '#8f0f1b', '#e0bfb4', '#754643', '#ef7e58', '#ff1c5d']
cmap_rr, ticks_rr, norm_rr, bounds_rr =  f_defineSingleColorPalette(colors_rr, mincm_rr, maxcm_rr, step_rr)



# composite figure for Wband and MRR data together
fig, axs = plt.subplots(4, 2, figsize=(24,20), sharex=True, constrained_layout=True)
grid = True
matplotlib.rc('xtick', labelsize=36)  # sets dimension of ticks in the plots
matplotlib.rc('ytick', labelsize=36) # sets dimension of ticks in the plots
  # sets dimension of ticks in the plots
# plotting W-band radar variables and color bars
mesh = axs[0,0].pcolormesh(timeLocal, rangeRadar, ZeLog.T, vmin=maxcm_ze, vmax=mincm_ze, cmap=cmap_ze, rasterized=True)
cbar = fig.colorbar(mesh, ax=axs[0,0], location='right', aspect=10, use_gridspec=grid)
cbar.set_label(label='Ze [dBZ]',  size=26)
mesh = axs[1,0].pcolormesh(timeLocal, rangeRadar, Vd.T, vmin=mincm_vd, vmax=maxcm_vd, cmap=cmap_vd, rasterized=True)
cbar = fig.colorbar(mesh, ax=axs[1,0], location='right', aspect=10, use_gridspec=grid)
cbar.set_label(label='Vd [ms$^{-1}$]',  size=26)
mesh = axs[2,0].pcolormesh(timeLocal, rangeRadar, Sw.T, vmin=mincm_sw, vmax=maxcm_sw, cmap=cmap_sw, rasterized=True)
cbar = fig.colorbar(mesh, ax=axs[2,0], location='right', aspect=10, use_gridspec=grid)
cbar.set_label(label='Spectral width [ms$^{-1}$]',  size=26)
mesh = axs[3,0].pcolormesh(timeLocal, rangeRadar, -Sk.T, vmin=mincm_sk, vmax=maxcm_sk, cmap=cmap_sk, rasterized=True)
cbar = fig.colorbar(mesh, ax=axs[3,0], location='right', aspect=10, use_gridspec=grid)
cbar.set_label(label='Skewness []',  size=26)
axs[3,0].set_xlabel('Time UTC [hh:mm]', fontsize=fontSizeX)

for ax, l in zip(axs[:,0].flatten(), ['(a) Reflectivity - W-band',  '(b) Mean Doppler velocity - W-band ', '(c) Spectral width - W-band', '(d) Skewness - W-band']):
    ax.text(-0.05, 1.05, l,  fontweight='black', fontsize=20, transform=ax.transAxes)
    ax.set_xlim(startTime, endTime)
    ax.set_ylabel('Height [m]', fontsize=fontSizeX)
    ax.set_ylim(100., 2200.)
    ax.plot(lcl_time, lcl, color='black', linestyle='dotted', linewidth=3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(3)
    ax.spines["left"].set_linewidth(3)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(n=3))
    ax.tick_params(which='minor', length=7, width=3)
    ax.tick_params(which='major', length=7, width=3)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(n=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.tick_params(axis='both', labelsize=26)


mesh = axs[0,1].pcolormesh(time_mrr, range_mrr,  Ze.T, vmin=mincm_ze_mrr, vmax=maxcm_ze_mrr, cmap=cmap_ze_mrr, rasterized=True)
cbar = fig.colorbar(mesh, ax=axs[0,1], location='right', aspect=10, use_gridspec=grid)
cbar.set_label(label='Ze [dBZ]',  size=fontSizeX)
mesh = axs[1,1].pcolormesh(time_mrr, range_mrr, -fall_speed.T, vmin=-8., vmax=0., cmap=cmap_vd_mrr, rasterized=True)
cbar = fig.colorbar(mesh, ax=axs[1,1], location='right', aspect=10, use_gridspec=grid)
cbar.set_label(label='Vd [ms$^{-1}$]',  size=fontSizeX)
mesh = axs[2,1].pcolormesh(time_mrr, range_mrr, RR.T, vmin=mincm_rr, vmax=maxcm_rr, cmap='viridis', rasterized=True)
cbar = fig.colorbar(mesh, ax=axs[2,1], location='right', aspect=10, use_gridspec=grid)
cbar.set_label(label='RR [mmh$^{-1}$]',  size=fontSizeX)
mesh = axs[3,1].scatter(timeLocal, LWP, c=color_arr, vmin=0., vmax=1000.)
#axs[2,1].text(0.1, 0.5, 'Begin text', horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes)
axs[3,1].plot([datetime(2020,2,12,16,31,45)], [2000.], 'o', color='black', markersize=10)
axs[3,1].plot([datetime(2020,2,12,16,31,45)], [1800.], 'o', color='red', markersize=10)
axs[3,1].text(0.83, 0.8, 'stab. ON', verticalalignment='center', transform=axs[3,1].transAxes, fontsize=20) #horizontalalignment='center',
axs[3,1].text(0.83, 0.72, 'stab. OFF',verticalalignment='center', transform=axs[3,1].transAxes, fontsize=20) # horizontalalignment='center',
#axs[2,1].text(x=datetime(2020,2,12,16,31,0), y=2000.,s='Stab. platform ON', fontsize=26)

count=0

for ax, l in zip(axs[:,1].flatten(), ['(e) Reflectivity - MRR-PRO',  '(f) Mean Doppler velocity - MRR-PRO', '(g) Spectral width - MRR-PRO', '(h) Liquid water path - W-band']):
    ax.text(-0.05, 1.05, l,  fontweight='black', fontsize=20, transform=ax.transAxes)
    ax.set_xlim(startTime, endTime)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(3)
    ax.spines["left"].set_linewidth(3)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(n=3))
    ax.tick_params(which='minor', length=7, width=3)
    ax.tick_params(which='major', length=7, width=3)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(n=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.tick_params(axis='both', labelsize=26)

    if (count <= 2):
        ax.set_ylabel('Height [m]', fontsize=fontSizeX)
        ax.set_ylim(100., 1200.)
        ax.plot(lcl_time, lcl, color='black', linestyle='dotted', linewidth=3)
    else:
        ax.set_ylabel('LWP [gm$^{-2}$]', fontsize=fontSizeX)
        ax.set_ylim(0., 2500.)
        ax.set_xlabel('Time UTC [hh:mm]', fontsize=fontSizeX)

    count=count+1

fig.savefig(Path_out+'Figure_paper_detail_all.png')

##############################################################################################################################################

# Plot overview plot with rectagle showing area that is enlarged in the next plot
#rect = mpl.patches.Rectangle((start, 0), width, 2000., linewidth=4, edgecolor='yellow', facecolor='none')
patch1 = matplotlib.patches.PathPatch(path1, facecolor='none', edgecolor='red', linewidth=4, linestyle=':')
patch2 = matplotlib.patches.PathPatch(path2, facecolor='none', edgecolor='red', linewidth=4, linestyle=':')
fig, axs = plt.subplots(2, 1, figsize=(20,14), constrained_layout=True)

# setting dates formatter
[a.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M')) for a in axs[:].flatten()]
matplotlib.rc('xtick', labelsize=36)  # sets dimension of ticks in the plots
matplotlib.rc('ytick', labelsize=36)  # sets dimension of ticks in the plots
grid            = True
mesh = axs[0].pcolormesh(timeLocal, rangeRadar,  ZeLog.T, vmin=maxcm_ze, vmax=mincm_ze, cmap=cmap_ze, rasterized=True)
axs[0].plot(timeLocal[N-1:], pd.Series(profile_mrr).rolling(window=N).mean().iloc[N-1:].values, color='black', linestyle='dotted', linewidth=4, label='MRR highest signal')
axs[0].plot(timeLocal, lcl, color='black', label='Lifting condensation level')
#axs[0].plot(time_mrr, profile_mrr, color='white')
axs[0].spines["top"].set_visible(False)
axs[0].spines["right"].set_visible(False)
axs[0].get_xaxis().tick_bottom()
axs[0].get_yaxis().tick_left()
axs[0].set_xlim(time_start, time_end)
axs[0].set_ylim(ymin_w, ymax_w)
axs[0].add_patch(patch1)
axs[0].xaxis.grid(True, which='minor')
axs[0].xaxis.set_minor_locator(ticker.MultipleLocator(5))

cbar = fig.colorbar(mesh, ax=axs[0], location='right', aspect=10, use_gridspec=grid)
cbar.set_label(label='Ze [dBZ]',  size=36)
axs[0].set_ylabel('Height [m]', fontsize=36)

#mrr_cs.Zea.plot(x='time', y='height', cmap=cmap_ze_mrr, vmin=-10., vmax=40.)
mesh = axs[1].pcolormesh(timeLocal, range_mrr, mrr_interp.Ze.values.T, vmin=mincm_ze_mrr, vmax=maxcm_ze_mrr, cmap='viridis', rasterized=True)
axs[1].plot(lcl_time, lcl, color='black', label='Lifting condensation level (LCL)')
axs[1].spines["top"].set_visible(False)
axs[1].spines["right"].set_visible(False)
axs[1].get_xaxis().tick_bottom()
axs[1].get_yaxis().tick_left()
axs[1].set_xlim(time_start, time_end)
axs[1].set_ylim(ymin_mrr, ymax_mrr)
cbar = fig.colorbar(mesh, ax=axs[1], location='right', aspect=10, use_gridspec=grid)
cbar.set_label(label='Ze - MRR [dBZ]',  size=36)
axs[1].set_ylabel('Height [m]', fontsize=36)
axs[1].set_xlabel('Time UTC [hh:mm]', fontsize=36)
axs[1].add_patch(patch2)
axs[1].legend(frameon=False, fontsize=36)
axs[1].xaxis.grid(True, which='minor')
axs[1].xaxis.set_minor_locator(ticker.MultipleLocator(5))
for ax, l in zip(axs.flatten(), ['a) Reflectivity - Wband', 'b) Reflectivity - MRR-PRO']):
    ax.text(-0.05, 1.05, l,  fontweight='black', fontsize=36, transform=ax.transAxes)
fig.savefig(Path_out+'_case_overview_both_instr.png')
#fig.savefig(Path_out+'_case_overview_both_instr.pdf')
#%%
##############################################################################################################################################
strasukamelo












# figure for MRR data obly
fig, axs = plt.subplots(3, 1, figsize=(20,20), constrained_layout=True)
# setting dates formatter
[a.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M')) for a in axs[:].flatten()]
matplotlib.rc('xtick', labelsize=26)  # sets dimension of ticks in the plots
matplotlib.rc('ytick', labelsize=26)  # sets dimension of ticks in the plots
# build colorbar
mesh = axs[0].pcolormesh(time_mrr, range_mrr,  Ze.T, vmin=mincm_ze_mrr, vmax=maxcm_ze_mrr, cmap=cmap_ze_mrr, rasterized=True)
#axs[0].set_title('Reflectivity ', loc='left')
axs[0].spines["top"].set_visible(False)
axs[0].spines["right"].set_visible(False)
axs[0].spines["bottom"].set_linewidth(3)
axs[0].spines["left"].set_linewidth(3)
axs[0].set_xlim(startTime, endTime)
cbar = fig.colorbar(mesh, ax=axs[0], location='right', aspect=10, use_gridspec=grid)
cbar.set_label(label='Ze [dBZ]',  size=fontSizeX)
axs[0].set_ylabel('Height [m]', fontsize=fontSizeX)
axs[0].set_ylim(100., 1000.)
axs[0].xaxis.set_minor_locator(AutoMinorLocator(n=3))
axs[0].tick_params(which='minor', length=7, width=3)
axs[0].tick_params(which='major', length=7, width=3)

mesh = axs[1].pcolormesh(time_mrr, range_mrr, -fall_speed.T, vmin=-8., vmax=0., cmap=cmap_vd_mrr, rasterized=True)

#axs[0].set_title('Reflectivity ', loc='left')
axs[1].spines["top"].set_visible(False)
axs[1].spines["right"].set_visible(False)
axs[1].spines["bottom"].set_linewidth(3)
axs[1].spines["left"].set_linewidth(3)
axs[1].set_xlim(startTime, endTime)
cbar = fig.colorbar(mesh, ax=axs[1], location='right', aspect=10, use_gridspec=grid)
cbar.set_label(label='Vd [ms$^{-1}$]',  size=fontSizeX)
axs[1].set_ylabel('Height [m]', fontsize=fontSizeX)
axs[1].set_ylim(100., 1000.)
axs[1].xaxis.set_minor_locator(AutoMinorLocator(n=3))
axs[1].tick_params(which='minor', length=7, width=3)
axs[1].tick_params(which='major', length=7, width=3)


mesh = axs[2].pcolormesh(time_mrr, range_mrr, RR.T, vmin=mincm_rr, vmax=maxcm_rr, cmap='viridis', rasterized=True)

#axs[0].set_title('Reflectivity ', loc='left')
axs[2].spines["top"].set_visible(False)
axs[2].spines["right"].set_visible(False)
axs[2].spines["bottom"].set_linewidth(3)
axs[2].spines["left"].set_linewidth(3)
axs[2].xaxis.set_minor_locator(AutoMinorLocator(n=3))
axs[2].tick_params(which='minor', length=7, width=3)
axs[2].tick_params(which='major', length=7, width=3)
axs[2].set_xlim(startTime, endTime)
cbar = fig.colorbar(mesh, ax=axs[2], location='right', aspect=10, use_gridspec=grid)
cbar.set_label(label='RR [mmh$^{-1}$]',  size=fontSizeX)
axs[2].set_ylabel('Height [m]', fontsize=fontSizeX)
axs[2].set_ylim(100., 1000.)
axs[2].set_xlabel('Time UTC [hh:mm]', fontsize=fontSizeX)


fig.savefig(Path_out+'_figure_paper_detail_MRR.png')
fig.savefig(Path_out+'_figure_paper_detail_MRR.pdf')



# figure for Wband radar data only
grid = True
fontSizeX = 26

fig, axs = plt.subplots(4, 1, figsize=(20,20), constrained_layout=True)
# setting dates formatter
[a.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M')) for a in axs[:].flatten()]
matplotlib.rc('xtick', labelsize=26)  # sets dimension of ticks in the plots
matplotlib.rc('ytick', labelsize=26)  # sets dimension of ticks in the plots
# build colorbar
mesh = axs[0].pcolormesh(timeLocal, rangeRadar,  ZeLog.T, vmin=maxcm_ze, vmax=mincm_ze, cmap=cmap_ze, rasterized=True)

#axs[0].set_title('Reflectivity ', loc='left')
axs[0].spines["top"].set_visible(False)
axs[0].spines["right"].set_visible(False)
axs[0].spines["bottom"].set_linewidth(3)
axs[0].spines["left"].set_linewidth(3)
axs[0].xaxis.set_minor_locator(AutoMinorLocator(n=3))
axs[0].tick_params(which='minor', length=7, width=3)
axs[0].tick_params(which='major', length=7, width=3)
axs[0].set_xlim(startTime, endTime)
cbar = fig.colorbar(mesh, ax=axs[0], location='right', aspect=10, use_gridspec=grid)
cbar.set_label(label='Ze [dBZ]',  size=26)
axs[0].set_ylabel('Height [m]', fontsize=fontSizeX)
axs[0].set_ylim(100., 2200.)
axs[0].plot(lcl_time, lcl, color='black', linestyle='dotted', linewidth=3, label='Lifting condensation level (LCL)')
axs[0].axvline(x=time_profile, color='black', linewidth=4, linestyle=':')
axs[0].legend()

mesh = axs[1].pcolormesh(timeLocal, rangeRadar,  Vd.T, vmin=mincm_vd, vmax=maxcm_vd, cmap=cmap_vd, rasterized=True)

#axs[0].set_title('Reflectivity ', loc='left')
axs[1].spines["top"].set_visible(False)
axs[1].spines["right"].set_visible(False)
axs[1].spines["bottom"].set_linewidth(3)
axs[1].spines["left"].set_linewidth(3)
axs[1].xaxis.set_minor_locator(AutoMinorLocator(n=3))
axs[1].tick_params(which='minor', length=7, width=3)
axs[1].tick_params(which='major', length=7, width=3)
axs[1].set_xlim(startTime, endTime)
cbar = fig.colorbar(mesh, ax=axs[1], location='right', aspect=10, use_gridspec=grid)
cbar.set_label(label='Vd [ms$^{-1}$]',  size=26)
axs[1].set_ylabel('Height [m]', fontsize=fontSizeX)
axs[1].set_ylim(100., 2200.)
axs[1].plot(lcl_time, lcl, color='black', linestyle='dotted', linewidth=3)
axs[1].axvline(x=time_profile, color='black', linewidth=4, linestyle=':')


mesh = axs[2].pcolormesh(timeLocal, rangeRadar, Sw.T, vmin=mincm_sw, vmax=maxcm_sw, cmap=cmap_sw, rasterized=True)

#axs[0].set_title('Reflectivity ', loc='left')
axs[2].spines["top"].set_visible(False)
axs[2].spines["right"].set_visible(False)
axs[2].spines["bottom"].set_linewidth(3)
axs[2].spines["left"].set_linewidth(3)
axs[2].xaxis.set_minor_locator(AutoMinorLocator(n=3))
axs[2].tick_params(which='minor', length=7, width=3)
axs[2].tick_params(which='major', length=7, width=3)
axs[2].set_xlim(startTime, endTime)
cbar = fig.colorbar(mesh, ax=axs[2], location='right', aspect=10, use_gridspec=grid)
cbar.set_label(label='Spectral width [ms$^{-1}$]',  size=26)
axs[2].set_ylabel('Height [m]', fontsize=fontSizeX)
axs[2].set_ylim(100., 2200.)
axs[2].plot(lcl_time, lcl, color='black', linestyle='dotted', linewidth=3)
axs[2].axvline(x=time_profile, color='black', linewidth=4, linestyle=':')



mesh = axs[3].pcolormesh(timeLocal, rangeRadar, -Sk.T, vmin=mincm_sk, vmax=maxcm_sk, cmap=cmap_sk, rasterized=True)

#axs[0].set_title('Reflectivity ', loc='left')
axs[3].spines["top"].set_visible(False)
axs[3].spines["right"].set_visible(False)
axs[3].spines["bottom"].set_linewidth(3)
axs[3].spines["left"].set_linewidth(3)
axs[3].xaxis.set_minor_locator(AutoMinorLocator(n=3))
axs[3].tick_params(which='minor', length=7, width=3)
axs[3].tick_params(which='major', length=7, width=3)
axs[3].set_xlim(startTime, endTime)
cbar = fig.colorbar(mesh, ax=axs[3], location='right', aspect=10, use_gridspec=grid)
cbar.set_label(label='Skewness []',  size=26)
axs[3].set_ylabel('Height [m]', fontsize=fontSizeX)
axs[3].set_ylim(100., 2200.)
axs[3].set_xlabel('Time UTC [hh:mm]', fontsize=fontSizeX)
axs[3].plot(lcl_time, lcl, color='black', linestyle='dotted', linewidth=3)
axs[3].axvline(x=time_profile, color='black', linewidth=4, linestyle=':')


for ax, l in zip(axs.flatten(), [ '(a) Reflectivity - W-band',  '(b) Mean Doppler velocity - W-band ', '(c) Spectral width - W-band', '(d) Skewness - W-band']):
    ax.text(-0.05, 1.05, l,  fontweight='black', fontsize=26, transform=ax.transAxes)
fig.savefig(Path_out+'_figure_paper_detail_WBAND.png')
fig.savefig(Path_out+'_figure_paper_detail_WBAND.pdf')


#%%

#time_profile = datetime(2020,1,28,10,24,0)

#spec_profile = spectra_file.sel(time=time_profile, method="nearest")
#spec = spec_profile.sze.values
#spec_hh = spec_profile.sze_hh.values

#ze_profile = 10*np.log(spec_profile.ze.values)
#sk_profile = spec_profile.skew.values
#spec_db = 10.*np.log10(spec)


# reading chirps separately
#h_chirp1 = rangeRadar[:153]
#spec1 = spec_db[:153,:]
#v1 = spec_profile.chirp_vel.values[0,:]

#h_chirp2 = rangeRadar[153:346]
#v2 = spec_profile.chirp_vel.values[1,:]
#v2_good = v2[np.where(~np.isnan(v2))[0]]
#spec2 = spec_db[153:346,np.where(~np.isnan(v2))[0]]

#h_chirp3 = rangeRadar[346:]
#v3 = spec_profile.chirp_vel.values[2,:]
#v3_good = v3[np.where(~np.isnan(v3))[0]]
#spec3 = spec_db[346:,np.where(~np.isnan(v3))[0]]

# plot spectrogram for the selected profile
#labelsizeaxes   = 20
#fontSizeTitle = 20
#fontSizeX = 20
#fontSizeY = 20
#cbarAspect = 10
#fontSizeCbar = 20

#fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12,6))
#rcParams['font.sans-serif'] = ['Tahoma']
#matplotlib.rcParams['savefig.dpi'] = 100
#plt.gcf().subplots_adjust(bottom=0.15)
#fig.tight_layout()
#ax = plt.subplot(1,1,1)
#ax.spines["top"].set_visible(False)
#ax.spines["right"].set_visible(False)
#ax.get_xaxis().tick_bottom()
#ax.get_yaxis().tick_left()
#matplotlib.rc('xtick', labelsize=26)  # sets dimension of ticks in the plots
#matplotlib.rc('ytick', labelsize=26)  # sets dimension of ticks in the plots

#mesh = ax.pcolormesh(v1, h_chirp1, spec1, cmap='jet', rasterized=True)
#ax.axhline(y=500., color='black', linestyle='dotted', linewidth=2, label='500m')
#ax.axhline(y=750., color='black', linestyle='dashed', linewidth=2, label='750m')
#ax.axhline(y=1000., color='black', linestyle='solid', linewidth=2, label='1000m')
#ax.axhline(y=1200., color='black', linestyle='dotted', linewidth=2, label='1200m')
#ax.axhline(y=1500., color='black', linestyle='dashed', linewidth=2, label='1500m')
#ax.axhline(y=1750., color='black', linestyle='solid', linewidth=2, label='1750m')
#ax.axhline(y=2000., color='black', linestyle='dotted', linewidth=2, label='2000m')
#ax.axhline(y=2250., color='black', linestyle='dashed', linewidth=2, label='2250m')
#ax.axhline(y=2500., color='black', linestyle='solid', linewidth=2, label='2500m')

#ax.pcolormesh(v2_good, h_chirp2, spec2, cmap='jet', rasterized=True)
#ax.pcolormesh(v3_good, h_chirp3, spec3, cmap='jet', rasterized=True)
#ax.set_xlabel('Doppler velocity [ms$^{-1}$]', fontsize=fontSizeX)
#ax.set_ylabel('Height [m]', fontsize=fontSizeX)
#cbar = fig.colorbar(mesh, aspect=10, use_gridspec=True)
#cbar.set_label(label='Power [dB]',  size=26)
#ax.legend(frameon=False)
#ax.set_ylim(800.,1200.)
#ax.set_xlim(-15.,10.)

#fig.savefig(Path_out+'_height_spectrogram.png')
#fig.savefig(Path_out+'_figure_paper_detail_WBAND.pdf')
#%%

# plot spectrograms at given heights
#labelsizeaxes   = 20
#fontSizeTitle = 20
#fontSizeX = 20
##fontSizeY = 20
#cbarAspect = 10
#fontSizeCbar = 20

# select heights for plotting
#h1_0 = f_closest(rangeRadar, 500.)
#h1_1 = f_closest(rangeRadar, 750.)
#h1_2 = f_closest(rangeRadar, 1000.)
#h1_3 = f_closest(rangeRadar, 1200.)
#h2_0 = f_closest(rangeRadar, 1500.)
#h2_1 = f_closest(rangeRadar, 1750.)
#h2_2 = f_closest(rangeRadar, 2000.)
#h2_3 = f_closest(rangeRadar, 2250.)
#h2_3 = f_closest(rangeRadar, 2500.)

#fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12,6))
#rcParams['font.sans-serif'] = ['Tahoma']
#matplotlib.rcParams['savefig.dpi'] = 100
#plt.gcf().subplots_adjust(bottom=0.15)
#fig.tight_layout()
#ax = plt.subplot(1,1,1)
#ax.spines["top"].set_visible(False)
#ax.spines["right"].set_visible(False)
#ax.get_xaxis().tick_bottom()
#ax.get_yaxis().tick_left()
#matplotlib.rc('xtick', labelsize=26)  # sets dimension of ticks in the plots
#matplotlib.rc('ytick', labelsize=26)  # sets dimension of ticks in the plots
#ax.plot()
#deltaP = 0.15
#ax.plot(v1, spec[f_closest(rangeRadar, 500.), :], label='500 m', color='blue', linewidth=3)

#ax.plot(v1, deltaP+spec[f_closest(rangeRadar, 750.), :], label='750 m', color='cyan', linewidth=3)
#ax.plot(v1, 2*deltaP+spec[f_closest(rangeRadar, 1000.), :], label='`1000 m', color='purple', linewidth=3)
#ax.plot(v1, 3*deltaP+spec[f_closest(rangeRadar, 1200.), :], label='1200 m', color='pink', linewidth=3)
#ax.plot(v2_good, 4*deltaP+spec[f_closest(rangeRadar, 1500.), np.where(~np.isnan(v2))[0]], label='1500 m', color='orange', linewidth=3)
#ax.plot(v2_good, 5*deltaP+spec[f_closest(rangeRadar, 1750.), np.where(~np.isnan(v2))[0]], label='1750 m', color='red', linewidth=3)
#ax.plot(v2_good, 6*deltaP+spec[f_closest(rangeRadar, 2000.), np.where(~np.isnan(v2))[0]], label='2000 m', color='green', linewidth=3)
#ax.plot(v2_good, 7*deltaP+spec[f_closest(rangeRadar, 2250.), np.where(~np.isnan(v2))[0]], label='2250 m', color='yellow', linewidth=3)

#ax.set_xlim([-2.,12.])
#ax.legend(frameon=False)

#%%



#fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12,6))
#rcParams['font.sans-serif'] = ['Tahoma']
#matplotlib.rcParams['savefig.dpi'] = 100
#plt.gcf().subplots_adjust(bottom=0.15)
#fig.tight_layout()
#ax = plt.subplot(1,1,1)
#ax.spines["top"].set_visible(False)
#ax.spines["right"].set_visible(False)
#ax.get_xaxis().tick_bottom()
#ax.get_yaxis().tick_left()
#matplotlib.rc('xtick', labelsize=26)  # sets dimension of ticks in the plots
#matplotlib.rc('ytick', labelsize=26)  # sets dimension of ticks in the plots
#ax.plot()
##deltaP = 0.15
#ax.plot(v2_good, spec[f_closest(rangeRadar, 2000.),  np.where(~np.isnan(v2))[0]], label='2000 m, sk = '+str(round(-sk_profile[f_closest(rangeRadar, 2000.)], 2)), color='black', linewidth=1.5, linestyle=":")
#ax.plot(v2_good, spec[f_closest(rangeRadar, 1950.),  np.where(~np.isnan(v2))[0]], label='1950 m, sk = '+str(round(-sk_profile[f_closest(rangeRadar, 1950.)], 2)), color='blue', linewidth=1.5, linestyle=":")
#ax.plot(v2_good, spec[f_closest(rangeRadar, 1900.),  np.where(~np.isnan(v2))[0]], label='1900 m, sk = '+str(round(-sk_profile[f_closest(rangeRadar, 1900.)], 2)), color='cyan', linewidth=1.5, linestyle=":")
#ax.plot(v2_good, spec[f_closest(rangeRadar, 1850.),  np.where(~np.isnan(v2))[0]], label='1850 m, sk = '+str(round(-sk_profile[f_closest(rangeRadar, 1850.)], 2)), color='green', linewidth=1.5, linestyle=":")
#ax.plot(v2_good, spec[f_closest(rangeRadar, 1800.),  np.where(~np.isnan(v2))[0]], label='1800 m, sk = '+str(round(-sk_profile[f_closest(rangeRadar, 2000.)], 2)), color='purple', linewidth=1.5, linestyle=":")
#ax.plot(v2_good, spec[f_closest(rangeRadar, 1750.),  np.where(~np.isnan(v2))[0]], label='1750 m, sk = '+str(round(-sk_profile[f_closest(rangeRadar, 1750.)], 2)), color='orange', linewidth=1.5, linestyle=":")
#ax.plot(v2_good, spec[f_closest(rangeRadar, 1700.),  np.where(~np.isnan(v2))[0]], label='1700 m, sk = '+str(round(-sk_profile[f_closest(rangeRadar, 1700.)], 2)), color='red', linewidth=1.5, linestyle=":")

#ax.set_xlim([-0.5,2.])
#ax.legend(frameon=False)



#%%

#fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12,6))
#rcParams['font.sans-serif'] = ['Tahoma']
##matplotlib.rcParams['savefig.dpi'] = 100
#plt.gcf().subplots_adjust(bottom=0.15)
#fig.tight_layout()
#ax = plt.subplot(1,1,1)
#ax.spines["top"].set_visible(False)
#ax.get_xaxis().tick_bottom()
#ax.get_yaxis().tick_left()
#matplotlib.rc('xtick', labelsize=26)  # sets dimension of ticks in the plots
#matplotlib.rc('ytick', labelsize=26)  # sets dimension of ticks in the plots
#ax.plot()
#deltaP = 0.15
#ax.plot(-sk_profile, rangeRadar)
#ax.set_ylim([0.,2000.])
