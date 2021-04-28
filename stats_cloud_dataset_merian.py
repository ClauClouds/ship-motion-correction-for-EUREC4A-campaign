#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon 19/04/2021

@author: cacquist
@date: 19.04.2021
@goal: calculate for essd paper
- most frequent cloud type in hours
- duration of non prec periods
- duration of prec periods from stratiform/cumulus

"""
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib

def calc_cloud_duration(cloud_base_arr, time):
    """
    Author: Claudia acquistapace
    Goal: calculate cloud duration
    Date: 19.04.2021, copied from PBL cloud paper
    input:
    - cloud base height array
    (dim: time, vals: nan if no cloud base, float with cb height if cloud)
    - time: datetime array
    output:
    cloud_duration: dict('timeStart':timeStart, time of start of the cloud
    'indStart':indStart,   index corresponding to cloud start
    'timeEnd':timeEnd,     time of end of the cloud
    'indEnd':indEnd,        index corresponding to cloud end
    'duration'duration)    time duration in seconds
    """

    cloudStart = 0
    cloud_duration = []
    duration_list = []
    for itime in range(len(time)):

        if (np.isnan(cloud_base_arr[itime]) == False) * (cloudStart == 0):
            cloudStart = 1
            timeStart = time[itime]
            indStart = itime
        if (np.isnan(cloud_base_arr[itime]) == True) * (cloudStart == 1):
            #print('sono qua')
            timeEnd = time[itime-1]
            indEnd = itime-1
            cloudStart = 0
            # calculate cloud duration
            duration = (timeEnd - timeStart).total_seconds()
            dict_cloud = {'timeStart':timeStart, 'indStart':indStart, \
            'timeEnd':timeEnd, 'indEnd':indEnd, \
            'duration':duration}
            timeStart = np.nan
            timeEnd = np.nan
            cloud_duration.append(dict_cloud)
            duration_list.append(duration)


    return(duration_list, cloud_duration)

# read file containing cloud type classification for the entire campaign
''' classification:
-3: unclassified,
 -2: no LCL data available,
 -1: no cloud observed,
0: shallow cloud = cloud top between LCL and LCL + 600 m),
1: stratiform cloud = cloud top between LCL + 600 m and 4 km height,
2: cloud only below LCL,
3: cloud only above 4 km height,
4: cloud above and below 4 km height

precipitation:
0: non-precipitating
1: precipitating
'''
# setting path for plots
path_fig = "/Volumes/Extreme SSD/work/006_projects/001_Prec_Trade_Cycle/plots/"


class_file = "/Volumes/Extreme SSD/work/006_projects/001_Prec_Trade_Cycle/\
Nils_contrib/cloud_lcl_classification.nc"
classification = xr.open_dataset(class_file)

# read cloud base height Dataset
cloud_base_file = "/Volumes/Extreme SSD/work/006_projects/001_Prec_Trade_Cycle/\
data_wind_lidar/wind_lidar_eurec4a.nc"
cloud_base = xr.open_dataset(cloud_base_file)

cb_serie = cloud_base['cb'].values
cb_time = pd.to_datetime(cloud_base['time'].values)

# interpolating classification data on wind lidar time resolution (10 s)
class_interp = classification.interp(time=cb_time, method='nearest')
prec_state = class_interp['precip'].values
cloud_type = class_interp['shape'].values
time_class = pd.to_datetime(class_interp['time'].values)

print('prec possible values: ', set(prec_state))
print(np.shape(np.where(prec_state == 1)[0]))


# calculating duration of precipitation periods
# defining array == 1 where precip, and nan elsewhere
prec_duration_flag = np.zeros(len(prec_state))
prec_duration_flag[np.where(prec_state == 1)[0]] = 1
prec_duration_flag[prec_duration_flag == 0] = np.nan
prec_duration, prec_dict_arr = calc_cloud_duration(prec_duration_flag, time_class)
prec_duration = np.asarray(prec_duration) # conversion to array in minutes
prec_duration = prec_duration[prec_duration > 0.]
print('min prec duration in minutes', np.nanmin(np.asarray(prec_duration)/60.))
print('max prec duration in minutes', np.nanmax(np.asarray(prec_duration)/60.))
print('mean prec duration in minutes', np.nanmedian(np.asarray(prec_duration)/60.))


# calculating clear sky duration
clear_sky_duration_flag = np.zeros(len(prec_state))
clear_sky_duration_flag.fill(np.nan)
clear_sky_duration_flag[(cloud_type == -1)] = 1.
clear_sky_duration, clear_sky_dict_arr = calc_cloud_duration(clear_sky_duration_flag, time_class)
clear_sky_duration = np.asarray(clear_sky_duration) # conversion to array in minutes
clear_sky_duration = clear_sky_duration[clear_sky_duration > 0.]
print('min clear sky duration in minutes', np.nanmin(np.asarray(clear_sky_duration)/60.))
print('max clear sky duration in minutes', np.nanmax(np.asarray(clear_sky_duration)/60.))
print('mean clear sky duration in minutes', np.nanmedian(np.asarray(clear_sky_duration)/60.))


# calculating duration of non precipitating periods
non_prec_duration_flag = np.zeros(len(prec_state))
# selecting no precip and no clouds
non_prec_duration_flag[np.where(prec_state != 1)[0]] = 1
non_prec_duration_flag[non_prec_duration_flag == 0 ] = np.nan
non_prec_duration, non_prec_dict_arr = calc_cloud_duration(non_prec_duration_flag, time_class)
non_prec_duration = np.asarray(non_prec_duration) # conversion to array in minutes
non_prec_duration = non_prec_duration[non_prec_duration > 0.]
print('min non prec duration ', np.nanmin(np.asarray(non_prec_duration)/60.))
print('max non prec duration ', np.nanmax(np.asarray(non_prec_duration)/60.))
print('mean non prec duration', np.nanmedian(np.asarray(non_prec_duration)/60.))


# calculation of cloudy periods
cloudy_duration_flag = np.zeros(len(prec_state))
cloudy_duration_flag.fill(np.nan)
cloudy_duration_flag[cloud_type > 0] = 1
cloud_duration, cloud_dict_arr = calc_cloud_duration(cloudy_duration_flag, time_class)
cloud_duration = np.asarray(cloud_duration) # conversion to array in minutes
cloud_duration = cloud_duration[cloud_duration > 0.]
print('min cloud duration in minutes', np.nanmin(np.asarray(cloud_duration)/60.))
print('max cloud duration in minutes', np.nanmax(np.asarray(cloud_duration)/60.))
print('mean cloud duration in minutes', np.nanmedian(np.asarray(cloud_duration)/60.))

#%%

# plotting histogram of vertical wind averaged in the first 300 m
labelsizeaxes = 12
fontSTitle = 12
fontSizeX = 12
fontSizeY = 12
cbarAspect = 10
fontSizeCbar = 12
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))
rcParams['font.sans-serif'] = ['Tahoma']
matplotlib.rcParams['savefig.dpi'] = 100
plt.gcf().subplots_adjust(bottom=0.15)
fig.tight_layout()
ax = plt.subplot(1, 1, 1)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()
matplotlib.rc('xtick', labelsize=labelsizeaxes)  # sets dimension of ticks
matplotlib.rc('ytick', labelsize=labelsizeaxes)  # sets dimension of ticks
ax.hist(cloud_duration/60, bins=60, color='red', label='cloud duration', density=True, histtype='step', cumulative=True)
ax.hist(non_prec_duration/60, bins=60, color='purple', density=True, label='non-precipitation periods', histtype='step', cumulative=True)
ax.hist(prec_duration/60, bins=60, color='blue', density=True, label='precipitation periods', histtype='step', cumulative=True)
ax.hist(clear_sky_duration/60, bins=60, color='green', density=True, label='cleary sky periods', histtype='step', cumulative=True)

ax.legend(frameon=False, loc='lower right')
ax.set_xlim([0.,40.])

ax.set_title(' histograms for duration', fontsize=fontSTitle, loc='left')
ax.set_xlabel("duration [$minutes$]", fontsize=fontSizeX)
ax.set_ylabel("occurrences [#]", fontsize=fontSizeY)
fig.tight_layout()
fig.savefig(path_fig+'_duration_histogram.png', format='png')


#%%

# calculating most frequenct cloud type in hours
shallow_flag = np.zeros(len(prec_state))
shallow_flag.fill(np.nan)
shallow_flag[cloud_type == 0] = 1
shallow_duration, shallow_dict_arr = calc_cloud_duration(shallow_flag, time_class)
shallow_duration = np.asarray(shallow_duration) # conversion to array in minutes
total_shallow_duration = np.nansum(shallow_duration)

print('total duration shallow clouds in hours, minutes', (total_shallow_duration/60.)//60., (total_shallow_duration/60.)% 60.)
#%%
stratif_flag = np.zeros(len(prec_state))
stratif_flag.fill(np.nan)
stratif_flag[cloud_type == 1] = 1
stratif_duration, stratif_dict_arr = calc_cloud_duration(stratif_flag, time_class)
stratif_duration = np.asarray(stratif_duration) # conversion to array in minutes
total_stratif_duration = np.nansum(stratif_duration)

print('total duration stratiform clouds in hours, minutes', (total_stratif_duration/60.)//60., (total_stratif_duration/60.)% 60.)
#%%

total_prec_duration = np.nansum(prec_duration)/60.
print('total duration precipitation in hours, minutes', total_prec_duration // 60., total_prec_duration % 60.)
total_non_prec_duration = np.nansum(non_prec_duration)/60.
print('total duration non-precipitation in hours, minutes', total_non_prec_duration//60., total_non_prec_duration % 60.)
total_clear_sky_duration = np.nansum(clear_sky_duration)/60.
print('total duration clear-sky conditions in hours, minutes', total_clear_sky_duration//60., total_clear_sky_duration % 60.)