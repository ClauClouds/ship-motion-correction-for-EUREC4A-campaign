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



path_data = '/home/cacquist/mrr_paper_plot/'
#mrr_file = '20200128_MRR_PRO_msm_eurec4a.nc'
#w_band_file = '20200128_wband_radar_msm_eurec4a_intake.nc'
mrr_file = '20200212_MRR_PRO_msm_eurec4a.nc'
w_band_file = '20200212_wband_radar_msm_eurec4a_intake.nc'
path_lcl = path_data
lcl_file = 'LCL_dataset.nc'
Path_out = path_data+'/plots/'


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
rangeRadar = w_band_cs['height'].values
timeLocal = pd.to_datetime(w_band_cs['time'].values)


Ze = w_band_cs['radar_reflectivity'].values
Ze[Ze == -999.]    = np.nan

#reading and slicing lcl data
lcl_data = xr.open_dataset(path_lcl+lcl_file)
lcl_cs = lcl_data.sel(time=slice(time_start, time_end))


# reading and slicing MRR data
mrr = xr.open_dataset(path_data+mrr_file)
mrr_cs = mrr.sel(time=slice(time_start, time_end))


#interpolating data on radar time stamps
mrr_interp = mrr_cs.interp(time=w_band_cs.time.values)
Ze_mrr = mrr_interp.Ze.values
range_mrr = mrr_interp.height.values
time_mrr = mrr_interp.time.values

lcl_interp = lcl_cs.interp(time=w_band_cs.time.values)
lcl = lcl_interp.lcl.values
lcl_time = lcl_interp.time.values


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


# Create rectangle x coordinates
startTime = datetime(2020,2,12,15,50,0,0)
endTime = startTime + timedelta(minutes = 15)
#datetime(2020,2,12,16,5,0,0)


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
mincm_ze = -50.
maxcm_ze = 20.
step_ze = 1
colors_ze =['#4f8c9d', '#1c4c5e', '#8ae1f9', '#8f0f1b', '#e0bfb4', '#754643', '#ef7e58', '#ff1c5d']
cmap_ze, ticks_ze, norm_ze, bounds_ze =  f_defineSingleColorPalette(colors_ze, mincm_ze, maxcm_ze, step_ze)

# settings for mrr radar reflectivity
mincm_ze_mrr = 0.
maxcm_ze_mrr = 40.
step_ze_mrr = 0.1
colors_ze_mrr =['#4f8c9d', '#1c4c5e', '#8ae1f9', '#8f0f1b', '#e0bfb4', '#754643', '#ef7e58', '#ff1c5d']
cmap_ze_mrr, ticks_ze_mrr, norm_ze_mrr, bounds_ze_mrr =  f_defineSingleColorPalette(colors_ze_mrr, mincm_ze_mrr, maxcm_ze_mrr, step_ze_mrr)


mincm_ze_mrr = 0.
maxcm_ze_mrr = 40.
step_ze_mrr = 0.1
#colors_ze_mrr =['#4f8c9d', '#1c4c5e', '#8ae1f9', '#8f0f1b', '#e0bfb4', '#754643', '#ef7e58', '#ff1c5d']
colors_ze_mrr = ["#56ebd3", "#9e4302", "#2af464", "#d6061a", "#1fa198"]

cmap_ze_mrr, ticks_ze_mrr, norm_ze_mrr, bounds_ze_mrr =  f_defineSingleColorPalette(colors_ze_mrr, mincm_ze_mrr, maxcm_ze_mrr, step_ze_mrr)





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


# Plot overview plot with rectagle showing area that is enlarged in the next plot
#rect = mpl.patches.Rectangle((start, 0), width, 2000., linewidth=4, edgecolor='yellow', facecolor='none')
patch1 = matplotlib.patches.PathPatch(path1, facecolor='none', edgecolor='blue', linewidth=4, linestyle=':')
patch2 = matplotlib.patches.PathPatch(path2, facecolor='none', edgecolor='blue', linewidth=4, linestyle=':')
fig, axs = plt.subplots(2, 1, figsize=(20,14), constrained_layout=True, sharex=True)

# setting dates formatter
[a.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M')) for a in axs[:].flatten()]
matplotlib.rc('xtick', labelsize=36)  # sets dimension of ticks in the plots
matplotlib.rc('ytick', labelsize=36)  # sets dimension of ticks in the plots
grid            = True
mesh = axs[0].pcolormesh(timeLocal, rangeRadar,  Ze.T, vmin=mincm_ze, vmax=maxcm_ze, cmap=cmap_ze, rasterized=True)
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
axs[0].tick_params(axis='both', labelsize=30)
axs[0].spines["bottom"].set_linewidth(3)
axs[0].spines["left"].set_linewidth(3)
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
axs[1].tick_params(axis='both', labelsize=30)
axs[1].spines["bottom"].set_linewidth(3)
axs[1].spines["left"].set_linewidth(3)
for ax, l in zip(axs.flatten(), ['a) Reflectivity - Wband', 'b) Reflectivity - MRR-PRO']):
    ax.text(-0.05, 1.05, l,  fontweight='black', fontsize=36, transform=ax.transAxes)
fig.savefig(Path_out+'Fig_11.png')
#fig.savefig(Path_out+'_case_overview_both_instr.pdf')
#%%
##############################################################################################################################################
