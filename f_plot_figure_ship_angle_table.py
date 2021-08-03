#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Created on 26 Mar 2021

    @author: cacquist
    @goal:elaborate the plot for the angles and the stable table for the paper essd
    for table working and table not working.





"""

import pandas as pd
import netCDF4 as nc4
from netCDF4 import Dataset
import xarray as xr
import numpy as np
from functions_essd import nearest
from datetime import datetime
from matplotlib import rcParams
import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# reading ship data for the entire campaign
# paths to the different data files and output directories for plots
pathFolderTree  = '/Volumes/Extreme SSD/ship_motion_correction_merian/'
pathFig         = pathFolderTree+'/plots/'
pathNcDataAnc   = pathFolderTree+'/ncdf_ancillary/'
ShipData        = pathFolderTree+'/ship_data/new/shipData_all2.nc'
StFile          = pathFolderTree+'/ship_data/new/status_v2.txt'



print('* reading ship data from ncdf file')
ShipDataset                  = xr.open_dataset(ShipData)
# extracting ship data for the day
datetimeStart  = datetime(2020,1,19,0,0,0)
datetimeEnd    = datetime(2020,2,19,0,0,0)
dataShipDay    = ShipDataset.sel(time=slice(datetimeStart,datetimeEnd))
rollShipArr    = dataShipDay['roll'].values
pitchShipArr   = dataShipDay['pitch'].values
yawShipArr     = dataShipDay['yaw'].values
heaveShipArr   = dataShipDay['heave'].values
datetimeShip   = pd.to_datetime(dataShipDay['time'].values)
NtimeShip      = len(datetimeShip)


data           = pd.read_csv(StFile, header = 1)
flagTable      = np.zeros(len(data))
datetimeTable  = []
datetimeWork   = []
for ind in range(0, len(data)):
    datetimeTable.append(pd.to_datetime(data.values[ind][0]))
    if data.values[ind][1] == True:
        flagTable[ind] = 1.
        datetimeWork.append(pd.to_datetime(data.values[ind][0]))

#%%
#fig, axs = plt.subplots(1, 1, figsize=(14,5), sharex=True, constrained_layout=True)
#[a.xaxis.set_major_formatter(mdates.DateFormatter('%d %b')) for a in axs.flatten()]
#axs[1].plot(datetimeTable, flagTable)
#%%
def f_findExtremesInterval(timeSerie):
    """
    goal : function to derive start and ending time of the nan gaps.
    input: timeSerie : time array in datetime format of times where the variable has a nan value
    output:
        timeStart array : time containing starting time of consecutive time gaps
        timeEnd array   : time containing ending time of consecutive time gaps

    """
    from datetime import datetime
    from datetime import timedelta
    # Construct dummy dataframe
    df = pd.DataFrame(timeSerie, columns=['time'])
    deltas = df['time'].diff()[0:]
    #print(deltas)
    gaps = deltas[deltas > timedelta(minutes=1)]
    #print(gaps)
    # build arrays to store data
    timeStopArr       = np.zeros((len(gaps)), dtype='datetime64[s]')
    timeRestartArr    = np.zeros((len(gaps)), dtype='datetime64[s]')
    durationSingleGaps = [" " for i in range(len(gaps))]
    #TotalDuration      = gaps.sum()

    # Print results
    #print(f'{len(gaps)} gaps with total gap duration: {gaps.sum()}')
    indArr = 0
    for i, g in gaps.iteritems():
        time_stop = df['time'][i - 1]
        time_restart = df['time'][i + 1]
        timeStopArr[indArr] = datetime.strftime(time_stop, "%Y-%m-%d %H:%M:%S")
        timeRestartArr[indArr] = datetime.strftime(time_restart, "%Y-%m-%d %H:%M:%S")
        durationSingleGaps[indArr] = str(g.to_pytimedelta())
        indArr = indArr + 1


        #print(f'time stop: {datetime.strftime(time_stop, "%Y-%m-%d %H:%M:%S")} | '
        #      f'Duration gap: {str(g.to_pytimedelta())} | '
        #       f'time Restart: {datetime.strftime(time_restart, "%Y-%m-%d %H:%M:%S")}')
    return(timeStopArr, timeRestartArr)

timeStopArr, timeRestartArr = f_findExtremesInterval(datetimeWork)

#fig, ax = plt.subplots()
#for i in range(len(timeStopArr)):#
#    ax.axvspan(pd.to_datetime(timeStopArr[i]), pd.to_datetime(timeRestartArr[i]), 0,1, alpha=0.5, color='grey')

#%%
# plot quicklook of filtered and corrected mdv for checking
import matplotlib.ticker as ticker
labelsizeaxes   = 26
fontSizeTitle   = 26
fontSizeX       = 26
fontSizeY       = 26
cbarAspect      = 26
fontSizeCbar    = 26
rcParams['font.sans-serif'] = ['Tahoma']
matplotlib.rcParams['savefig.dpi'] = 100
plt.rcParams.update({'font.size':24})
grid            = True
fig, axs = plt.subplots(4, 1, figsize=(18,14), sharex=True, constrained_layout=True)

# build colorbar
for i in range(len(timeStopArr)):
    axs[0].axvspan(pd.to_datetime(timeStopArr[i]), pd.to_datetime(timeRestartArr[i]), 0,1, alpha=0.3, color='grey')
axs[0].plot(datetimeShip, heaveShipArr,  color='slateblue',  rasterized=True)
axs[0].axhline(0., xmin=0, xmax=1, linestyle=':', alpha=0.5, color='white')

#axs[0].set_title('Original', loc='left')
axs[0].spines["top"].set_visible(False)
axs[0].spines["right"].set_visible(False)
axs[0].get_xaxis().tick_bottom()
axs[0].get_yaxis().tick_left()
axs[0].set_xlim(datetimeShip[0], datetimeShip[-1])
axs[0].set_ylim(-5., 5.)
axs[0].set_ylabel('Heave [m]')
axs[0].axhline(0., xmin=0, xmax=1, linestyle=':', alpha=0.5, color='grey')


#axs[0].set_xlabel('time [hh:mm]', fontsize=fontSizeX)
#axs[1].set_xlabel('time [mm:ss]', fontsize=fontSizeX)
#axs[1].set_xlabel('time [mm:ss]', fontsize=fontSizeX)
#[a.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M')) for a in axs[:].flatten()]
for i in range(len(timeStopArr)):
    axs[1].axvspan(pd.to_datetime(timeStopArr[i]), pd.to_datetime(timeRestartArr[i]), 0,1, alpha=0.3, color='grey')
mesh = axs[1].plot(datetimeShip, rollShipArr,  color='blueviolet',  rasterized=True)
axs[1].axhline(0., xmin=0, xmax=1, linestyle=':', alpha=0.5, color='black')
#axs[2].set_title('Corrected and smoothed', fontsize=fontSizeX, loc='left')
#axs[1].spines["top"].set_visible(False)
#axs[1].spines["right"].set_visible(False)
#axs[1].get_xaxis().tick_bottom()
#axs[1].get_yaxis().tick_left()
#axs[1].set_xlim(datetimeShip[0], datetimeShip[-1])
axs[1].set_ylabel('Roll [$^{\circ}$]')
axs[1].set_ylim(-15., 15.)


for i in range(len(timeStopArr)):
    axs[2].axvspan(pd.to_datetime(timeStopArr[i]), pd.to_datetime(timeRestartArr[i]), 0,1, alpha=0.3, color='grey')
mesh = axs[2].plot(datetimeShip, pitchShipArr,  color='dodgerblue',  rasterized=True)
axs[2].axhline(0., xmin=0, xmax=1, linestyle=':', alpha=0.5, color='black')
axs[2].set_ylabel('Pitch [$^{\circ}$]')
#axs[2].plot(datetimeTable, flagTable)

#axs[2].set_title('Corrected and smoothed', fontsize=fontSizeX, loc='left')
#axs[2].spines["top"].set_visible(False)
#axs[2].spines["right"].set_visible(False)
#axs[2].get_xaxis().tick_bottom()
#axs[2].get_yaxis().tick_left()
#axs[2].set_xlim(datetimeShip[0], datetimeShip[-1])
axs[2].set_ylim(-7., 7.)

#axs[3].spines["top"].set_visible(False)
#axs[3].spines["right"].set_visible(False)
#axs[3].get_xaxis().tick_bottom()
#axs[3].get_yaxis().tick_left()


axs[3].scatter(datetimeShip, yawShipArr,  color='orchid',  marker="o", s=0.1, rasterized=True)
axs[3].axhline(90., xmin=0, xmax=1, linestyle=':', alpha=0.5, color='grey')
axs[3].axhline(180., xmin=0, xmax=1, linestyle=':', alpha=0.5, color='grey')
axs[3].axhline(270., xmin=0, xmax=1, linestyle=':', alpha=0.5, color='grey')

axs[3].set_ylim(0., 360.)
#axs[1].set_title('Corrected', fontsize=fontSizeX, loc='left')
#xs[3].set_xlim(datetimeShip[0], datetimeShip[-1])
axs[3].set_ylabel('Yaw [$^{\circ}$]')
axs[3].set_xlabel('Time [dd Mon]')
axs[3].yaxis.set_ticks([0., 90., 180., 270., 360.])

#fig.colorbar(mesh, ax=axs[:], label='Mean Doppler velocity [$ms^{-1}$]', location='right', aspect=60, use_gridspec=grid)
for ax, l in zip(axs.flatten(), ['(a) Heave', '(b) Roll', '(c) Pitch', '(d) Yaw' ]):
    ax.text(-0.05, 1.08, l,  fontweight='black', transform=ax.transAxes)
    ax.set_xlim(datetimeShip[0], datetimeShip[-1])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(3)
    ax.spines["left"].set_linewidth(3)
    ax.tick_params(which='minor', length=7, width=3)
    ax.tick_params(which='major', length=7, width=3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    ax.tick_params(axis='both', labelsize=26)
fig.savefig(pathFig+'_figure_paper.png')
fig.savefig(pathFig+'_figure_paper.pdf')

strasuka
#%%
for i, line in enumerate(f):

                try:

                    date_str = line[:17]  # extract date (not always only first row with comment)
                    self.date = np.append(self.date, datetime.datetime.strptime(date_str, '%y/%m/%d %H:%M:%S'))

                except ValueError:

                    try:

                        date_str = line[2:19]  # extract date (not always only first row with comment)
                        self.date = np.append(self.date, datetime.datetime.strptime(date_str, '%y/%m/%d %H:%M:%S'))

                    except ValueError:

                        print('Can not match date format to: "{}"'.format(date_str))
