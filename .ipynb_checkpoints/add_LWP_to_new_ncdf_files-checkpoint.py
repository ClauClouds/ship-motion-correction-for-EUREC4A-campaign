#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue 9 June 2021

@author: cacquist
@ goals:
    add LWP estimations to ncdf files postprocessed by rosa
    recalculate doppler correction for the mean doppler velocity using the old files with the correction stored

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
from datetime import datetime
from datetime import timedelta
#import atmos
from pathlib import Path

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
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib
import os.path
import pandas as pd
import matplotlib as mpl

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
def f_plot_height_spectrogram(time_sel, nc_data, pathFig, ymin, ymax):


    '''
    function to plot height spectrogram of w band radar data
    '''

    # building date for the output file __name__
    hour = str(time_sel.hour)
    day = str(time_sel.day)
    month = str(time_sel.month)
    year = str(time_sel.year)
    if len(hour) == 1:
        hour= '0'+hour
    time_string = year+month+day+'_'+hour
    nc_sel = nc_data.sel(time=time_sel, method='nearest')
    spec_profile = nc_sel.spec.values
    v_doppler_profile = nc_sel.Doppler_velocity.values

    # converting spec to log units
    spec_db= 10*np.log10(spec_profile)
    height = nc_sel.range.values

    print(len(height))
    # reading index where label changes
    ind_1 = nc_sel.range_offsets.values[1]
    ind_2 = nc_sel.range_offsets.values[2]

    # reading chirps separately
    h_chirp1 = height[:ind_1]
    spec1 = spec_db[:ind_1,:]
    v1 = v_doppler_profile[0,:]

    h_chirp2 = height[ind_1:ind_2]
    v2 = v_doppler_profile[ind_1+1,:]
    v2_good = v2[np.where(~np.isnan(v2))[0]]
    spec2 = spec_db[ind_1:ind_2,np.where(~np.isnan(v2))[0]]

    h_chirp3 = height[ind_2:]
    v3 = v_doppler_profile[ind_2+1,:]
    v3_good = v3[np.where(~np.isnan(v3))[0]]
    spec3 = spec_db[ind_2:,np.where(~np.isnan(v3))[0]]

    # plot spectrogram for the selected profile
    labelsizeaxes   = 20
    fontSizeTitle = 20
    fontSizeX = 20
    fontSizeY = 20
    cbarAspect = 10
    fontSizeCbar = 20

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
    matplotlib.rc('xtick', labelsize=26)  # sets dimension of ticks in the plots
    matplotlib.rc('ytick', labelsize=26)  # sets dimension of ticks in the plots
    for ind_height, height_sel in enumerate(height):
        if ind_height == 0:
            v = v_doppler_profile[ind_height,:]
            v = v[np.where(~np.isnan(v))[0]]
            mesh = ax.pcolormesh(v, height_sel, spec_db[ind_height:,np.where(~np.isnan(v))[0]], cmap='jet', rasterized=True)
        else:
            v = v_doppler_profile[ind_height,:]
            v = v[np.where(~np.isnan(v))[0]]
            ax.pcolormesh(v, height_sel, spec_db[ind_height:,np.where(~np.isnan(v))[0]], cmap='jet', rasterized=True)

    #mesh = ax.pcolormesh(v1, h_chirp1, spec1, cmap='jet', rasterized=True)
    #ax.pcolormesh(v2_good, h_chirp2, spec2, cmap='jet', rasterized=True)
    #ax.pcolormesh(v3_good, h_chirp3, spec3, cmap='jet', rasterized=True)
    ax.set_xlabel('Doppler velocity [ms$^{-1}$]', fontsize=fontSizeX)
    ax.set_ylabel('Height [m]', fontsize=fontSizeX)
    cbar = fig.colorbar(mesh, aspect=10, use_gridspec=True)
    cbar.set_label(label='Power [dB]',  size=26)
    ax.legend(frameon=False)
    ax.set_ylim(ymin,ymax)
    ax.set_xlim(-8.,1.)
    fig.tight_layout()
    fig.savefig(pathFig+time_string+'_height_spectrogram.png')
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
def f_calculateExactRadarTime(millisec, chirpIntegrations, datetimeRadar):
    '''
    date:23/11/2020
    author : Claudia Acquistapace
    contact: cacquist@uni-koeln.de
    goal   : function to calculate the exact radar time composing the millisecond part,
             the chirp integration time and summing everything up to get for each time
             step, the exact central time stamp of the time step.
             Exact time of the radar is
             t_radar_i = t_radar + sampleTms/1000 - sum(j=i,N) chirpIntegrations[j] + chirpIntegrations[i]/2
             because the time recorded is the final time of the time interval

    input:
        millisec: numpy array containing milliseconds to be added, expressed in seconds
        chirpIntegrations: numpy array of integration time of each chirp, expressed in nanoseconds
        datetimeRadar: datetime array of the radar time stamps
    output:
        datetimeChirp matrix type (datetime64[ns]) with dimensions (3,len(datetimeRadar))
        each row corresponds to the new calculated chirp datetime array as indicated below:
        timeChirp1 = datetime[0,:]
        timeChirp2 = datetime[1,:]
        timeChirp3 = datetime[2,:]
        '''

    # converting time in seconds since 1970
    timeRadar          = datetimeRadar[:].astype('datetime64[s]').astype('int')  # time in seconds since 1970

    # calculating chirp starting time
    timeChirp          = np.zeros((3,len(datetimeRadar)))
    datetimeChirp      = np.empty((3,len(datetimeRadar)), dtype='datetime64[ns]')

    # calculating time stamps of each chirp
    for i_chirp in range(len(chirpIntegrations)):
        timeChirp[i_chirp,:] = timeRadar +  millisec * 10.**(-3) -  np.sum(chirpIntegrations[i_chirp:]) + chirpIntegrations[i_chirp]/2

    for ind_chirp in range(len(chirpIntegrations)):
        for ind_time in range(len(timeRadar)):
            datetimeChirp[ind_chirp, ind_time] = pd.to_datetime(datetime.utcfromtimestamp(timeChirp[ind_chirp, ind_time]), unit='ns')

    return(datetimeChirp)
def f_findMdvTimeSerie_VMerge(values, datetime, rangeHeight, NtimeStampsRun, pathFig, chirp):
    '''
    author: Claudia Acquistapace
    date: 25 november 2020
    modified: 16 December 2020 to work for all conditions (table work and not working)
    goal : identify, given a mean doppler velocity matrix, a sequence of lenght
    NtimeStampsRun, of values in the matrix at a given height
    that contains the minimum possible amount of nan values in it.

    Parameters
    ----------
    INPUT:
    values : TYPE ndarray(time, height)
        DESCRIPTION : matrix of values for which it is necessary to find a serie of non nan values of given lenght
    datetime : datetime
        DESCRIPTION: time array associated with the matrix of values
    rangeHeight : ndarray
        DESCRIPTION: height array associated with the matrix of values
    NtimeStampsRun : scalar
        DESCRIPTION: length of the sequence of values that are read for scanning the matrix
        (expressed in interval of the time resolution)
    pathFig : string
        DESCRIPTION: string for the output path of the selected plot
    chirp: string
        DESCRIPTION: string indicating the chirp of the radar data processed
    OUTPUT:
    valuesTimeSerie  : type(ndarray) - time serie of the lenght prescribed by NtimeStampsRun corresponding
    to the minimum amount of nan values found in the matrix of values
    i_height_sel     : type scalar - index of the height for the selected serie of values
    timeSerie        : type datetime - time array corresponding to the selected time serie of values
    valuesColumnMean : type ndarray - values of the variable averaged over height for the selected time interval
    -------

    '''
    import pandas as pd
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib import rcParams
    import matplotlib

    # extracting date from timestamp format
    date = '20'+pd.to_datetime(datetimeRadar[0]).strftime(format="%y%m%d-%H")

    #  concept: scan the matrix using running mean for every height, and check the number of nans in the selected serie.
    if len(datetime)-NtimeStampsRun > 0:
        nanAmountMatrix = np.zeros((len(datetime)-NtimeStampsRun, len(rangeHeight)))
        nanAmountMatrix.fill(np.nan)
        for indtime in range(len(datetime)-NtimeStampsRun):
            mdvChunk = values[indtime:indtime+NtimeStampsRun, :]
            df = pd.DataFrame(mdvChunk, index = datetime[indtime:indtime+NtimeStampsRun], columns=rangeHeight)

            # count number of nans in each height
            nanAmountMatrix[indtime,:] = df.isnull().sum(axis=0).values


        # find indeces where nanAmount is minimal
        ntuples          = np.where(nanAmountMatrix == np.nanmin(nanAmountMatrix))
        i_time_sel       = ntuples[0][0]
        i_height_sel     = ntuples[1][0]

        # extract corresponding time Serie of mean Doppler velocity values for the chirp
        valuesTimeSerie  = values[i_time_sel:i_time_sel+NtimeStampsRun, i_height_sel]
        timeSerie        = datetime[i_time_sel:i_time_sel+NtimeStampsRun]
        heightSerie      = np.repeat(rangeHeight[i_height_sel], NtimeStampsRun)


        ###### adding test for columns ########
        valuesColumn     = values[i_time_sel:i_time_sel+NtimeStampsRun, :]
        valuesColumnMean = np.nanmean(valuesColumn, axis=1)
    else:
        nanAmountMatrix = np.zeros((len(datetime), len(rangeHeight)))
        nanAmountMatrix[~np.isnan(values)] = 1
        # count number of nans in each height
        NnansCol = []
        for indh in range(len(rangeHeight)):
            NnansCol.append(np.sum(~np.isnan(values[:,indh])))
        print(NnansCol)
        i_height_sel = np.argmax(NnansCol)
        print('height found ', i_height_sel)
        i_time_sel = np.arange(0,len(datetime)-1)
        valuesTimeSerie = values[:, i_height_sel]
        timeSerie = datetime
        heightSerie = np.repeat(rangeHeight[i_height_sel], len(datetime))
        valuesColumnMean = np.nanmean(values, axis=1)

    # plotting quicklooks of the values map and the picked time serie interval
    labelsizeaxes    = 12
    fontSizeTitle    = 12
    fontSizeX        = 12
    fontSizeY        = 12
    cbarAspect       = 10
    fontSizeCbar     = 12
    fig, ax          = plt.subplots(nrows=1, ncols=1, figsize=(12,6))
    rcParams['font.sans-serif'] = ['Tahoma']
    matplotlib.rcParams['savefig.dpi'] = 100
    plt.gcf().subplots_adjust(bottom=0.15)
    fig.tight_layout()
    ax               = plt.subplot(2,1,1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    matplotlib.rc('xtick', labelsize=labelsizeaxes)  # sets dimension of ticks in the plots
    matplotlib.rc('ytick', labelsize=labelsizeaxes)  # sets dimension of ticks in the plots
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.xaxis.set_minor_formatter(mdates.DateFormatter("%H:%M"))
    ax.xaxis_date()
    if len(datetime)-NtimeStampsRun > 0:
        cax              = ax.pcolormesh(datetime[:-NtimeStampsRun], rangeHeight, nanAmountMatrix.transpose(), vmin=0., vmax=200., cmap='viridis')
        ax.set_xlim(datetime[0], datetime[-200])                                 # limits of the x-axes

    else:
        cax              = ax.pcolormesh(datetime[:], rangeHeight, nanAmountMatrix.transpose(), vmin=0., vmax=200., cmap='viridis')
        ax.set_xlim(datetime[0], datetime[-1])                                 # limits of the x-axes

    #ax.scatter(timeSerie, heightSerie, s=nanAmountSerie, c='orange', marker='o')
    ax.plot(timeSerie, heightSerie, color='orange', linewidth=7.0)
    ax.set_ylim(rangeHeight[0],rangeHeight[-1]+200.)                                               # limits of the y-axesn  cmap=plt.cm.get_cmap("viridis", 256)
    ax.set_title('time-height plot for the day : '+date, fontsize=fontSizeTitle, loc='left')
    ax.set_xlabel("time [hh:mm]", fontsize=fontSizeX)
    ax.set_ylabel("height [m]", fontsize=fontSizeY)
    cbar             = fig.colorbar(cax, orientation='vertical', aspect=cbarAspect)
    cbar.set_label(label='Nan Amount [%]', size=fontSizeCbar)
    cbar.ax.tick_params(labelsize=labelsizeaxes)


    ax = plt.subplot(2,1,2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    matplotlib.rc('xtick', labelsize=labelsizeaxes)  # sets dimension of ticks in the plots
    matplotlib.rc('ytick', labelsize=labelsizeaxes)  # sets dimension of ticks in the plots
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.xaxis.set_minor_formatter(mdates.DateFormatter("%H:%M"))
    ax.xaxis_date()
    ax.plot(timeSerie,valuesTimeSerie, color='black', label='selected values at '+str(heightSerie[0]))
    ax.legend(frameon=False)
    ax.set_xlim(timeSerie[0], timeSerie[-1])                                 # limits of the x-axes
    fig.tight_layout()
    fig.savefig(pathFig+date+'_chirp_'+chirp+'_mdvSelected4Timeshift.png', format='png')

    return(valuesTimeSerie, i_height_sel, timeSerie, valuesColumnMean)
def f_calcTimeShift(w_radar_meanCol, DeltaTimeShift, w_ship_chirp, timeSerieRadar, pathFig, chirp, date, hour):
    '''             w_radar_meanCol, DeltaTimeShift, w_ship_chirpSel, timeRadar_valid, pathFig, chirp, date, hour
    author: Claudia Acquistapace, Jan. H. Schween
    date; 25/11/2020
    goal: calculate and estimation of the time lag between the radar time stamps and the ship time stamp

    NOTE: adding or subtracting the obtained time shift depends on what you did
    during the calculation of the covarainces: if you added/subtracted time _shift
    to t_radar you have to do the same for the 'exact time'
    if you added/subtracted it to t_ship you have to subtract/add it to t_radar.
    Here is the time shift anaylysis as plot:
    <ww> is short for <w'_ship*w'_radar> i.e. covariance between vertical speeds from
    ship movements and radar its maximum gives an estimate for optimal agreement in
    vertical velocities of ship and radar
    <Delta w^2> is short for <(w[i]-w[i-1])^2> where w = w_rad - 2*w_ship - this
    is wat i called yesterday a measure for the stripeness. Its minimum gives an
    estimate how to get the smoothest w data

    Parameters
    ----------
    INPUT:
    w_radar : ndarray
        DESCRIPTION: mdv time serie of NtimeStampsRun elements selected from the mdv matrix of the radar
    DeltaTimeShift : array of time shifts to be used to calculate shifted series and derive covariances
        DESCRIPTION.
    Cs : TYPe: interpolation
        DESCRIPTION: cubic interpolation of ship velocity
    w_ship_valid,
    timeShip_valid,
    timeSerieRadar : datetime array
        DESCRIPTION.time array corresponding to the w_radar serie of values
    pathFig : TYPE string
        DESCRIPTION. output path for quicklooks
    i_chirp : TYPE string
        DESCRIPTION. string indicating the chirp processed
    date : TYPE string
        DESCRIPTION. date
    hour : TYPE string
        DESCRIPTION. hour
    OUTPUT:
    timeShift_chirp: TYPE ndarray of dimension equal to the number of chirps
        DESCRIPTION: time lag for each chirp in seconds
    -------

    '''
    from scipy.interpolate import CubicSpline

    labelsizeaxes   = 12
    fontSizeTitle = 12
    fontSizeX = 12
    fontSizeY = 12
    rcParams['font.sans-serif'] = ['Tahoma']
    matplotlib.rcParams['savefig.dpi'] = 100
    plt.gcf().subplots_adjust(bottom=0.15)

    # calculating variation for w_radar
    w_prime_radar = w_radar_meanCol - np.nanmean(w_radar_meanCol)

    print(np.shape(w_prime_radar))
    # calculating covariance between w-ship and w_radar where w_ship is ahifted for each deltaT given gy DeltaTimeShift
    cov_ww      = np.zeros(len(DeltaTimeShift))
    deltaW_ship = np.zeros(len(DeltaTimeShift))

    for i in range(len(DeltaTimeShift)):

        # calculate w_ship interpolating it on the new time array (timeShip+deltatimeShift(i))
        T_corr = pd.to_datetime(timeSerieRadar) + timedelta(seconds=DeltaTimeShift[i])

        # interpolating w_ship on the shifted time serie
        cs_ship = CubicSpline(timeSerieRadar, w_ship_chirp)
        w_ship_shifted = cs_ship(pd.to_datetime(T_corr))

        #calculating w_prime_ship with the new interpolated serie
        w_ship_prime = w_ship_shifted - np.nanmean(w_ship_shifted)

        # calculating covariance of the prime series
        cov_ww[i] = np.nanmean(w_ship_prime*w_prime_radar)

        # calculating sharpness deltaW_ship
        w_corrected = w_radar_meanCol - w_ship_shifted
        delta_w = (np.ediff1d(w_corrected))**2
        deltaW_ship[i] = np.nanmean(delta_w)


    #calculating max of covariance and min of deltaW_ship
    minDeltaW = np.nanmin(deltaW_ship)
    indMin    = np.where(deltaW_ship == minDeltaW)
    maxCov_w  = np.nanmax(cov_ww)
    indMax    = np.where(cov_ww == maxCov_w)
    print('time shift found for the chirp '+str(DeltaTimeShift[indMin][0]))

    # calculating time shift for radar data
    timeShift_chirp = DeltaTimeShift[indMin][0]

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12,6))
    fig.tight_layout()
    ax = plt.subplot(1,1,1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    matplotlib.rc('xtick', labelsize=labelsizeaxes)  # sets dimension of ticks in the plots
    matplotlib.rc('ytick', labelsize=labelsizeaxes)  # sets dimension of ticks in the plots
    ax.plot(DeltaTimeShift, cov_ww, color='red', linestyle=':', label='cov_ww')
    ax.axvline(x=DeltaTimeShift[indMax], color='red', linestyle=':', label='max cov_w')
    ax.plot(DeltaTimeShift, deltaW_ship, color='red', label='Deltaw^2')
    ax.axvline(x=DeltaTimeShift[indMin], color='red', label='min Deltaw^2')
    ax.legend(frameon=False)
    #ax.xaxis_date()
    ax.set_ylim(-0.1,2.)                                               # limits of the y-axesn  cmap=plt.cm.get_cmap("viridis", 256)
    ax.set_xlim(DeltaTmin,DeltaTmax)                                 # limits of the x-axes
    ax.set_title('covariance and sharpiness for chirp '+str(i_chirp)+': '+date+' '+hour+':'+str(int(hour)+1)+', time lag found : '+str(DeltaTimeShift[indMin]), fontsize=fontSizeTitle, loc='left')
    ax.set_xlabel("time Shift [seconds]", fontsize=fontSizeX)
    ax.set_ylabel('w [m s-1]', fontsize=fontSizeY)
    fig.tight_layout()
    fig.savefig(pathFig+date+'_'+hour+'_'+str(i_chirp)+'_timeShiftQuicklook.png', format='png')

    return(timeShift_chirp)
def f_convertHourString2Int(radarFileName, n_char):

    '''
    author: Claudia Acquistapace
    date  : 17/12/2020
    goal  : extract from radar file name the string corresponding to the hour and
    read the corresponding integer. The function conforms to the following radar file
    name convention: 'msm94_msm_200120_000001_P07_ZEN_v2.nc'
    input : radar file name (string)
    output: hourString, hourInt
    '''
    hourString = radarFileName[-17:-15]
    if hourString[0] == '0':
        hourInt = int(hourString[-1])
    else:
        hourInt = int(hourString)
    return(hourString, hourInt)

# generating array of days for the dataset
Eurec4aDays     = pd.date_range(datetime(2020,1,19),datetime(2020,2,19),freq='d')
NdaysEurec4a    = len(Eurec4aDays)
#%%
for indDay in range(NdaysEurec4a):

    dayEu           = Eurec4aDays[indDay]
    dayEu           = Eurec4aDays[9]
    yy              = str(dayEu)[0:4]
    mm              = str(dayEu)[5:7]
    dd              = str(dayEu)[8:10]
    date            = yy+mm+dd
    # setting dates strings
    date            = dd+mm+yy      #'04022020'
    dateRadar       = yy[0:2]+mm+dd #'200204'
    dateReverse     = yy+mm+dd      #'20200204'


    # establishing paths for data input and output
    pathRadar       = '/net/norte/EUREC4Aprocessed_20210609'+yy+'/'+mm+'/'+dd+'/'
    pathFig         = '/work/cacquist/w_band_eurec4a_LWP_corr/plots/'+yy+'/'+mm+'/'+dd+'/'
    file_list       = np.sort(glob.glob(pathRadar+'msm94_msm_'+dateRadar+'*ZEN.nc'))
    tech_file_list = np.sort(glob.glob(pathRadar+'msm94_msm_'+dateRadar+'*_ZEN_technical.nc'))
    pathOutData     = '/work/cacquist/w_band_eurec4a_LWP_corr/'+yy+'/'+mm+'/'+dd+'/'

    # loop on hours for the selected day
    for indHour in range(len(file_list)):

        # reading radar file name
        radarFileName = file_list[indHour]

        # reading hour string and int quantity
        hour, hourInt =  f_convertHourString2Int(radarFileName, len(pathRadar))

        print('hour: ',hour)

        # check if file has already been processed or not
        if os.path.isfile(pathOutData+date+'_'+hour+'msm94_msm_ZEN_corrected.nc'):
            print(date+'_'+hour+' - already processed')
        else:

            # check if folder for plots and for outputs exists, otherwise creates it
            PathFigHour     = pathFig+hour+'/'
            Path(PathFigHour).mkdir(parents=True, exist_ok=True)
            Path(pathOutData).mkdir(parents=True, exist_ok=True)

            # reading corresponding correction term file
            corr_term_filename = np.sort(glob.glob('/data/obs/campaigns/eurec4a/msm/wband_radar/ncdf/'+yy+'/'+mm+'/'+dd+'/'+dd+mm+yy+'_'+hour+'*'))

            # reading correction file
            w_ship_shifted = xr.open_dataset(corr_term_filename[0])

            print('* read radar data for the selected hour')

            if os.path.exists(file_list[indHour]):
                radarData          = xr.open_dataset(file_list[indHour])
                techData           = xr.open_dataset(tech_file_list[indHour])
                mdv                = radarData['vm'].values
                datetimeRadar      = radarData['time'].values
                rangeRadar         = radarData['range'].values
                mdv[mdv == -999.]  = np.nan
                range_offset       = rangeRadar[radarData['range_offsets'].values]
                indRangeChirps     = radarData['range_offsets'].values
                millisec           = techData['sample_tms'].values/1000.
                NtimeRadar         = len(datetimeRadar)
                sampleDur          = techData['SampDur'].values # full sample duration [s]
                MaxVel             = techData['nqv'].values
                seq_avg            = techData['seq_avg'].values # number of averaged chirps in each chirp
                chirpRepFreq       = (4 * MaxVel * 94.*10)/ 3.
                chirpDuration      = 1/chirpRepFreq
                chirpIntegrations  = chirpDuration*seq_avg
                Nchirps            = len(chirpIntegrations)

                # calculating exact radar time stamp for every chirp (assumption: radar time stamp is the final time stamp of the interval)
                datetimeChirps     = f_calculateExactRadarTime(millisec, chirpIntegrations, datetimeRadar)
                timeChirp1         = pd.to_datetime(datetimeChirps[0,:])
                timeChirp2         = pd.to_datetime(datetimeChirps[1,:])
                timeChirp3         = pd.to_datetime(datetimeChirps[2,:])



                # calculation of time lag between radar time serie and ship time serie
                print('* calculate time shift between ship and radar time for each chirp ')

                # reading ship data
                denTimeSerie      = w_ship_shifted['den_scalar_product'].values
                timeShip          = w_ship_shifted['time_shifted'].values
                w_ship            = w_ship_shifted['correction_term'].values

                # assigning lenght of the mean doppler velocity time serie for calculating time shift
                # with 3 sec time resolution, 200 corresponds to 10 min
                NtimeStampsRun   = 200


                # time shift array to be tested
                DeltaTmin        = -3.
                DeltaTmax        = 3.
                res              = 0.05
                DimDeltaT        = (DeltaTmax- DeltaTmin)/res
                DeltaTimeShift   = np.arange(DeltaTmin, DeltaTmax, step=res)

                # calculating time shift and corrected time array for radar mean doppler velocity for each chirp
                timeShiftArray   = np.zeros((Nchirps))
                timeShiftArray.fill(-999.)
                heightSelArray   = np.zeros((Nchirps))
                heightSelArray.fill(-999.)
                timeBeginSamples = np.zeros((Nchirps), dtype='datetime64[ns]')
                timeEndSamples = np.zeros((Nchirps), dtype='datetime64[ns]')
                timeExactFinal   = np.zeros((Nchirps, len(datetimeRadar)), dtype='datetime64[ns]')


                for i_chirp in range(0, Nchirps):
                    print('- processing chirp '+str(i_chirp))

                    #assigning string identifying the chirp that is processed
                    chirp = 'chirp_'+str(i_chirp)

                    # selecting index of min and max height of the chirp
                    i_h_min     = f_closest(rangeRadar, range_offset[i_chirp])
                    if i_chirp+1 != 3:
                        i_h_max = f_closest(rangeRadar, range_offset[i_chirp+1])
                    else:
                        i_h_max = -1

                    # reading radar time array of the chirp
                    if i_chirp == 0:
                        timeSerieRadar   = timeChirp1
                    elif i_chirp == 1:
                        timeSerieRadar   = timeChirp2
                    else:
                        timeSerieRadar   = timeChirp3

                    print('- finding best mdv time seried between '+str(rangeRadar[i_h_min])+' and '+str(rangeRadar[i_h_max]))

                    # reading mdv values for the selected chirp
                    mvd_chirp  = mdv[:,i_h_min:i_h_max]
                    dimHchirp  = np.shape(mvd_chirp)[1]
                    rangeChirp = rangeRadar[i_h_min:i_h_max]


                    # search for at least 10 min of consecutive w obs in the chirp
                    w_radar, indHeightBest, timeRadarSel, w_radar_meanCol = f_findMdvTimeSerie_VMerge(mvd_chirp, \
                                                                            timeSerieRadar, \
                                                                            rangeChirp, \
                                                                            NtimeStampsRun, \
                                                                            PathFigHour, \
                                                                            chirp)
                    # adapting index for the chirp matrix to the entire matrix by adding i_h_min-1
                    indHeightBest = indHeightBest+i_h_min-1

                    # reading correction time serie corresponding to the selected height in the selected time interval
                    i_time_ship_best  = (pd.to_datetime(timeShip) >= timeRadarSel[0]) * (pd.to_datetime(timeShip) <= timeRadarSel[-1])
                    w_ship_best       = w_ship[i_time_ship_best, indHeightBest]

                    heightSelArray[i_chirp] = indHeightBest
                    timeBeginSamples[i_chirp] = pd.to_datetime(timeRadarSel[0])
                    if len(timeRadarSel) >= 40:
                        timeEndSamples[i_chirp] = pd.to_datetime(timeRadarSel[40])
                        TimeEndPlot     = timeRadarSel[40]

                    else:
                        timeEndSamples[i_chirp] = pd.to_datetime(timeRadarSel[-1])
                        TimeEndPlot     = timeRadarSel[-1]

                    print(' - calculating time shift for the chirp '+str(i_chirp))
                    # calculating time shift for the chirp
                    if np.sum(np.where(~np.isnan(w_radar))) != 0:

                        i_valid    = np.where(~np.isnan(w_radar_meanCol))
                        w_valid    = w_radar_meanCol[i_valid]
                        time_valid = timeRadarSel[i_valid]
                        if (len(w_valid) > 10):
                            # interpolating mean radar mdv speed on the ship time resolution for time shift calculation
                            CS_rad                  = CubicSpline(time_valid, w_valid)
                            WChirpshipRes           = CS_rad(pd.to_datetime(timeShip[i_time_ship_best]))

                            # calculating time shift
                            timeShiftArray[i_chirp] = f_calcTimeShift(WChirpshipRes, \
                                                                      DeltaTimeShift, \
                                                                      w_ship_best, \
                                                                      pd.to_datetime(timeShip[i_time_ship_best]), \
                                                                      PathFigHour, \
                                                                      chirp, \
                                                                      date, \
                                                                      hour)
                        else:
                            timeShiftArray[i_chirp] = np.nan
                    else:

                        timeShiftArray[i_chirp] = np.nan

                    print('- recalculating exact time for ship data ')
                    # recalculating exact time including time shift due to lag
                    if ~np.isnan(timeShiftArray[i_chirp]):
                        timeExact  = pd.to_datetime(timeSerieRadar) - timedelta(seconds=timeShiftArray[i_chirp])

                    else:
                        timeExact  = pd.to_datetime(timeSerieRadar)

                    timeExactFinal[i_chirp, :] = pd.to_datetime(timeExact)

                    # interpolating ship data on radar time
                    Cs_ship        = CubicSpline(timeShip, w_ship[:, indHeightBest])
                    W_shipRadar    = Cs_ship(pd.to_datetime(timeSerieRadar))

                    # interpolating ship data on the exact time again, with the right time found after the calculating
                    # and the addition of the time shift
                    CS_exact        = CubicSpline(timeExact, W_shipRadar)
                    W_ship_exact    = CS_exact(pd.to_datetime(timeSerieRadar))
                    timePlot        = pd.to_datetime(timeExact)
                    TimeBeginPlot   = timeRadarSel[0]


                    labelsizeaxes   = 12
                    fontSizeTitle   = 12
                    fontSizeX       = 12
                    fontSizeY       = 12
                    cbarAspect      = 10
                    fontSizeCbar    = 12
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
                    ax.xaxis_date()
                    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
                    major_ticks = np.arange(TimeBeginPlot, TimeEndPlot, 50000000, dtype='datetime64')
                    minor_ticks = np.arange(TimeBeginPlot, TimeEndPlot, 1000000, dtype='datetime64')
                    ax.tick_params(which = 'both', direction = 'out')
                    ax.set_xticks(major_ticks)
                    ax.set_xticks(minor_ticks, minor=True)
                    ax.grid(which='both')
                    ax.grid(which='minor', alpha=0.2)
                    ax.grid(which='major', alpha=0.5)
                    ax.plot(timeRadarSel, w_radar_meanCol, color='red', label='mean w_radar over column')
                    ax.plot(timeRadarSel, w_radar, linewidth = 0.2, color='red', label='w_radar at one height')
                    ax.plot(timeSerieRadar, W_shipRadar, color='blue', linewidth=0.2, label='w_ship original')
                    ax.plot(timePlot, W_shipRadar, color='blue', label='w_ship shifted of deltaT found')
                    ax.scatter(timeSerieRadar, W_ship_exact, color='green', label='w_ship shifted interpolated on radar exact time')
                    ax.set_ylim(-4.,2.)
                    ax.legend(frameon=False)
                    ax.set_xlim(TimeBeginPlot,TimeEndPlot)                                 # limits of the x-axes
                    ax.set_title('velocity for time delay calculations : '+date+' '+hour+':'+str(int(hour)+1)+' shift = '+str(timeShiftArray[i_chirp]), fontsize=fontSizeTitle, loc='left')
                    ax.set_xlabel("time [hh:mm:ss]", fontsize=fontSizeX)
                    ax.set_ylabel('w [m s-1]', fontsize=fontSizeY)
                    fig.tight_layout()
                    fig.savefig(PathFigHour+date+'_'+hour+'_chirp_'+str(i_chirp)+'_quicklook_correction.png', format='png')




                print(' ### time shifts found for each chirp: ', timeShiftArray)

                print('* calculating corrected mean Doppler velocity')

                # calculating now correction term based on the time shift found
                mdv_corrected = np.zeros((len(datetimeRadar), len(rangeRadar)))
                mdv_fake      = np.zeros((len(datetimeRadar), len(rangeRadar)))
                # loop on range gates: for each height, find the exact radar time
                for indHeight in range(len(rangeRadar)):
                    mdvTimeSerie = mdv[:, indHeight]

                    # assigning radar time chirp based on the range selected in the loop
                    if indHeight <= indRangeChirps[1]:
                        timeExactShipChirp = timeExactFinal[0, :]
                        timeSerieRadar   = timeChirp1
                    if  (indHeight > indRangeChirps[1]) * (indHeight <= indRangeChirps[2]):
                        timeExactShipChirp = timeExactFinal[1, :]
                        timeSerieRadar   = timeChirp2
                    elif (indHeight > indRangeChirps[2]) * (indHeight <= 550):
                        timeExactShipChirp = timeExactFinal[2, :]
                        timeSerieRadar   = timeChirp3

                    # interpolating w_ship and correction denominator on radar time stamps
                    Cs_ship        = CubicSpline(timeShip, w_ship[:, indHeight])
                    W_shipRadar    = Cs_ship(pd.to_datetime(timeSerieRadar))
                    Cs_den         = CubicSpline(timeShip, denTimeSerie)
                    denShipRadar   = Cs_den(pd.to_datetime(timeSerieRadar))


                    # interpolating w_ship shifted of the time gap on the radar time stamps
                    CS_exact_w     = CubicSpline(timeExactShipChirp, W_shipRadar)
                    W_ship_exact   = CS_exact_w(pd.to_datetime(timeSerieRadar))

                    # interpolating den shifted of the time gap on the radar time stamps
                    CS_exact_den   = CubicSpline(timeExactShipChirp, denShipRadar)
                    den_exact      = CS_exact_den(pd.to_datetime(timeSerieRadar))

                    # calculating corrected mdv for radar time stamps
                    mdv_corrected[:, indHeight] = - mdvTimeSerie/den_exact + W_ship_exact/den_exact
                    mdv_fake[:, indHeight]      = - mdvTimeSerie/denShipRadar + W_shipRadar/denShipRadar



            else:
                print ('File for '+dateRadar+'_'+hour+' does not exist')

            # applying rolling average to the data
            print('* Applying running mean on 3 time steps for improving data quality')
            if np.sum(~np.isnan(mdv_corrected)) != 0:
                df        = pd.DataFrame(mdv_corrected, index=datetimeRadar, columns=rangeRadar)
                mdv_roll3 = df.rolling(window=3,center=True, axis=0).apply(lambda x: np.nanmean(x))
            else:
                mdv_r3    = np.zeros((len(datetimeRadar), len(rangeRadar)), dtype=float)
                mdv_r3[:] = np.nan
                mdv_roll3 = pd.DataFrame(mdv_r3, index=datetimeRadar, columns=rangeRadar)

            # saving corrrected mean doppler velocities matrices to be added to radar data
            # save xarray dataset containing the correction terms for the hour
            print('* saving data in ncdf format')
            dims2             = ['time','height']
            coords2           = {"time":datetimeRadar, "height":rangeRadar}
            mdv_corr          = xr.DataArray(dims=dims2, coords=coords2, data=mdv_corrected,
                                     attrs={'long_name':'Mean Doppler velocity corrected \
                                            using algorithm described in \
                                                https://github.com/ClauClouds/ship-motion-correction-for-EUREC4A-campaign',
                                            'units':'ms-1'})
            mdv_corr_smooth   = xr.DataArray(dims=dims2, coords=coords2, data=mdv_roll3.values,
                                     attrs={'long_name':'Mean Doppler velocity corrected \
                                            using algorithm described in \
                                                https://github.com/ClauClouds/ship-motion-correction-for-EUREC4A-campaign \
                                                and smoothed using 3 time steps running mean',
                                            'units':'ms-1'})

            global_attributes = {'CREATED_BY'     : 'Claudia Acquistapace',
                                 'CREATED_ON'       : str(datetime.now()),
                                 'FILL_VALUE'       : 'NaN',
                                 'PI_NAME'          : 'Claudia Acquistapace',
                                 'PI_AFFILIATION'   : 'University of Cologne (UNI), Germany',
                                 'PI_ADDRESS'       : 'Institute for geophysics and meteorology, Pohligstrasse 3, 50969 Koeln',
                                 'PI_MAIL'          : 'cacquist@meteo.uni-koeln.de',
                                 'DO_NAME'          : 'University of Cologne - Germany',
                                 'DO_AFFILIATION'   : 'University of Cologne - Germany',
                                 'DO_address'       : 'Institute for geophysics and meteorology, Pohligstrasse 3, 50696 Koeln',
                                 'DO_MAIL'          : 'cacquist@meteo.uni-koeln.de',
                                 'DS_NAME'          : 'University of Cologne - Germany',
                                 'DS_AFFILIATION'   : 'University of Cologne - Germany',
                                 'DS_address'       : 'Institute for geophysics and meteorology, Pohligstrasse 3, 50696 Koeln',
                                 'DS_MAIL'          : 'cacquist@meteo.uni-koeln.de',
                                 'DATA_DESCRIPTION' : 'hourly cloud radar measurements on Maria S. Merian (msm) ship during EUREC4A campaign',
                                 'DATA_DISCIPLINE'  : 'Atmospheric Physics - Remote Sensing Radar Profiler',
                                 'DATA_GROUP'       : 'Experimental;Profile;Moving',
                                 'DATA_LOCATION'    : 'Research vessel Maria S. Merian - Atlantic Ocean ',
                                 'DATA_SOURCE'      : 'Radar.Standard.Moments processed using the standard code available at \
                                     uni_koeln https://github.com/igmk/w-radar/tree/master/scripts',
                                 'DATA_PROCESSING'  : 'additional data processing is performed for applying ship motion correction. \
                                     the code used is available at https://github.com/ClauClouds/ship-motion-correction-for-EUREC4A-campaign',
                                 'INSTRUMENT_MODEL' : '94 GHz dual polatization FMCW band radar',
                                 'MDF_PROGRAM_USED' : 'EUREC4A',
                                 'COMMENT'          : 'The file is the product of two processings. First data are \
                                     processed using the standard matlab script from UNI. Then , the ship motion \
                                         correction algorithm is applied to the mean Doppler velocity field. The code \
                                             and a description of the algorithm are available at (see DATA_PROCESSING. \
                                                 Two additional variables are provided: \
                                   * vm_corrected, corresponding to the mean Doppler velocity field \
                                  obtained by applying the ship motion correction algorithm,  \
                                  * vm_corrected_smoothed,   running mean over 3 time steps (approx 9 sec interval) \
                                      of the variable vm_corrected. Mean Doppler velocity is smoothed over 9 sec \
                                          time interval for quicklooks applications' }

            print('adding LWP variable to the dataset')
            # reading corresponding LWP Data
            LWP_filename = np.sort(glob.glob('/work/cacquist/w_band_eurec4a_LWP_corr/Reprocessed/netcdf/'+dateRadar+'_'+hour+'*'))
            print(LWP_filename)
            nc_data = xr.open_dataset(radarFileName)
            LWP_data = xr.open_dataset(LWP_filename[0])
            LWP = LWP_data.LWP.values
            LWP_da = xr.DataArray(data=LWP,
                     dims=["time"],
                     coords=dict(
                                time=nc_data.time.values,
                                ),
                     attrs=dict(
                                long_name='atmosphere_cloud_liquid_water_content',
                                description="Amount of liquid water integrated from surface to the highest radar range gate",
                                units="gm-2",
                     ))

            # adding LWP variable to the dataset
            radarData = radarData.assign({'LWP':LWP_da})

            print('***********************************')




            # storing the new variables in the file
            radarData['vm_corrected']          = mdv_corr
            radarData['vm_corrected_smoothed'] = mdv_corr_smooth
            radarData.attrs                    = global_attributes
            radarData.to_netcdf(pathOutData+date+'_'+hour+'msm94_msm_ZEN_corrected.nc')
