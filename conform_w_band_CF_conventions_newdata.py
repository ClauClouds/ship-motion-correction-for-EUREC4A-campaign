#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 21.04.2021
@author: Claudia Acquistapace
@goal: produce daily reduced files for eurec4a intake data browser

"""
from datetime import datetime
import xarray as xr
import numpy as np
import pandas as pd
import glob


#NOTE: TO BE RUN ON SECAIRE


# Preprocessing of the data compact.nc
def generate_preprocess(): # can take arguments

    '''
    author: Claudia Acquistapace
    date  : 29/10/2020
    goal  : generate a function that  does the preprocessing of the data
    '''
    import numpy as np
    import xarray as xr

    # Generate a preprocess function that takes a dataset as only argument and does all the processing needed
    def preprocess(ds):
        '''
        author: Claudia Acquistapace
        date : 29/10/2020, modified jan 2021
        goal : preprocess wband radar ncdf data to make them readable with xr.open_fmdataset instruction
        The preprocessing includes:
            - removal of non useful variables :
                var2DropList = ['source_rr', 'source_rh', 'source_ta', 'source_pa', 'source_wspeed', 'source_wdir', 'wl']
            - redefinition of scalar variables as time series
                varList = ['lat','lon','zsl','freq_sb','hpbw']
        input:
            dataset xarray of data from radar

        output:
            new dataset with var2DropList variables removed and varList variables modified

        '''
        # assign new coordinate for dependencies not specified
        ds = ds.assign_coords({'scalar':np.arange(1),'number.chirp.sequences':np.arange(3)})

        # retrieving the list of variables of the dataset
        var2DropList = list(ds.keys())
        # escluding from the list of variables to remove the  moments to be plotted
        Var2keep = ['vm_corrected_smoothed', 'ze', 'sw', 'sk','rr', 'rh', 'ta',\
                    'pa', 'wspeed', 'wdir', 'LWP', 'tb']
        for ind in range(len(Var2keep)):
            var2DropList.remove(Var2keep[ind])
        print('variable list ', var2DropList)
        # drop variables that have no interest
        #var2DropList = ['source_rr', 'source_rh', 'source_ta', 'source_pa', 'source_wspeed', 'source_wdir', 'wl', 'nqv']

        for indDrop in range(len(var2DropList)):
            ds = ds.drop(var2DropList[indDrop])

        # Convert all variables with scalar dimension to time
        # list of variables to convert
        #varList = ['lat','lon','zsl','freq_sb','hpbw']
        #varUnits = ['degrees', 'degrees', 'm', 's-1', 'degrees']
        #varLongNames = ['latitude',
        #        'longitude',
        ##        'Altitude above mean sea level the instrument is located',
        #        'Central transmission frequency ',
        #        'Antenna half power beam width']

        #loop on variables to convert
        #for iVar in range(len(varList)):
        #    var = varList[iVar]
        #
        #    # storing single value of the variable
        #    varValue = ds[var].values
        #
        #    # deleting the variable from the dataset
        #    ds = ds.drop(var)#

            # creating new variable time serie
        #    varTimeSerie = np.repeat(varValue, len(ds['time'].values))#

            # saving variable in the dataset
        #    ds[var] = (['time'],varTimeSerie)
        #    dim     = ['time']
        #    coord   = {'time':ds['time'].values}
        #    VarDataArray = xr.DataArray(dims=dim, coords=coord, data=varTimeSerie,
        #                                  attrs={'long_name':varLongNames[iVar],
        #                                         'units':varUnits[iVar]})
            # adding new Var variable to the dataset
       #     ds = ds.assign({var:VarDataArray})

        #ds['time'] = pd.to_datetime(ds.time.values.astype(int).astype(str), format='%Y%m%d') + pd.to_timedelta(ds.time.values%1, unit='D').round('1s')
        return ds
    return preprocess


# generating array of days for the dataset
Eurec4aDays  = pd.date_range(datetime(2020, 1, 19), datetime(2020, 2, 19), freq='d')
NdaysEurec4a = len(Eurec4aDays)
radar_name = 'msm'

# reading ship data for lat/lon
shipFile = '/data/obs/campaigns/eurec4a/msm/ship data/shipData_all2.nc'
shipData = xr.open_dataset(shipFile)


for indDay in range(NdaysEurec4a):
    # select corresponding  date
    #dayEu           = Eurec4aDays[indDay]
    dayEu           = Eurec4aDays[9]

    # extracting strings for yy, dd, mm
    yy              = str(dayEu)[0:4]
    mm              = str(dayEu)[5:7]
    dd              = str(dayEu)[8:10]

    # setting dates strings
    date            = dd+mm+yy      #'04022020'
    dateRadar       = yy[0:2]+mm+dd #'200204'
    dateReverse     = yy+mm+dd      #'20200204'


    # paths to the different data files and output directories for plots
    pathFolderTree  = '/net/ostro/cacquist/w_band_eurec4a_LWP_corr/'+yy+'/'+mm+'/'+dd+'/'
    path_out = '/net/ostro/cacquist/w_band_eurec4a_LWP_corr/'+yy+'/'+mm+'/'+dd+'/'

    pathRadar          = '/net/ostro/cacquist/w_band_eurec4a_LWP_corr/'+yy+'/'+mm+'/'+dd+'/'
    radarFileList      = np.sort(glob.glob(pathRadar+date+'_*msm94_msm_ZEN_corrected.nc'))
    Nfiles             = len(radarFileList)

    print('reading mean Doppler velocity from radar files. Nfiles :', len(radarFileList))#


    print('processing day : '+yy+'/'+mm+'/'+dd)
    print("*************************************************")
    day_data = xr.open_mfdataset(radarFileList,
                                   concat_dim = 'time',
                                   data_vars = 'minimal',
                                   preprocess = generate_preprocess(),
                                  )



    # substitute xarray time coordinate with unix time type int
    time = day_data['time'].values
    rr = day_data['rr'].values
    rh = day_data['rh'].values
    ta = day_data['ta'].values
    pa = day_data['pa'].values
    w_speed = day_data['wspeed'].values
    w_dir = day_data['wdir'].values
    lwp = day_data['LWP'].values
    ze = day_data['ze'].values
    vd = day_data['vm_corrected_smoothed'].values
    sw = day_data['sw'].values
    sk = day_data['sk'].values
    height = day_data['height'].values
    tb = day_data['tb'].values

    # selecting ship data
    ship_data_day = shipData.sel(time=slice(pd.to_datetime(time[0]), pd.to_datetime(time[-1])))

    # interpolating ship data on radar time stamps
    ship_day_interp = ship_data_day.interp(time=time, method='nearest')
    # reading lat/lon for the trajectory
    lat_day = ship_day_interp['lat'].values
    lon_day = ship_day_interp['lon'].values



    # saving data in CF convention dataset
    WbandData = xr.Dataset(
        data_vars={
            "rain_rate": (('time',), rr, {'long_name': 'surface rain rate', 'units':'mm h-1', "standard_name": "rainfall_rate"}),
            'relative_humidity':(('time',), rh*0.01, {'long_name': 'surface relative humidity', 'units':'1', "standard_name": "relative_humidity"}),
            'air_temperature': (('time',), ta, {'long_name': 'surface temperature', 'units':'degC', "standard_name": "air_temperature"}),
            'air_pressure':(('time',), pa, {'long_name': 'surface air pressure', 'units':'hPa', "standard_name": "surface_air_pressure"}),
            'wind_speed':(('time',), w_speed, {'long_name': 'surface horizontal wind speed', "standard_name": "wind_speed", 'units':'m s-1'}),
            'wind_direction':(('time',), w_dir, {'long_name': 'surface wind direction', 'units':'degrees', "standard_name": "wind_from_direction"}),
            'liquid_water_path':(('time',), lwp,{'long_name': 'liquid water path', 'standard_name':'atmosphere_cloud_liquid_water_content', 'units':'g m-2','comment':'retrieval based on neural networks developed by the manufacturer'}),
            'brightness_temperature':(('time',), tb,{'long_name': 'brightness temperature at 89 GHz', 'standard_name':'brightness_temperature','units':'K'}),
            'mean_doppler_velocity':(('time','height'), vd, {'long_name': 'mean Doppler velocity', 'units':'m s-1', 'comment':'mean Doppler velocity after correction for ship motions and smoothing'}),
            'radar_reflectivity':(('time','height'), 10*np.log10(ze), {'long_name': 'equivalent reflectivity factor', 'standard_name':'equivalent_reflectivity_factor', 'units':'dBZ'}),
            'spectral_width':(('time','height'), sw, {'long_name': 'spectral width', 'units':'m s-1'}),
            'skewness':(('time','height'), sk, {'long_name': 'skewness', 'units':''}),
        },
        coords={
            "time": (('time',), day_data['time'].values, {"axis": "T","standard_name": "time"}), # leave units intentionally blank, to be defined in the encoding
            "height": (('height',), height, {"axis": "Z","positive": "up","units": "m", "long_name":'radar_range_height'}),
            "lat": (('time',), lat_day, {"axis": "Y", "standard_name": "latitude", "units": "degree_north"}),
            "lon": (('time',), lon_day, { "axis": "X", "standard_name": "longitude","units": "degree_east"}),
        },
        attrs={'CREATED_BY'     : 'Claudia Acquistapace',
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
                        'DATA_DESCRIPTION' : 'hourly MRR measurements on Maria S. Merian (msm) ship during EUREC4A campaign',
                        'DATA_DISCIPLINE'  : 'Atmospheric Physics - Remote Sensing Radar Profiler',
                        'DATA_GROUP'       : 'Experimental;Profile;Moving',
                        'DATA_LOCATION'    : 'Research vessel Maria S. Merian - Atlantic Ocean',
                        'DATA_SOURCE'      : 'wband data postprocessed',
                        'DATA_PROCESSING'  : 'ship motion correction, the code used is available at https://github.com/ClauClouds/ship-motion-correction-for-EUREC4A-campaign',
                        'INSTRUMENT_MODEL' : '94 GHz (W-band) radar, manufactured by RPG GmbH',
                        'COMMENT'          : 'daily files reduced for eurec4a intake book tool, not for publication purposes. For publications, check https://doi.org/10.25326/156' }
    )



    # assign istrument id
    instrument_id = xr.DataArray("msm_wband_radar",dims=(),attrs={"cf_role": "trajectory_id"},)
    WbandData = WbandData.assign({"instrument": instrument_id,})

    # assign additional attributes following CF convention
    WbandData = WbandData.assign_attrs({
            "Conventions": "CF-1.8",
            "title": WbandData.attrs["DATA_DESCRIPTION"],
            "institution": WbandData.attrs["DS_AFFILIATION"],
            "history": "".join([
                "source: " + WbandData.attrs["DATA_SOURCE"] + "\n",
                "processing: " + WbandData.attrs["DATA_PROCESSING"] + "\n",
                " adapted to enhance CF compatibility\n",
            ]),  # the idea of this attribute is that each applied transformation is appended to create something like a log
            "featureType": "trajectoryProfile",
        })

    # storing ncdf data
    WbandData.to_netcdf(path_out+dateReverse+'_wband_radar_msm_eurec4a_intake.nc', encoding={"radar_reflectivity":{"zlib":True, "complevel":9},\
                                                                                             "mean_doppler_velocity": {"dtype": "f4", "zlib": True, "complevel":9}, \
                                                                                             "spectral_width": {"zlib": True, "complevel":9}, \
                                                                                             "skewness": {"zlib": True, "complevel":9}, \
                                                                                             "lat": {"dtype": "f4"} , \
                                                                                             "lon": {"dtype": "f4"}, \
                                                                                             "time": {"units": "seconds since 2020-01-01", "dtype": "i4"}})
