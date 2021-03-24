

def calc_velocity(dist_km, time_start, time_end):
    """Return 0 if time_start == time_end, avoid dividing by 0"""
    return dist_km / (time_end - time_start) if time_end > time_start else 0





def comprehense_hours_of_day(day, pathCnt):
    #day would be like: day = "200219"
    datestring = day[4]+day[5]+day[2]+day[3]+"20"+day[0]+day[1]
    day_file = open(pathCnt+"totalcnt"+datestring+".dat", "w")
    for hour in range(0,24):
        if hour < 10:
            hourstring = "0" + str(hour)
        else:
            hourstring = str(hour)
        try:
            print(pathCnt+"rphdrive_cnt" + day + hourstring + ".dat")
            hour_file = open(pathCnt+"/rphdrive_cnt" + day + hourstring + ".dat", "r")
        except:
            print("file not found: " + day + hourstring + ".dat")
            continue
        for line in hour_file:
            line_to_copy = hour_file.readline()
            day_file.write(line_to_copy)
        hour_file.close()
    day_file.close()
    

def f_calculateMomentsCol(spec,v_doppl):
    """
    author: Claudia Acquistapace
    date  : 01/07/2020
    goal  : calculate Radar Doppler moments from the radar Doppler spectrum and the Doppler velocity
    input : spec - Doppler spectrum array of dimension (height,Vdoppler)
            v_doppler - Doppler velocity array
    output: momentsArray: array of dimension (height,6), with:
       - (:,0) reflectivity height array in linear units, 
       - (:,1) Mean Doppler velocity height array
       - (:,2) Spectral width height array
       - (:,3) Skewness height array
       - (:,4) Kurtosis height array
       - (:,5) reflectivity in dB units height array
       
    """
    import numpy as np
    
    dim_v = len(spec[0,:])
    dim_h = len(spec[:,0])
    v_doppl = np.tile(v_doppl,(dim_h,1))
    moment_matrix = np.zeros((dim_h,6))
    moment_matrix.fill(np.nan)
    
    #reflectivity
    spec[np.isnan(spec)] = 0.
    Ze = np.nansum(spec[:,:], axis=1)
    Ze_db = 10*np.log10(Ze)       
    
    #mean_doppl_vel
    num = spec[:,:]*v_doppl
     #(125, 512)
    num[np.where(np.isnan(num))] == 0.
    VD = 1/Ze*(np.nansum(num, axis=1))
    
    # spectral width
    sw_num = spec[:,:]*(v_doppl-np.tile(VD, (dim_v,1)).T)**2
    sw_num[np.where(np.isnan(sw_num))] == 0.
    SW = np.sqrt(np.nansum(sw_num, axis=1)/Ze)

    # skewness
    sk_num = spec[:,:]*(v_doppl-np.tile(VD, (dim_v,1)).T)**3
    sk_num[np.where(np.isnan(sk_num))] == 0.
    SK = (np.nansum(sk_num, axis=1))*1/(Ze*(SW**3))
    
    # kurtosis
    ku_num = spec[:,:]*(v_doppl-np.tile(VD, (dim_v,1)).T)**4
    ku_num[np.where(np.isnan(ku_num))] == 0.
    KU = (np.nansum(ku_num, axis=1))*1/(Ze*(SW**4))
    
    moment_matrix[:,0] = Ze
    moment_matrix[:,1] = VD
    moment_matrix[:,2] = SW
    moment_matrix[:,3] = SK
    moment_matrix[:,4] = KU
    moment_matrix[:,5] = Ze_db

    return(moment_matrix)    
    


def f_calc_Vcourse(ShipData):
    import numpy as np
    
    # calculating Vship as deltax/deltat where delta x is the distance in km and 
    # delta t is the difference in seconds between consecutive time stamps
    vs_hav          = []
    vs_coursex      = []
    vs_coursey      = []
    lat             = ShipData.lat.values
    lon             = ShipData.lon.values
    lat[lat==-999.] = np.nan
    lon[lon==-999.] = np.nan
    
    
    for ind in range(len(lat)-1):
        print(ind, len(lat))
        dist_km     = getDistanceFromLatLonInKm(lat[ind],lon[ind],lat[ind+1],lon[ind+1]) # km
        time_start  = ShipData.time.values[ind]   # sec
        time_end    = ShipData.time.values[ind+1] # sec
        vsVal       = 1000.* calc_velocity(dist_km, time_start, time_end)
        vs_hav.append(vsVal) # calculation of v [m/s]
    
        # calculating x and y components of V_course by multiplying for sin(heading), cos(heading)
        vs_coursex.append(vsVal* np.cos(ShipData.heading_INHDT.values[ind]))
        vs_coursey.append(vsVal* np.sin(ShipData.heading_INHDT.values[ind]))    
        
    vs_hav.append(np.nan)
    vs_coursey.append(np.nan)
    vs_coursex.append(np.nan)
    vs_hav          = np.asarray(vs_hav)
    vs_coursey      = np.asarray(vs_coursey)
    vs_coursex      = np.asarray(vs_coursex)
    
    vs_coursey[vs_hav > 20.] = np.nan
    vs_coursex[vs_hav > 20.] = np.nan
    vs_hav[vs_hav > 20.]     = np.nan

    
    return(vs_hav, vs_coursey, vs_coursex)


def f_calcTablegaps(cntFile):
    '''
      Created on Mar Oct 06 17:06:20 2020
    
      @author: cacquist
      @goal: read the ship data cnt file for one day and extract 
      data relative to gaps in which the table is not working 
      to an xarray dataset. In particular, it returns a dataset containing all 
      last recorded times before the gap, all first time recorded after the gaps, 
      gaps duration and last recorded roll and pitch before the gap.
    
    
      inputs:
        cntFile 
        
      Output: 
         'tableStopTime'
         'tableRestartTime'
         'durationSingleGaps'
         'rollLastPosArr'
         'pitchLastPosArr'
      '''
    from datetime import datetime, timedelta
    import pandas as pd
    import numpy as np
    import xarray as xr
    
    DF = pd.read_csv(cntFile, \
                         header=None, delimiter=r"\s+", error_bad_lines=False)
    DF = DF.dropna()
    DF_CNT = pd.DataFrame(DF.values,  columns=['date', 'hour', 'decSec', 'tbl', \
                                    'rolltbl', 'pitchtbl', 'shp', \
                                        'rollShp', 'pitchShp', \
                                            'offset1','offset2'])
     
    # Construct dummy dataframe
    dates = pd.to_datetime(DF_CNT.date.values+' '+DF_CNT.hour.values, yearfirst=True)
    df = pd.DataFrame(dates, columns=['date'])
    deltas = df['date'].diff()[1:]
    gaps = deltas[deltas > timedelta(seconds=2)]

    # build arrays to store data
    tableStopArr       = np.zeros((len(gaps)), dtype='datetime64[s]')
    tableRestartArr    = np.zeros((len(gaps)), dtype='datetime64[s]')
    durationSingleGaps = [" " for i in range(len(gaps))] 
    #TotalDuration      = gaps.sum()
    rollArr            = np.zeros([len(gaps)])
    pitchArr           = np.zeros([len(gaps)])

    
    # Print results
    print(f'{len(gaps)} gaps with total gap duration: {gaps.sum()}')
    indArr = 0
    for i, g in gaps.iteritems():
        table_stop = df['date'][i - 1]
        table_restart = df['date'][i + 1]
        tableStopArr[indArr] = datetime.strftime(table_stop, "%Y-%m-%d %H:%M:%S")
        tableRestartArr[indArr] = datetime.strftime(table_restart, "%Y-%m-%d %H:%M:%S")
        durationSingleGaps[indArr] = str(g.to_pytimedelta())
        rollArr[indArr] = DF_CNT.rolltbl.values[np.where(dates == datetime.strftime(table_stop, "%Y-%m-%d %H:%M:%S"))][-1]
        pitchArr[indArr] = DF_CNT.pitchtbl.values[np.where(dates == datetime.strftime(table_stop, "%Y-%m-%d %H:%M:%S"))][-1]
        indArr = indArr + 1    
        
        
        print(f'table stop: {datetime.strftime(table_stop, "%Y-%m-%d %H:%M:%S")} | '
              f'Duration: {str(g.to_pytimedelta())} | '
              f'Table Restart: {datetime.strftime(table_restart, "%Y-%m-%d %H:%M:%S")}')
        
    #DF_CNT.roll.values[]
    
    numberStopsArr = np.arange(start=0, stop=len(gaps), step=1)
    tableGapsDataset = xr.Dataset({'tableStopTime':(['numberStopsArr'], tableStopArr),
                                  'tableRestartTime':(['numberStopsArr'], tableRestartArr),
                                  'durationSingleGaps':(['numberStopsArr'], durationSingleGaps),
                                  'rollLastPosArr':(['numberStopsArr'], rollArr),
                                  'pitchLastPosArr':(['numberStopsArr'], pitchArr),
                                  },
                            coords={'numberStopsArr':numberStopsArr})
    
    return(tableGapsDataset)


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


def convert_timedelta(duration):
    days, seconds = duration.days, duration.seconds
    hours = days * 24 + seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = (seconds % 60)
    return days, hours, minutes, seconds


def f_findClosestSoundings2MissingTableData(missingTimesArr,soundings_dataset):
    import numpy as np
    import xarray as xr
    from datetime import datetime
    #from function_essd import convert_timedelta
    
    listDays                  = []
    listHours                 = []
    listMinutes               = []
    listTotSecs               = []
    timeCorrectionQualityFlag = np.zeros(len(missingTimesArr))
    timeCorrectionQualityFlag.fill(np.nan)
    latSel                    = np.zeros((len(missingTimesArr),len(soundings_dataset.height)))
    lonSel                    = np.zeros((len(missingTimesArr),len(soundings_dataset.height)))
    closestWindSpeed          = np.zeros((len(missingTimesArr),len(soundings_dataset.height)))
    closestWindDir            = np.zeros((len(missingTimesArr),len(soundings_dataset.height)))
    
    # selecting radiosondes launched that are closest in time to the missing time selected, for each missing time.
    datetimeClosestSoundingFound = []
    for indLoop in range(len(missingTimesArr)):

        soundingsTimeSelected_dataset = soundings_dataset.sel(time=missingTimesArr[indLoop], method='nearest')
        #print(datetime.strptime(np.datetime_as_string(soundingsTimeSelected_dataset['time'].values,unit='s'), '%Y-%m-%dT%H:%M:%S'))
        #print(missingTimesArr[indLoop])
        #CTdatetime = datetime.datetime.utcfromtimestamp(soundingsTimeSelected_dataset.time.values.tolist()/1e9)
        datetimeClosestSoundingFound.append(soundingsTimeSelected_dataset.time.values)
        
        
        # calculating time difference in days, hours, mins, secs between the 
        # launch time of the selected radiosonde and the missing time
        soundingDatetimeSel = datetime.strptime(np.datetime_as_string(soundingsTimeSelected_dataset['time'].values,unit='s'), \
                                                '%Y-%m-%dT%H:%M:%S')
        if (soundingDatetimeSel > missingTimesArr[indLoop]):
            diff = soundingDatetimeSel - missingTimesArr[indLoop]
        else:
            diff = missingTimesArr[indLoop] - soundingDatetimeSel
            
            
        days    = diff.days
        totSecs = diff.seconds
        seconds = diff.seconds
        hours   = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = (seconds % 60)
        
        # appending days, hours and mins in lists of elements corresponding to each missing time
        listDays.append(days)
        listHours.append(hours)
        listMinutes.append(minutes)
        listTotSecs.append(totSecs)
        
        #print('delay of the radiosonde found closest :' )
        #print(days, hours, minutes, seconds)
        
        # reading lat lon, wind speed and direction
        latSel[indLoop,:]           = soundingsTimeSelected_dataset.lat[:]
        lonSel[indLoop,:]           = soundingsTimeSelected_dataset.lon[:]
        closestWindSpeed[indLoop,:] = soundingsTimeSelected_dataset.windSpeed_RS[:]
        closestWindDir[indLoop,:]   = soundingsTimeSelected_dataset.windDir_RS[:]


        # classification of sondes based on time distance from the missing value:
        if (days == 0.) * (hours <= 1.):
            timeCorrectionQualityFlag[indLoop] = 1.  
            # within one hour delay
            #print('qui')
        elif (days == 0.) * (hours > 1.) * (hours < 2.):
            timeCorrectionQualityFlag[indLoop] = 2.
            # between 1 and 2 hours delay
            #print('quo')
        elif (days == 0.) *  (hours >= 2.) :
            timeCorrectionQualityFlag[indLoop] = 3.
            # more than 2 hours delay
            #print('quA')
        else:
            timeCorrectionQualityFlag[indLoop] = 4.
            # more than 2 hours, also different days
            #print('NADA')
        #print('sonde classified as :', timeCorrectionQualityFlag[indLoop])


    # build output dataset containing for each time stamp of missingtimesarray, all the values
    closestSondesFound = xr.Dataset(data_vars = {'qualityFlag':(('time'), timeCorrectionQualityFlag),
                                                 'datetimeClosestSoundingFound':(('time'), datetimeClosestSoundingFound), 
                                                 'Ndays':(('time'), listDays),
                                                 'Nhours':(('time'), listHours),
                                                 'Nminutes':(('time'), listMinutes),
                                                 'totDelaySeconds':(('time'), listTotSecs),
                                                 'lat' :(('time','height'), latSel),
                                                 'lon' :(('time','height'), lonSel),
                                                 'wind_speed':(('time','height'),closestWindSpeed),
                                                 'wind_dir':(('time','height'),closestWindDir)},
                                       coords = {'time': missingTimesArr,
                                                 'height':soundings_dataset.height})
    return(closestSondesFound)




def f_haversineFormula(lat1_deg,lon1_deg,lat2_deg,lon2_deg):
    import math
    R = 6373.0  #radius of the Earth [km]
    dist_array = []
    
    # reshape arrays on the same time dimension if the time dimension is different
    lat1 = math.radians(lat1_deg)
    lon1 = math.radians(lon1_deg)
    lat2 = math.radians(lat2_deg)
    lon2 = math.radians(lon2_deg)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    distance = R * c
    dist_array.append(distance)
    return(dist_array)

def f_interpShipData(array, orderValue):
    '''
    author: claudia Acquistapace
    date: 26 october 2020
    goasl: interpolate missing values of ship data time series with cubic interpolation.
    It works only on one value to be interpolated
    '''
    import pandas as pd
    s        = pd.Series(array)
    interpol = s.interpolate(method='polynomial', order=orderValue)
    return(interpol)


def find_runs(x):
    """Find runs of consecutive items in an array.
    source:  https://gist.github.com/alimanfoo/c5977e87111abe8127453b21204c1065
    
    """
    import numpy as np
    # ensure array
    x = np.asanyarray(x)
    if x.ndim != 1:
        raise ValueError('only 1D array supported')
    n = x.shape[0]

    # handle empty array
    if n == 0:
        return np.array([]), np.array([]), np.array([])

    else:
        # find run starts
        loc_run_start = np.empty(n, dtype=bool)
        loc_run_start[0] = True
        np.not_equal(x[:-1], x[1:], out=loc_run_start[1:])
        run_starts = np.nonzero(loc_run_start)[0]

        # find run values
        run_values = x[loc_run_start]

        # find run lengths
        run_lengths = np.diff(np.append(run_starts, n))

        return run_values, run_starts, run_lengths



# reading ship data for the entire campaign
def f_readShipDataset(shipDataName):

  '''
  Created on Mer Sep 30 16:06:20 2020

  @author: cacquist
  @goal: read the ship data file and extract data to an xarray dataset


  inputs:
    shipDataName (ship path+filename string)
    
  Output: ShipData xarray dataset containing 
  time [sec since 1970-01-01]
  lat 
  lon
  heading
  heave
  pitch
  roll
  absWindDir
  absWindSpeed
  relWindDir
  relWindSpeed
  heading2
  '''
  import pandas as pd
  import xarray as xr

  dataset       = pd.read_csv(shipDataName, skiprows=[1,2], usecols=['seconds since 1970','SYS.STR.PosLat','SYS.STR.PosLon',\
                                                  'Seapath.INHDT.Heading','Seapath.PSXN.Heave','Seapath.PSXN.Pitch',\
                                                  'Seapath.PSXN.Roll','Weatherstation.PEUMA.Absolute_wind_direction',\
                                                  'Weatherstation.PEUMA.Absolute_wind_speed','Weatherstation.PEUMA.Relative_wind_direction',\
                                                  'Weatherstation.PEUMA.Relative_wind_speed', 'Seapath.PSXN.Heading'], low_memory=False)
  xrDataset     = dataset.to_xarray()   
      
  ShipData      = xrDataset.rename({'seconds since 1970':'time',\
                    'SYS.STR.PosLat':'lat',\
                    'SYS.STR.PosLon':'lon',\
                    'Seapath.INHDT.Heading':'heading_INHDT',\
                    'Seapath.PSXN.Heave':'heave',\
                    'Seapath.PSXN.Pitch':'pitch',\
                    'Seapath.PSXN.Roll':'roll',\
                    'Weatherstation.PEUMA.Absolute_wind_direction':'absWindDir',\
                    'Weatherstation.PEUMA.Absolute_wind_speed':'AbsWindSpeed',\
                    'Weatherstation.PEUMA.Relative_wind_direction':'relWindDir',\
                    'Weatherstation.PEUMA.Relative_wind_speed':'relWindSpeed',\
                    'Seapath.PSXN.Heading':'heading_PSXN'})
  return(ShipData)



def list_duplicates(seq):
    seen = set()
    seen_add = seen.add
    return [idx for idx,item in enumerate(seq) if item in seen or seen_add(item)]






def f_readAndMergeRadarDataDay_DopplerCorrection(radarFileList):
    """
    author : Claudia Acquistapace
    date   : 20.10.2020
    goal   : read a list of hourly radar data files and produce a unified xarray dataset for the day, 
    containing only the mean doppler velocity matrix merged together
    
    input:
        fileRadarList: array of strings. the dimension is given by the number of files
        For each file the string should include the general path to the radar data file
    output: 
        xarray dataset of the entire data from the file list
        
    """
    
    import xarray as xr
    
    MergedData = xr.Dataset()
    for fileName in radarFileList[0:len(radarFileList)]:
        print(fileName)
        Data        = xr.open_dataset(fileName)
        print(type(Data['vm']))
        TempDataSet = xr.Dataset({'range':Data['range'],         # range array
                                  'time':Data['time'],           # time array
                                   'vm':Data['vm'],              # Mean Doppler velocity field [m/s]
                                   'ind_range_offsets':Data['range_offsets'],
                                  }) 
        Data.close()
        MergedData = xr.merge([MergedData, TempDataSet])
    return(MergedData)
    



# Version 1.0 released by David Romps on September 12, 2017.
# 
# When using this code, please cite:
# 
# @article{16lcl,
#   Title   = {Exact expression for the lifting condensation level},
#   Author  = {David M. Romps},
#   Journal = {Journal of the Atmospheric Sciences},
#   Year    = {2017},
#   Volume  = {in press},
# }
#
# This lcl function returns the height of the lifting condensation level
# (LCL) in meters.  The inputs are:
# - p in Pascals
# - T in Kelvins
# - Exactly one of rh, rhl, and rhs (dimensionless, from 0 to 1):
#    * The value of rh is interpreted to be the relative humidity with
#      respect to liquid water if T >= 273.15 K and with respect to ice if
#      T < 273.15 K. 
#    * The value of rhl is interpreted to be the relative humidity with
#      respect to liquid water
#    * The value of rhs is interpreted to be the relative humidity with
#      respect to ice
# - ldl is an optional logical flag.  If true, the lifting deposition
#   level (LDL) is returned instead of the LCL. 
# - min_lcl_ldl is an optional logical flag.  If true, the minimum of the
#   LCL and LDL is returned.
def lcl(p,T,rh=None,rhl=None,rhs=None,return_ldl=False,return_min_lcl_ldl=False):

    import math
    import scipy.special
    import numpy as np 
    
    # Parameters
    Ttrip = 273.16     # K
    ptrip = 611.65     # Pa
    E0v   = 2.3740e6   # J/kg
    E0s   = 0.3337e6   # J/kg
    ggr   = 9.81       # m/s^2
    rgasa = 287.04     # J/kg/K 
    rgasv = 461        # J/kg/K 
    cva   = 719        # J/kg/K
    cvv   = 1418       # J/kg/K 
    cvl   = 4119       # J/kg/K 
    cvs   = 1861       # J/kg/K 
    cpa   = cva + rgasa
    cpv   = cvv + rgasv

    # The saturation vapor pressure over liquid water
    def pvstarl(T):
        return ptrip * (T/Ttrip)**((cpv-cvl)/rgasv) * \
         math.exp( (E0v - (cvv-cvl)*Ttrip) / rgasv * (1/Ttrip - 1/T) )
   
    # The saturation vapor pressure over solid ice
    def pvstars(T):
        return ptrip * (T/Ttrip)**((cpv-cvs)/rgasv) * \
         math.exp( (E0v + E0s - (cvv-cvs)*Ttrip) / rgasv * (1/Ttrip - 1/T) )

    # Calculate pv from rh, rhl, or rhs
    rh_counter = 0
    if rh  is not None:
        rh_counter = rh_counter + 1
    if rhl is not None:
        rh_counter = rh_counter + 1
    if rhs is not None:
        rh_counter = rh_counter + 1
    if rh_counter != 1:
        print(rh_counter)
        exit('Error in lcl: Exactly one of rh, rhl, and rhs must be specified')
    if rh is not None:
        # The variable rh is assumed to be 
        # with respect to liquid if T > Ttrip and 
        # with respect to solid if T < Ttrip
        if T > Ttrip:
            pv = rh * pvstarl(T)
        else:
            pv = rh * pvstars(T)
        rhl = pv / pvstarl(T)
        rhs = pv / pvstars(T)
    elif rhl is not None:
        pv = rhl * pvstarl(T)
        rhs = pv / pvstars(T)
        if T > Ttrip:
            rh = rhl
        else:
            rh = rhs
    elif rhs is not None:
        pv = rhs * pvstars(T)
        rhl = pv / pvstarl(T)
        if T > Ttrip:
            rh = rhl
        else:
            rh = rhs
    if pv > p:
        return np.nan

    # Calculate lcl_liquid and lcl_solid
    qv = rgasa*pv / (rgasv*p + (rgasa-rgasv)*pv)
    rgasm = (1-qv)*rgasa + qv*rgasv
    cpm = (1-qv)*cpa + qv*cpv
    if rh == 0:
        return cpm*T/ggr
    aL = -(cpv-cvl)/rgasv + cpm/rgasm
    bL = -(E0v-(cvv-cvl)*Ttrip)/(rgasv*T)
    cL = pv/pvstarl(T)*math.exp(-(E0v-(cvv-cvl)*Ttrip)/(rgasv*T))
    aS = -(cpv-cvs)/rgasv + cpm/rgasm
    bS = -(E0v+E0s-(cvv-cvs)*Ttrip)/(rgasv*T)
    cS = pv/pvstars(T)*math.exp(-(E0v+E0s-(cvv-cvs)*Ttrip)/(rgasv*T))
    lcl = cpm*T/ggr*( 1 - \
       bL/(aL*scipy.special.lambertw(bL/aL*cL**(1/aL),-1).real) )
    ldl = cpm*T/ggr*( 1 - \
      bS/(aS*scipy.special.lambertw(bS/aS*cS**(1/aS),-1).real) )

    # Return either lcl or ldl
    if return_ldl and return_min_lcl_ldl:
        exit('return_ldl and return_min_lcl_ldl cannot both be true')
    elif return_ldl:
        return ldl
    elif return_min_lcl_ldl:
        return min(lcl,ldl)
    else:
        return lcl
    




def f_readAndMergeRadarDataDay(radarFileList):
    """
    author : Claudia Acquistapace
    date   : 20.10.2020
    goal   : read a list of hourly radar data files and produce a unified xarray dataset for the day, 
    containing all the data merged together
    
    input:
        fileRadarList: array of strings. the dimension is given by the number of files
        For each file the string should include the general path to the radar data file
    output: 
        xarray dataset of the entire data from the file list
        
    """
    
    import xarray as xr
    
    MergedData = xr.Dataset()
    for fileName in radarFileList[0:len(radarFileList)]:
        print(fileName)
        Data        = xr.open_dataset(fileName)
        TempDataSet = xr.Dataset({'range':Data['range'],         # range array
                              'time':Data['time'],               # time array
                              'Ze':Data['ze'],                   # reflectivity field [mm^6/m^3]
                              'vm':Data['vm'],                   # Doppler velocity field [m/s]
                              'sigma':Data['sw'],                # Doppler spectral width field [m/s]
                              'skew':Data['skew'],               # Doppler spectrum skewness 
                              'ldr':Data['ldr'],                 # linear depolarisation ratio [mm^6/m^3]
                              'lwp':Data['lwp'],                 # LWP array [g/m^2]
                              'Tb':Data['tb'],                   # bridness temperature [K] 
                              'T_pc':Data['t_pc'],               # Radar-PC temperature [K] 
                              'T_trans':Data['t_trans'],         # Radar transmitter temperature [K] 
                              'T_rec':Data['t_rec'],             # Radar receiver temperature [K]                                           
                              'TransPow':Data['p_trans'],        # Radar transmitted power temperature [W] 
                              'RR':Data['rr'],                   # Rain rate from the weather station [mm/h]
                              'T_env':Data['ta'],                # temperature from the weather station [K]
                              'rh':Data['rh'],                   # Relative humidity from the weather station [%]
                              'ff':Data['wspeed'],               # wins speed from the weather station [km/h]
                              'fff':Data['wdir'],                # wins direction from the weather station [deg]
                              'ind_range_offsets':Data['range_offsets'],
                              }) 
        Data.close()
        MergedData = xr.merge([MergedData, TempDataSet])
    return(MergedData)
    
def f_tableFlag(cnt_file_day):
    
    import xarray as xr
    import pandas as pd
    import numpy as np
    
    # reading cnt file from the stable table and find times of gaps and table position when the table got stuck.
    dataset = f_calcTablegaps(cnt_file_day)

#    tableGapsDataset = xr.Dataset({'tableStopTime':(['numberStopsArr'], tableStopArr),
#                                  'tableRestartTime':(['numberStopsArr'], tableRestartArr),
#                                  'durationSingleGaps':(['numberStopsArr'], durationSingleGaps),
#                                  'rollLastPosArr':(['numberStopsArr'], rollArr),
#                                  'pitchLastPosArr':(['numberStopsArr'], pitchArr),
##                                  },
#                           coords={'numberStopsArr':numberStopsArr})


    # reading all data excluding the string lines due to table getting stuck.
    DF = pd.read_csv(cnt_file_day, header=None, delimiter=r"\s+", error_bad_lines=False)#}, dtype={'Hˆhe [m]':str}) sep='\t', skipinitialspace=True, \
    DF = DF.dropna()

    # exctracting datetime array from the cnt file (high resolution data > 1s)
    dateArr = DF.values[:,0]
    timeArr = DF.values[:,1]
    
    from datetime import datetime
    timesWithDataArr = []
    for indTime in range(len(dateArr)):
        datetime = datetime.strptime(dateArr[indTime]+' '+timeArr[indTime], '%y/%m/%d %H:%M:%S')
        timesWithDataArr.append(datetime)


    # converting time serie in pandas series
    timeSerieData = pd.Series(np.repeat(np.nan, len(timesWithDataArr)),\
                          index=timesWithDataArr) 
    
    # downsampling the time serie to 1s resolution
    prova = timeSerieData.resample('1S').max()
    
    # defining dataset for output flag: nan when the table is working, contains the roll and pitch position 
    # at the relative last recorded time for all the time stamps in which the table is stuck
    flagStableTable =  xr.Dataset({'pitch':(['time'],  np.repeat(np.nan, len(prova.index))),
                                   'roll':(['time'],  np.repeat(np.nan, len(prova.index)))},
                         coords = {'time':(['time'], prova.index)})
    
    # storing roll and pitch in the time intervals in which the table is stuck
    Nloops = len(dataset.numberStopsArr)
    for ind in range(Nloops):
        
        #read datetime stop time
        timeStop = dataset.tableStopTime.values[ind]
        timeRestart = dataset.tableRestartTime.values[ind]
        dim = np.shape(flagStableTable.pitch.sel(time=slice(timeStop, timeRestart)))
        flagStableTable.pitch.sel(time=slice(timeStop, timeRestart)).values[:] = np.repeat(dataset['pitchLastPosArr'].values[ind], dim)
        flagStableTable['roll'].sel(time=slice(timeStop, timeRestart)).values[:] = np.repeat(dataset['rollLastPosArr'].values[ind],dim)
        print(timeStop)
        print(dataset['pitchLastPosArr'].values[ind])
        print(dataset['rollLastPosArr'].values[ind])
        
    return(flagStableTable, dataset)








def getDistanceFromLatLonInKm(lat1,lon1,lat2,lon2):
    from math import sin, cos, sqrt, atan2, radians

    R = 6371 # Radius of the earth in km
    dLat = radians(lat2-lat1)
    dLon = radians(lon2-lon1)
    rLat1 = radians(lat1)
    rLat2 = radians(lat2)
    a = sin(dLat/2) * sin(dLat/2) + cos(rLat1) * cos(rLat2) * sin(dLon/2) * sin(dLon/2) 
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    d = R * c # Distance in km
    return d



def nearest(items, pivot):
    return min(items, key=lambda x: abs(x - pivot))




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
        Var2keep = ['vm_corrected_smoothed', 'ze', 'sw', 'skew','rr', 'rh', 'ta',\
                    'pa', 'wspeed', 'wdir', 'tb', 'lwp', 'blower_status', 'p_trans', 't_trans', 't_rec', 't_pc']
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



# Preprocessing of the data compact.nc

#def generate_preprocess(): # can take arguments

#    '''
#    author: Claudia Acquistapace
#    date  : 29/10/2020
#    goal  : generate a function that  does the preprocessing of the data
#    '''
#    import numpy as np
#    import xarray as xr
    
    # Generate a preprocess function that takes a dataset as only argument and does all the processing needed
#    def preprocess(ds):
#        '''
#        author: Claudia Acquistapace
#        date : 29/10/2020
#        goal : preprocess wband radar ncdf data to make them readable with xr.open_fmdataset instruction
#        The preprocessing includes:
#            - removal of non useful variables : 
#                var2DropList = ['source_rr', 'source_rh', 'source_ta', 'source_pa', 'source_wspeed', 'source_wdir', 'wl']
#            - redefinition of scalar variables as time series
#                varList = ['lat','lon','zsl','freq_sb','hpbw']
#        input: 
##            dataset xarray of data from radar
#            
#        output:
#            new dataset with var2DropList variables removed and varList variables modified
#            
#        '''
#        # assign new coordinate for dependencies not specified
#        ds = ds.assign_coords({'scalar':np.arange(1),'number.chirp.sequences':np.arange(3)})
#
#        # drop variables that have no interest
#        #var2DropList = ['source_rr', 'source_rh', 'source_ta', 'source_pa', 'source_wspeed', 'source_wdir', 'wl', 'nqv']
#        #var2DropList = [ 'wl', 'nqv']
#        #print(var2DropList)
#        #for indDrop in range(len(var2DropList)):
#        #    ds = ds.drop(var2DropList[indDrop])
#        
#        # Convert all variables with scalar dimension to time 
#        # list of variables to convert  
#        varList = ['lat','lon','zsl','freq_sb','hpbw']
#        varUnits = ['degrees', 'degrees', 'm', 's-1', 'degrees']
#        varLongNames = ['latitude',
#                'longitude',
#                'Altitude above mean sea level the instrument is located',
#                'Central transmission frequency ',
#                'Antenna half power beam width']
        
#        #loop on variables to convert
#        for iVar in range(len(varList)):
#            var = varList[iVar]
            
#            # storing single value of the variable 
#            varValue = ds[var].values
            
#            # deleting the variable from the dataset
#            ds = ds.drop(var)

#            # creating new variable time serie
#            varTimeSerie = np.repeat(varValue, len(ds['time'].values))

#            # saving variable in the dataset
#            ds[var] = (['time'],varTimeSerie)
#            dim     = ['time']
#            coord   = {'time':ds['time'].values}
#            VarDataArray = xr.DataArray(dims=dim, coords=coord, data=varTimeSerie,
#                                          attrs={'long_name':varLongNames[iVar],
#                                                 'units':varUnits[iVar]})
#            # adding new Var variable to the dataset
#            ds = ds.assign({var:VarDataArray})
            
#        #ds['time'] = pd.to_datetime(ds.time.values.astype(int).astype(str), format='%Y%m%d') + pd.to_timedelta(ds.time.values%1, unit='D').round('1s')
#        return ds
#    return preprocess





# Preprocessing of the data zen.nc
def generate_preprocess_zen(): # can take arguments

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
        date : 29/10/2020
        goal : preprocess wband radar ncdf data to make them readable with xr.open_fmdataset instruction
        The preprocessing includes:
            - removal of non useful variables : 
                var2DropList = [  'wl']
            - redefinition of scalar variables as time series
                varList = ['lat','lon','zsl','freq_sb','hpbw']
        input: 
            dataset xarray of data from radar
            
        output:
            new dataset with var2DropList variables removed and varList variables modified
            
        '''
        # assign new coordinate for dependencies not specified
        ds = ds.assign_coords({'scalar':np.arange(1),'number.chirp.sequences':np.arange(3)})

        # drop variables that have no interest
        var2DropList = ['wl', 'nqv']
        for indDrop in range(len(var2DropList)):
            ds = ds.drop(var2DropList[indDrop])
        
        # Convert all variables with scalar dimension to time 
        # list of variables to convert  
        varList = ['lat','lon','zsl','freq_sb','hpbw']
        varUnits = ['degrees', 'degrees', 'm', 's-1', 'degrees']
        varLongNames = ['latitude',
                'longitude',
                'Altitude above mean sea level the instrument is located',
                'Central transmission frequency ',
                'Antenna half power beam width']
        
        #loop on variables to convert
        for iVar in range(len(varList)):
            var = varList[iVar]
            
            # storing single value of the variable 
            varValue = ds[var].values
            
            # deleting the variable from the dataset
            ds = ds.drop(var)

            # creating new variable time serie
            varTimeSerie = np.repeat(varValue, len(ds['time'].values))

            # saving variable in the dataset
            ds[var] = (['time'],varTimeSerie)
            dim     = ['time']
            coord   = {'time':ds['time'].values}
            VarDataArray = xr.DataArray(dims=dim, coords=coord, data=varTimeSerie,
                                          attrs={'long_name':varLongNames[iVar],
                                                 'units':varUnits[iVar]})
            # adding new Var variable to the dataset
            ds = ds.assign({var:VarDataArray})
            
        #ds['time'] = pd.to_datetime(ds.time.values.astype(int).astype(str), format='%Y%m%d') + pd.to_timedelta(ds.time.values%1, unit='D').round('1s')
        return ds
    return preprocess






def f_readingRadiosondeData(pathSoundings,fileSounding):
    import numpy as np
    import netCDF4 as nc4
    from netCDF4 import Dataset
    import xarray as xr
    
    dataRS = Dataset(pathSoundings+fileSounding)
    timeRS_launch = dataRS.variables['launch_time'][:]
    units_time          = 'seconds since 1970-01-01T00:00:00Z' 
    datetimeRS = nc4.num2date(timeRS_launch, units_time, only_use_cftime_datetimes=False) 
    windDir_RS = dataRS.variables['wdir'][:]
    windSpeed_RS = dataRS.variables['wspd'][:]
    #T_RS = dataRS.variables['ta'][:]
    #P_RS = dataRS.variables['p'][:]
    #P0 = 1013.25 # hPa
    geo_h_RS =  dataRS.variables['alt'][:]
    lat = dataRS.variables['lat'][:]
    lon = dataRS.variables['lon'][:]
    #launch_time = dataRS.variables['launch_time'][:]
    #Nsoundings = len(P_RS[:,0])
    #Nheight = len(P_RS[0,:])

    # derivation of geometrical height bbased on geopotential height using formula 
    #in https://www.gspteam.com/GSPsupport/OnlineHelp/index.html?pressure_altitude.htm
    #Relation between geometric altitude and geopotential altitude:
    Re = 6356766. # [m] the nominal radius of the earth
    geom_height = (Re * geo_h_RS)/(np.repeat(Re, len(geo_h_RS)) - geo_h_RS)


    soundings_dataset = xr.Dataset(data_vars = {'windDir_RS':(('time','height'), windDir_RS),
                                                'windSpeed_RS':(('time','height'), windSpeed_RS), 
                                                'lat':(('time','height'), lat),
                                                'lon':(('time','height'), lon)},
                                     coords  = {'time':datetimeRS,
                                                'height':geom_height})
    return(soundings_dataset)



#from f_calc_Vrot import f_calcRMatrix

def f_calcRinvMatrix(roll, pitch, yaw):
    
    '''
    Created on Mer Sep 30 16:06:20 2020

    @author: cacquist
    @goal: calculate the inverse rotational matrix R_inv given in Eq 16. 
    The matrix is calculated with the roll, pitch and yaw values obtained in eq 
    13, 14, 15.
    
    
    inputs:
        table roll at time T0 , 
        table pitch at time T0, 
        table yaw at time T0 (corresponding to ship yaw),
        
    Output: matrix R_inv
    '''
    
    # reading ship angles for each time stamp and converting them in radiants
    theta = np.deg2rad(roll)
    phi   = np.deg2rad(pitch)
    psi   = np.deg2rad(yaw)
    
    cosTheta = np.cos(theta)
    senTheta = np.sin(theta)
    cosPhi   = np.cos(phi)
    senPhi   = np.sin(phi)
    cosPsi   = np.cos(psi)
    senPsi   = np.sin(psi)

    # definition of the rotation matrix for roll angle (theta)
    A_inv           = np.array([[         1,              0,              0], \
                               [          0,       cosTheta,       senTheta], \
                               [          0,      -senTheta,       cosTheta]])
        
    # rotation matrix for pitch angle (phi)
    B_inv           = np.array([[     cosPhi,             0,        -senPhi], \
                               [           0,             1,              0], \
                               [      senPhi,             0,         cosPhi]])
    
    # definition of the rotation matrix for the yaw (psi)
    C_inv            = np.array([[    cosPsi,        senPsi,            0], \
                                [    -senPsi,        cosPsi,            0], \
                                [          0,             0,             1]])
        
        
    # calculation of the rotation matrix
    R_inv            = np.matmul(C_inv, np.matmul(B_inv, A_inv))
    
    return(R_inv)



def f_calcRMatrix(roll, pitch, yaw):

    '''
    Created on Mer Sep 30 16:06:20 2020

    @author: cacquist
    @goal: calculate the rotational matrix R given in Eq 9 
    The matrix is calculated with the roll, pitch and yaw values for one time stamp
    
    inputs:
         roll at time t , 
         pitch at time t, 
         yaw at time t
        
    Output: matrix R
    '''
    
    import numpy as np
    # reading ship angles for each time stamp and converting them in radiants
    theta = np.deg2rad(roll)
    phi   = np.deg2rad(pitch)
    psi   = np.deg2rad(yaw)

    cosTheta = np.cos(theta)
    senTheta = np.sin(theta)
    cosPhi   = np.cos(phi)
    senPhi   = np.sin(phi)
    cosPsi   = np.cos(psi)
    senPsi   = np.sin(psi)
    
    # definition of the rotation matrix for roll angle (theta)
    A              = np.array([[           1,             0,              0], \
                               [           0,      cosTheta,      -senTheta], \
                               [           0,      senTheta,       cosTheta]])
        
    # rotation matrix for pitch angle (phi)
    B              = np.array([[      cosPhi,             0,         senPhi], \
                               [           0,             1,              0], \
                               [     -senPhi,             0,         cosPhi]])
    
    # definition of the rotation matrix for the yaw (psi)
    C              = np.array([[      cosPsi,       -senPsi,              0], \
                               [      senPsi,        cosPsi,              0], \
                               [           0,             0,              1]])
        
        
    # calculation of the rotation matrix
    R              = np.matmul(C, np.matmul(B, A))
    
    return(R)
