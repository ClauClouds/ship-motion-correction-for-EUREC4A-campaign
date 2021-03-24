
#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Created on Mon Sep 28 16:06:20 2020

    @author: cacquist
    @goal: calculate the unit vector components of the pointing direction of the radar
    for table working and table not working.
    
    inputs:
        flag for table working or not working, 
        time of the interruption of the table, 
        hipdata(roll, pitch, yaw)
        
    Output: Ep_x
            Ep_y
            Ep_z
"""

import pandas as pd
import netCDF4 as nc4
from netCDF4 import Dataset
import xarray as xr
import numpy as np
from functions_essd import nearest
from datetime import datetime

# reading ship data for the entire campaign
# paths to the different data files and output directories for plots
pathFolderTree  = '/Volumes/Extreme SSD/ship_motion_correction_merian/'
pathFig         = pathFolderTree+'/plots/'
pathNcDataAnc   = pathFolderTree+'/ncdf_ancillary/'
ShipData        = pathFolderTree+'/ship_data/new/shipData_all2.nc'   
 
print('* reading ship data from ncdf file')
ShipDataset                  = xr.open_dataset(ShipData)

#RollTtest = ShipDataset['roll'].values 
#indNans = np.where(np.isnan(RollTtest))
#for ind in range(len(indNans[0])):
#    print(pd.to_datetime(ShipDataset['time'].values)[indNans[0][ind]])

# generating array of days for the dataset
Eurec4aDays  = pd.date_range(datetime(2020,1,19),datetime(2020,2,19),freq='d')
NdaysEurec4a = 1#len(Eurec4aDays)

#%%
for indDayEu in range(NdaysEurec4a):
    

        
    # select a date
    dayEu           = Eurec4aDays[30]
    
    # extracting strings for yy, dd, mm
    yy              = str(dayEu)[0:4]
    mm              = str(dayEu)[5:7]
    dd              = str(dayEu)[8:10]
    
    # setting starting and ending time for the day
    datetimeStart   = datetime(int(yy),int(mm), int(dd),0,0,0)
    datetimeEnd     = datetime(int(yy),int(mm), int(dd),23,59,59)
    
    
    print('processing date :'+yy+'-'+mm+'-'+dd)
    print('**************************************')
    print('* reading stable table data of roll and pitch')
    StableTableFlag = Dataset(pathFolderTree+'/stable_table_processed_data/'+yy+mm+dd+'_stableTableFlag.nc',mode='r')
    timeTable       = StableTableFlag.variables['time'][:]
    #unitsTimeTable  = StableTableFlag.variables['time'].units[:]
    #datetimeTable   = nc4.num2date(timeTable, unitsTimeTable, only_use_cftime_datetimes=False)
    #rollTable       = StableTableFlag.variables['roll'][:].data
    #pitchTable      = StableTableFlag.variables['pitch'][:].data
    
    print('* reading times in which the table got stuck, and restarted and last recorded roll and pitch at that times')
    LastBeforeStuck = Dataset(pathFolderTree+'/stable_table_processed_data/'+yy+mm+dd+'_lastBeforeStuckDataset.nc',mode='r')
    timeStopArr     = LastBeforeStuck.variables['tableStopTime'][:]
    unitsTimeStop   = LastBeforeStuck.variables['tableStopTime'].units[:]
    timeRestartArr  = LastBeforeStuck.variables['tableRestartTime'][:]
    unitsTimeRest   = LastBeforeStuck.variables['tableRestartTime'].units[:]
    rollStopArr     = LastBeforeStuck.variables['rollLastPosArr'][:]
    pitchStopArr    = LastBeforeStuck.variables['pitchLastPosArr'][:]
    
    
    # extracting ship data for the day
    dataShipDay    = ShipDataset.sel(time=slice(datetimeStart,datetimeEnd))
    rollShipArr    = dataShipDay['roll'].values 
    pitchShipArr   = dataShipDay['pitch'].values 
    yawShipArr     = dataShipDay['yaw'].values 
    heaveShipArr   = dataShipDay['heave'].values
    datetimeShip   = pd.to_datetime(dataShipDay['time'].values)
    NtimeShip      = len(datetimeShip)
    


        
    if len(timeStopArr) == 0:
        a = 3
        print('no gaps in the stable table data')
        print('Ep = [0,0,-1] for the whole day')
        EpArr           = np.zeros((3, NtimeShip))  #np.shape(Ep0Arr[:,np.isnan(timeFinalArr)])[1])
        EpArr[2,:]      = -1.
        Ep0Arr          = EpArr
        datetimeStop    = np.datetime64("NaT")
        datetimeRestart = np.datetime64("NaT")
        
        
        EpDataset  = xr.Dataset({'Ep':(['axis','time'], EpArr),
                            'Ep_0':(['axis','time'], Ep0Arr)},
                    coords={'time':datetimeShip,
                            'axis':np.arange(3)})

    
        # saving data in a ncdf file.
        EpDataset.to_netcdf(pathNcDataAnc+yy+mm+dd+'_Ep_dataset.nc')
        
        
    else:
        # reading times at which table got stuck and restarted from stable table data
        datetimeStop    = nc4.num2date(timeStopArr, unitsTimeStop, only_use_cftime_datetimes=False)
        datetimeRestart = nc4.num2date(timeRestartArr, unitsTimeRest, only_use_cftime_datetimes=False)
        

        # defining arrays for output
        NtimeSteps      = len(timeTable)                                    # dimension of time in stable table data
        timeFinalArr    = np.zeros((NtimeShip), dtype='datetime64[s]')      # array containing closest datetimeShip values to the table time stamps at which the table is stuck for
        timeFinalArr.fill(np.datetime64("NaT"))                             # they stored time stamps are duplicates for the whole stable table data gap of the very first time moment the table got stuck.
        indexT_FinalArr = np.zeros([NtimeShip], dtype=np.int64)             # index set to zero for the whole day. When different from zero, it is because table is stuck
        EpArr           = np.zeros([3, NtimeShip], dtype=np.double)         # EP array
        timeEp          = np.zeros((NtimeShip), dtype='datetime64[s]')
        Ep0Arr          = np.zeros([3, NtimeShip], dtype=np.double)
        rollTShipRes    = np.zeros((NtimeShip), dtype=np.double)            # array to store the roll value from the table corresponding to the first time the table got stuck. The value is replicated for the whole data gap
        pitchTShipRes   = np.zeros((NtimeShip), dtype=np.double)            # array to store the pitch value from the table corresponding to the first time the table got stuck. The value is replicated for the whole data gap
        rollTShipRes.fill(np.nan)
        pitchTShipRes.fill(np.nan)
        
        #tableWorkingArr = np.zeros([NtimeSteps])
        #timeStuckArr    = np.zeros((NtimeSteps), dtype='datetime64[s]')
        #timeTableArr    = np.zeros((NtimeSteps), dtype='datetime64[s]')


        # calculation of Ep_0 for table not working and for table working 
        
        # setting Ep and Ep0 equal to [0,0,-1] for each time step.
        EpArr      = np.empty((3, NtimeShip))  #np.shape(Ep0Arr[:,np.isnan(timeFinalArr)])[1])
        EpArr[2,:] = -1.
        Ep0Arr     = EpArr
        
        # loop on the gaps of the day
        for indFindTimeFinal in range(len(datetimeStop)):
            
            print('processing gap n: '+str(indFindTimeFinal))
            #saving timeFinalArr containing duplicates, for all times in the gap, of the datetimeShip value closest to the time in
            # in which the table got stuck.
            timeFinalArr[(datetimeShip >= datetimeStop[indFindTimeFinal]) * \
                         (datetimeShip < datetimeRestart[indFindTimeFinal])] =  nearest(datetimeShip, datetimeStop[indFindTimeFinal])
            
            # find stop and restart times in the datetimeship array corresponding to datetimeStop and restart of the table
            timeStopFound    = nearest(datetimeShip, datetimeStop[indFindTimeFinal])
            timeRestartFound = nearest(datetimeShip, datetimeRestart[indFindTimeFinal])
            
            # finding index of datetimeShip corresponding to the time the table stopped for the ship 
            indFound         = np.where(datetimeShip == timeStopFound)[0]
            
            print('time Stop: found = '+str(timeStopFound)+', from table '+str( datetimeStop[indFindTimeFinal]))
            print('time Restart: found = '+str(timeRestartFound)+', from table '+str( datetimeRestart[indFindTimeFinal]))
            
            # saving indeces for reading angles
            # assigning index to read datetimeShip of the time the table stopped to all values of datetime 
            # between table stop and table restart
            print('values of roll and pitch saved in each gap')
            print(rollStopArr[indFindTimeFinal])
            print(pitchStopArr[indFindTimeFinal])
            print('########################################')
            print('########################################')
            print('########################################')
            
            # storing the values of roll and pitch as copies for all tiem steps of the gaps
            indexT_FinalArr[(datetimeShip >= datetimeStop[indFindTimeFinal]) * \
                         (datetimeShip < datetimeRestart[indFindTimeFinal])] = indFound[0]
            rollTShipRes[(datetimeShip >= datetimeStop[indFindTimeFinal]) * \
                         (datetimeShip < datetimeRestart[indFindTimeFinal])] = rollStopArr[indFindTimeFinal]
            pitchTShipRes[(datetimeShip >= datetimeStop[indFindTimeFinal]) * \
                         (datetimeShip < datetimeRestart[indFindTimeFinal])] = pitchStopArr[indFindTimeFinal]
            
        
        #print('* calculation of pointing direction for the radar when table is on')
        
        # calculation of Ep_0, pointing direction when table is at rest
        #EpTableOn = np.empty((3, NtimeShip))  #np.shape(Ep0Arr[:,np.isnan(timeFinalArr)])[1])
        #EpTableOn[2,:] = -1.
        #print('* Ep for table on : ', EpTableOn)

        
        # R_inv equal to identity for cases when table is working (indexT_FinalArr == 0)
        #identityMatrix = np.identity(3)     # definition of Identity matrix 
        #R_inv[:,:, indexT_FinalArr==0] = np.repeat(identityMatrix, \
        #                                           np.sum(indexT_FinalArr==0)).\
        #                                    reshape((3, 3,np.sum(indexT_FinalArr==0))) 
        #print('** case for table working - done')
    
                              

        # calculation of R_inv[To] for all final positions in which the table got stuck (indexT_FinalArr != 0)                            
        print('*** calculating roll, pitch and yaw of the time in which the table got stuck')
        RollStuckArr  = rollShipArr[indexT_FinalArr[indexT_FinalArr != 0]] - rollTShipRes[indexT_FinalArr != 0]
        PitchStuckArr = pitchShipArr[indexT_FinalArr[indexT_FinalArr != 0]] - pitchTShipRes[indexT_FinalArr != 0]
        YawStuckArr   = yawShipArr[indexT_FinalArr[indexT_FinalArr != 0]] 
              
        def f_calcR_inv(RollArr,PitchArr,YawArr,dimTime,indTimeCalc):
            '''
            author: Claudia Acquistapace
            date : 27/10/2020
            goal: function to calculate inverse matrix given roll, pitch, yaw
            input:
                roll array in degrees
                pitch array in degrees
                yaw array in degrees
                dimtime: dimension of time array for the definition of R_inv as [3,3,dimTime]
                indTimeCalc: index of time !=0 stamps for which to calculate the R_inv matrix
            output: 
                RInv[3,3,Dimtime]
            '''
            
            # definition of cos/sin arrays for the stuck angles arrays
            cosTheta = np.cos(np.deg2rad(RollArr))
            senTheta = np.sin(np.deg2rad(RollArr))
            cosPhi   = np.cos(np.deg2rad(PitchArr))
            senPhi   = np.sin(np.deg2rad(PitchArr))
            cosPsi   = np.cos(np.deg2rad(YawArr))
            senPsi   = np.sin(np.deg2rad(YawArr))
            
            # definition of the R_inv matrix array per elements
            R_inv = np.zeros([3, 3, dimTime], dtype=np.double)
            A_inv = np.zeros([3, 3, dimTime], dtype=np.double)
            B_inv = np.zeros([3, 3, dimTime], dtype=np.double)
            C_inv = np.zeros([3, 3, dimTime], dtype=np.double)
            R_inv.fill(np.nan)
            A_inv.fill(0.)
            B_inv.fill(0.)
            C_inv.fill(0.)
            
            A_inv[0,0,indTimeCalc!=0] = 1
            A_inv[1,1,indTimeCalc!=0] = cosTheta
            A_inv[1,2,indTimeCalc!=0] = senTheta
            A_inv[2,1,indTimeCalc!=0] = -senTheta
            A_inv[2,2,indTimeCalc!=0] = cosTheta
            
            B_inv[0,0,indTimeCalc!=0] = cosPhi
            B_inv[1,1,indTimeCalc!=0] = 1
            B_inv[2,2,indTimeCalc!=0] = cosPhi
            B_inv[0,2,indTimeCalc!=0] = -senPhi
            B_inv[2,0,indTimeCalc!=0] = senPhi
            
            C_inv[0,0,indTimeCalc!=0] = cosPsi
            C_inv[0,1,indTimeCalc!=0] = senPsi
            C_inv[2,2,indTimeCalc!=0] = 1
            C_inv[1,0,indTimeCalc!=0] = -senPsi
            C_inv[1,1,indTimeCalc!=0] = cosPsi
            
            # calculation of the rotation matrix
            #vf = np.vectorize(np.matmul, signature='(n,n),(n,n)->(n,n)')
            A_inv = np.moveaxis(A_inv[:,:,indTimeCalc!=0], 2, 0)
            B_inv = np.moveaxis(B_inv[:,:,indTimeCalc!=0], 2, 0)
            C_inv = np.moveaxis(C_inv[:,:,indTimeCalc!=0], 2, 0)
            #R_inv_off = np.matmul(C_inv, np.matmul(B_inv, A_inv))
            # debug version
            R_inv_off = np.matmul(A_inv, np.matmul(B_inv, C_inv))
    
            R_inv_off = np.moveaxis(R_inv_off,0,2)
            # assigning R_inv_off matrices to the R_inv global matrix
            R_inv[:,:, indTimeCalc!=0] = R_inv_off
            return(R_inv)
        
        print('* calculation of R inverse matrix for time with stuck table: ')
        R_inv = f_calcR_inv(RollStuckArr,PitchStuckArr,YawStuckArr,NtimeShip,indexT_FinalArr)
        
        def f_calcRMatrix(rollShipArr,pitchShipArr,yawShipArr,NtimeShip):
            '''
            author: Claudia Acquistapace
            date : 27/10/2020
            goal: function to calculate R matrix given roll, pitch, yaw
            input:
                roll array in degrees
                pitch array in degrees
                yaw array in degrees
                dimtime: dimension of time array for the definition of R_inv as [3,3,dimTime]
            output: 
                R[3,3,Dimtime]
            '''            
            # calculation of the rotational matrix for each time stamp of the ship data for the day
            cosTheta = np.cos(np.deg2rad(rollShipArr))
            senTheta = np.sin(np.deg2rad(rollShipArr))
            cosPhi   = np.cos(np.deg2rad(pitchShipArr))
            senPhi   = np.sin(np.deg2rad(pitchShipArr))
            cosPsi   = np.cos(np.deg2rad(yawShipArr))
            senPsi   = np.sin(np.deg2rad(yawShipArr))
            
            R = np.zeros([3, 3, NtimeShip])
            A = np.zeros([3, 3, NtimeShip])
            B = np.zeros([3, 3, NtimeShip])
            C = np.zeros([3, 3, NtimeShip])
            R.fill(np.nan)
            A.fill(0.)
            B.fill(0.)
            C.fill(0.)
            
            # indexing for the matrices
            #[0,0]  [0,1]  [0,2]
            #[1,0]  [1,1]  [1,2]
            #[2,0]  [2,1]  [2,2]
            A[0,0,:] = 1
            A[1,1,:] = cosTheta
            A[1,2,:] = -senTheta
            A[2,1,:] = senTheta
            A[2,2,:] = cosTheta
            
            B[0,0,:] = cosPhi
            B[1,1,:] = 1
            B[2,2,:] = cosPhi
            B[0,2,:] = senPhi
            B[2,0,:] = -senPhi
            
            C[0,0,:] = cosPsi
            C[0,1,:] = -senPsi
            C[2,2,:] = 1
            C[1,0,:] = senPsi
            C[1,1,:] = cosPsi
                
            # calculation of the rotation matrix
            A = np.moveaxis(A, 2, 0)
            B = np.moveaxis(B, 2, 0)
            C = np.moveaxis(C, 2, 0)
            R = np.matmul(C, np.matmul(B, A))
            R = np.moveaxis(R, 0, 2)
            return(R)   
        
        print('* calculation of R matrix: ')
        R = f_calcRMatrix(rollShipArr,pitchShipArr,yawShipArr,NtimeShip)
        # definition of cos/sin arrays for the stuck angles arrays

        
        # find indeces where timeFinalArr is nan.: they correspond to stable table working
        print('* Calculation of Ep_0 for table not working R_inv*[0,0,-1] and nan for table working')
        # defining the function scalar product to iterate on time dimensione
        func       = np.vectorize(np.dot, signature='(n,n),(n)->(n)')
        # flipping dimension of the matrix to be in the form (dimtime, 3,3)
        R_inv2         = np.moveaxis(R_inv, 2, 0)
        # defining matrix [0,0,-1]* dimtime for pointing direction of radar on ground 
        EpTableOn      = np.zeros((3, NtimeShip))  #np.shape(Ep0Arr[:,np.isnan(timeFinalArr)])[1])
        EpTableOn[2,:] = -1.
        # flipping dimension to adjust for product with rotation inverse matrix
        EpTableOn2     = np.moveaxis(EpTableOn, 1, 0)
        # calculation of Ep0 when table is stuck
        Ep0Arr         = func(R_inv2,EpTableOn2)   #  = np.nan when table works / values when table does not work.
        
        # calculation of the pointing vector Ep for all times as R * Ep0
        R              = np.moveaxis(R, 2, 0)
        EpRotated      = func(R,Ep0Arr)  
        EpRotated      = np.moveaxis(EpRotated, 1, 0)

        print('* saving  Ep values when table does not work in Ep array')
        EpArr[:,indexT_FinalArr!=0] = EpRotated[:,indexT_FinalArr!=0]
                # flipping coordinates 
        Ep0Arr         = np.moveaxis(Ep0Arr, 1, 0)

        print('########################################')
        print('########################################')
        print('########################################')         
        print('EPx max ', np.nanmax(EpArr[0,:]))
        print('EPx min ', np.nanmin(EpArr[0,:]))
        print('EPy max ', np.nanmax(EpArr[1,:]))
        print('EPy min ', np.nanmin(EpArr[1,:]))
        print('EPz max ', np.nanmax(EpArr[2,:]))
        print('EPz min ', np.nanmin(EpArr[2,:]))    
        print('########################################')
        print('########################################')
        print('########################################') 
        
        # storing data in a ncdf file
        R          = np.moveaxis(R, 0, 2)
        EpDataset  = xr.Dataset({'Ep':(['axis','time'], EpArr),
                            'Ep_0':(['axis','time'], Ep0Arr),
                            'R':(['axis','axis','time'], R),
                            'R_inv':(['axis','axis','time'], R_inv),
                            'rollStuck':(['timeStuck'],rollStopArr),
                            'pitchStuck':(['timeStuck'],pitchStopArr),
                            'timeRestart':(['timeStuck'],datetimeRestart)},
                    coords={'time':datetimeShip,
                            'timeStuck':datetimeStop,
                            'axis':np.arange(3)})

    
        # saving data in a ncdf file.
        EpDataset.to_netcdf(pathNcDataAnc+yy+mm+dd+'_Ep_dataset.nc')
  
 
    # saving quickook to check how the processing works
    MetadataDict = {'code':'preprocessShipData.py', 'folder':'/Lavoro/python_scripts/DFG_codes/essd_paper/'}
    ymax_z = -0.5
    ymin_z = -1.#np.nanmax(EpPlot)
    ymax = -0.
    ymin = -1.#np.nanmax(EpPlot)
  
    # test plot for Ep
    # plot mean Doppler velocity time serie
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib import rcParams
    labelsizeaxes = 12
    fontSizeTitle = 12
    fontSizeX = 14
    fontSizeY = 14
    cbarAspect = 10
    fontSizeCbar = 14
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = ['Tahoma']

    Nrows = 3
    Ncols = 1
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(15, 10))
    fig = plt.gcf()
    fig.suptitle("Ep components for the day : "+yy+mm+dd, fontsize=14)
    axes = plt.subplot(Nrows, Ncols, 1)
    axes.spines["top"].set_visible(False)
    axes.spines["right"].set_visible(False)
    axes.get_xaxis().tick_bottom()
    axes.get_yaxis().tick_left()
    axes.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    axes.xaxis.set_minor_formatter(mdates.DateFormatter("%H:%M"))
    axes.xaxis_date()
    axes.plot(datetimeShip, EpArr[0,:], color='green')
    if len(timeStopArr) != 0:
        for ind in range(len(datetimeStop)):
            axes.axvspan(datetimeStop[ind], datetimeRestart[ind], alpha=0.5, color='grey')
    axes.set_xlim(datetimeStart, datetimeEnd)
    axes.set_ylim(ymin,ymax)                                               # limits of the y-axesn  cmap=plt.cm.get_cmap("viridis", 256)
    #ax.set_title('title string', fontsize=fontSizeTitle, loc='left')
    axes.set_xlabel("time [hh:mm]", fontsize=fontSizeX)
    axes.set_ylabel("Epx []", fontsize=fontSizeY)
    

    axes = plt.subplot(Nrows, Ncols, 2)
    axes.spines["top"].set_visible(False)
    axes.spines["right"].set_visible(False)
    axes.get_xaxis().tick_bottom()
    axes.get_yaxis().tick_left()
    axes.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    axes.xaxis.set_minor_formatter(mdates.DateFormatter("%H:%M"))
    axes.xaxis_date()
    axes.plot(datetimeShip, EpArr[1,:], color='orange')
    if len(timeStopArr) != 0:
        for ind in range(len(datetimeStop)):
            axes.axvspan(datetimeStop[ind], datetimeRestart[ind], alpha=0.5, color='grey')
    axes.set_xlim(datetimeStart, datetimeEnd)
    axes.set_ylim(ymin,ymax)                                               # limits of the y-axesn  cmap=plt.cm.get_cmap("viridis", 256)
    #ax.set_title('title string', fontsize=fontSizeTitle, loc='left')
    axes.set_xlabel("time [hh:mm]", fontsize=fontSizeX)
    axes.set_ylabel("Epy []", fontsize=fontSizeY)
    
    
    axes = plt.subplot(Nrows, Ncols, 3)
    axes.spines["top"].set_visible(False)
    axes.spines["right"].set_visible(False)
    axes.get_xaxis().tick_bottom()
    axes.get_yaxis().tick_left()
    axes.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    axes.xaxis.set_minor_formatter(mdates.DateFormatter("%H:%M"))
    axes.xaxis_date()
    axes.plot(datetimeShip, EpArr[2,:], color='red')
    if len(timeStopArr) != 0:
        for ind in range(len(datetimeStop)):
            axes.axvspan(datetimeStop[ind], datetimeRestart[ind], alpha=0.5, color='grey')
    axes.set_xlim(datetimeStart, datetimeEnd)
    axes.set_ylim(ymin_z,ymax_z)                                               # limits of the y-axesn  cmap=plt.cm.get_cmap("viridis", 256)
    #ax.set_title('title string', fontsize=fontSizeTitle, loc='left')
    axes.set_xlabel("time [hh:mm]", fontsize=fontSizeX)
    axes.set_ylabel("Epz []", fontsize=fontSizeY)

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(pathFig+'/Ep_quicklooks/'+yy+mm+dd+'_Ep_quicklooks.png', format='png')#, metadata=MetadataDict)
    
    
    
