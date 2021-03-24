
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Thu Sep 17 14:30:15 2020

@author: cacquist
@goal: calculate the velocity term Vrot that has to be considered for calculating the correction 
for ship motions (see eq. 11) This term is always present, either in the three components
( table not working) or just z component when the stable table is working (see eq. 25)
It also derives the translational velocity V_trasl, determined by the heave rate
The routine reads in the ship dataset, and calculates the quantities for each time stamp
input: 
    position of the instruments on the ship,
    roll time serie [degrees],
    pitch time serie [degrees], 
    yaw time serie [degrees], 
    
output: 
        Vrot_x component
        Vrot_y component
        Vrot_z component

"""

import pandas as pd
import numpy as np
import pandas as pd
import netCDF4 as nc4
from netCDF4 import Dataset
import xarray as xr
import matplotlib.pyplot as plt

### instrument position coordinates [+5.15m; +5.40m;−15.60m]
x_ship = 5.15 # [m]
y_ship = 5.4  # [m]
z_ship = -15.6 # [m]
instrPosition_dataset = [x_ship, y_ship, z_ship]

# paths to the different data files and output directories for plots
pathFolderTree  = '/Volumes/Extreme SSD/ship_motion_correction_merian/'
pathFig         = pathFolderTree+'/plots/'
pathNcDataAnc   = pathFolderTree+'/ncdf_ancillary/'
ShipFile        = pathFolderTree+'/ship_data/new/shipData_all2.nc'   
 
print('* reading ship data from ncdf file')
# reading ship data for the entire campaign
ShipData       = xr.open_dataset(ShipFile)
rollArr        = ShipData['roll'].values 
pitchArr       = ShipData['pitch'].values 
yawArr         = ShipData['yaw'].values 
heaveArr       = ShipData['heave'].values
datetimeShip   = ShipData['time'].values
#unitsShipArr   = 'seconds since 1970-01-01 00:00:00'
#datetimeShip   = nc4.num2date(timeShipArr, unitsShipArr, only_use_cftime_datetimes=False)

#%%
# calculation of heave, roll and pitch rates
deltaTimeSec    = np.ones(len(heaveArr)-1)
heaveRate       = np.empty_like(heaveArr)
thetaRate       = np.empty_like(rollArr)
phiRate         = np.empty_like(pitchArr)
psiRate         = np.empty_like(yawArr)

#timeSec = shipAngles_dataset.time.values.total_seconds()
heaveRateArr    = np.ediff1d(heaveArr)/deltaTimeSec
thetaRateArr    = np.ediff1d(np.deg2rad(rollArr))/deltaTimeSec
phiRateArr      = np.ediff1d(np.deg2rad(pitchArr))/deltaTimeSec
psiRateArr      = np.ediff1d(np.deg2rad(yawArr))/deltaTimeSec

# append last null element to the speeds for processing the data
heaveRateArr    = np.concatenate([[0.], heaveRateArr])
thetaRateArr    = np.concatenate([[0.], thetaRateArr])
phiRateArr      = np.concatenate([[0.], phiRateArr])
psiRateArr      = np.concatenate([[0.], phiRateArr])
print('rates calculated')


# calculation of the components of the V_rot
print('calculation of x component of Vrot')
Vrot_x =   instrPosition_dataset[0] *  phiRateArr   * np.sin(np.deg2rad(pitchArr)) + \
           instrPosition_dataset[1] * (phiRateArr   * np.cos(np.deg2rad(pitchArr)) * np.sin(np.deg2rad(rollArr)) + \
                                       thetaRateArr * np.sin(np.deg2rad(pitchArr)) * np.cos(np.deg2rad(rollArr))) + \
           instrPosition_dataset[2] * (phiRateArr   * np.cos(np.deg2rad(pitchArr)) * np.cos(np.deg2rad(rollArr)) - \
                                       thetaRateArr * np.sin(np.deg2rad(pitchArr)) * np.sin(np.deg2rad(rollArr)))


print('calculation of y component of Vrot')             
Vrot_y = - instrPosition_dataset[1] *  thetaRateArr * np.sin(np.deg2rad(rollArr)) - \
           instrPosition_dataset[2] *  thetaRateArr * np.cos(np.deg2rad(rollArr))


print('calculation of z component of Vrot')
Vrot_z = - instrPosition_dataset[0] *  thetaRateArr * np.cos(np.deg2rad(rollArr)) + \
           instrPosition_dataset[1] * (thetaRateArr * np.cos(np.deg2rad(pitchArr)) * np.cos(np.deg2rad(rollArr)) - \
                                       phiRateArr   * np.sin(np.deg2rad(pitchArr)) * np.sin(np.deg2rad(rollArr))) - \
           instrPosition_dataset[2] * (thetaRateArr * np.cos(np.deg2rad(pitchArr)) * np.sin(np.deg2rad(rollArr)) + \
                                       phiRateArr   * np.sin(np.deg2rad(pitchArr)) * np.cos(np.deg2rad(rollArr)))


      
# calculating V_rot with Jan's method (double check for results)
               
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
# calculation of the rotational matrix
NtimeShip       = len(datetimeShip)
R               = f_calcRMatrix(rollArr,pitchArr,yawArr,NtimeShip)

# calculation of the radar position r_ship [m]
r_ship          = np.zeros((3,NtimeShip))

for i in range(NtimeShip):
    r_ship[:,i] = np.dot(R[:,:,i],instrPosition_dataset)
    
delta_rx = np.ediff1d(r_ship[0,:])
delta_ry = np.ediff1d(r_ship[1,:])
delta_rz = np.ediff1d(r_ship[2,:])
deltatime =  np.ediff1d(datetimeShip)*10.**(-9)
delta_T = np.zeros(NtimeShip)q
for ind in range(len(deltatime)):
    delta_T[ind] = float(deltatime[ind])
#%%
w_rot_x2 = np.concatenate([[0.], delta_rx])/delta_T
w_rot_y2 = np.concatenate([[0.], delta_ry])/delta_T
w_rot_z2 = np.concatenate([[0.], delta_rz])/delta_T


       

#%%
# saving data in ncdf ancillary file
import xarray as xr
V_rot_traslDataset = xr.Dataset({'Vrot_x':(['time'], Vrot_x),
                                  'Vrot_y':(['time'], Vrot_y),
                                  'Vrot_z':(['time'], Vrot_z),
                                  'Vrot_x2':(['time'], w_rot_x2),
                                  'Vrot_y2':(['time'], w_rot_y2),
                                  'Vrot_z2':(['time'], w_rot_z2),
                                  'V_trasl':(['time'], heaveRateArr),
                                  },
                        coords={'time':datetimeShip})


# saving data in a ncdf file.
V_rot_traslDataset.to_netcdf(pathNcDataAnc+'V_rot_V_trasl_dataset.nc')
              
