#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thur 1 Jul 2021

@author: cacquist
@ goals:
    Rename wband quicklooks to match the convention from the w band

"""
import numpy as np
import glob
import xarray as xr
import pandas as pd
from datetime import datetime
from datetime import timedelta
from pathlib import Path
import os


# generating array of days for the dataset
Eurec4aDays     = pd.date_range(datetime(2020,1,19),datetime(2020,2,19),freq='d')
NdaysEurec4a    = len(Eurec4aDays)
path_files      = '/Volumes/Extreme SSD/ship_motion_correction_merian/plots/quicklooks/'
#hours_list = ['00', '01', '02', '03', '04',  '05', '06', '07', '08', '09', '10', '11', '12', '13', '14','15','16','17','18','19','20','21','22','23']
#%%


for indDay in range(NdaysEurec4a):

    dayEu           = Eurec4aDays[indDay]
    yy              = str(dayEu)[0:4]
    mm              = str(dayEu)[5:7]
    dd              = str(dayEu)[8:10]
    date            = yy+mm+dd
    # setting dates strings
    date            = dd+mm+yy      #'04022020'
    dateRadar       = yy[0:2]+mm+dd #'200204'
    dateReverse     = yy+mm+dd      #'20200204'

    print('processing date: ', date)
    path_out =  '/Volumes/Extreme SSD/ship_motion_correction_merian/plots/quicklooks_hourly/'+yy+mm+dd+'/'

    file_names = np.sort(glob.glob(path_files+yy+mm+dd+'*.png'))
    for indfile, file_name in enumerate(file_names):
        src = file_name
        dst = path_out+file_name[len(path_files):]
        os.rename(src, dst)

strasuka


var_w_band = ['metstation', 'Sk', 'Sw', 'Vd', 'Ze']
var_mrr = ['RR', 'LWC', 'Zea', 'W']


for indDay in range(NdaysEurec4a):

    dayEu           = Eurec4aDays[indDay]
    yy              = str(dayEu)[0:4]
    mm              = str(dayEu)[5:7]
    dd              = str(dayEu)[8:10]
    date            = yy+mm+dd
    # setting dates strings
    date            = dd+mm+yy      #'04022020'
    dateRadar       = yy[0:2]+mm+dd #'200204'
    dateReverse     = yy+mm+dd      #'20200204'

    print('processing date: ', date)
    path_out = path_files


    for var in var_w_band:
        src = path_out+yy+mm+dd+'_'+var+'_quicklook_WBAND.png'
        dst = path_out+yy+mm+dd+'_cd_'+var+'_quicklook_WBAND.png'

        if os.path.isfile(src):
            # moving the file to new destination
            os.rename(src, dst)
        elif os.path.isfile(dst):
            print('wband file of '+var+'already changed')
        else:
            print('houston wband!')

    for var_m in var_mrr:
        src_mrr = path_out+yy+mm+dd+'_'+var_m+'_quicklook_MRR.png'
        dst_mrr = path_out+yy+mm+dd+'_cd_'+var_m+'_quicklook_MRR.png'
        if os.path.isfile(src_mrr):
            # renaming the files from MRR
            os.rename(src_mrr, dst_mrr)
        elif os.path.isfile(dst_mrr):
            print('mrr file of '+var_m+'already changed')
        else:
            print('houston mrr!')

    # change the status file
    src_status = path_out+yy+mm+dd+'_msm_status.png'
    dst_status = path_out+yy+mm+dd+'_cd_status_quicklook_WBAND.png'
    if os.path.isfile(src_status):
    # renaming status files
        os.rename(src_status, dst_status)
    elif os.path.isfile(dst_status):
        print('status file already changed')
    else:
        print('houston status!')

    # changing the map file
    src_map = path_out+yy+mm+dd+'_MariaSMerianTrack.png'
    dst_map = path_out+yy+mm+dd+'_cd_MariaSMerianTrack.png'
    if os.path.isfile(src_map):
    # renaming status files
        os.rename(src_map, dst_map)

    elif os.path.isfile(dst_map):
        print('map file already changed')
    else:
        print('houston map!')
