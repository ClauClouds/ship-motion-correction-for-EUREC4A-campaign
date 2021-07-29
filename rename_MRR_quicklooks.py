#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thur 1 Jul 2021

@author: cacquist
@ goals:
    Rename mrr quicklooks to match the convention from the w band

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
path_files      = '/Volumes/Extreme SSD/ship_motion_correction_merian/mrr/plots/hourly_plots/'
hours_list = ['00', '01', '02', '03', '04',  '05', '06', '07', '08', '09', '10', '11', '12', '13', '14','15','16','17','18','19','20','21','22','23']
#%%
var_list = ['RR', 'W', 'Zea', 'LWC']

var_w_band = ['metstation', 'Sk', 'Sw', 'Vd', 'Ze']



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
    path_out = '/Volumes/Extreme SSD/ship_motion_correction_merian/plots/quicklooks_hourly/'+yy+mm+dd+'/'

    for hh in hours_list:
        for var in var_w_band:
            if var == 'metstation':
                src = path_out+yy+mm+dd+'_'+hh+'_msm_'+var+'.png'
            else:
                src = path_out+yy+mm+dd+'_'+hh+'_msm_'+var+'_quicklooks.png'
            print(src)
            if os.path.isfile(src):
                print('sono qui')
                dst = path_out+yy+mm+dd+'_'+hh+'_'+var+'_quicklook_WBAND.png'

                # moving the file to new destination
                os.rename(src, dst)


MRRPROCESSED = True
if ~MRRPROCESSED:
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
        path_out = '/Volumes/Extreme SSD/ship_motion_correction_merian/plots/quicklooks_hourly/'+yy+mm+dd+'/'

        for hh in hours_list:
            for var in var_list:
                src = path_files+dd+mm+yy+'_'+hh+'_'+var+'_quicklook_MRR.png'
                if os.path.isfile(src):
                    dst = path_out+yy+mm+dd+'_'+hh+'_'+var+'_quicklook_MRR.png'

                    # moving the file to new destination
                    os.rename(src, dst)
