#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 14:12:23 2021

@author: claudia
"""

import numpy as np
import glob
import xarray as xr
import pandas as pd
from datetime import datetime
from datetime import timedelta
from pathlib import Path

# code to read 5 s int time data and resample them on 10 s resolution
path = '/Volumes/Extreme SSD/ship_motion_correction_merian/corrected_data/mrr_final/formatted/commented/'
file_list = np.sort(glob.glob('/Volumes/Extreme SSD/ship_motion_correction_merian/corrected_data/mrr_final/formatted/commented/*.nc'))


for indFile, file in enumerate(file_list):
    
    MRRdata = xr.open_dataset(file)
    file_out = file[len(path):]
    print(file_out)
    # add comment on fall speed variable
    attrs={'CREATED_BY'     : 'Claudia Acquistapace and Albert Garcia Benadi',
                    'ORCID-AUTHORS'   : "Claudia Acquistapace: 0000-0002-1144-4753, Albert Garcia Benadi : 0000-0002-5560-4392",
                    'DOI'             :  "10.25326/233 , https://doi.org/10.25326/233",
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
                    'DATA_DESCRIPTION' : 'daily MRR measurements on Maria S. Merian (msm) ship during EUREC4A campaign',
                    'DATA_DISCIPLINE'  : 'Atmospheric Physics - Remote Sensing Radar Profiler',
                    'DATA_GROUP'       : 'Experimental;Profile;Moving',
                    'DATA_LOCATION'    : 'Research vessel Maria S. Merian - Atlantic Ocean',
                    'DATA_SOURCE'      : 'MRR-PRO data postprocessed',
                    'DATA_PROCESSING'  : 'ship motion correction and filtering of interference the code used is available at https://github.com/ClauClouds/ship-motion-correction-for-EUREC4A-campaign',
                    'INSTRUMENT_MODEL' : 'MRR PRO (24 Ghz radar)',
                    'COMMENT'          : 'The MRR pro belongs to Jun. Prof. Heike Kalesse, University of Leipzig (DE)' }

    MRRdata.attrs  = attrs
   # assign additional attributes following CF convention
    MRRdata = MRRdata.assign_attrs({
            "Conventions": "CF-1.8",
            "title": MRRdata.attrs["DATA_DESCRIPTION"],
            "institution": MRRdata.attrs["DS_AFFILIATION"],
            "history": "".join([
                "source: " + MRRdata.attrs["DATA_SOURCE"] + "\n",
                "processing: " + MRRdata.attrs["DATA_PROCESSING"] + "\n",
                "postprocessing with de-aliasing developed by Albert Garcia Benadi " + '\n',
                "adapted to enhance CF compatibility\n",
            ]),  # the idea of this attribute is that each applied transformation is appended to create something like a log
            "featureType": "trajectoryProfile",
        })

    
    MRRdata.to_netcdf('/Volumes/Extreme SSD/ship_motion_correction_merian/corrected_data/mrr_final/formatted/commented/with_DOI/'+file_out)
    
