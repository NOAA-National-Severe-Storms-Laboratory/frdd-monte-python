from wofs.util import config
from wofs.util.MultiProcessing import multiprocessing_per_date
import numpy as np
from datetime import datetime
from IdentifyForecastTracks import function_for_multiprocessing

""" usage: stdbuf -oL python -u run_IdentifyForecastTracks.py 2 > & log & """
###########################################################################
# Label forecast storm objects valid at a single time 
###########################################################################
debug = True
# WOFS_UPDRAFT_SWATH_OBJECTS_20180501-2300_12.nc
if debug:
    print("\n Working in DEBUG MODE...\n")
    date = '20180501'    
    time = '2300'
    fcst_time_idx = 6
    kwargs = {'time_indexs': np.arange( config.N_TIME_IDX_FOR_HR+1 ) + fcst_time_idx, 
              'fcst_time_idx': fcst_time_idx,
               'debug': debug}
    function_for_multiprocessing( date, time, kwargs )
    
else:
    datetimes  = config.datetimes_ml
    print ( len(datetimes) ) 
    for fcst_time_idx in config.fcst_time_idx_set:
        print('\n Start Time:', datetime.now().time())
        print("Forecast Time Index: ", fcst_time_idx)
        kwargs = {'time_indexs': np.arange( config.N_TIME_IDX_FOR_HR+1 ) + fcst_time_idx, 'fcst_time_idx': fcst_time_idx}
        multiprocessing_per_date( datetimes=datetimes, n_date_per_chunk=8, func=function_for_multiprocessing, kwargs=kwargs)
        print('End Time: ', datetime.now().time())
