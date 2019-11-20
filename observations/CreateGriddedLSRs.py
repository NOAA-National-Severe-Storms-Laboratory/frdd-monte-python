import numpy as np
import xarray as xr

# Personal Imports 
import sys, os
from os.path import join, exists
sys.path.append('/home/monte.flora/wofs/data')
sys.path.append('/home/monte.flora/wofs/util')
sys.path.append('/home/monte.flora/wofs/processing')
from basic_functions import get_key, personal_datetime, check_file_path, to_ordered_dict
from loadWWAs import load_reports, load_tornado_warning
import config
from MultiProcessing import multiprocessing_per_date
from scipy.ndimage import maximum_filter
from skimage.measure import regionprops
get_time = personal_datetime( )

""" usage: stdbuf -oL python -u CreateGriddedLSRs.py 2 > & log & """
###########################################################################
# Label forecast storm objects valid at a single time 
###########################################################################
debug = False 
###########################################

def function_for_multiprocessing( date, time, kwargs):
    '''
    Function for multiprocessing
    '''
    _, adjusted_date, time = get_time.initial_datetime( date_dir = str(date), time_dir = time)
    lsr_xy = load_reports( date, (adjusted_date, time))
    lsr_xy = [ (x,y) for x,y in lsr_xy if x < 249 and y < 249 and x > 0 and y > 0 ]
    gridded_lsrs = np.zeros((250, 250))
    for i, pair in enumerate(lsr_xy):
        gridded_lsrs[pair[0],pair[1]] = i 

    ds = xr.Dataset( {'LSRs':(['y', 'x'], gridded_lsrs)}) 
    fname = 'LSRs_%s-%s.nc' % (adjusted_date, time)
    out_path = join( config.LSR_SAVE_PATH, date )

    if debug:
        ds.to_netcdf(fname)
    else:
        fname = join(out_path, fname)
        print (fname)
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        ds.to_netcdf( fname )
        ds.close( )

if debug: 
    print("\n Working in DEBUG MODE...\n") 
    date = '20170516'; time = '0000'
    function_for_multiprocessing( date, time, kwargs={} )

else:
    datetimes = config.obs_datetimes
    multiprocessing_per_date( datetimes=datetimes, n_date_per_chunk=8, func=function_for_multiprocessing, kwargs={})
