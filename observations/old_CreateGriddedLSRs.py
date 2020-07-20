import numpy as np
import xarray as xr

# Personal Imports 
import os
from os.path import join, exists
from wofs.util.basic_functions import get_key, personal_datetime, check_file_path, to_ordered_dict
from wofs.data.loadWWAs import load_reports, load_tornado_warning
from wofs.util import config
from machine_learning.main.run_ensemble_feature_extraction import run_parallel
from scipy.ndimage import maximum_filter
from skimage.measure import regionprops
get_time = personal_datetime( )

""" usage: stdbuf -oL python -u CreateGriddedLSRs.py 2 > & log & """
###########################################################################
# Label forecast storm objects valid at a single time 
###########################################################################
debug = False 
###########################################

def worker(date, time):
    '''
    Function for multiprocessing
    '''
    _, adjusted_date, time = get_time.initial_datetime( date_dir = str(date), time_dir = time)
    for duration in [30,60]:
        hail_xy, torn_xy, wind_xy = load_reports( date, (adjusted_date, time), all_lsrs=False, forecast_length=duration)
        lsr_tuple
        for xy in lsr_xy_tuple:



    lsr_xy = [ (x,y) for x,y in lsr_xy if x < 249 and y < 249 and x > 0 and y > 0 ]
    gridded_lsrs = np.zeros((250, 250))
    for i, pair in enumerate(lsr_xy):
        gridded_lsrs[pair[0],pair[1]] = i 

    data = {'LSR_3km_grid':(['y', 'x'], gridded_lsrs),
            'LSR_15km_grid':(['y', 'x'], maximum_filter(gridded_lsrs,5)),
            'LSR_30km_grid':(['y', 'x'], maximum_filter(gridded_lsrs,10))
            }

    ds = xr.Dataset(data) 
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
    worker( date, time, kwargs={} )

else:
    multiprocessing_per_date( datetimes=datetimes, n_date_per_chunk=8, func=worker, kwargs={})

    dates = config.ml_dates
    times = config.observation_times

    run_parallel(
            func = worker,
            nprocs_to_use = 0.4,
            iterator = itertools.product(dates, times)
            )






