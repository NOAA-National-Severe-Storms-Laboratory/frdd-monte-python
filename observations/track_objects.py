import sys, os 
from os.path import join
import xarray as xr 
import numpy as np 
# Personal Modules 
sys.path.append('/home/monte.flora/wofs/processing')
sys.path.append('/home/monte.flora/wofs/util')
sys.path.append('/home/monte.flora/wofs/data')
from ObjectIdentification import ObjectIdentification, QualityControl 
from loadMRMSData import MRMSData 
from loadLSRs import loadLSR
from loadWWAs import loadWWA
from basic_functions import personal_datetime, check_file_path, to_ordered_dict
from MultiProcessing import multiprocessing_per_date
import config 
from scipy.ndimage import gaussian_filter, maximum_filter
from hagelslag.processing.tracker import extract_storm_objects, track_storms
from hagelslag.processing.ObjectMatcher import shifted_centroid_distance
import matplotlib.pyplot as plt
from ObjectMatching import ObjectMatching
from ObjectTracking import ObjectTracking 

# USER-DEFINED PARAMETERS 
level = 'dbz' #('low-level','mid-level','dbz') 
debug = True 
################################
obj_id = ObjectIdentification( )
qc = QualityControl( )
matching_dist = 3
obj_track  = ObjectTracking( dist_max = 10, one_to_one=False ) 

qc_params = [ ('min_area', 12.)]
qc_params_official = to_ordered_dict( qc_params ) 
obj_id_params = {'bdry_thresh': 45.} #config.variable_attrs[level]['mrms_bdry']}

def function_for_multiprocessing( date, time, kwargs ): 
    '''
    Objectively identifies composite reflectivity object every 5 minutes 
    '''
    in_path = join( config.MRMS_DBZ_PATH, date )
    mrms_files = sorted( os.listdir( in_path ) )[29:50] 
    objects_at_different_times = [ ] 
    dbz_at_different_times = [ ]
    num_unique_objects = 0
    for mrms_file in mrms_files:
        ds = xr.open_dataset( join(in_path, mrms_file) )
        objects = ds['Storm Objects'].values 
        dbz = ds['DBZ'].values
        modified_objects = objects+num_unique_objects
        modified_objects[objects==0] = 0 
        objects_at_different_times.append( modified_objects ) 
        dbz_at_different_times.append( dbz )
        ds.close( )
        num_unique_objects += len( np.unique( objects )[1:])

    objects = np.array( objects_at_different_times ) #Shape: (138, 250, 250)
    original_dbz = np.array( dbz_at_different_times )
    #tracked_objects = obj_track.track_objects( objects, original_dbz )     

    x, y = np.meshgrid(np.arange(250), np.arange(250))
    storm_objs = extract_storm_objects(objects, original_dbz, x, y, np.arange(len(objects)))
    out_storms = track_storms(storm_objs, np.arange(len(objects)), [shifted_centroid_distance], np.array([30]), np.array([1])) 

    data = {'Storm Objects': (['Time', 'Y', 'X'], tracked_objects)}
    data['DBZ'] =  (['Time', 'Y', 'X'], dbz_at_different_times)
    ds = xr.Dataset( data )
    ds.to_netcdf( 'test.nc' ) 

if debug: 
    print ("Working in DEBUG MODE!.....") 
    function_for_multiprocessing( date = '20180501', time = None, kwargs={ } ) 
else:
    #datetimes = config.obs_datetimes
    multiprocessing_per_date(datetimes=datetimes, n_date_per_chunk=4, func=function_for_multiprocessing, kwargs={})  



