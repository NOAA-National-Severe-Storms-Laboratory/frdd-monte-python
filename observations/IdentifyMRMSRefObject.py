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

# USER-DEFINED PARAMETERS 
level = 'dbz' #('low-level','mid-level','dbz') 
debug = True 
################################
obj_id = ObjectIdentification( )
qc = QualityControl( )

qc_params = [ ('min_area', 10.)]
qc_params_official = to_ordered_dict( qc_params ) 

method = 'single_threshold'
obj_id_params = {'bdry_thresh': config.variable_attrs[level]['mrms_bdry']}

#method = 'watershed'
#obj_id_params = {'min_thresh': 45, 'max_thresh': 80 , 'data_increment': 5, 'delta': 80, 'size_threshold_pixels': 100, 'local_max_area': 50 }

def function_for_multiprocessing( date, time, kwargs ): 
    '''
    Objectively identifies composite reflectivity object every 5 minutes 
    '''
    in_path = join(config.MRMS_PATH, str(date)) 
    out_path = join( config.MRMS_DBZ_PATH, date )
    check_file_path( out_path )
    mrms_files = os.listdir( in_path )  
    for mrms_file in mrms_files:
        ds = xr.open_dataset( join(in_path, mrms_file) )
        dbz = ds['DZ_CRESSMAN'].values 
        object_labels, object_props = obj_id.label( input_data = dbz, method=method,  **obj_id_params )
        qc_object_labels, _ = qc.quality_control( input_data = dbz, object_labels = object_labels, object_properties=object_props, qc_params=qc_params_official )
        data = { 'Storm Objects' : (['Y', 'X'], qc_object_labels) }
        data['DBZ'] = (['Y', 'X'], dbz )
        ds = xr.Dataset( data )
        ds.to_netcdf( join( out_path, 'MRMS_STORM_OBJECTS_'+mrms_file))

        ds.close()
        del data

if debug: 
    print "Working in DEBUG MODE!....." 
    function_for_multiprocessing( date = '20180501', time = None, kwargs={ } ) 
else:
    #datetimes = config.obs_datetimes
    multiprocessing_per_date(datetimes=datetimes, n_date_per_chunk=4, func=function_for_multiprocessing, kwargs={})  



