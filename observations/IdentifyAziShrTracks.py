import sys, os 
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
level = 'low-level' #('low-level','mid-level','dbz') 
min_thresh = 0.002 # (temporarily changed from 0.002)
debug = True
################################
obj_id = ObjectIdentification( )
mrms = MRMSData( ) 
get_time = personal_datetime( )
qc = QualityControl( )

watershed_params_official = {'min_thresh': int(min_thresh*1e4), 'max_thresh': int(0.008*1e4) , 'data_increment': 1, 'delta': 80, 'size_threshold_pixels': 200, 'local_max_area': 100 }
qc_params = [ ('min_area', 10.), ('merge_thresh', 2.), ('min_time', [2]), ('max_thresh', 0.0035)]
qc_params_official = to_ordered_dict( qc_params ) 

def function_for_multiprocessing( date, time , kwargs ): 
    '''
    Objectively-identifies rotation tracks in hour-long azimuthal shear data. 
    '''
    datetime_obj, adjusted_date, time = get_time.initial_datetime( date_dir = date, time_dir = time) 
    az_shr_composite, az_shr_time_argmax_indices = mrms.load_mrms_time_composite( var_name =config.VARIABLE_ATTRIBUTES[level]['var_mrms'], date=date, time=time ) 
    processed_data = 1e4*az_shr_composite
    object_labels, object_props = obj_id.label( input_data = processed_data, method='watershed',  **watershed_params_official )
    
    qc_params_official['min_time'].append( az_shr_time_argmax_indices )
    qc_object_labels, _ = qc.quality_control( input_data = az_shr_composite, object_labels = object_labels, object_properties=object_props, qc_params=qc_params_official )

    out_path = os.path.join( config.MRMS_TRACKS_PATH, date )
    check_file_path( out_path )  
    filename = os.path.join( out_path, 'MRMS_%s_TRACKS_%s%s.nc' % ( config.VARIABLE_ATTRIBUTES[level]['var_mrms'], adjusted_date, time ) )

    data = { 'Rotation Tracks' : (['Y', 'X'], qc_object_labels) }
    data['Azimuthal Shear'] = (['Y', 'X'], az_shr_composite)
    ds = xr.Dataset( data )
    if debug: 
        ds.to_netcdf( 'MRMS_%s_TRACKS_%s%s.nc' % ( config.VARIABLE_ATTRIBUTES[level]['var_mrms'], adjusted_date, time ) ) 
    else:
        ds.to_netcdf( filename )

    ds.close()
    del ds, data, az_shr_composite, az_shr_time_argmax_indices, processed_data  

# 20180601, 0200
# 20180501, 0000
# 20170516, 0000
# 20180529, 2300
# 20180531, 0100
# 20180516 1900
if debug: 
    print ("Working in DEBUG MODE!.....") 
    function_for_multiprocessing( date = '20180501', time = '2330', kwargs={ } ) 
else:
    datetimes = config.obs_datetimes
    multiprocessing_per_date(datetimes=datetimes, n_date_per_chunk=4, func=function_for_multiprocessing, kwargs={})  



