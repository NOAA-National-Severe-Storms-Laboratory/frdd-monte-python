from skimage.measure import regionprops
import numpy as np
import xarray as xr 

# Personal Imports 
import sys, os
from os.path import join, exists
sys.path.append('/home/monte.flora/hagelslag/hagelslag/processing')
sys.path.append('/home/monte.flora/wofs/data')
sys.path.append('/home/monte.flora/wofs/util')
sys.path.append('/home/monte.flora/wofs/processing')

from EnhancedWatershedSegmenter import rescale_data
from basic_functions import get_key, personal_datetime
from ObjectIdentification import ObjectIdentification
from ObjectMatching import ObjectMatching
from loadEnsembleData import EnsembleData, calc_time_max
from loadMRMSData import MRMSData
from StoringObjectProperties import initialize_dict, append_object_properties, save_object_properties  
import config
from MultiProcessing import multiprocessing_per_date

###########################################################################
# Label forecast storm objects valid at a single time 
###########################################################################
fcst_time_idx = 12  
label_option = 'watershed_and_single_thresh' 
variable_key = 'dbz'

time_indexs = [ 0 ] 

# Personal Classes 
obj_match = ObjectMatching( dist_max = config.matching_dist, one_to_one=True )
obj_id = ObjectIdentification( )
mrms = MRMSData( )
get_time = personal_datetime( )

property_name_list = initialize_dict( )
n_props = len(property_name_list)

def function_for_multiprocessing( date, time, kwargs):
    a_dict = {'date':date, 'time':time}     
    # Only need to load the forecast data at 60-min ( as a start ) 
    instance = EnsembleData( date_dir=date, time_dir=time, base_path = 'summary_files')
    ens_data = instance.load( var_name_list=[config.variable_attrs[variable_key]['var_newse'] ], time_indexs=time_indexs, tag=get_key( config.newse_tag, config.variable_attrs[variable_key]['var_newse'] ))
    ens_data = ens_data[0,0,:,:] 

    # Load MRMS 60-min rotation tracks 
    valid_date_and_time, initial_date_and_time = get_time.determine_forecast_valid_datetime( date_dir=str(date), time_dir=time, fcst_time_idx=fcst_time_idx )
    # Get the correct date and time for the observed rotation tracks
    labels_obs, az_shear = mrms.load_az_shr_tracks( var=config.variable_attrs['low-level']['var_mrms'], date_dir=str(date), valid_date=valid_date_and_time[0], valid_time=valid_date_and_time[1], tag='watershed_and_single_thresh' ) 
    obs_object_props = regionprops( labels_obs.astype(int), az_shear, coordinates='rc' )

    # Cycle through the individual ensemble members 
    ens_object_labels = np.zeros(( ens_data.shape ))
    object_properties_for_all_objects = [ ]
    for mem in range( np.shape( ens_data )[0] ):
        # Label storm objects 
        fcst_object_labels, fcst_object_props = obj_id.label( input_data=ens_data[mem,:,:], method = 'single_threshold', bdry_thresh = config.variable_attrs[variable_key]['newse_bdry'] )
        # Quality control the identified objects 
        qc_fcst_object_labels, qc_fcst_object_props = obj_id.quality_control(fcst_object_labels, fcst_object_props, input_data = ens_data[mem,:,:], **config.qc_params_dbz )
        # Match Storm Objects to observed low-level rotation
        matched_obs_labels, matched_fcst_labels, _  = obj_match.match_objects( labels_obs, qc_fcst_object_labels )
        
         # Identify the storms as MCS or non-MCS 
        fcst_object_labels_small_thresh, fcst_object_props_small_thresh = obj_id.label( input_data=ens_data[mem,:,:], method = 'single_threshold', bdry_thresh = 35. )
        # Quality control the identified objects 
        qc_fcst_object_labels_small_thresh, qc_fcst_object_props_small_thresh = obj_id.quality_control(fcst_object_labels_small_thresh, fcst_object_props_small_thresh, input_data = ens_data[mem,:,:], **config.qc_params_dbz ) 

        storm_labels = obj_id.identify_storm_mode( qc_fcst_object_labels, qc_fcst_object_labels_small_thresh, qc_fcst_object_props, qc_fcst_object_props_small_thresh)

        # Store forecast object properties 
        ens_object_labels[mem,:,:] = qc_fcst_object_labels
        object_properties_for_all_objects.append( append_object_properties ( qc_fcst_object_labels, qc_fcst_object_props, matched_fcst_labels, storm_labels, n_props, mem, **a_dict ))   

    object_properties_for_all_objects = np.concatenate( object_properties_for_all_objects, axis = 0 ) 
    data = { object_property : (['Object'], object_properties_for_all_objects[:,i] ) for i, object_property in enumerate(property_name_list) } 
    data['Objects'] = (['Ensemble Member', 'y', 'x'], ens_object_labels)
    ds = xr.Dataset( data )
    out_path = join( config.object_storage_dir_ml, date )
    if not exists( out_path ):
        try:
           os.mkdir( out_path )
        except OSError:
           print(out_path, "exists!")
    ds.to_netcdf(path = join(out_path, 'WOFS_%s_OBJECTS_%s-%s_%02d.nc' % (config.variable_attrs[variable_key]['title'], date, time, fcst_time_idx)) )
    ###ds.to_netcdf( 'test.nc' )


#date = '20180601'    
#time = '0200'
#function_for_multiprocessing( date, time, kwargs={} )

datetimes  = config.datetimes_ml
kwargs = { }
multiprocessing_per_date( datetimes=datetimes, n_date_per_chunk=15, func=function_for_multiprocessing, kwargs=kwargs)

