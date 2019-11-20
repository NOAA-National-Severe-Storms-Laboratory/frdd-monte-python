import numpy as np
import xarray as xr 

# Personal Imports 
import sys, os
from os.path import join, exists
sys.path.append('/home/monte.flora/hagelslag/hagelslag/processing')
sys.path.append('/home/monte.flora/wofs/data')
sys.path.append('/home/monte.flora/wofs/util')
sys.path.append('/home/monte.flora/wofs/processing')
from basic_functions import get_key, personal_datetime, check_file_path, to_ordered_dict
from ObjectIdentification import ObjectIdentification, QualityControl
from loadEnsembleData import EnsembleData, calc_time_max
from ObjectMatching import ObjectMatching, match_to_lsrs
from loadMRMSData import MRMSData
from loadWWAs import load_reports, load_tornado_warning, load_svr_warning
from StoringObjectProperties import object_properties_list, append_object_properties, save_object_properties
from loadMRMSData import MRMSData
import config
from MultiProcessing import multiprocessing_per_date
from scipy.ndimage import maximum_filter, gaussian_filter
from datetime import datetime
obj_id  = ObjectIdentification( )
qc = QualityControl( )
get_time = personal_datetime( )
obj_match = ObjectMatching( dist_max = 14., time_max=1, one_to_one=True )
mrms = MRMSData( )
object_properties = object_properties_list( )


""" usage: stdbuf -oL python -u IdentifyForecastTracks.py 2 > & log & """
###########################################################################
# Label forecast storm objects valid at a single time 
###########################################################################
debug = False 
variable_key = 'updraft'
###########################################
WATERSHED_PARAMS = config.VARIABLE_ATTRIBUTES[variable_key]['watershed_params'] 
QC_PARAMS = config.VARIABLE_ATTRIBUTES[variable_key]['qc_params']
QC_PARAMS = to_ordered_dict( QC_PARAMS )

def function_for_multiprocessing( date, time, kwargs):
    date_and_time_dict = {'date':date, 'time':time}
    out_path = join( config.OBJECT_SAVE_PATH, date )

    # READ THE TORNADO WARNINGS, AZIMUTHAL SHEAR TRACKS, AND LSRs
    valid_date_and_time, initial_date_and_time = get_time.determine_forecast_valid_datetime( date_dir=str(date), time_dir=time, fcst_time_idx=fcst_time_idx )
    obs_rotation_tracks, az_shear = mrms.load_az_shr_tracks( var=config.VARIABLE_ATTRIBUTES['low-level']['var_mrms'], name='Rotation Tracks', date_dir=str(date), valid_datetime=valid_date_and_time )
    torn_warn_xy = load_tornado_warning(str(date), initial_date_and_time, time_window=15)
    svr_warn_xy = load_svr_warning(str(date), initial_date_and_time, time_window=15)
    lsr_xy = load_reports(str(date), valid_date_and_time, time_window=15) 

    instance = EnsembleData( date_dir=date, time_dir=time, base_path = 'summary_files')
    input_data = instance.load( variables=[config.VARIABLE_ATTRIBUTES[variable_key]['var_newse'] ], time_indexs=kwargs['time_indexs'], tag=get_key( config.newse_tag, config.VARIABLE_ATTRIBUTES[variable_key]['var_newse'] ) )
    ens_data, ens_data_indices = calc_time_max( input_data=input_data[:,0,:,:,:], argmax=True) 
    valid_date_and_time, initial_date_and_time = get_time.determine_forecast_valid_datetime( date_dir=str(date), time_dir=time, fcst_time_idx=fcst_time_idx )
    ens_object_labels = np.zeros((ens_data.shape ))
    object_properties_for_all_objects = [ ]
    for mem in range( np.shape( ens_data )[0] ):
        fcst_object_labels, fcst_object_props = obj_id.label( input_data=ens_data[mem,:,:], method = 'watershed', **WATERSHED_PARAMS) 
        QC_PARAMS['min_time'].append( ens_data_indices[mem,:,:]) 
        qc_fcst_object_labels, qc_fcst_object_props = qc.quality_control( input_data = ens_data[mem,:,:], object_labels = fcst_object_labels, object_properties=fcst_object_props, qc_params=QC_PARAMS )
        ens_object_labels[mem,:,:] = qc_fcst_object_labels
        
        forecast_labels_matched_to_torn_warn = match_to_lsrs( object_properties=qc_fcst_object_props, lsr_points=torn_warn_xy, dist_to_lsr=10 )
        forecast_labels_matched_to_lsrs = match_to_lsrs( object_properties=qc_fcst_object_props, lsr_points=lsr_xy, dist_to_lsr=10 )
        forecast_labels_matched_to_svr_warn = match_to_lsrs( object_properties=qc_fcst_object_props, lsr_points=svr_warn_xy, dist_to_lsr=10 )

        times_a = [valid_date_and_time[0] +' '+ valid_date_and_time[1]]; times_b = times_a
        _,forecast_labels_matched_to_azshr,_ = obj_match.match_objects( [obs_rotation_tracks], [qc_fcst_object_labels], times_a, times_b )
        forecast_labels_matched_to_azshr = [ int(string.split('_' )[1]) for string in forecast_labels_matched_to_azshr]
        forecast_labels_matched_to_azshr  = {region.label: 1.0 if region.label in forecast_labels_matched_to_azshr else 0.0 for region in qc_fcst_object_props}


        kwargs = {'date':date, 'time':time, 'mem': mem}
        object_properties_for_all_objects.append( append_object_properties( object_props = qc_fcst_object_props,
                                                                            matched_to_torn_warn = forecast_labels_matched_to_torn_warn,
                                                                            matched_to_lsrs = forecast_labels_matched_to_lsrs,
                                                                            matched_to_azshr = forecast_labels_matched_to_azshr,
                                                                            matched_to_svr_warn = forecast_labels_matched_to_svr_warn,
                                                                            **kwargs ))
        
    data = { }
    data['Objects'] = (['Ensemble Member', 'y', 'x'], ens_object_labels)
    data['Raw Data'] = (['Ensemble Member', 'y', 'x'], ens_data)
    
    object_properties_for_all_objects = np.concatenate( object_properties_for_all_objects, axis = 0 )
    for i, object_property in enumerate(object_properties):
       data[object_property] = (['Object'], object_properties_for_all_objects[:,i] )

    del ens_object_labels, ens_data, ens_data_indices

    ds = xr.Dataset( data )
    fname = join(out_path, 'WOFS_%s_OBJECTS_%s-%s_%02d.nc' % (config.VARIABLE_ATTRIBUTES[variable_key]['title'], date, time, fcst_time_idx))
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    if debug:
        ds.to_netcdf( 'WOFS_%s_OBJECTS_%s-%s_%02d.nc' % (config.VARIABLE_ATTRIBUTES[variable_key]['title'], date, time, fcst_time_idx) )
    else:
        ds.to_netcdf( fname )
        ds.close( )
        del data

if debug: 
    print("\n Working in DEBUG MODE...\n") 
    date = '20170516'    
    time = '0000'
    fcst_time_idx = 6 
    kwargs = {'time_indexs': np.arange( config.N_TIME_IDX_FOR_HR+1 ) + fcst_time_idx, 'fcst_time_idx': fcst_time_idx}
    function_for_multiprocessing( date, time, kwargs )

else:
    datetimes  = config.datetimes_ml
    for fcst_time_idx in config.fcst_time_idx_set:
        print('Start Time:', datetime.now().time())
        print("Forecast Time Index: ", fcst_time_idx)
        kwargs = {'time_indexs': np.arange( config.N_TIME_IDX_FOR_HR+1 ) + fcst_time_idx, 'fcst_time_idx': fcst_time_idx}
        multiprocessing_per_date( datetimes=datetimes, n_date_per_chunk=8, func=function_for_multiprocessing, kwargs=kwargs)
        print('End Time: ', datetime.now().time())
