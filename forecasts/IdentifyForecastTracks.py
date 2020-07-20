import numpy as np
import xarray as xr 
from os.path import join, exists
import os 
import pandas as pd 

from wofs.util.basic_functions import get_key, personal_datetime, check_file_path, to_ordered_dict
from wofs.processing.ObjectIdentification import label_regions, QualityControl
from wofs.data.loadEnsembleData import EnsembleData, calc_time_max
from wofs.processing.ObjectMatching import ObjectMatching, match_to_lsrs
from wofs.data.loadMRMSData import MRMSData
from wofs.data.loadWWAs import load_reports, load_tornado_warning, load_svr_warning, is_severe
from wofs.util.StoringObjectProperties import get_object_properties, save_object_properties
from wofs.data.loadMRMSData import MRMSData
from wofs.util import config
from scipy.ndimage import maximum_filter, gaussian_filter
from datetime import datetime

qc = QualityControl( )
get_time = personal_datetime( )
obj_match = ObjectMatching( cent_dist_max = 10., min_dist_max = 10., time_max=0, one_to_one=True )
mrms = MRMSData( )

""" usage: stdbuf -oL python -u IdentifyForecastTracks.py 2 > & log & """
###########################################################################
# Label forecast storm objects valid at a single time 
###########################################################################
variable_key = 'updraft'
###########################################
WATERSHED_PARAMS = config.VARIABLE_ATTRIBUTES[variable_key]['watershed_params'] 
QC_PARAMS = config.VARIABLE_ATTRIBUTES[variable_key]['qc_params']
QC_PARAMS = to_ordered_dict( QC_PARAMS )

def _load_verification( date, valid_date_and_time, initial_date_and_time, time_window=15, load_az_shr=None, **kwargs ):
    '''
    Calls the load functions for the local storm reports (tornado,
        severe wind, hail) and tornado and severe weather warning
        polygons.
    -----------------
    Inputs:
        date, str, date of reports (YYMMDDDD)
        time, str, center of the time window of the reports (HHmm)
        time_window, int, length of half of the window to search 
                          for LSRS/warning polygons (in minutes)
    '''
    if len(kwargs['time_indexs']) == 7:
        forecast_length=30
    elif len(kwargs['time_indexs']) == 13:
        forecast_length=60

    obs_rotation_tracks = [ ]
    if load_az_shr is not None: 
        obs_rotation_tracks, az_shear = mrms.load_az_shr_tracks( var=config.VARIABLE_ATTRIBUTES['low-level']['var_mrms'], name='Rotation Tracks', date_dir=str(date), valid_datetime=valid_date_and_time )
    
    torn_warn_xy = load_tornado_warning(str(date), initial_date_and_time, time_window=15)
    svr_warn_xy = load_svr_warning(str(date), initial_date_and_time, time_window=15)
    hail_xy, torn_xy, wind_xy = load_reports(str(date), valid_date_and_time, time_window=15, forecast_length=forecast_length)

    severe_wx_obs = {'tornado_warn_ploys': torn_warn_xy,
                     'severe_wx_warn_polys': svr_warn_xy,
                     'severe_hail': hail_xy,
                     'severe_wind': wind_xy,
                     'tornado': torn_xy} 
 
    return severe_wx_obs


def _load_forecast_data( date_and_time_dict, **kwargs ):
    '''
    '''
    instance = EnsembleData( date_dir=date_and_time_dict['date'], 
                             time_dir=date_and_time_dict['time'], 
                             base_path = 'summary_files')
    input_data = instance.load( 
                                variables=[config.VARIABLE_ATTRIBUTES[variable_key]['var_newse'] ], 
                                time_indexs=kwargs['time_indexs'], 
                                tag=get_key( config.newse_tag, 
                                             config.VARIABLE_ATTRIBUTES[variable_key]['var_newse'] ) 
                              )
    
    ensemble_data_over_time = input_data[config.VARIABLE_ATTRIBUTES[variable_key]['var_newse']].values 
    ens_data, ens_data_indices = calc_time_max( 
                                                input_data=ensemble_data_over_time, 
                                                argmax=True
                                              )

    return ens_data, ens_data_indices


def _identify_and_match_tracks( data, time_argmax, verification_dict ):
    '''
    Identify objects and then match them
    '''
    fcst_object_labels, fcst_object_props = label_regions( input_data = data, 
                                                           method = 'single_threshold', 
                                                           params = {'bdry_thresh':10.}
                                                         )
    QC_PARAMS['min_time'].append( time_argmax ) 
    qc_fcst_object_labels, qc_fcst_object_props = qc.quality_control( input_data = data, 
                                                                      object_labels = fcst_object_labels, 
                                                                      object_properties=fcst_object_props, 
                                                                      qc_params=QC_PARAMS )
    
    matched_at_30km = { 'matched_to_{}_30km'.format(atype):match_to_lsrs( object_properties=qc_fcst_object_props, 
                                                                          lsr_points=verification_dict[atype], dist_to_lsr=10 ) for atype in verification_dict.keys() }
    matched_at_15km = { 'matched_to_{}_15km'.format(atype):match_to_lsrs( object_properties=qc_fcst_object_props, 
                                                                          lsr_points=verification_dict[atype], dist_to_lsr=5 ) for atype in verification_dict.keys() }

    matched_at_30km = is_severe(matched_at_30km, '30km')
    matched_at_15km = is_severe(matched_at_15km, '15km')
    all_matched = {**matched_at_30km, **matched_at_15km}

    if 'az_shr_tracks' in verification_dict.keys():
        _,forecast_labels_matched_to_azshr,_ = obj_match.match_objects( obs_rotation_tracks, qc_fcst_object_labels )
        forecast_labels_matched_to_azshr  = {region.label: 1.0 if region.label in forecast_labels_matched_to_azshr else 0.0 for region in qc_fcst_object_props}
        
        all_matched['matched_to_azshr_30km'] = forecast_labels_matched_to_azshr 

    return qc_fcst_object_labels, qc_fcst_object_props, all_matched

def _save_netcdf( data, date_and_time_dict, **kwargs ):
    '''
    '''
    debug = kwargs.get('debug')

    out_path = join( config.OBJECT_SAVE_PATH, date_and_time_dict['date'] )
    ds = xr.Dataset( data )
    fname = join(out_path, 'WOFS_%s_OBJECTS_%s-%s_%02d.nc' % ( config.VARIABLE_ATTRIBUTES[variable_key]['title'], 
                                                               date_and_time_dict['date'], 
                                                               date_and_time_dict['time'], 
                                                               date_and_time_dict['fcst_time_idx']))
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    print ('Writing {}...'.format(fname))
    if debug:
        ds.to_netcdf( 'WOFS_%s_OBJECTS_%s-%s_%02d.nc' % (config.VARIABLE_ATTRIBUTES[variable_key]['title'], 
                                                         date_and_time_dict['date'], 
                                                         date_and_time_dict['time'], 
                                                         date_and_time_dict['fcst_time_idx']) )
    else:
        ds.to_netcdf( fname )
        ds.close( )
        del data

def function_for_multiprocessing( date, time, kwargs):
    print ('\t {} {}'.format(date, time))
    load_az_shr = False
    date_and_time_dict = {'date':date, 'time':time, 'fcst_time_idx': kwargs['fcst_time_idx']}
    valid_date_and_time, initial_date_and_time = \
                                get_time.determine_forecast_valid_datetime( date_dir=str(date), 
                                                                            time_dir=time, 
                                                                            fcst_time_idx=date_and_time_dict['fcst_time_idx'] )
    verification_dict = _load_verification( date, valid_date_and_time, initial_date_and_time )
    if load_az_shr:
        obs_rotation_tracks, az_shear = mrms.load_az_shr_tracks( var=config.VARIABLE_ATTRIBUTES['low-level']['var_mrms'], 
                                                                 name='Rotation Tracks', 
                                                                 date_dir=str(date), 
                                                                 valid_datetime=valid_date_and_time ) 
        verification_dict['az_shr_tracks'] = obs_rotation_tracks


    ens_data, ens_data_indices =  _load_forecast_data( date_and_time_dict, **kwargs )
    ens_object_labels = np.zeros((ens_data.shape ))
   
    object_properties_for_all_objects = [ ]
    for mem in range( np.shape( ens_data )[0] ):
        print(mem) 
        qc_forecast_object_labels, qc_forecast_object_props, all_matched = _identify_and_match_tracks( data=ens_data[mem,:,:], 
                                                                             time_argmax = ens_data_indices[mem,:,:],
                                                                             verification_dict =verification_dict ) 
        ens_object_labels[mem,:,:] = qc_forecast_object_labels
        object_properties_for_all_objects.append( get_object_properties( ens_mem_idx = mem, 
                                                                         object_props = qc_forecast_object_props, 
                                                                         matched = all_matched ) )
    
    df = pd.concat( object_properties_for_all_objects ) 

    data = { }
    data['Objects'] = (['Ensemble Member', 'y', 'x'], ens_object_labels)
    data['Raw Data'] = (['Ensemble Member', 'y', 'x'], ens_data)
   
    for object_property in df.columns:
        data[object_property] = (['Object'], df[object_property].values)

    del ens_object_labels, ens_data, ens_data_indices

    _save_netcdf( data, date_and_time_dict, **kwargs )
