import numpy as np
import xarray as xr 
# Personal Imports 
import sys, os
sys.path.append('/home/monte.flora/wofs/util') 
sys.path.append('/home/monte.flora/wofs/processing')
from os.path import join, exists
from basic_functions import check_file_path
import config
from MultiProcessing import multiprocessing_per_date
from EnsembleProbability import EnsembleProbability
from ObjectIdentification import ObjectIdentification, QualityControl
from scipy.ndimage import gaussian_filter

###########################################################################
# Label forecast storm objects valid at a single time 
###########################################################################
debug = False
variable_key = 'low-level'
ens_probs = EnsembleProbability( ) 
obj_id = ObjectIdentification( )
qc = QualityControl( )
num_qc = 4
watershed_params_fcst = {'min_thresh': 2,
                         'max_thresh': 75  ,
                         'data_increment': 10,
                         'delta': 100,
                         'size_threshold_pixels': 200,
                         'local_max_area': 50 }

qc_params = {'min_area':12.}
max_nghbrd = 13

def function_for_multiprocessing( date, time, kwargs):
    in_path = join( config.OBJECT_SAVE_PATH, date )
    fname = join( in_path, 'WOFS_%s_OBJECTS_%s-%s_%02d.nc' % (config.variable_attrs[variable_key]['title'], date, time, kwargs['fcst_time_idx']))
    ds = xr.open_dataset( fname ) 
    forecast_objects = [ ds['Objects (QC%s)'%(i+1) ].values for i in range(num_qc) ]
    data = { }
    for i, forecast_objects in enumerate(forecast_objects): 
        fcst_probabilities = ens_probs.calc_ensemble_probability( ensemble_data = forecast_objects, intensity_threshold=1, max_nghbrd=max_nghbrd )
        processed_data = np.round(100.*fcst_probabilities,0)

        labels_fcst, props_fcst = obj_id.label( processed_data, method='watershed', **watershed_params_fcst)
        qc_object_labels, qc_object_props = qc.quality_control(object_labels=labels_fcst, object_properties=props_fcst, input_data = fcst_probabilities , qc_params=qc_params ) 
        
        data['Ensemble Probability (QC%s)'%(i+1)] = (['y', 'x'], fcst_probabilities)
        data['Probability Objects (QC%s)'%(i+1)] = (['y', 'x'], qc_object_labels)    
    ds = xr.Dataset( data )
    if debug:
        ds.to_netcdf( 'WOFS_%s_PROBS_%s-%s_%02d.nc' % (config.variable_attrs[variable_key]['title'], date, time, kwargs['fcst_time_idx']) )
    else:
        out_path = join( config.WOFS_PROBS_PATH, date )
        check_file_path( out_path )
        ds.to_netcdf(path = join(out_path, 'WOFS_%s_PROBS_%s-%s_%02d_max_nghbrd=%s.nc' % (config.variable_attrs[variable_key]['title'], date, time, kwargs['fcst_time_idx'], max_nghbrd)) )

    ds.close( )
    del data

if debug: 
    print("\n Working in DEBUG MODE...\n") 
    date = '20180501'    
    time = '2300'
    fcst_time_idx = 6 
    kwargs = {'fcst_time_idx':fcst_time_idx}
    function_for_multiprocessing( date, time, kwargs )
else:
    datetimes  = config.datetimes_ml
    #aset = [0]
    #for fcst_time_idx in aset:
    for fcst_time_idx in config.fcst_time_idx_set:
        print("Forecast Time Index: ", fcst_time_idx)
        kwargs = {'fcst_time_idx':fcst_time_idx} 
        multiprocessing_per_date( datetimes=datetimes, n_date_per_chunk=4, func=function_for_multiprocessing, kwargs=kwargs)
