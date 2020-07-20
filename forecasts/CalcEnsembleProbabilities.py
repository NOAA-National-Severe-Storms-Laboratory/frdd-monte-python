import numpy as np
import xarray as xr
from os.path import join, exists
import os
import pandas as pd

from wofs.util.basic_functions import get_key, personal_datetime, check_file_path, to_ordered_dict
from wofs.data.loadEnsembleData import EnsembleData
from wofs.util import config
from datetime import datetime
from wofs.processing.ObjectIdentification import label_ensemble_objects, QualityControl

qc = QualityControl( )
def _load_forecast_data( date_and_time_dict, **kwargs ):
    '''
    '''
    instance = EnsembleData( date_dir=date_and_time_dict['date'],
                             time_dir=date_and_time_dict['time'],
                             base_path = 'summary_files')
    input_data = instance.load(
                                variables=['w_up'],
                                time_indexs=kwargs['time_indexs'],
                                tag=get_key( config.newse_tag,
                                             'w_up' )
                              )

    ensemble_data_over_time = input_data['w_up'].values
    time_max_ensemble_data = np.amax(ensemble_data_over_time, axis = 0)    

    return time_max_ensemble_data

def calc_ensemble_probs( data, thresh ):

    binary_data = np.where(data>thresh,True,False)
    ensemble_probabilities = np.mean( binary_data, axis=0)

    return ensemble_probabilities

# WOFS_UPDRAFT_SWATH_OBJECTS_20180501-2300_06.nc
date = '20180501'
time = '2300'
fcst_time_idx = 6 

kwargs = {'time_indexs': np.arange( config.N_TIME_IDX_FOR_HR+1 ) + fcst_time_idx}
date_and_time_dict = {'date': date, 'time': time}

data = _load_forecast_data( date_and_time_dict, **kwargs)

ds = xr.open_dataset('WOFS_UPDRAFT_SWATH_OBJECTS_20180501-2300_06.nc')
objects = ds['Objects'].values
binary = np.where(objects>0, True, False)
ensemble_probabilities = np.mean(binary, axis=0)
full_objects, full_object_props  = label_ensemble_objects(ensemble_probabilities=ensemble_probabilities)


data = { }
data['Probabilities'] = (['y', 'x'], ensemble_probabilities)
data['Objects'] = (['y', 'x'], full_objects)
ds = xr.Dataset(data)
ds.to_netcdf('test.nc')



