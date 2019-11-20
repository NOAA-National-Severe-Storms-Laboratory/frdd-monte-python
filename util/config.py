import sys , os
from os.path import join
from basic_functions import load_date_list, get_newse_datetimes
import netCDF4
import numpy as np 
import xarray as xr

#############################
# NEWS-E VERIFICATION CONFIG

SUMMARY_FILES_PATH = '/oldscratch/skinnerp/2018_newse_post/summary_files' #'/work/skinnerp/summary_files_official/'
WOFS_PROBS_PATH = '/oldscratch/mflora/WOFS_PROB_FCSTS/'
MRMS_TRACKS_PATH = '/oldscratch/mflora/MRMS_ROTATION_TRACKS/'
MRMS_DBZ_PATH = '/oldscratch/mflora/MRMS_DBZ/'
WOFS_RESULTS_PATH = '/oldscratch/mflora/WOFS_VERIFICATION/'
WOFS_DATA_PATH = '/scratch/mflora/WOFS_DATA/'
MRMS_PATH = '/oldscratch/cpotvin/MRMS' #'/work/skinnerp/mrms_verif_files'

BASE_PATH_ML = '/scratch/mflora/ML_DATA'
OBJECT_SAVE_PATH = join( BASE_PATH_ML, 'MODEL_OBJECTS') 
LSR_SAVE_PATH = join( BASE_PATH_ML, 'GRIDDED_LSRS')
ML_MODEL_SAVE_PATH = join( BASE_PATH_ML, 'MODEL_SAVES')
ML_FCST_PATH = join( BASE_PATH_ML, 'FORECASTS')
ML_INPUT_PATH = join( BASE_PATH_ML, 'INPUT_DATA')
ML_RESULTS_PATH = join( BASE_PATH_ML, 'RESULTS')
ML_DATA_STORAGE_PATH = join( BASE_PATH_ML, 'DATA')

N_ENS_MEM  = 18
N_TIME_IDX_FOR_HR = 12
DIST_FROM_EDGE = 6
fcst_time_idx_set = [ 0, 6, 12, 18, 24 ]

VARIABLE_ATTRIBUTES = {
'low-level': { 'var_mrms': 'LOW_CRESSMAN' , 
               'var_newse' : 'uh_0to2', 
               'mrms_bdry' : 0.0038, 
               'newse_bdry' : 14., 
               'name': 
               'Low-level Updraft Helicity', 
               'title': 'LOW-MESO',  
               'qc_params': [ ('min_area', 10.), ('merge_thresh', 4.), ('min_time', [2]), ('max_thresh', 20)], 
               'watershed_params': {'min_thresh': 10, 'max_thresh': 250, 'data_increment': 5, 'delta': 80, 'size_threshold_pixels': 200, 'local_max_area':100 }},

'mid-level': { 'var_mrms': 'MID_CRESSMAN', 
               'var_newse' : 'uh_2to5', 
               'mrms_bdry' : 0.0041, 
               'newse_bdry' : 65., 
               'name': 'Mid-level Updraft Helicity', 
               'title' : 'MID-MESO',
               'qc_params': [ ('min_area', 10.), ('merge_thresh', 4.), ('min_time', [2]), ('max_thresh', 80)], 
               'watershed_params': {'min_thresh': 40, 'max_thresh': 250, 'data_increment': 5, 'delta': 80, 'size_threshold_pixels': 200, 'local_max_area':100 }},

'updraft':  {  'var_mrms': None, 
               'var_newse' : 'w_up', 
               'mrms_bdry' : None, 
               'newse_bdry' : 10., 
               'name': 'Column-max Updraft Swath', 
               'title' : 'UPDRAFT_SWATH',
               'qc_params': [ ('min_area', 10.), ('merge_thresh', 1.), ('min_time', [2])] , 
               'watershed_params': {'min_thresh': 12, 'max_thresh': 80, 'data_increment': 1, 'delta': 100, 'size_threshold_pixels': 250, 'local_max_area':75 }},

'dbz'     : { 'var_mrms': 'DZ_CRESSMAN' , 'var_newse' : 'comp_dz', 'mrms_bdry' : 41.121, 'newse_bdry' : 45., 'name': 'Reflectivity' , 'title':'STORM'},
} 

qc_params_dbz = { 'min_area': 10*9., 'merge_thresh': 6. }
qc_params_obs = { 'min_area': 5*9., 'merge_thresh': 12. , 'min_time' : 2 }
qc_params_uh  = { 'min_area': 12.*9, 'merge_thresh': 12., 'min_time' : 2 }
qc_params_w = { 'min_area': 10.*9, 'merge_thresh': 3. }

observation_times = ['1800', '1830', '1900', '1930', '2000', '2030', '2100', '2130', '2200', '2230', '2300', '2330', '0000', '0030', '0100', '0130',
                    '0200', '0230', '0300', '0330', '0400', '0430', '0500', '0530', '0600']

data = xr.open_dataset('/home/monte.flora/wofs/util/wofs_dates_for_verification.nc')
verify_forecast_dates = data['Dates'].values 

verify_forecast_dates = np.array([int(date) for date in os.listdir('/scratch/mflora/ML_DATA/INPUT_DATA')]) 

verification_times =  [ '1900', '2000', '2100', '2200', '2300', '0000', '0100', '0200', '0300' ]
def datetime_dict_verify( list_of_dates, list_of_times ):
    datetimes = { }
    for date in list_of_dates:
        datetimes[str(date)] = list_of_times
    return datetimes

verification_datetimes = datetime_dict_verify( verify_forecast_dates, verification_times )
obs_datetimes = datetime_dict_verify( verify_forecast_dates, observation_times )

f_ens = netCDF4.Dataset( SUMMARY_FILES_PATH+'/20170501/0000/news-e_ENS_00_20170502_0000_0000.nc', 'r' ) 
f_env = netCDF4.Dataset( SUMMARY_FILES_PATH+'/20170501/0000/news-e_ENV_00_20170502_0000_0000.nc', 'r') 
newse_tag = {'ENS': list(f_ens.variables.keys( )) , 'ENV' : list(f_env.variables.keys( ))} 

############################################################################
# MACHINE LEARNING CONFIG 
############################################################################
instance = get_newse_datetimes(option='mach_learn')
datetimes_ml = instance.datetime_dict( verify_forecast_dates )
qc_params_ml = { 'min_area': 12*9., 'merge_thresh': 6. }


