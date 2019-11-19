import sys, os
sys.path.append('/home/monte.flora/wofs/data')
sys.path.append('/home/monte.flora/wofs/util')

from loadEnsembleData import load_mesocyclone_tracks
from loadMRMSData import MRMSData
from ObjectMatching import ObjectMatching
from basic_functions import get_key, personal_datetime, check_file_path
import numpy as np 

date = '20180501'
times = ['2300', '2330', '0000']
fcst_time_idx = 0 
var_name = 'Objects (QC3)'
mrms = MRMSData( )
get_time = personal_datetime()
variable_key = 'low-level'

valid_datetimes = [get_time.determine_forecast_valid_datetime( date_dir=str(date), time_dir=time, fcst_time_idx=fcst_time_idx )[0] for time in times]
object_set_a = [ mrms.load_az_shr_tracks( var='LOW_CRESSMAN', name='Rotation Objects (QC9)', date_dir=str(date), valid_datetime=valid_datetime)[0] for valid_datetime in valid_datetimes ]
object_set_b = [ load_mesocyclone_tracks(variable_key, var_name, date, time, fcst_time_idx )[1,:,:] for time in times ] 

obj_match = ObjectMatching( dist_max = 15., time_max = 30., one_to_one=True )

times_a = [ valid_datetime[0] + ' ' + valid_datetime[1] for valid_datetime in valid_datetimes ] 
times_b = [ valid_datetime[0] + ' ' + valid_datetime[1] for valid_datetime in valid_datetimes ]

matched_pairs, _ = obj_match.match_objects( object_set_a, object_set_b, times_a, times_b ) 

print(matched_pairs) 


