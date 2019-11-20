import config 

import sys
sys.path.append('/oldhome/monte.flora/NEWSeProbs/misc_python_scripts')
from basic_functions import determine_valid_datetime
datetimes   = config.verification_datetimes

fcst_time_idx = 18 

for date in datetimes.keys()[:1]:
	for time in datetimes[date]: 
		inst  = determine_valid_datetime( date, time, fcst_time_idx)
                valid_forecast_date, valid_forecast_time, initial_date, initial_time = inst.valid_dateTime( return_obj=False )

		print "Date dir: ", date, "| Time dir: ", time, "| Initial Date: ", initial_date, "| Initial Time: ", initial_time, "| Valid Date: ", valid_forecast_date, "| Valid time:  ", valid_forecast_time 
		print "\n"


