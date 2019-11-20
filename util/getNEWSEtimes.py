import sys 
sys.path.append( '/home/monte.flora/NEWSeProbs/misc_python_scripts' ) 
import os
from basic_functions import load_date_list 

dates    = load_date_list( )

def get_times( date, opt ):  
	path     = '/oldscratch/skinnerp/2018_newse_post/summary_files/'
	times_for_each_date = [ ]
	basePath = os.path.join( path, date)
	times    = os.listdir(basePath)
	if opt == 'verify':
	# remove off-hour forecasts
		if '2017' in str(date):  
			times = [ time for time in times if int(time[2:]) != 30 ]
		times_for_each_date.append( times )
	elif opt == 'mach_learn': 
		times_for_each_date.append( times ) 

	return times_for_each_date

def combine_date_and_time( dates, opt ): 
	datetimes = [ ] 
	for date in dates: 
		times = get_times( date, opt)[0] 
		for time in times: 
			datetimes.append( date + time )  

	return datetimes 

def datetime_dict( dates, opt ): 
	datetimes = { } 
	for date in dates: 
		datetimes[date] = get_times(date, opt)[0] 

	return datetimes 


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]
    
