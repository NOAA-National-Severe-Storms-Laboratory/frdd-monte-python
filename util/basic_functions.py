import datetime
import pickle
import os
import numpy as np
from collections import OrderedDict

class personal_datetime:
    def convert_datetime_obj_to_components( self, datetime_obj ):
        '''
        Converts the output of datetime to separate components (year, month, day)
        '''
        datetime_str = str( datetime_obj )
        reduced_datetime_str = datetime_str.replace(":", "").replace(" ", "").replace('-', '')
        date = reduced_datetime_str[:8]
        time = reduced_datetime_str[8:12]

        return date, time

    def initial_datetime( self, date_dir, time_dir ):
        year   = int( date_dir[:4]  )
        month  = int( date_dir[4:6] )
        day    = int( date_dir[6:]  )
        hour   = int( time_dir[:2]  )
        minute = int( time_dir[2:]  )
        # The way NEWS-e and MRMS files are structured by Pat, 
        # for next day times (i.e., < 18 ), the date_dir is a
        # day behind the initialization date
        d = 0
        if hour < 18:
            d = 1

        datetime_obj  = datetime.datetime( year, month, day, hour, minute ) + datetime.timedelta( days = d )
        initial_date, initial_time  = self.convert_datetime_obj_to_components( datetime_obj )

        return datetime_obj, initial_date, initial_time

    def determine_forecast_valid_datetime( self, date_dir, time_dir, fcst_time_idx, return_obj=False ):
        initial_datetime_obj, initial_date, initial_time  = self.initial_datetime( date_dir, time_dir )
        valid_datetime_obj = initial_datetime_obj + datetime.timedelta( minutes = 5 * fcst_time_idx )
        valid_forecast_date, valid_forecast_time  = self.convert_datetime_obj_to_components( valid_datetime_obj )

        if return_obj: 
            return valid_datetime_obj, valid_forecast_date, valid_forecast_time, adjusted_date, initial_time 
        else:
            return (valid_forecast_date, valid_forecast_time), (initial_date, initial_time)    

class get_newse_datetimes: 
    def __init__( self, option, summary_files_dir = '/oldscratch/skinnerp/2018_newse_post/summary_files/' ): 
        self.option = option 
        self.summary_files_dir = summary_files_dir 

    def get_times( self, date ):
        times_for_each_date = [ ]
        basePath = os.path.join( self.summary_files_dir, str(date) ) 
        times    = os.listdir(basePath)
        if self.option == 'verify':
        # remove off-hour forecasts for 2017 since they are only 90 mins
            if '2017' in str(date):
                times = [ time for time in times if int(time[2:]) != 30 ]
                times_for_each_date.append( times )
        elif self.option == 'mach_learn':
            times_for_each_date.append( times )

        return times_for_each_date

    def datetime_dict( self, list_of_dates ):
            datetimes = { }
            for date in list_of_dates:
                    datetimes[date] = self.get_times( date )[0]
            return datetimes

def to_ordered_dict( list_of_key_item_tuples ):
    ordered_dict = OrderedDict( )
    for items in list_of_key_item_tuples:
        ordered_dict[items[0]] = items[1]
    
    return ordered_dict 

def check_file_path( path_to_dir): 
    '''
    Checks if file path exists and if not creates it.
    '''
    if not os.path.exists( path_to_dir ):
        try:
            os.mkdir( path_to_dir )
        except OSError:
            print(('{} already exists!'.format( path_to_dir )))

def convert_to_seconds( time ):
    '''
    Converts time in hour and minutes to total number of seconds 
    '''
    hour = int(time[:2])
    minute = int(time[2:])
    time_in_seconds = hour * 3600 + minute * 60

    return time_in_seconds

def run_script(cmd, cmd2=0):
    print(("Executing command:  " + cmd))
    if cmd2 == 0:
        os.system(cmd)
    else:
        os.system( cmd & cmd2 )
    #print cmd + "  is finished...."
    return

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def get_key(adict, val):
    for key, value in list(adict.items()):
         if val in value:
             return key

def fill_object_with_single_value( objects, input_data ):
    prob_objects = np.zeros(( objects.shape ))
    for label in np.unique(objects)[1:]:
        print(label)
        prob_objects[objects == label] = np.amax( input_data[objects==label] )
    return prob_objects

def load_date_list(   ):
        with open("/home/monte.flora/NEWSeProbs/misc_python_scripts/newse_code/newse_dates.txt", "rb") as fp:   # Unpickling
                dates = pickle.load(fp)
        return dates

def extract_training_subset( predictors, outcomes, date_list, time_list, mem_list, obj_label_list, date_subset ):
        dateIndices = [i for i, x in enumerate(date_list.astype(str)) if str(x) in date_subset]

        predictor_subset = predictors[ dateIndices, : ]
        outcome_subset   = outcomes[ dateIndices ]

        date_list_subset      = date_list[dateIndices]
        time_list_subset      = time_list[dateIndices]
        obj_label_subset      = obj_label_list[dateIndices]
        mem_list_subset       = mem_list[dateIndices]

        return predictor_subset , outcome_subset, date_list_subset, time_list_subset, obj_label_subset, mem_list_subset
