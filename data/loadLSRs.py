import wrf 
import pandas as pd 
from datetime import datetime, timedelta
import numpy as np 
import sys
sys.path.append('/home/monte.flora/wofs/data')
from loadWRFGrid import WRFData

class loadLSR: 
    def __init__(self, date_dir, date, time, time_window = 15, forecast_length = 60,  
                    dtype = {'VALID': np.int64, 'LAT':np.float64, 'LON':np.float64, 'MAG':np.float64, 'TYPETEXT':object},
                    cols = ['VALID', 'LAT', 'LON', 'MAG', 'TYPETEXT'],
                    fname = '/home/monte.flora/lsr_201704010000_201908010000.csv'):
        self.date = str(date)
        self.time = time
        self.df = pd.read_csv( fname, usecols=cols, dtype=dtype, na_values = 'None' )
        self.time_window = time_window
        self.forecast_length = forecast_length
        self._get_time_window( )     
        wrf_data = WRFData( date=date_dir, time='2000', time_indexs=[0] )
        self.wrfin = wrf_data._generate_filename_list( mem_idx=1 )

    def _to_datetime(self):
        '''
        convert strings of date and time into a full string of date and time 
        '''
        year = int(self.date[:4])
        month = int(self.date[4:6])
        day = int(self.date[6:])
        hour = int(self.time[:2])
        minute = int(self.time[2:])
        return datetime( year = year, month=month, day=day, hour=hour, minute=minute ) 
        
    def _get_time_window(self): 
        '''
        Get beginning and ending of the time window to search for LSRs
        '''
        initial_datetime = self._to_datetime( ) 
        begin_time = str( initial_datetime - timedelta( minutes = self.time_window ))[:-2]
        end_time = str( initial_datetime + timedelta( minutes = self.forecast_length+self.time_window ))[:-2]
   
        begin_time = int(begin_time.replace(':', '').replace('-','').replace(' ','')) 
        end_time = int(end_time.replace(':', '').replace('-','').replace(' ',''))  

        self.begin_time = begin_time
        self.end_time = end_time 

        return self

    def load_hail_reports(self, magnitude=1.0): 
        '''
        Load the Hail LSRs.
        '''
        severe_hail_reports =  self.df.loc[ (self.df['MAG'] >= magnitude) & (self.df['TYPETEXT'] == 'HAIL') & (self.df['VALID'] >= self.begin_time) & (self.df['VALID'] <= self.end_time) ]
        return ( severe_hail_reports['LAT'].values, severe_hail_reports['LON'].values)

    def load_tornado_reports(self):
        '''
        Load the tornado reports.
        '''
        tornado_reports = self.df.loc[ (self.df['TYPETEXT'] == 'TORNADO') & (self.df['VALID'] >= self.begin_time) & (self.df['VALID'] <= self.end_time) ]    
        return (tornado_reports['LAT'].values, tornado_reports['LON'].values)       
   
    def load_wind_reports(self):
        '''
        Load the wind reports.
        '''
        wind_reports = self.df.loc[ (self.df['TYPETEXT'] == 'TSTM WND DMG') & (self.df['VALID'] >= self.begin_time) & (self.df['VALID'] <= self.end_time) ]
        return (wind_reports['LAT'].values, wind_reports['LON'].values) 

    def to_xy(self, lats, lons):
        '''
        Converts lats and lons to x,y on the WRF grid 
        '''
        xy = wrf.ll_to_xy( wrfin = self.wrfin, latitude =lats, longitude = lons, meta = False )
        return xy 

'''
def remove_objects_unmatched_to_lsrs( input_data, object_labels, object_properties, **qc_params):
    date = qc_params['lsr']['date']
    time = qc_params['lsr']['time']
    lsr_dist = round(qc_params['lsr']['lsr_dist'],10)
    lsr_points = qc_params['lsr']['lsr_points'] 
    wrf_data = WRFData( date=date, time=time, time_indexs=[0] ) 
    wrfin = wrf_data._generate_filename_list( mem_idx=1 )
    xy =  wrf.ll_to_xy( wrfin = wrfin, latitude =lsr_points[0] , longitude = lsr_points[1], meta = False )
    points = list(zip(xy[1,:],xy[0,:]))
    qc_object_labels = np.zeros( object_labels.shape, dtype=int)
    j=1
    for region in object_properties:
        kdtree = spatial.cKDTree( region.coords )
        dist_btw_region_and_lsr, _ = kdtree.query( points )   
        if round( np.amin( dist_btw_region_and_lsr ), 10) < lsr_dist:
            qc_object_labels[np.where(object_labels == region.label)] = j
            j+=1
   
    qc_object_properties = regionprops( qc_object_labels, input_data, coordinates='rc' )
    return qc_object_labels, qc_object_properties 

def load_lsr_data( date, time, fname = '/home/monte.flora/lsr_201704010000_201908010000.csv'):
    dtype = {'VALID': np.int64, 'LAT':np.float64, 'LON':np.float64, 'MAG':np.float64, 'TYPETEXT':object}
    df = pd.read_csv( fname, usecols=['VALID', 'LAT', 'LON', 'MAG', 'TYPETEXT'], dtype=dtype, na_values = 'None' )
    year = int(date[:4])
    month = int(date[4:6])
    day = int(date[6:])
    hour = int(time[:2])
    minute = int(time[2:]) 
    x = datetime( year = year, month=month, day=day, hour=hour, minute=minute )
    begin_time = str( x - timedelta( minutes = 15 ))[:-2]
    end_time = str( x + timedelta( minutes = 75 ))[:-2]

    begin_time = int(begin_time.replace(':', '').replace('-','').replace(' ','')) 
    end_time = int(end_time.replace(':', '').replace('-','').replace(' ',''))

    print begin_time, end_time 
    severe_hail_reports = df.loc[ (df['MAG'] >= 1.0) & (df['TYPETEXT'] == 'HAIL') & (df['VALID'] >= begin_time) & (df['VALID'] <= end_time) ]
    tornado_reports = df.loc[(df['TYPETEXT'] == 'TORNADO') & (df['VALID'] >= begin_time) & (df['VALID'] <= end_time)]
    wind_reports = df.loc[ (df['TYPETEXT'] == 'TSTM WND DMG') & (df['VALID'] >= begin_time) & (df['VALID'] <= end_time)]

    lsr_points = {'hail': {'lons': severe_hail_reports['LON'].values, 'lats':severe_hail_reports['LAT'].values}, 
                   'tornado': {'lons': tornado_reports['LON'].values, 'lats':tornado_reports['LAT'].values},
                   'wind': {'lons':wind_reports['LON'].values, 'lats':wind_reports['LAT'].values}}

    lsr_lats = np.concatenate(( severe_hail_reports['LAT'].values, tornado_reports['LAT'].values, wind_reports['LAT'].values ))
    lsr_lons = np.concatenate(( severe_hail_reports['LON'].values, tornado_reports['LON'].values, wind_reports['LON'].values ))

    lsr_pos = ( lsr_lats, lsr_lons )

    return lsr_points, lsr_pos  

date = qc_params['lsr']['date']
        time = qc_params['lsr']['time']
        lsr_dist = round(qc_params['lsr']['lsr_dist'],10)
        lsr_points = qc_params['lsr']['lsr_points']
        wrf_data = WRFData( date=date, time=time, time_indexs=[0] )
        wrfin = wrf_data._generate_filename_list( mem_idx=1 )
        xy =  wrf.ll_to_xy( wrfin = wrfin, latitude =lsr_points[0] , longitude = lsr_points[1], meta = False )
        points = list(zip(xy[1,:],xy[0,:]))

'''
