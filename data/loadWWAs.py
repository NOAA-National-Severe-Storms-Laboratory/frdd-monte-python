import wrf 
import pandas as pd 
from datetime import datetime, timedelta
import numpy as np 
import sys
sys.path.append('/home/monte.flora/wofs/data')
from loadWRFGrid import WRFData
from loadLSRs import loadLSR
import shapefile
from os.path import join

class loadWWA(loadLSR): 
    def __init__(self, date_dir, date, time, time_window = 15, forecast_length = 60):  
        self.date = str(date)
        self.time = time
        self._to_pandas( )
        self.time_window = time_window
        self.forecast_length = forecast_length
        self._get_time_window( )     
        #wrf_data = WRFData( date=date_dir, time='2000', time_indexs=[0] )
        #self.wrfin = wrf_data._generate_filename_list( mem_idx=1 )

    def _get_fname(self): 
        '''
        '''
        base_path = '/home/monte.flora/wwa_shapefiles'
        fname_dict = {
                      '2017': join(base_path, 'wwa_201701010000_201712312359'), 
                      '2018': join(base_path, 'wwa_201801010000_201812312359') 
                     } 
        return fname_dict[self.date[:4]] 

    def _to_pandas(self):
        '''
        load shapefile into a pandas dataframe.
        '''
        shp_file = shapefile.Reader(self._get_fname() )
        fields = [x[0] for x in shp_file.fields][1:]
        records = [y[:] for y in shp_file.records()]
        shps = [s.points for s in shp_file.shapes()]

        #write the records into a dataframe
        shapefile_dataframe = pd.DataFrame(columns=fields, data=records)

        #add the coordinate data to a column called "coords"
        shapefile_dataframe = shapefile_dataframe.assign(coords=shps)
        shapefile_dataframe.astype( dtype = {'ISSUED': np.int64, 'EXPIRED': np.int64 } )  
        self.df = shapefile_dataframe 

    def _to_coordinates(self, polygons):
        '''
        Convert list of polygons to separate lists of lats and lons 
        converted to x and y in model grid.
        Args:
            polygons, list of lat/lon coordinate pair defining boundary of the warning polygon
        '''
        lons = [ ]
        lats = [ ]
        for coords in polygons:
            lons.extend( [ pair[0] for pair in coords ] )
            lats.extend( [ pair[1] for pair in coords ] )

        return (lats, lons)

    def load_tornado_warning_polygon(self, return_polygons=False): 
        '''
        Loads the tornado warning polygons.
        '''    
        tornado_warning_polygons = self.df.loc[ (self.df['PHENOM'] == 'TO') & (self.df['ISSUED'].astype(int) >= self.begin_time) & (self.df['EXPIRED'].astype(int) <= self.end_time) ]
        if return_polygons:
            return tornado_warning_polygons['coords']
        else:
            return self._to_coordinates( polygons = tornado_warning_polygons['coords'].values ) 

    def load_svr_tstorm_warning_polygon(self):
        '''
        Loads the tornado warning polygons.
        '''    
        svr_tstorm_polys = self.df.loc[ (self.df['PHENOM'] == 'SV') & (self.df['ISSUED'].astype(int) >= self.begin_time) & (self.df['EXPIRED'].astype(int) <= self.end_time) ]
        return self._to_coordinates( polygons = svr_tstorm_polys['coords'].values )

    def load_flash_flood_polygon(self):
        '''
        Loads the tornado warning polygons.
        '''    
        flash_flood_polys = self.df.loc[ (self.df['PHENOM'] == 'FF') & (self.df['ISSUED'].astype(int) >= self.begin_time) & (self.df['EXPIRED'].astype(int) <= self.end_time) ]
        return self._to_coordinates( polygons = flash_flood_polys['coords'].values )

def load_reports(date, initial_date_and_time, time_window=15):
    '''
    Load Local storm reports
    '''
    load_lsr = loadLSR(date_dir=date, date=initial_date_and_time[0], time=initial_date_and_time[1], time_window = time_window)
    load_wwa = loadWWA(date_dir=date, date=initial_date_and_time[0], time=initial_date_and_time[1], time_window = time_window)
    hail_ll = load_lsr.load_hail_reports( )
    torn_ll = load_lsr.load_tornado_reports( )
    wind_ll = load_lsr.load_wind_reports( )
    torn_wwa_ll = load_wwa.load_tornado_warning_polygon( )
    lsr_lons = np.concatenate((hail_ll[1], torn_ll[1], wind_ll[1], torn_wwa_ll[1]))
    lsr_lats = np.concatenate((hail_ll[0], torn_ll[0], wind_ll[0], torn_wwa_ll[0]))
    lsr_xy = load_lsr.to_xy( lats=lsr_lats, lons=lsr_lons)
    lsr_xy = list(zip(lsr_xy[1,:],lsr_xy[0,:]))

    return lsr_xy

def load_tornado_warning(date, initial_date_and_time, time_window=15):
    '''
    Load tornado warning polygons within +/- 15 minutes of the initialization time. 
    '''
    load_lsr = loadLSR(date_dir=date, date=initial_date_and_time[0], time=initial_date_and_time[1], time_window = time_window, forecast_length=0)
    load_wwa = loadWWA(date_dir=date, date=initial_date_and_time[0], time=initial_date_and_time[1], time_window = time_window, forecast_length=0)
    
    torn_wwa_ll = load_wwa.load_tornado_warning_polygon( )
    lsr_xy = load_lsr.to_xy( lats=torn_wwa_ll[0], lons=torn_wwa_ll[1])
    lsr_xy = list(zip(lsr_xy[1,:],lsr_xy[0,:]))

    return lsr_xy

def load_svr_warning(date, initial_date_and_time, time_window = 15): 
    '''
    Load severe weather warning polygons within +/- 15 min of the initialization time.
    '''
    load_lsr = loadLSR(date_dir=date, date=initial_date_and_time[0], time=initial_date_and_time[1], time_window = time_window, forecast_length=0)
    load_wwa = loadWWA(date_dir=date, date=initial_date_and_time[0], time=initial_date_and_time[1], time_window = time_window, forecast_length=0)

    svr_wwa_ll = load_wwa.load_svr_tstorm_warning_polygon( )
    lsr_xy = load_lsr.to_xy( lats=svr_wwa_ll[0], lons=svr_wwa_ll[1])
    lsr_xy = list(zip(lsr_xy[1,:],lsr_xy[0,:]))
    
    return lsr_xy
 
    


