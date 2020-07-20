import netCDF4 as nc 
import numpy as np 
import os
from os.path import join, exists 
import datetime
import xarray as xr

# Personal Modules
from wofs.util.basic_functions import personal_datetime
from wofs.util import config

get_time = personal_datetime( )

class MRMSData: 
    def NCFileList( self, date_dir, time, nt  ):
        '''
        Returns multiple netcdf Datasets 
        ''' 
        # Example file name: /work1/skinnerp/MRMS_verif/mrms_cressman/20170516/20170516-184500.nc ; each time is separate, every 5 min  
        list_of_filenames = [ ]
        datetime_obj, _, _ = get_time.initial_datetime(date_dir=date_dir, time_dir=time  )

        for i in range( nt+1 ):
            date_temp, time_temp = get_time.convert_datetime_obj_to_components( datetime_obj )                                  
            filename = join( join( config.MRMS_PATH, date_dir), '%s-%s00.nc' % (date_temp, time_temp) ) 
            if exists( filename ) : 
                ncfile = nc.Dataset( filename, 'r' )        
                list_of_filenames.append( ncfile ) 
            datetime_obj += datetime.timedelta( minutes = 5 )
        return list_of_filenames
    
    def load_mrms_time_composite( self, var_name, date, time, nt = 12 ): 
        ncfiles = self.NCFileList( date, time, nt )
        data = [ ] 
        for f in ncfiles: 
            data.append( f.variables[var_name][:,:] )  
            f.close( ) 
            del f
        # Returns ny, nx 
        return np.amax( data, axis = 0 ) , np.argmax( data, axis = 0 )             
   
    def radarMask( self ):
        mrms_files = os.listdir( self.basePath )
        f = nc.Dataset( mrms_files[0], 'r' )
        radar_mask = f.variables['RADMASK'][:]
        f.close( )
        del f
        return radar_mask

    def load_single_mrms_time( self, date_dir, valid_datetime, var_name ):
        in_path = join( config.MRMS_PATH, date_dir )
        filename = join( in_path, '%s-%s00.nc' % (valid_datetime[0], valid_datetime[1]) )
        if os.path.exists( filename ):
            ncfile = nc.Dataset( filename, 'r' )
            return ncfile.variables[var_name][:]
        else: 
            print(filename, "does not exist!")
            return None 
        # May need to introduce code that reads in the file 5 mins before or after if this one doesn't exist 

    def load_az_shr_tracks( self, var, name, date_dir, valid_datetime ): 
        '''
        Load the objectively-identified hour-long azimuthal shear (rotation) tracks.
        '''    
        in_path = join( config.MRMS_TRACKS_PATH, date_dir )
        fname = join( in_path, 'MRMS_%s_TRACKS_%s%s.nc' % ( var, valid_datetime[0], valid_datetime[1] ))
        ds = xr.open_dataset( fname )

        return ds[name].values, ds['Azimuthal Shear'].values 

    def load_gridded_lsrs(self, var, date_dir, valid_datetime):
        '''
        Load Gridded LSRS.
        '''
        fname = join( join( config.LSR_SAVE_PATH, date_dir ),  'LSRs_%s-%s.nc' % (valid_datetime[0], valid_datetime[1]))
        ds = xr.open_dataset( fname )
        return ds[var].values

    def load_mrms_dbz( self, date_dir, valid_datetime, tag):
        '''
        Load the objectively-identified MRMS reflectivity objects.
        '''
        in_path = os.path.join( config.MRMS_DBZ_PATH, date_dir ) 
        fname = os.path.join( in_path, '%s_%s%s_%s.nc' % ( 'DZ_CRESSMAN', valid_datetime[0], valid_datetime[1], tag ) )
        ds = xr.open_dataset( fname )

        return ds['DBZ Objects (40 dbz)'].values, ds['DBZ Objects (35 dbz)'].values, ds['DBZ'].values 




