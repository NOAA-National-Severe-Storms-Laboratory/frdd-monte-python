import netCDF4 as nc 
import os, sys
from glob import glob
from os.path import join
import numpy as np
sys.path.append('/home/monte.flora/wofs/util')
import config
import xarray as xr 

class EnsembleData:
    """
    EnsembleData loads the netCDF4 summary file data created by Patrick Skinner 
    to summarize the NEWS-e WRF output. Loads ensemble worth of data for one 
    particular variable at the time indexs given.  

        ATTRIBUTES:
            initial_date  , string , date when forecast was initialized (format: YYYYDDMM)    
            initial_time  , string , time when forecast was initialized (format: HHmm )
            var_name_list , list   , list containing variable names  
            fcst_time_idx , integer, parameter for forecast where lower bound is not the initial_time (default = 0)   
            basePath      , string , Summary file path including initial_date and initial_time
            nt          , integer, Number of time steps for 1 hour ( default = 12 for 5 min time step) 
            ne          , integer, Ensemble size (default = 18 for NEWS-e) 
        """ 
    def __init__( self, date_dir, time_dir, base_path ):
        self.date_dir = date_dir
        self.time_dir = time_dir
        self.mode = base_path
        if base_path == 'summary_files':
            self.basePath  = join( join( config.SUMMARY_FILES_PATH, self.date_dir), self.time_dir)
        elif base_path == 'wofs_data':
            self.basePath  = join( join( config.WOFS_DATA_PATH, self.date_dir), self.time_dir)

    def _build_filename_list(self, time_indexs, tag, verbose=False):
        '''
        Generates the file names to be loaded. 
        Args, 
            : time_indexs, list of time indexs to load (e.g., [0,1,2,3]) 
            : tag, string for Pat's different netCDF files 
                 options = 'ENV', 'ENS'
            : verbose, boolean, if True print statements 
        Return, 
            : files, a list of strings of the different file names to load 
        '''
        all_files  = os.listdir( self.basePath )
        files  = [ ]
        if 'summary_files' in self.basePath:
            for t in time_indexs:
                #print 'news-e_%s_%02d*' % (tag, t )
                fname = glob(join( self.basePath, 'news-e_{}_{02d}*'.format(tag, t )))
                try:
                    files.append( nc.Dataset( fname[0], 'r' ))     
                except IndexError:
                    if verbose:
                        print ("{} is empty or does not exist! \ 
                               {} may not have files in it at time index {}".format(fname, self.basePath, t)
        
        elif 'WOFS_DATA' in self.basePath:
            for t in time_indexs:
                fname = glob(join( self.basePath, '*_%s.nc' % ( t )))
                try:
                    files.append( nc.Dataset( fname[0], 'r' )) 
                except IndexError:
                    if verbose:
                        print ("{} is empty or does not exist! \ 
                               {} may not have files in it at time index {}".format(fname, self.basePath, t)
        return files

    def load(self, variables, time_indexs, tag=None ):
        '''
        Load Ensemble data into array of shape (NT, NV, NE, NY, NX)
        
        Args,
            : variables, list of variable names in the netCDF files
            : time_indexs, list of time indexs to load (e.g., [0,1,2,3])
            : tag, string for Pat's different netCDF files 
                 options = 'ENV', 'ENS'

        Returns,
            : ens_data, numpy array of data, shape =(NT,NV,NE,NY,NX) 
        ''' 
        ens_data  = [ ] 
        append = ens_data.append
        files  = self._build_filename_list( time_indexs=time_indexs, tag=tag)
        for f in files:
            data_per_file = [ ] #nv, ne, ny, nx
            append([f.variables[var][:] for var in variables]) 
            f.close( ) 
            del f 
        ens_data = np.array( ens_data )
        if self.mode == 'summary_files':
            order_of_idxs = np.array([0, 9, 10, 11, 12, 13, 14, 15, 16, 17, 1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int32)
            ens_data = ens_data[:,:,order_of_idxs,:,:]
       
        return ens_data


def calc_time_max( input_data, time_axis=0, argmax=False ):
        '''
        Calculate the time max swath.
        optional to calculate argmax over the time axis as well
        
        Args, 
            : input_data, array-like data with time as an axis 
            : time_axis, integer index of the time axis of input_data
            : argmax, boolean, if True also returns the np.argmax 
                               of input_data along time axis
        Return,
            : time_max, np.amax of input_data along the given time_axis 
            : time_argmax, np.argmax of input_data along the given time_axis (optional) 
        '''
        time_max = np.amax( input_data, axis = time_axis )
        if argmax:
            time_argmax = np.argmax( input_data, axis = time_axis )
            return time_max, time_argmax
        else:
            return time_max

def calc_time_tendency( input_data, future_time_idx, past_time_idx ):
    '''
    Calculate the time tendency value between states at two different times.
    Assumes axis = 0 is the time axis 
    '''
    diff = input_data[future_time_idx,:] - input_data[past_time_idx,:] 

    return diff 

def load_mesocyclone_tracks(variable_key, var_name, date, time, fcst_time_idx ):
    '''
    Load ensemble of objectively-identified forecast mesoscyclone tracks.
    '''
    in_path = join( config.OBJECT_SAVE_PATH, date )
    fname =  join(in_path, 'WOFS_%s_OBJECTS_%s-%s_%02d.nc' % (config.variable_attrs[variable_key]['title'], date, time, fcst_time_idx))
    ds = xr.open_dataset( fname )

    return ds[var_name].values 
    



