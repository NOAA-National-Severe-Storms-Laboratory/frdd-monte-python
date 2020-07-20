import netCDF4 as nc 
import os
from glob import glob
from os.path import join
import numpy as np
from wofs.util import config
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
            if '2019' in date_dir:
                self.basePath  = join( join( config.SUMMARY_FILES_PATH_2019, self.date_dir), self.time_dir)
            else:
                self.basePath  = join( join( config.SUMMARY_FILES_PATH, self.date_dir), self.time_dir)
        elif base_path == 'wofs_data':
            self.basePath  = join( join( config.WOFS_DATA_PATH, self.date_dir), self.time_dir)

    def generate_filelist(self, time_indexs, tag):
        """
        Generates the file names to be loaded.
        """
        try:
            all_files  = os.listdir( self.basePath )
        except:
            raise Exception(f'{self.basePath} does not exist!')

        if 'summary_files' in self.basePath:
            files = [ glob(join( self.basePath, f'news-e_{tag}_{t:02d}*')) for t in time_indexs ]
        elif 'WOFS_DATA' in self.basePath:
            files = [ glob(join( self.basePath, f'*_{t}.nc')) for t in time_indexs ]

        try:
            files = [f[0] for f in files]
        except:
            raise Exception('files list is empty. No file matching the pattern exists! Check for possible error!.')

        return files

    def load_multiple_nc_files(self, vars_to_load, time_indexs, tag=None):
        """
        Load multiple netcdf files using xr.open_mfdataset
        Log: April 20, 2017. It was key to load the ncfiles separately
        rather than with load_mfdatset since load_mfdataset cause a MemoryError
        Args:
        --------------
            nc_file_paths : list
                list of netcdf file paths
            concat_dim : string
                name of the dimension that is being concatenated
                E.g., if nc_file_paths are valid for different times
                then concat_dim = 'time'
        Returns:
        ---------------
            multiple_datasets : xr.Dataset
                An xr.Dataset containing data from nc_file_paths
        """
        nc_file_paths = self.generate_filelist(time_indexs, tag)
        if tag is not None:
            drop_vars = ['hgt', 'xlat', 'xlon']
        else:
            drop_vars = [ ]
        
        multiple_datasets = [ ]
        for i, ncfile in enumerate(nc_file_paths):
            dataset = xr.open_dataset(ncfile, drop_variables=drop_vars)
            dataset_loaded = [dataset[var].values for var in vars_to_load]
            multiple_datasets.append(dataset_loaded)
            dataset.close()
            del dataset, dataset_loaded

        multiple_datasets = np.array(multiple_datasets).squeeze()
        
        if len(nc_file_paths) == 1:
            multiple_datasets_dict = {var: multiple_datasets[i].squeeze() for i, var in enumerate(vars_to_load)}
        else:
            multiple_datasets_dict = {var: multiple_datasets[:,i].squeeze() for i, var in enumerate(vars_to_load)}

        return multiple_datasets_dict

    def load_xr_dataset(self, time_indexs, tag=None, verbose=False):
        '''
        Generates the file names to be loaded. 
        '''
        files = self.generate_filelist() 
        try:
            ds = xr.open_mfdataset( files, concat_dim = 'time', parallel=True)
            return ds
        except:
            raise (f'Could not load all files for the xarray dataset!')

    def load(self, variables, time_indexs, tag=None ):
        '''
        Loads the variables at the different time indexs and returns
        a single xarray dataset

        ---------
        Args:
            variables, list of variables to load
            time_indexs, list of forecast time indexs to load (between 0-36)

        Returns: 
                ds, xarray dataset 
                [ Shape for a given variable: (NT, NE, NY, NX)]
        ''' 
        order_of_idxs = np.array([0, 9, 10, 11, 12, 13, 14, 15, 16, 17, 1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int32)
        ds  = self.load_xr_dataset( time_indexs=time_indexs, tag=tag)
        if ds is None:
            return None
        ds = ds[ variables ] 
        if self.mode == 'summary_files':
            # IT WORKS!!!!
            data = {}
            for v in variables:
                data[v] = (['time', 'NE', 'NY', 'NX'], ds[v].values[:,order_of_idxs,:,:])
            ds = xr.Dataset(data)

        return ds


def calc_time_max( input_data, time_axis=0, argmax=False ):
        '''
        Calculate the time max swath.
        optional to calculate argmax over the time axis as well
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
    



