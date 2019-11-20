import numpy as np 
import xarray as xr
from glob import glob
from os.path import join
import os 

mrmsPath_2018 = '/oldscratch/skinnerp/2018_newse_post/mrms/'
mrmsPath_2017 = '/work/skinnerp/MRMS_verif/mrms_cressman/'

base_paths = [ mrmsPath_2017, mrmsPath_2018 ] 

good_dates = [ ]
for in_path in base_paths:
    potential_dates = [ ]
    dates = [ x.split('/')[-1] for x in glob( join(in_path, '20*')) ]
    potential_dates.extend( dates ) 
 
    for date in potential_dates:
        if len(os.listdir( join(in_path, date)))>0:
            good_dates.append( date )

good_dates.remove( '20180517' ) 
good_dates.remove( '20180518' )
good_dates.remove( '20180602' )
#good_dates.remove( '20180611' ) 
good_dates.remove( '20180616' ) 
good_dates.remove( '20180617' ) 
good_dates.remove( '20180626' )
good_dates.remove( '20180618' )
good_dates.remove( '20180628' )

good_dates.remove( '20180526' ) 
good_dates.remove( '20170825' ) # Not Spring
good_dates.remove( '20170910' ) # Not spring 
good_dates.remove( '20180430' ) # Testing date 

good_dates = [ int(x) for x in good_dates ] 
good_dates_shuffled = np.random.shuffle( good_dates )

ds = xr.Dataset( data ) 
ds.to_netcdf( 'wofs_dates_for_verification.nc')

