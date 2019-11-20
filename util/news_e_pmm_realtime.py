#!/usr/bin/env python
import matplotlib
matplotlib.use('Agg')
from mpl_toolkits.basemap import Basemap
from scipy import signal
from scipy import *
from scipy import ndimage
from skimage.morphology import label
from skimage.measure import regionprops
import math
from math import radians, tan, sin, cos, pi, atan, sqrt, pow, asin, acos
import pylab as P
import numpy as np
from numpy import NAN
import sys
import netCDF4
from optparse import OptionParser
from netcdftime import utime
import os
import time as timeit
from optparse import OptionParser
import news_e_post_cbook
from news_e_post_cbook import *
import news_e_plotting_cbook_v2
from news_e_plotting_cbook_v2 import *

####################################### Parse Options: ######################################################

parser = OptionParser()
parser.add_option("-d", dest="exp_dir", type="string", default= None, help="Input Directory (of summary files)")
parser.add_option("-s", dest="start_t", type="int", help = "Start timestep")
parser.add_option("-e", dest="end_t", type="int", help = "End timestep")
parser.add_option("-t", dest="nt", type="int", help = "Total number of timesteps")

(options, args) = parser.parse_args()

if ((options.exp_dir == None) or (options.start_t == None) or (options.end_t == None) or (options.nt == None)):
    print()
    parser.print_help()
    print()
    sys.exit(1)
else:
    exp_dir = options.exp_dir
    start_t = options.start_t
    end_t = options.end_t
    nt = options.nt

output_path = os.path.join(exp_dir, 'pmm_dz.nc')
domain = 'full'

#################################### User-Defined Variables:  #####################################################

edge            = 7 		#number of grid points to remove from near domain boundaries
thin		= 6		#thinning factor for quiver values (e.g. 6 means slice every 6th grid point)

fcst_len 	= 5400. 	#length of forecast to consider (s)
radius_max      = 3             #grid point radius for maximum value filter
radius_gauss    = 2             #grid point radius of convolution operator
neighborhood 	= 15		#grid point radius of prob matched mean neighborhood

#################################### Read Data:  #####################################################

files = os.listdir(exp_dir)
files.sort()
ne = len(files)

for f, file in enumerate(files):
   if ((file[0:3] != 'pmm') and (file[0:3] != 'dum')):
      exp_file = os.path.join(exp_dir, file)
      try:
         fin = netCDF4.Dataset(exp_file, "r")
         print("Opening %s \n" % exp_file)
      except:
         print("%s does not exist! \n" % exp_file)
         sys.exit(1)

      if (f == 0): 
         date = file[0:10]
         init_label = 'Init: ' + date + ', ' + file[11:13] + file[14:16] + ' UTC'
      
         time = fin.variables['TIME'][start_t:end_t]

         xlat = fin.variables['XLAT'][:]
         xlon = fin.variables['XLON'][:]

         xlat = xlat[edge:-edge,edge:-edge]
         xlon = xlon[edge:-edge,edge:-edge]

         ny = xlat.shape[0]
         nx = xlat.shape[1]

         sw_lat_full = xlat[0,0]
         sw_lon_full = xlon[0,0]
         ne_lat_full = xlat[-1,-1]
         ne_lon_full = xlon[-1,-1]

         cen_lat = fin.CEN_LAT
         cen_lon = fin.CEN_LON
         stand_lon = fin.STAND_LON
         true_lat1 = fin.TRUE_LAT1
         true_lat2 = fin.TRUE_LAT2

         dz = np.zeros((ne, (end_t - start_t), xlat.shape[0], xlat.shape[1]))
#         wz_0to2 = np.zeros((ne, (end_t - start_t), xlat.shape[0], xlat.shape[1]))

      dz[f,:,:,:] = fin.variables['DZ_COMP'][start_t:end_t,edge:-edge,edge:-edge]
#      wz_0to2[f,:,:,:] = fin.variables['WZ_0TO2'][start_t:end_t,edge:-edge,edge:-edge]

      fin.close()
      del fin

mean_dz = np.mean(dz, axis=0)
#mean_wz = np.mean(wz_0to2, axis=0)

#################################### Calc Prob Matched Mean:  #############################################

print('prob match mean part') 

pmm_dz = mean_dz * 0.
#pmm_wz = mean_wz * 0.
print(end_t-start_t)

for t in range(0, (end_t - start_t)):
   pmm_dz[t,:,:] = prob_match_mean(dz[:,t,:,:], mean_dz[t,:,:], neighborhood)
#   pmm_wz[t,:,:] = prob_match_mean(wz_0to2[:,t,:,:], mean_wz[t,:,:], neighborhood)
   print(t)

########## Save output as netcdf file: #################

if (start_t == 0):
   try:
      fout = netCDF4.Dataset(output_path, "w")
   except:
      print("Could not create %s!\n" % output_path)

   fout.createDimension('NX', nx)
   fout.createDimension('NY', ny)
   fout.createDimension('NT', nt)

   setattr(fout,'CEN_LAT',cen_lat)
   setattr(fout,'CEN_LON',cen_lon)
   setattr(fout,'STAND_LON',stand_lon)
   setattr(fout,'TRUE_LAT1',true_lat1)
   setattr(fout,'TRUE_LAT2',true_lat2)

   fout.createVariable('TIME', 'f4', ('NT',))
   fout.createVariable('XLAT', 'f4', ('NY','NX',))
   fout.createVariable('XLON', 'f4', ('NY','NX',))
   fout.createVariable('HGT', 'f4', ('NY','NX',))

   fout.createVariable('PMM_DZ', 'f4', ('NT','NY','NX',))
#   fout.createVariable('PMM_WZ', 'f4', ('NT','NY','NX',))

   fout.variables['XLAT'][:] = xlat
   fout.variables['XLON'][:] = xlon

else:
   try:
      fout = netCDF4.Dataset(output_path, "a")
   except:
      print("Could not create %s!\n" % output_path)

fout.variables['TIME'][start_t:end_t] = time
fout.variables['PMM_DZ'][start_t:end_t,:,:] = pmm_dz
#fout.variables['PMM_WZ'][start_t:end_t,:,:] = pmm_wz

fout.close()
del fout

