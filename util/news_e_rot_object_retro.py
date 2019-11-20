#!/usr/bin/env python
import matplotlib
matplotlib.use('Agg')
from mpl_toolkits.basemap import Basemap
import scipy
from scipy import signal
from scipy import *
from scipy import ndimage
import skimage
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
import radar_info
from radar_info import *

####################################### Parse Options: ######################################################

parser = OptionParser()
parser.add_option("-d", dest="exp_dir", type="string", default= None, help="Input Directory (of NEWS-e summary files)")
parser.add_option("-o", dest="out_path", type="string", help = "Output File Path")
parser.add_option("-f", dest="aws_path", type="string", help = "Path to AWS qc File")
parser.add_option("-m", dest="aws_ml_path", type="string", help = "Path to Midlevel AWS qc File")
parser.add_option("-z", dest="dz_path", type="string", help = "Path to MRMS dBZ qc File")

(options, args) = parser.parse_args()

if ((options.exp_dir == None) or (options.out_path == None) or (options.aws_path == None) or (options.aws_ml_path == None) or (options.dz_path == None)):
    print()
    parser.print_help()
    print()
    sys.exit(1)
else:
    exp_dir = options.exp_dir
    out_path = options.out_path
    aws_path = options.aws_path
    aws_ml_path = options.aws_ml_path
    dz_path = options.dz_path

#################################### User-Defined Variables:  #####################################################

#### generic variables in order to call mymap (as quickly as possible)

damage_files     = ''
area_thresh      = 1000.
resolution       = 'c'

edge            = 7		#number of grid points to remove from near domain boundaries
radius_max      = 3             #grid point radius for maximum value filter
radius_gauss    = 2             #grid point radius of convolution operator
time_window     = 3		#Number of forecast times +- current time to make swath
area_thresh     = 4		#Mininum area of rotation object
eccent_thresh   = 0.4		#Eccentricity threshold of rotation object (NOT USED) 

kernel = gauss_kern(radius_gauss) #Smoothing kernel

aws_thresh      = 0.005  ### thresholds calculated using empirical conversion formulas from aws_thresh
wz_0to2_thresh  = -16.5 * aws_thresh**2 + .773 * aws_thresh - 0.000178
uh_0to2_thresh  = 44400 * aws_thresh**2 + 6760. * aws_thresh - 15.3
uh_2to5_thresh  = -48200 * aws_thresh**2 + 32800 * aws_thresh - 93.3

print('thresholds: ', aws_thresh, wz_0to2_thresh, uh_0to2_thresh, uh_2to5_thresh)

#Read radmask from 0-2 aws file (rest of MRMS data are read in at end of script:

try:
   aws_in = netCDF4.Dataset(aws_path, "r")
   print("Opening %s \n" % aws_path)
except:
   print("%s does not exist! \n" %aws_path)
   sys.exit(1)

radmask = aws_in.variables['RADMASK'][:,:]

aws_in.close()
del aws_in

######################################################################################################
#################################### Read Data:  #####################################################
######################################################################################################

files = []
files_temp = os.listdir(exp_dir)
for f, file in enumerate(files_temp):
   if (file[0] == '2'):
      files.append(file)

files.sort()
ne = len(files)

################## Get number of forecast times (fcst_nt) from first summary file: ########################

exp_file = os.path.join(exp_dir, files[0])

try:
   fin = netCDF4.Dataset(exp_file, "r")
   print("Opening %s \n" % exp_file)
except:
   print("%s does not exist! \n" % exp_file)
   sys.exit(1)

temp_time = fin.variables['TIME'][:]
nt = len(temp_time)

print('nt is: ', nt)

fin.close()
del fin

for f, file in enumerate(files):
   exp_file = os.path.join(exp_dir, file)

   try:
      fin = netCDF4.Dataset(exp_file, "r")
      print("Opening %s \n" % exp_file)
   except:
      print("%s does not exist! \n" % exp_file)
      sys.exit(1)

   if (f == 0):
      time = fin.variables['TIME'][0:nt]

############ Get grid/projection info, chop 'edge' from boundaries: ####################
 
      xlat = fin.variables['XLAT'][:]
      xlon = fin.variables['XLON'][:]
#      xlat = xlat[edge:-edge,edge:-edge]
#      xlon = xlon[edge:-edge,edge:-edge]

      sw_lat_full = xlat[0,0]
      sw_lon_full = xlon[0,0]
      ne_lat_full = xlat[-1,-1]
      ne_lon_full = xlon[-1,-1]

      cen_lat = fin.CEN_LAT
      cen_lon = fin.CEN_LON
      stand_lon = fin.STAND_LON
      true_lat1 = fin.TRUE_LAT1
      true_lat2 = fin.TRUE_LAT2

######################### Initialize variables: ####################################

      wz_0to2 = np.zeros((ne, len(time), xlat.shape[0], xlat.shape[1]))
      uh_0to2 = np.zeros((ne, len(time), xlat.shape[0], xlat.shape[1]))
      uh_2to5 = np.zeros((ne, len(time), xlat.shape[0], xlat.shape[1]))
      dz = np.zeros((ne, len(time), xlat.shape[0], xlat.shape[1]))

######################## Read in summary file variables: ################################

   wz_0to2[f,:,:,:] = fin.variables['WZ_0TO2'][0:nt,:,:]
   uh_0to2[f,:,:,:] = fin.variables['UH_0TO2'][0:nt,:,:]
   uh_2to5[f,:,:,:] = fin.variables['UH_2TO5'][0:nt,:,:]
   dz[f,:,:,:] = fin.variables['DZ_COMP'][0:nt,:,:]

######################### Initialize rotation track variables: ####################################

window_time = time[time_window:-time_window]

wz_0to2_window_qc = np.zeros((ne, nt, xlat.shape[0], xlat.shape[1]))
uh_0to2_window_qc = np.zeros((ne, nt, xlat.shape[0], xlat.shape[1]))
uh_2to5_window_qc = np.zeros((ne, nt, xlat.shape[0], xlat.shape[1]))

wz_0to2_conv_window_qc = np.zeros((ne, nt, xlat.shape[0], xlat.shape[1]))
uh_0to2_conv_window_qc = np.zeros((ne, nt, xlat.shape[0], xlat.shape[1]))
uh_2to5_conv_window_qc = np.zeros((ne, nt, xlat.shape[0], xlat.shape[1]))

wz_0to2_window_mask_qc = np.zeros((ne, nt, xlat.shape[0], xlat.shape[1]))
uh_0to2_window_mask_qc = np.zeros((ne, nt, xlat.shape[0], xlat.shape[1]))
uh_2to5_window_mask_qc = np.zeros((ne, nt, xlat.shape[0], xlat.shape[1]))

wz_0to2_conv_window_mask_qc = np.zeros((ne, nt, xlat.shape[0], xlat.shape[1]))
uh_0to2_conv_window_mask_qc = np.zeros((ne, nt, xlat.shape[0], xlat.shape[1]))
uh_2to5_conv_window_mask_qc = np.zeros((ne, nt, xlat.shape[0], xlat.shape[1]))

######################### Find rotation track objects: ####################################

for n in range(0, ne):
   for t in range(0, len(time)):
      print(t)
      if ((t >= time_window) and (t < (len(time)-time_window))): #if within time window
         wz_0to2_window_temp = wz_0to2[n,(t-time_window):(t+time_window),:,:]
         uh_0to2_window_temp = uh_0to2[n,(t-time_window):(t+time_window),:,:]
         uh_2to5_window_temp = uh_2to5[n,(t-time_window):(t+time_window),:,:]

         wz_0to2_window = np.max(wz_0to2_window_temp, axis=0)
         uh_0to2_window = np.max(uh_0to2_window_temp, axis=0)
         uh_2to5_window = np.max(uh_2to5_window_temp, axis=0)

         wz_0to2_window_indices = np.argmax(wz_0to2_window_temp, axis=0)
         uh_0to2_window_indices = np.argmax(uh_0to2_window_temp, axis=0)
         uh_2to5_window_indices = np.argmax(uh_2to5_window_temp, axis=0)
      
######################### Mask regions too close/far from nearest WSR-88D: ####################################

         wz_0to2_window_mask = np.where(radmask > 0, 0., wz_0to2_window)
         uh_0to2_window_mask = np.where(radmask > 0, 0., uh_0to2_window)
         uh_2to5_window_mask = np.where(radmask > 0, 0., uh_2to5_window)
      
         wz_0to2_window_indices_mask = np.where(radmask > 0, 0., wz_0to2_window_indices)
         uh_0to2_window_indices_mask = np.where(radmask > 0, 0., uh_0to2_window_indices)
         uh_2to5_window_indices_mask = np.where(radmask > 0, 0., uh_2to5_window_indices)

         wz_0to2_init = np.where(wz_0to2_window >= wz_0to2_thresh, wz_0to2_window, 0.)
         uh_0to2_init = np.where(uh_0to2_window >= uh_0to2_thresh, uh_0to2_window, 0.)
         uh_2to5_init = np.where(uh_2to5_window >= uh_2to5_thresh, uh_2to5_window, 0.)

         wz_0to2_init_mask = np.where(wz_0to2_window_mask >= wz_0to2_thresh, wz_0to2_window_mask, 0.)
         uh_0to2_init_mask = np.where(uh_0to2_window_mask >= uh_0to2_thresh, uh_0to2_window_mask, 0.)
         uh_2to5_init_mask = np.where(uh_2to5_window_mask >= uh_2to5_thresh, uh_2to5_window_mask, 0.)

         wz_0to2_window_qc_temp = wz_0to2_init * 0.
         uh_0to2_window_qc_temp = uh_0to2_init * 0.
         uh_2to5_window_qc_temp = uh_2to5_init * 0.

         wz_0to2_window_qc_temp_mask = wz_0to2_init_mask * 0.
         uh_0to2_window_qc_temp_mask = uh_0to2_init_mask * 0.
         uh_2to5_window_qc_temp_mask = uh_2to5_init_mask * 0.

         wz_0to2_window_int = np.where(wz_0to2_window >= wz_0to2_thresh, 1, 0)
         uh_0to2_window_int = np.where(uh_0to2_window >= uh_0to2_thresh, 1, 0)
         uh_2to5_window_int = np.where(uh_2to5_window >= uh_2to5_thresh, 1, 0)

         wz_0to2_window_int_mask = np.where(wz_0to2_window_mask >= wz_0to2_thresh, 1, 0)
         uh_0to2_window_int_mask = np.where(uh_0to2_window_mask >= uh_0to2_thresh, 1, 0)
         uh_2to5_window_int_mask = np.where(uh_2to5_window_mask >= uh_2to5_thresh, 1, 0)

         wz_0to2_labels = skimage.measure.label(wz_0to2_window_int)
         uh_0to2_labels = skimage.measure.label(uh_0to2_window_int)
         uh_2to5_labels = skimage.measure.label(uh_2to5_window_int)

         wz_0to2_labels_mask = skimage.measure.label(wz_0to2_window_int_mask)
         uh_0to2_labels_mask = skimage.measure.label(uh_0to2_window_int_mask)
         uh_2to5_labels_mask = skimage.measure.label(uh_2to5_window_int_mask)

         wz_0to2_labels = wz_0to2_labels.astype(int)
         uh_0to2_labels = uh_0to2_labels.astype(int)
         uh_2to5_labels = uh_2to5_labels.astype(int)

         wz_0to2_labels_mask = wz_0to2_labels_mask.astype(int)
         uh_0to2_labels_mask = uh_0to2_labels_mask.astype(int)
         uh_2to5_labels_mask = uh_2to5_labels_mask.astype(int)

         wz_0to2_props = regionprops(wz_0to2_labels, wz_0to2_init)
         uh_0to2_props = regionprops(uh_0to2_labels, uh_0to2_init)
         uh_2to5_props = regionprops(uh_2to5_labels, uh_2to5_init)

         wz_0to2_props_mask = regionprops(wz_0to2_labels_mask, wz_0to2_init_mask)
         uh_0to2_props_mask = regionprops(uh_0to2_labels_mask, uh_0to2_init_mask)
         uh_2to5_props_mask = regionprops(uh_2to5_labels_mask, uh_2to5_init_mask)

################### Remove objects smaller than area threshold: ###############

         wz_0to2_qc_obs = []
         for i in range(0, len(wz_0to2_props)):
            if ((wz_0to2_props[i].area > area_thresh)): # and (cress_props[i].eccentricity > eccent_thresh)):
               wz_0to2_qc_obs.append(i)

         wz_0to2_qc_obs_mask = []
         for i in range(0, len(wz_0to2_props_mask)):
            if ((wz_0to2_props_mask[i].area > area_thresh)): # and (cress_props[i].eccentricity > eccent_thresh)):
               wz_0to2_qc_obs_mask.append(i)

         uh_0to2_qc_obs = []
         for i in range(0, len(uh_0to2_props)):
            if ((uh_0to2_props[i].area > area_thresh)): # and (cress_props[i].eccentricity > eccent_thresh)):
               uh_0to2_qc_obs.append(i)

         uh_0to2_qc_obs_mask = []
         for i in range(0, len(uh_0to2_props_mask)):
            if ((uh_0to2_props_mask[i].area > area_thresh)): # and (cress_props[i].eccentricity > eccent_thresh)):
               uh_0to2_qc_obs_mask.append(i)

         uh_2to5_qc_obs = []
         for i in range(0, len(uh_2to5_props)):
            if ((uh_2to5_props[i].area > area_thresh)): # and (cress_props[i].eccentricity > eccent_thresh)):
               uh_2to5_qc_obs.append(i)

         uh_2to5_qc_obs_mask = []
         for i in range(0, len(uh_2to5_props_mask)):
            if ((uh_2to5_props_mask[i].area > area_thresh)): # and (cress_props[i].eccentricity > eccent_thresh)):
               uh_2to5_qc_obs_mask.append(i)

################### Remove objects that don't meet continuity threshold: ###############

         for i in range(0, len(wz_0to2_qc_obs)):
            temp_object = np.where(wz_0to2_labels == (wz_0to2_qc_obs[i]+1), wz_0to2_window_indices, 0.)
            temp_num_times = np.unique(temp_object)
            if (len(temp_num_times) > 2): #require swath object to contain values from at least 2 different times 
               wz_0to2_window_qc_temp = np.where(wz_0to2_labels == (wz_0to2_qc_obs[i]+1), wz_0to2_init, wz_0to2_window_qc_temp)

         for i in range(0, len(wz_0to2_qc_obs_mask)):
            temp_object = np.where(wz_0to2_labels_mask == (wz_0to2_qc_obs_mask[i]+1), wz_0to2_window_indices_mask, 0.)
            temp_num_times = np.unique(temp_object)
            if (len(temp_num_times) > 2): #require swath object to contain values from at least 2 different times 
               wz_0to2_window_qc_temp_mask = np.where(wz_0to2_labels_mask == (wz_0to2_qc_obs_mask[i]+1), wz_0to2_init_mask, wz_0to2_window_qc_temp_mask)

         for i in range(0, len(uh_0to2_qc_obs)):
            temp_object = np.where(uh_0to2_labels == (uh_0to2_qc_obs[i]+1), uh_0to2_window_indices, 0.)
            temp_num_times = np.unique(temp_object)
            if (len(temp_num_times) > 2): #require swath object to contain values from at least 2 different times 
               uh_0to2_window_qc_temp = np.where(uh_0to2_labels == (uh_0to2_qc_obs[i]+1), uh_0to2_init, uh_0to2_window_qc_temp)

         for i in range(0, len(uh_0to2_qc_obs_mask)):
            temp_object = np.where(uh_0to2_labels_mask == (uh_0to2_qc_obs_mask[i]+1), uh_0to2_window_indices_mask, 0.)
            temp_num_times = np.unique(temp_object)
            if (len(temp_num_times) > 2): #require swath object to contain values from at least 2 different times 
               uh_0to2_window_qc_temp_mask = np.where(uh_0to2_labels_mask == (uh_0to2_qc_obs_mask[i]+1), uh_0to2_init_mask, uh_0to2_window_qc_temp_mask)

         for i in range(0, len(uh_2to5_qc_obs)):
            temp_object = np.where(uh_2to5_labels == (uh_2to5_qc_obs[i]+1), uh_2to5_window_indices, 0.)
            temp_num_times = np.unique(temp_object)
            if (len(temp_num_times) > 2): #require swath object to contain values from at least 2 different times 
               uh_2to5_window_qc_temp = np.where(uh_2to5_labels == (uh_2to5_qc_obs[i]+1), uh_2to5_init, uh_2to5_window_qc_temp)

         for i in range(0, len(uh_2to5_qc_obs_mask)):
            temp_object = np.where(uh_2to5_labels_mask == (uh_2to5_qc_obs_mask[i]+1), uh_2to5_window_indices_mask, 0.)
            temp_num_times = np.unique(temp_object)
            if (len(temp_num_times) > 2): #require swath object to contain values from at least 2 different times 
               uh_2to5_window_qc_temp_mask = np.where(uh_2to5_labels_mask == (uh_2to5_qc_obs_mask[i]+1), uh_2to5_init_mask, uh_2to5_window_qc_temp_mask)

         wz_0to2_window_qc[n,t,:,:] = wz_0to2_window_qc_temp
         uh_0to2_window_qc[n,t,:,:] = uh_0to2_window_qc_temp
         uh_2to5_window_qc[n,t,:,:] = uh_2to5_window_qc_temp

         wz_0to2_window_mask_qc[n,t,:,:] = wz_0to2_window_qc_temp_mask
         uh_0to2_window_mask_qc[n,t,:,:] = uh_0to2_window_qc_temp_mask
         uh_2to5_window_mask_qc[n,t,:,:] = uh_2to5_window_qc_temp_mask

################### Run maximum value and convolution filters: ###############

         wz_0to2_maxfilter_window = get_local_maxima2d(wz_0to2_window_qc_temp, radius_max)
         wz_0to2_conv_window_qc[n,t,:,:] = signal.convolve2d(wz_0to2_maxfilter_window, kernel, 'same')

         wz_0to2_maxfilter_window_mask = get_local_maxima2d(wz_0to2_window_qc_temp_mask, radius_max)
         wz_0to2_conv_window_mask_qc[n,t,:,:] = signal.convolve2d(wz_0to2_maxfilter_window_mask, kernel, 'same')

         uh_0to2_maxfilter_window = get_local_maxima2d(uh_0to2_window_qc_temp, radius_max)
         uh_0to2_conv_window_qc[n,t,:,:] = signal.convolve2d(uh_0to2_maxfilter_window, kernel, 'same')

         uh_0to2_maxfilter_window_mask = get_local_maxima2d(uh_0to2_window_qc_temp_mask, radius_max)
         uh_0to2_conv_window_mask_qc[n,t,:,:] = signal.convolve2d(uh_0to2_maxfilter_window_mask, kernel, 'same')

         uh_2to5_maxfilter_window = get_local_maxima2d(uh_2to5_window_qc_temp, radius_max)
         uh_2to5_conv_window_qc[n,t,:,:] = signal.convolve2d(uh_2to5_maxfilter_window, kernel, 'same')

         uh_2to5_maxfilter_window_mask = get_local_maxima2d(uh_2to5_window_qc_temp_mask, radius_max)
         uh_2to5_conv_window_mask_qc[n,t,:,:] = signal.convolve2d(uh_2to5_maxfilter_window_mask, kernel, 'same')

####################################### Read AWS Variables: ######################################################

#aws 0-2:

try:
   aws_in = netCDF4.Dataset(aws_path, "r")
   print("Opening %s \n" % aws_path)
except:
   print("%s does not exist! \n" %aws_path)
   sys.exit(1)

vtimes = aws_in.variables['TIME'][:]

vt_min = (np.abs(time[0] - vtimes)).argmin()
vt_max = vt_min + nt #(np.abs(time[-1] - vtimes)).argmin()

aws_0to2 = aws_in.variables['VAR_CRESS_WINDOW_QC'][vt_min:vt_max,:,:]

aws_in.close()
del aws_in

#aws 2-5: 

try:
   aws_in = netCDF4.Dataset(aws_ml_path, "r")
   print("Opening %s \n" % aws_ml_path)
except:
   print("%s does not exist! \n" %aws_ml_path)
   sys.exit(1)

vtimes = aws_in.variables['TIME'][:]

vt_min = (np.abs(time[0] - vtimes)).argmin()
vt_max = vt_min + nt #(np.abs(time[-1] - vtimes)).argmin()

aws_2to5 = aws_in.variables['VAR_CRESS_WINDOW_QC'][vt_min:vt_max,:,:]

aws_in.close()
del aws_in

#dz:

try:
   dz_in = netCDF4.Dataset(dz_path, "r")
   print("Opening %s \n" % dz_path)
except:
   print("%s does not exist! \n" %dz_path)
   sys.exit(1)

ztimes = dz_in.variables['TIME'][:]

zt_min = (np.abs(time[0] - ztimes)).argmin()
zt_max = zt_min + nt #(np.abs(time[-1] - ztimes)).argmin()

mrms_dz = dz_in.variables['VAR_CRESSMAN'][zt_min:zt_max,:,:]

dz_in.close()
del dz_in

print(mrms_dz.shape, aws_0to2.shape, dz.shape, wz_0to2_window_qc.shape)

################### Output rotation track objects as netcdf file: ###############

try:
   fout = netCDF4.Dataset(out_path, "w")
except:
   print("Could not create %s!\n" % out_path)

fout.createDimension('NE', ne)
fout.createDimension('NT', len(time))
fout.createDimension('NX', xlat.shape[1])
fout.createDimension('NY', xlat.shape[0])

fout.createVariable('TIME', 'f4', ('NT',))
fout.createVariable('XLAT', 'f4', ('NY','NX',))
fout.createVariable('XLON', 'f4', ('NY','NX',))
fout.createVariable('RADMASK', 'f4', ('NY','NX',))

fout.createVariable('WZ_0TO2_WINDOW_QC', 'f4', ('NE', 'NT','NY','NX',))
fout.createVariable('UH_0TO2_WINDOW_QC', 'f4', ('NE', 'NT','NY','NX',))
fout.createVariable('UH_2TO5_WINDOW_QC', 'f4', ('NE', 'NT','NY','NX',))
#fout.createVariable('WZ_0TO2_CONV_WINDOW_QC', 'f4', ('NE', 'NT','NY','NX',))
#fout.createVariable('UH_0TO2_CONV_WINDOW_QC', 'f4', ('NE', 'NT','NY','NX',))
#fout.createVariable('UH_2TO5_CONV_WINDOW_QC', 'f4', ('NE', 'NT','NY','NX',))
fout.createVariable('WZ_0TO2_WINDOW_MASK', 'f4', ('NE', 'NT','NY','NX',))
fout.createVariable('UH_0TO2_WINDOW_MASK', 'f4', ('NE', 'NT','NY','NX',))
fout.createVariable('UH_2TO5_WINDOW_MASK', 'f4', ('NE', 'NT','NY','NX',))
#fout.createVariable('WZ_0TO2_CONV_WINDOW_MASK', 'f4', ('NE', 'NT','NY','NX',))
#fout.createVariable('UH_0TO2_CONV_WINDOW_MASK', 'f4', ('NE', 'NT','NY','NX',))
#fout.createVariable('UH_2TO5_CONV_WINDOW_MASK', 'f4', ('NE', 'NT','NY','NX',))

fout.createVariable('DZ', 'f4', ('NE', 'NT','NY','NX',))
fout.createVariable('MRMS_DZ', 'f4', ('NT','NY','NX',))
fout.createVariable('MRMS_AWS_0TO2', 'f4', ('NT','NY','NX',))
fout.createVariable('MRMS_AWS_2TO5', 'f4', ('NT','NY','NX',))

fout.variables['TIME'][:] = time 
fout.variables['XLAT'][:] = xlat
fout.variables['XLON'][:] = xlon
fout.variables['RADMASK'][:] = radmask

fout.variables['WZ_0TO2_WINDOW_QC'][:] = wz_0to2_window_qc
fout.variables['UH_0TO2_WINDOW_QC'][:] = uh_0to2_window_qc
fout.variables['UH_2TO5_WINDOW_QC'][:] = uh_2to5_window_qc
#fout.variables['WZ_0TO2_CONV_WINDOW_QC'][:] = wz_0to2_conv_window_qc
#fout.variables['UH_0TO2_CONV_WINDOW_QC'][:] = uh_0to2_conv_window_qc
#fout.variables['UH_2TO5_CONV_WINDOW_QC'][:] = uh_2to5_conv_window_qc
fout.variables['WZ_0TO2_WINDOW_MASK'][:] = wz_0to2_window_mask_qc
fout.variables['UH_0TO2_WINDOW_MASK'][:] = uh_0to2_window_mask_qc
fout.variables['UH_2TO5_WINDOW_MASK'][:] = uh_2to5_window_mask_qc
#fout.variables['WZ_0TO2_CONV_WINDOW_MASK'][:] = wz_0to2_conv_window_mask_qc
#fout.variables['UH_0TO2_CONV_WINDOW_MASK'][:] = uh_0to2_conv_window_mask_qc
#fout.variables['UH_2TO5_CONV_WINDOW_MASK'][:] = uh_2to5_conv_window_mask_qc

fout.variables['DZ'][:] = dz
fout.variables['MRMS_DZ'][:] = mrms_dz
fout.variables['MRMS_AWS_0TO2'][:] = aws_0to2
fout.variables['MRMS_AWS_2TO5'][:] = aws_2to5

fout.close()
del fout

