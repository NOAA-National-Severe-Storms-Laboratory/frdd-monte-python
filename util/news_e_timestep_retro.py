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
parser.add_option("-o", dest="outdir", type="string", help = "Output Directory (for images)")
parser.add_option("-t", dest="t", type="int", help = "Timestep to process")

(options, args) = parser.parse_args()

if ((options.exp_dir == None) or (options.outdir == None) or (options.t == None)):
    print()
    parser.print_help()
    print()
    sys.exit(1)
else:
    exp_dir = options.exp_dir
    outdir = options.outdir
    t = options.t

domain = 'full'

#################################### User-Defined Variables:  #####################################################

edge            = 7 		#number of grid points to remove from near domain boundaries
thin		= 8		#thinning factor for quiver values (e.g. 6 means slice every 6th grid point)

fcst_len 	= 5400. 	#length of forecast to consider (s)
radius_max      = 9 #3             #grid point radius for maximum value filter
radius_gauss    = 6 #2             #grid point radius of convolution operator
neighborhood 	= 15		#grid point radius of prob matched mean neighborhood

plot_alpha 	= 0.6		#transparency value for filled contour plots

#################################### Threshold Values:  #####################################################

wz_thresh 	= [0.001, 0.002, 0.003, 0.004, 0.005, 0.006]		#vertical vorticity thresh (s^-1)
uh2to5_thresh 	= [20., 40., 60., 80., 100., 120.]		#updraft helicity thresh (2-5 km) (m^2 s^-2)
uh0to2_thresh 	= [10., 20., 30., 40., 50., 60.]		#updraft helicity thresh (0-2 km) (m^2 s^-2)
ws_thresh 	= [10., 15., 20., 25., 30., 35.]		#wind speed thresholds (m s^-1)
#rain_thresh     = [1., 7., 13., 19., 25., 31.] 			#qpf thresholds (mm)
rain_thresh     = [0.01, 0.25, 0.5, 0.75, 1., 1.25]                        #qpf thresholds (in)
dz_thresh       = [35., 40., 45., 50., 55., 60.]		#radar reflectivity thresholds (dBZ)

perc            = [10, 20, 30, 40, 50, 60, 70, 80, 90]		#Ens. percentiles to calc/plot

#################################### Define colormaps:  #####################################################

#dz_cmap         = ctables.Carbone42	#Carbone42 colormap from soloii

cin_cmap = matplotlib.colors.ListedColormap([cb_colors.purple7, cb_colors.purple6, cb_colors.purple5, cb_colors.purple4, cb_colors.blue4, cb_colors.blue3, cb_colors.blue2, cb_colors.gray1])

wz_cmap       = matplotlib.colors.ListedColormap([cb_colors.blue2, cb_colors.blue3, cb_colors.blue4, cb_colors.red2, cb_colors.red3, cb_colors.red4, cb_colors.red5, cb_colors.red6, cb_colors.red7])

dz_cmap       = matplotlib.colors.ListedColormap([cb_colors.green5, cb_colors.green4, cb_colors.green3, cb_colors.orange2, cb_colors.orange4, cb_colors.orange6, cb_colors.red6, cb_colors.red4, cb_colors.purple3, cb_colors.purple5])

wind_cmap = matplotlib.colors.ListedColormap([cb_colors.gray1, cb_colors.gray2, cb_colors.gray3, cb_colors.orange2, cb_colors.orange3, cb_colors.orange4, cb_colors.orange5, cb_colors.orange6, cb_colors.orange7, cb_colors.orange8])

wz_cmap_extend = matplotlib.colors.ListedColormap([cb_colors.blue2, cb_colors.blue3, cb_colors.blue4, cb_colors.red2, cb_colors.red3, cb_colors.red4, cb_colors.red5, cb_colors.red6, cb_colors.red7, cb_colors.purple7, cb_colors.purple6, cb_colors.purple5, cb_colors.purple4])

cape_cmap = matplotlib.colors.ListedColormap([cb_colors.blue2, cb_colors.blue3, cb_colors.blue4, cb_colors.orange2, cb_colors.orange3, cb_colors.orange4, cb_colors.orange5, cb_colors.red4, cb_colors.red5, cb_colors.red6, cb_colors.red7, cb_colors.purple7, cb_colors.purple6, cb_colors.purple5])

td_cmap_ncar = matplotlib.colors.ListedColormap(['#ad598a', '#c589ac','#dcb8cd','#e7cfd1','#d0a0a4','#ad5960', '#8b131d', '#8b4513','#ad7c59', '#c5a289','#dcc7b8','#eeeeee', '#dddddd', '#bbbbbb', '#e1e1d3', '#e1d5b1','#ccb77a','#ffffe5','#f7fcb9', '#addd8e', '#41ab5d', '#006837', '#004529', '#195257', '#4c787c'])

temp_cmap = matplotlib.colors.ListedColormap([cb_colors.purple4, cb_colors.purple5, cb_colors.purple6, cb_colors.purple7, cb_colors.blue8, cb_colors.blue7, cb_colors.blue6, cb_colors.blue5, cb_colors.blue4, cb_colors.blue3, cb_colors.green7, cb_colors.green6, cb_colors.green5, cb_colors.green4, cb_colors.green3, cb_colors.green2, cb_colors.orange2, cb_colors.orange3, cb_colors.orange4, cb_colors.orange5, cb_colors.red5, cb_colors.red6, cb_colors.red7, cb_colors.red8, cb_colors.purple3, cb_colors.purple4, cb_colors.purple5, cb_colors.purple6])  

#################################### Basemap Variables:  #####################################################

resolution 	= 'h'
area_thresh 	= 1000.

damage_files = '' #['/Volumes/fast_scr/pythonletkf/vortex_se/2013-11-17/shapefiles/extractDamage_11-17/extractDamagePaths']

#################################### Contour Levels:  #####################################################

prob_levels    		= np.arange(0.1,1.1,0.1)		#(%)
wz_levels      	 	= np.arange(0.002,0.01175,0.00075)	#(s^-1)
uh2to5_levels 	  	= np.arange(40.,560.,40.)		#(m^2 s^-2)
uh0to2_levels  		= np.arange(15.,210.,15.)		#(m^2 s^-2)
cape_levels     	= np.arange(250.,4000.,250.)		#(J Kg^-1)
cin_levels      	= np.arange(-200.,25.,25.)		#(J Kg^-1)
temp_levels             = np.arange(-20., 125., 5.)
td_levels		= np.arange(-16., 88., 4.)		#(deg F)
ws_levels_low       	= np.arange(5.,35.,3.)			#(m s^-1)
ws_levels_high       	= np.arange(15.,45.,3.)			#(m s^-1)
srh_levels      	= np.arange(40.,640.,40.)		#(m^2 s^-2)
stp_levels      	= np.arange(0.25,7.75,0.5)		#(unitless)
rain_levels             = np.arange(0.0,2.25,0.15)              #(in)
dz_levels       	= np.arange(5.0,75.,5.)			#(dBZ)
dz_levels2       	= np.arange(20.0,75.,5.)		#(dBZ)

pmm_dz_levels 		= [35., 50.]				#(dBZ) 
pmm_dz_colors_gray	= [cb_colors.gray8, cb_colors.gray8]	#gray contours
#pmm_dz_colors_blue	= [cb_colors.blue6, cb_colors.blue8]	#blue contours
#pmm_dz_colors_red	= [cb_colors.red6, cb_colors.red8]	#red contours
#pmm_dz_colors_green	= [cb_colors.green6, cb_colors.green8]	#red contours

td_spec_levels          = [60.]					#extra contours for important Td levels
td_spec_colors          = [cb_colors.purple6]			#colors for important Td levels

#################################### Initialize plot attributes using 'web plot' objects:  #####################################################

ws_plot = web_plot('',                   \
                   '',			\
                   'Probability Matched Mean - Composite Reflectivity (dBZ)',                   \
                   cb_colors.gray6,      \
                   '',            \
                   pmm_dz_levels,        \
                   '',                   \
                   '',                   \
                   pmm_dz_colors_gray,        \
                   cb_colors.orange9,		\
                   'none',		\
                   wind_cmap,              \
                   'max',                \
                   plot_alpha,               \
                   neighborhood)  

dz_plot = web_plot('',                   \
                   '',			\
                   '',                   \
                   cb_colors.gray6,      \
                   dz_levels2,            \
                   [80., 90.],        \
                   '',                   \
                   '',                   \
                   ['none', 'none'],        \
                   cb_colors.purple7,		\
                   'none',		\
                   dz_cmap,              \
                   'max',                \
                   plot_alpha,               \
                   neighborhood)  

rain_plot = web_plot('rain',                 \
                   'Ens. Mean Accumulated Rainfall (inches)',			\
                   'Probability Matched Mean - Composite Reflectivity (dBZ)',                   \
                   cb_colors.gray6,      \
                   rain_levels,          \
                   pmm_dz_levels,        \
                   '',                   \
                   '',                   \
                   pmm_dz_colors_gray,        \
                   cb_colors.purple8,		\
                   'none',		\
                   cape_cmap,              \
                   'max',                \
                   plot_alpha,               \
                   neighborhood)  

prob_plot = web_plot('',                 \
                   '',			\
                   'Probability Matched Mean - Composite Reflectivity (dBZ)',                   \
                   cb_colors.gray6,      \
                   prob_levels,          \
                   pmm_dz_levels,        \
                   '',                   \
                   '',                   \
                   pmm_dz_colors_gray,        \
                   'none',		\
                   'none',		\
                   wz_cmap,              \
                   'neither',            \
                   plot_alpha,               \
                   neighborhood)  

cape_plot = web_plot('mlcape',                 \
                   'Ens. Mean 75 hPa MLCAPE (J Kg$^{-1}$)',			\
                   'Probability Matched Mean - Composite Reflectivity (dBZ)',                   \
                   cb_colors.gray6,      \
                   cape_levels,          \
                   pmm_dz_levels,        \
                   '',                   \
                   '',                   \
                   pmm_dz_colors_gray,        \
                   cb_colors.purple3,		\
                   'none',		\
                   cape_cmap,              \
                   'max',                \
                   plot_alpha,               \
                   neighborhood)  

cin_plot = web_plot('mlcin',                  \
                   'Ens. Mean 75 hPa MLCIN (J Kg$^{-1}$)',			\
                   'Probability Matched Mean - Composite Reflectivity (dBZ)',                   \
                   cb_colors.gray6,      \
                   cin_levels,           \
                   pmm_dz_levels,        \
                   '',                   \
                   '',                   \
                   pmm_dz_colors_gray,        \
                   cb_colors.gray1,		\
                   cb_colors.purple8,		\
                   cin_cmap,             \
                   'both',               \
                   plot_alpha,               \
                   neighborhood)  

stp_plot = web_plot('stp',                  \
                   'Ens. Mean Significant Tornado Parameter',			\
                   'Probability Matched Mean - Composite Reflectivity (dBZ)',                   \
                   cb_colors.gray6,      \
                   stp_levels,           \
                   pmm_dz_levels,        \
                   '',                   \
                   '',                   \
                   pmm_dz_colors_gray,        \
                   cb_colors.purple3,		\
                   'none',		\
                   cape_cmap,              \
                   'max',                \
                   plot_alpha,               \
                   neighborhood)  

srh_plot = web_plot('',                  \
                   '',			\
                   'Probability Matched Mean - Composite Reflectivity (dBZ)',                   \
                   cb_colors.gray6,      \
                   srh_levels,           \
                   pmm_dz_levels,        \
                   '',                   \
                   '',                   \
                   pmm_dz_colors_gray,        \
                   cb_colors.purple3,		\
                   'none',		\
                   cape_cmap,              \
                   'max',                \
                   plot_alpha,               \
                   neighborhood)  

temp_plot = web_plot('temp',                 \
                   'Ens. Mean 2 m Temperature ($^{\circ}$F)',			\
                   'Probability Matched Mean - Composite Reflectivity (dBZ)',                   \
                   cb_colors.gray6,      \
                   temp_levels,          \
                   pmm_dz_levels,        \
                   '',                   \
                   '',                   \
                   pmm_dz_colors_gray,        \
                   'none',		\
                   'none',		\
                   temp_cmap,              \
                   'neither',               \
                   plot_alpha,               \
                   neighborhood)  

td_plot = web_plot('td',                   \
                   'Ens. Mean 2 m Dewpoint Temp ($^{\circ}$F)',			\
                   'Probability Matched Mean - Composite Reflectivity (dBZ)',                   \
                   cb_colors.gray6,      \
                   td_levels,            \
                   pmm_dz_levels,        \
                   td_spec_levels,                   \
                   td_spec_colors,                   \
                   pmm_dz_colors_gray,        \
                   'none',		\
                   'none',		\
                   td_cmap_ncar,              \
                   'neither',               \
                   plot_alpha,               \
                   neighborhood)  

######################################################################################################
#################################### Read Data:  #####################################################
######################################################################################################

################### Read probability matched mean composite reflectivity: ############################

pmm_file = os.path.join(exp_dir, 'pmm_dz.nc')
try:
   fin = netCDF4.Dataset(pmm_file, "r")
   print("Opening %s \n" % pmm_file)
except:
   print("%s does not exist! \n" % pmm_file)
   sys.exit(1)

pmm_dz = fin.variables['PMM_DZ'][t,:,:] 
fin.close()
del fin

##################### Get list of summary files to process: ##############################

files = []
files_temp = os.listdir(exp_dir)
for f, file in enumerate(files_temp):
   if (file[0] == '2'):
      files.append(file)

files.sort()
ne = len(files)  #number of ensemble members

############### for each ensemble member summary file: #############################

for f, file in enumerate(files):
   exp_file = os.path.join(exp_dir, file)
   try:
      fin = netCDF4.Dataset(exp_file, "r")
      print("Opening %s \n" % exp_file)
   except:
      print("%s does not exist! \n" % exp_file)
      sys.exit(1)

############## Get grid/forecast time information from first summary file: ##################

   if (f == 0): 

############# Get/process date/time info, handle 00Z shift for day but not month/year ############

      date = file[0:10]
      init_label = 'Init: ' + date + ', ' + file[11:13] + file[14:16] + ' UTC'
      init_hr = int(file[11:13])
 
      time = fin.variables['TIME'][t]
      if (time is ma.masked):
         time = fin.variables['TIME'][t-1] + 300. 

      valid_hour = np.floor(time / 3600.)
      valid_min = np.floor((time - valid_hour * 3600.) / 60.)

      if (valid_hour > 23):
         valid_hour = valid_hour - 24
         if (init_hr > 20):
            temp_day = int(date[-2:])+1
            temp_day = str(temp_day)
            if (len(temp_day) == 1):
               temp_day = '0' + temp_day
            date = date[:-2] + temp_day

      valid_hour = str(int(valid_hour))
      valid_min = str(int(valid_min))

      if (len(valid_hour) == 1):
         valid_hour = '0' + valid_hour
      if (len(valid_min) == 1):
         valid_min = '0' + valid_min

      valid_label = 'Valid: ' + date + ', ' + valid_hour + valid_min + ' UTC'

############ Get grid/projection info, chop 'edge' from boundaries: ####################

      xlat = fin.variables['XLAT'][:]
      xlon = fin.variables['XLON'][:]

      xlat = xlat[edge:-edge,edge:-edge]
      xlon = xlon[edge:-edge,edge:-edge]

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

      u_10 = np.zeros((ne, xlat.shape[0], xlat.shape[1]))
      v_10 = np.zeros((ne, xlat.shape[0], xlat.shape[1]))
      ws_max_10 = np.zeros((ne, xlat.shape[0], xlat.shape[1]))
      u_500 = np.zeros((ne, xlat.shape[0], xlat.shape[1]))
      v_500 = np.zeros((ne, xlat.shape[0], xlat.shape[1]))
      
      lfc_ml = np.zeros((ne, xlat.shape[0], xlat.shape[1]))
      cape_ml = np.zeros((ne, xlat.shape[0], xlat.shape[1]))
      cin_ml = np.zeros((ne, xlat.shape[0], xlat.shape[1]))
      stp_ml = np.zeros((ne, xlat.shape[0], xlat.shape[1]))
      bunk_r_u = np.zeros((ne, xlat.shape[0], xlat.shape[1]))
      bunk_r_v = np.zeros((ne, xlat.shape[0], xlat.shape[1]))
      srh_0to1 = np.zeros((ne, xlat.shape[0], xlat.shape[1]))
      srh_0to3 = np.zeros((ne, xlat.shape[0], xlat.shape[1]))
      shear_u_0to6 = np.zeros((ne, xlat.shape[0], xlat.shape[1]))
      shear_v_0to6 = np.zeros((ne, xlat.shape[0], xlat.shape[1]))
      shear_u_0to1 = np.zeros((ne, xlat.shape[0], xlat.shape[1]))
      shear_v_0to1 = np.zeros((ne, xlat.shape[0], xlat.shape[1]))

      dz = np.zeros((ne, xlat.shape[0], xlat.shape[1]))
      dz_2km = np.zeros((ne, xlat.shape[0], xlat.shape[1])) ###actually reading in 1km DZ for 2016 -> will be 2km in future

      t_2 = np.zeros((ne, xlat.shape[0], xlat.shape[1]))
      p_2 = np.zeros((ne, xlat.shape[0], xlat.shape[1]))
      qv_2 = np.zeros((ne, xlat.shape[0], xlat.shape[1]))

######################## Read in summary file variables: ################################

   u_10[f,:,:] = fin.variables['U_10'][t,edge:-edge,edge:-edge]
   v_10[f,:,:] = fin.variables['V_10'][t,edge:-edge,edge:-edge]
   ws_max_10[f,:,:] = fin.variables['WS_MAX_10'][t,edge:-edge,edge:-edge]
   u_500[f,:,:] = fin.variables['U_500'][t,edge:-edge,edge:-edge]
   v_500[f,:,:] = fin.variables['V_500'][t,edge:-edge,edge:-edge]

   lfc_ml[f,:,:] = fin.variables['LFC_ML'][t,edge:-edge,edge:-edge]
   cape_ml[f,:,:] = fin.variables['CAPE_ML'][t,edge:-edge,edge:-edge]
   cin_ml[f,:,:] = fin.variables['CIN_ML'][t,edge:-edge,edge:-edge]
   stp_ml[f,:,:] = fin.variables['STP_ML'][t,edge:-edge,edge:-edge]
   bunk_r_u[f,:,:] = fin.variables['BUNK_R_U'][t,edge:-edge,edge:-edge]
   bunk_r_v[f,:,:] = fin.variables['BUNK_R_V'][t,edge:-edge,edge:-edge]
   srh_0to1[f,:,:] = fin.variables['SRH_0TO1'][t,edge:-edge,edge:-edge]
   srh_0to3[f,:,:] = fin.variables['SRH_0TO3'][t,edge:-edge,edge:-edge]
   shear_u_0to6[f,:,:] = fin.variables['SHEAR_U_0TO6'][t,edge:-edge,edge:-edge]
   shear_v_0to6[f,:,:] = fin.variables['SHEAR_V_0TO6'][t,edge:-edge,edge:-edge]
   shear_u_0to1[f,:,:] = fin.variables['SHEAR_U_0TO1'][t,edge:-edge,edge:-edge]
   shear_v_0to1[f,:,:] = fin.variables['SHEAR_V_0TO1'][t,edge:-edge,edge:-edge]

   dz[f,:,:] = fin.variables['DZ_COMP'][t,edge:-edge,edge:-edge]
#   dz_2km[f,:,:] = fin.variables['DZ_1KM'][t,edge:-edge,edge:-edge]
   dz_2km[f,:,:] = fin.variables['DZ_2KM'][t,edge:-edge,edge:-edge] ##toggle for changing from 1km agl to 2km agl at end of 2016
   dz_max[f,:,:] = fin.variables['DZ_MAX'][t,edge:-edge,edge:-edge] ##toggle for changing from 1km agl to 2km agl at end of 2016

   t_2[f,:,:] = fin.variables['T_2'][t,edge:-edge,edge:-edge]
   p_2[f,:,:] = fin.variables['P_SFC'][t,edge:-edge,edge:-edge]
   qv_2[f,:,:] = fin.variables['QV_2'][t,edge:-edge,edge:-edge]

   fin.close()
   del fin

######################## check for missing/unrealistic values and replace with zeros: #############################################

u_10 = np.where(u_10 > 100000., 0., u_10)
v_10 = np.where(v_10 > 100000., 0., v_10)
ws_max_10 = np.where(ws_max_10 > 100000., 0., ws_max_10)
u_500 = np.where(u_500 > 100000., 0., u_500)
v_500 = np.where(v_500 > 100000., 0., v_500)

cape_ml = np.where(cape_ml > 100000., 0., cape_ml)
cin_ml = np.where(cin_ml > 100000., 0., cin_ml)
stp_ml = np.where(stp_ml > 100000., 0., stp_ml)
bunk_r_u = np.where(bunk_r_u > 100000., 0., bunk_r_u)
bunk_r_v = np.where(bunk_r_v > 100000., 0., bunk_r_v)

srh_0to1 = np.where(srh_0to1 > 100000., 0., srh_0to1)
srh_0to3 = np.where(srh_0to3 > 100000., 0., srh_0to3)
shear_u_0to6 = np.where(shear_u_0to6 > 100000., 0., shear_u_0to6)
shear_v_0to6 = np.where(shear_v_0to6 > 100000., 0., shear_v_0to6)
shear_u_0to1 = np.where(shear_u_0to1 > 100000., 0., shear_u_0to1)
shear_v_0to1 = np.where(shear_v_0to1 > 100000., 0., shear_v_0to1)

dz = np.where(dz > 100000., 0., dz)
dz_2km = np.where(dz_2km > 100000., 0., dz_2km)

t_2 = np.where(t_2 > 100000., 0., t_2)
p_2 = np.where(p_2 > 100000., 0., p_2)
qv_2 = np.where(qv_2 > 100000., 0., qv_2)

qv_2 = np.where(qv_2 < 0., 0., qv_2)
td_2 = calc_td(t_2, p_2, qv_2)
ws_500 = np.sqrt(u_500**2 + v_500**2)

##################### Mask CIN/STP values where there is no LFC (inclusion can artificially lower ensemble mean) #####################

masked_cin_ml = np.ma.masked_where((lfc_ml == 0.), (cin_ml))
masked_stp_ml = np.ma.masked_where((lfc_ml == 0.), (stp_ml))

##################### Calculate ensemble mean values: ###################################

mean_cape = np.mean(cape_ml, axis=0)
mean_cin = np.mean(masked_cin_ml, axis=0)
mean_stp = np.mean(masked_stp_ml, axis=0)
mean_bunk_r_u = np.mean(bunk_r_u, axis=0)
mean_bunk_r_v = np.mean(bunk_r_v, axis=0)
mean_srh_0to1 = np.mean(srh_0to1, axis=0)
mean_srh_0to3 = np.mean(srh_0to3, axis=0)
mean_shear_u_0to6 = np.mean(shear_u_0to6, axis=0)
mean_shear_v_0to6 = np.mean(shear_v_0to6, axis=0)
mean_shear_u_0to1 = np.mean(shear_u_0to1, axis=0)
mean_shear_v_0to1 = np.mean(shear_v_0to1, axis=0)

mean_dz_max = np.mean(dz_max, axis=0)
mean_dz_2km = np.mean(dz_2km, axis=0)
mean_dz = np.mean(dz, axis=0)
mean_u_10 = np.mean(u_10, axis=0)
mean_v_10 = np.mean(v_10, axis=0)
mean_u_500 = np.mean(u_500, axis=0)
mean_v_500 = np.mean(v_500, axis=0)

mean_ws_10 = np.mean(ws_max_10, axis=0)
mean_ws_500 = np.sqrt(mean_u_500**2 + mean_v_500**2)
mean_shear_0to6 = np.sqrt(mean_shear_u_0to6**2 + mean_shear_v_0to6**2)
mean_shear_0to1 = np.sqrt(mean_shear_u_0to1**2 + mean_shear_v_0to1**2)
mean_bunk = np.sqrt(mean_bunk_r_u**2 + mean_bunk_r_v**2)

mean_t_2 = np.mean(t_2, axis=0)
mean_td_2 = np.mean(td_2, axis=0)

mean_tf_2 = (mean_t_2 - 273.15) * 1.8 + 32.     #convert to deg. F
mean_tdf_2 = (mean_td_2 - 273.15) * 1.8 + 32.     #convert to deg. F

#################### Thin arrays used for quiver plots (removes extra vectors): #################################

quiv_xlon = xlon[0:-1:thin,0:-1:thin]
quiv_xlat = xlat[0:-1:thin,0:-1:thin]
quiv_u_10 = mean_u_10[0:-1:thin,0:-1:thin]
quiv_v_10 = mean_v_10[0:-1:thin,0:-1:thin]
quiv_u_500 = mean_u_500[0:-1:thin,0:-1:thin]
quiv_v_500 = mean_v_500[0:-1:thin,0:-1:thin]
quiv_shear0to6_u = mean_shear_u_0to6[0:-1:thin,0:-1:thin]
quiv_shear0to6_v = mean_shear_v_0to6[0:-1:thin,0:-1:thin]
quiv_shear0to1_u = mean_shear_u_0to1[0:-1:thin,0:-1:thin]
quiv_shear0to1_v = mean_shear_v_0to1[0:-1:thin,0:-1:thin]
quiv_bunk_u = mean_bunk_r_u[0:-1:thin,0:-1:thin]
quiv_bunk_v = mean_bunk_r_v[0:-1:thin,0:-1:thin]

################################# Make Figure Template: ###################################################

print('basemap part')

map, fig, ax1, ax2, ax3 = create_fig(sw_lat_full, sw_lon_full, ne_lat_full, ne_lon_full, true_lat1, true_lat2, cen_lat, stand_lon, damage_files, resolution, area_thresh)

x, y = list(map(xlon[:], xlat[:]))
xx, yy = map.makegrid(xlat.shape[1], xlat.shape[0], returnxy=True)[2:4]   #equidistant x/y grid for streamline plots

##########################################################################################################
###################################### Make Plots: #######################################################
##########################################################################################################

print('plot part')

################# Rotate quiver values according to map projection: #####################

q_u, q_v = map.rotate_vector(quiv_u_10[:,:], quiv_v_10[:,:], quiv_xlon, quiv_xlat, returnxy=False)
q_u_500, q_v_500 = map.rotate_vector(quiv_u_500[:,:], quiv_v_500[:,:], quiv_xlon, quiv_xlat, returnxy=False)
shear0to6_q_u, shear0to6_q_v = map.rotate_vector(quiv_shear0to6_u[:,:], quiv_shear0to6_v[:,:], quiv_xlon, quiv_xlat, returnxy=False)
shear0to1_q_u, shear0to1_q_v = map.rotate_vector(quiv_shear0to1_u[:,:], quiv_shear0to1_v[:,:], quiv_xlon, quiv_xlat, returnxy=False)
bunk_q_u, bunk_q_v = map.rotate_vector(quiv_bunk_u[:,:], quiv_bunk_v[:,:], quiv_xlon, quiv_xlat, returnxy=False)

######################## Environment Plots: #####################

env_plot(map, fig, ax1, ax2, ax3, x, y, temp_plot, mean_tf_2[:,:], pmm_dz[:,:], t, init_label, valid_label, domain, outdir, q_u, q_v, 500, 5, 0, spec='False', quiv='True') 

temp_plot.name = 'temp3'
env_plot(map, fig, ax1, ax2, ax3, x, y, temp_plot, mean_tf_2[:,:], pmm_dz[:,:], t, init_label, valid_label, domain, outdir, q_u, q_v, 500, 5, 0, spec='False', quiv='True') 

env_plot(map, fig, ax1, ax2, ax3, x, y, td_plot, mean_tdf_2[:,:], pmm_dz[:,:], t, init_label, valid_label, domain, outdir, q_u, q_v, 500, 5, 0, spec='True', quiv='True') 

td_plot.name = 'td3'
env_plot(map, fig, ax1, ax2, ax3, x, y, td_plot, mean_tdf_2[:,:], pmm_dz[:,:], t, init_label, valid_label, domain, outdir, q_u, q_v, 500, 5, 0, spec='True', quiv='True') 

env_plot(map, fig, ax1, ax2, ax3, x, y, cape_plot, mean_cape[:,:], pmm_dz[:,:], t, init_label, valid_label, domain, outdir, '', '', '', 5, 0, spec='False', quiv='False') 

cape_plot.name = 'mlcape3'
env_plot(map, fig, ax1, ax2, ax3, x, y, cape_plot, mean_cape[:,:], pmm_dz[:,:], t, init_label, valid_label, domain, outdir, '', '', '', 5, 0, spec='False', quiv='False') 

env_plot(map, fig, ax1, ax2, ax3, x, y, cin_plot, mean_cin[:,:], pmm_dz[:,:], t, init_label, valid_label, domain, outdir, '', '', '', 5, 0, spec='False', quiv='False') 

env_plot(map, fig, ax1, ax2, ax3, x, y, stp_plot, mean_stp[:,:], pmm_dz[:,:], t, init_label, valid_label, domain, outdir, '', '', '', 5, 0, spec='False', quiv='False') 

srh_plot.name = 'srh0to1'
srh_plot.var1_title = 'Ens. Mean 0 - 1 km Storm Relative Helicity (m$^{2}$ s$^{-2}$)'
env_plot(map, fig, ax1, ax2, ax3, x, y, srh_plot, mean_srh_0to1[:,:], pmm_dz[:,:], t, init_label, valid_label, domain, outdir, '', '', '', 5, 0, spec='False', quiv='False') 

srh_plot.name = 'srh0to13'
env_plot(map, fig, ax1, ax2, ax3, x, y, srh_plot, mean_srh_0to1[:,:], pmm_dz[:,:], t, init_label, valid_label, domain, outdir, '', '', '', 5, 0, spec='False', quiv='False') 

srh_plot.name = 'srh0to3'
srh_plot.var1_title = 'Ens. Mean 0 - 3 km Storm Relative Helicity (m$^{2}$ s$^{-2}$)'
env_plot(map, fig, ax1, ax2, ax3, x, y, srh_plot, mean_srh_0to3[:,:], pmm_dz[:,:], t, init_label, valid_label, domain, outdir, '', '', '', 5, 0, spec='False', quiv='False') 

ws_plot.name = 'ws_500'
ws_plot.var1_title = 'Ens. Mean 500 m Wind Speed (m s$^{-1}$)'
ws_plot.var1_levels = ws_levels_low 
env_plot(map, fig, ax1, ax2, ax3, x, y, ws_plot, mean_ws_500[:,:], pmm_dz[:,:], t, init_label, valid_label, domain, outdir, q_u_500, q_v_500, 1000, 5, 0, spec='False', quiv='True') 

ws_plot.name = 'shear0to6'
ws_plot.var1_title = 'Ens. Mean 0 - 6 km Shear (m s$^{-1}$)'
ws_plot.var1_levels = ws_levels_high 
env_plot(map, fig, ax1, ax2, ax3, x, y, ws_plot, mean_shear_0to6[:,:], pmm_dz[:,:], t, init_label, valid_label, domain, outdir, shear0to6_q_u, shear0to6_q_v, 1000, 5, 0, spec='False', quiv='True') 

ws_plot.name = 'shear0to1'
ws_plot.var1_title = 'Ens. Mean 0 - 1 km Shear (m s$^{-1}$)'
ws_plot.var1_levels = ws_levels_low 
env_plot(map, fig, ax1, ax2, ax3, x, y, ws_plot, mean_shear_0to1[:,:], pmm_dz[:,:], t, init_label, valid_label, domain, outdir, shear0to1_q_u, shear0to1_q_v, 1000, 5, 0, spec='False', quiv='True') 

ws_plot.name = 'bunk'
ws_plot.var1_title = 'Ens. Mean Bunkers Storm Motion (m s$^{-1}$)'
ws_plot.var1_levels = ws_levels_low 
env_plot(map, fig, ax1, ax2, ax3, x, y, ws_plot, mean_bunk[:,:], pmm_dz[:,:], t, init_label, valid_label, domain, outdir, bunk_q_u, bunk_q_v, 1000, 5, 0, spec='False', quiv='True')

######################## Reflectivity Plots: #####################

dz_plot.name = 'compdz'
dz_plot.var1_title = 'Probability Matched Mean Composite Reflectivity (dBZ)'
env_plot(map, fig, ax1, ax2, ax3, x, y, dz_plot, pmm_dz[:,:], pmm_dz[:,:], t, init_label, valid_label, domain, outdir, '', '', '', 5, 0, spec='False', quiv='False') 

dz_plot.name = 'compdz3'
env_plot(map, fig, ax1, ax2, ax3, x, y, dz_plot, pmm_dz[:,:], pmm_dz[:,:], t, init_label, valid_label, domain, outdir, '', '', '', 5, 0, spec='False', quiv='False') 

dz_plot.name = 'dz2km'
dz_plot.var1_title = 'Ensemble Mean 1 km AGL Reflectivity (dBZ)'
env_plot(map, fig, ax1, ax2, ax3, x, y, dz_plot, mean_dz_2km[:,:], mean_dz_2km[:,:], t, init_label, valid_label, domain, outdir, '', '', '', 5, 0, spec='False', quiv='False') 

dz_plot.name = 'dzmax'
dz_plot.var1_title = 'Ensemble Mean Column Max Reflectivity (dBZ)'
env_plot(map, fig, ax1, ax2, ax3, x, y, dz_plot, mean_dz_max[:,:], mean_dz_max[:,:], t, init_label, valid_label, domain, outdir, '', '', '', 5, 0, spec='False', quiv='False')

 
