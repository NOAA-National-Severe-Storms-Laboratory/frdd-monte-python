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
import ctables
import news_e_post_cbook
from news_e_post_cbook import *
import news_e_plotting_cbook_v2
from news_e_plotting_cbook_v2 import *

####################################### File Variables: ######################################################

parser = OptionParser()
parser.add_option("-d", dest="exp_dir", type="string", default= None, help="Input Directory (of summary files)")
parser.add_option("-o", dest="outdir", type="string", help = "Output Directory (for images)")
parser.add_option("-n", dest="name", type="string", help = "Name of variable to process")
parser.add_option("-t", dest="max_t", type="int", help = "Maximum timestep available")
parser.add_option("-s", dest="start_t", type="int", help = "First timestep to plot")

(options, args) = parser.parse_args()

if ((options.exp_dir == None) or (options.outdir == None) or (options.name == None) or (options.max_t == None) or (options.start_t == None)):
   print()
   parser.print_help()
   print()
   sys.exit(1)
else:
   exp_dir = options.exp_dir
   outdir = options.outdir
   name = options.name
   max_t = options.max_t
   start_t = options.start_t

domain = 'full'

#################################### User-Defined Variables:  #####################################################

edge            = 7 		#number of grid points to remove from near domain boundaries
thin		= 6		#thinning factor for quiver values (e.g. 6 means slice every 6th grid point)

fcst_len 	= 5400. 	#length of forecast to consider (s)
radius_max      = 9 #3            #grid point radius for maximum value filter
radius_gauss    = 6 #2            #grid point radius of convolution operator
neighborhood 	= 15		#grid point radius of prob matched mean neighborhood

plot_alpha 	= 0.6		#transparency value for filled contour plots

################################### Set plot object values for different variables:  #####################################################

if (name == 'wz0to2'):
   var_name = 'WZ_0TO2'
   var_thresh 	= [0.001, 0.002, 0.003, 0.004, 0.005, 0.006]	   #vertical vorticity thresh (s^-1)
   var_label    = '0-2 km Vertical Vort.'
   var_units    = 's$^{-1}$'
if (name == 'wz2to5'):
   var_name = 'WZ_2TO5'
   var_thresh 	= [0.001, 0.002, 0.003, 0.004, 0.005, 0.006]	   #vertical vorticity thresh (s^-1)
   var_label    = '2-5 km Vertical Vort.'
   var_units    = 's$^{-1}$'
elif (name == 'uh0to2'): 
   var_name = 'UH_0TO2'
   var_thresh 	= [10., 20., 30., 40., 50., 60.]		   #updraft helicity thresh (0-2 km) (m^2 s^-2)
   var_label    = '0-2 km Updraft Hel.'
   var_units    = 'm$^{2}$ s$^{-2}$'
elif (name == 'uh2to5'): 
   var_name = 'UH_2TO5'
   var_thresh 	= [20., 40., 60., 80., 100., 120.]		   #updraft helicity thresh (2-5 km) (m^2 s^-2)
   var_label    = '2-5 km Updraft Hel.'
   var_units    = 'm$^{2}$ s$^{-2}$'
elif (name == 'rain'): 
   var_name = 'RAIN'
   var_thresh   = [0.01, 0.25, 0.5, 0.75, 1., 1.25]                #qpf thresholds (in)
   var_label    = 'Accumulated Rainfall'
   var_units    = 'inches'
elif (name == 'graupmax'): 
   var_name = 'GRAUPEL_MAX'
   var_thresh   = [4., 8., 12., 16., 20., 24.] 		   #graup max thresholds (kg m^-2)
   var_label    = 'Col. Integrated Graupel'
   var_units    = 'Kg m$^{-2}$'
elif (name == 'ws'): 
   var_name = 'WS_MAX_10'
   var_thresh 	= [10., 15., 20., 25., 30., 35.]		   #wind speed thresholds (m s^-1)
   var_label    = '10 m Max Wind Speed'
   var_units    = 'm s$^{-1}$'
elif (name == 'dz'): 
   var_name = 'DZ_COMP'
   var_thresh   = [35., 40., 45., 50., 55., 60.]		   #radar reflectivity thresholds (dBZ)
   var_label    = 'Simulated Composite Reflectivity'
   var_units    = 'dBZ'
elif (name == 'dz1km'): 
   var_name = 'DZ_1KM'
   var_thresh   = [35., 40., 45., 50., 55., 60.]		   #radar reflectivity thresholds (dBZ)
   var_label    = 'Simulated 1 km Reflectivity'
   var_units    = 'dBZ'
elif (name == 'wup'): 
   var_name = 'W_UP_MAX'
   var_thresh 	= [4., 8., 12., 16., 20., 24.]		           #max updraft thresh (m s^-1)
   var_label    = 'Max Updraft'
   var_units    = 'm s$^{-1}$'
elif (name == 'wdn'): 
   var_name = 'W_DN_MAX'
   var_thresh 	= [3., 6., 9., 12., 15., 18.]		   #max downdraft thresh (m s^-1)
   var_label    = 'Min Downdraft'
   var_units    = 'm s$^{-1}$'
elif (name == 'w1km'): 
   var_name = 'W_1KM'
   var_thresh 	= [1.25, 2.5, 3.75, 5., 6.25, 7.5]		           #1km updraft thresh (s^-1)
   var_label    = '1 km Updraft'
   var_units    = 'm s$^{-1}$'

perc            = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]		   #Ens. percentiles to calc/plot

#################################### Define colormaps:  #####################################################

wz_cmap       = matplotlib.colors.ListedColormap([cb_colors.blue2, cb_colors.blue3, cb_colors.blue4, cb_colors.red2, cb_colors.red3, cb_colors.red4, cb_colors.red5, cb_colors.red6, cb_colors.red7])

dz_cmap       = matplotlib.colors.ListedColormap([cb_colors.green5, cb_colors.green4, cb_colors.green3, cb_colors.orange2, cb_colors.orange4, cb_colors.orange6, cb_colors.red6, cb_colors.red4, cb_colors.purple3, cb_colors.purple5])

wind_cmap = matplotlib.colors.ListedColormap([cb_colors.gray1, cb_colors.gray2, cb_colors.gray3, cb_colors.orange2, cb_colors.orange3, cb_colors.orange4, cb_colors.orange5, cb_colors.orange6, cb_colors.orange7])

wz_cmap_extend = matplotlib.colors.ListedColormap([cb_colors.blue2, cb_colors.blue3, cb_colors.blue4, cb_colors.red2, cb_colors.red3, cb_colors.red4, cb_colors.red5, cb_colors.red6, cb_colors.red7, cb_colors.purple7, cb_colors.purple6, cb_colors.purple5])

cape_cmap = matplotlib.colors.ListedColormap([cb_colors.gray1, cb_colors.gray2, cb_colors.gray3, cb_colors.orange3, cb_colors.orange4, cb_colors.orange5, cb_colors.orange6, cb_colors.red7, cb_colors.red6, cb_colors.red5, cb_colors.red4, cb_colors.purple4, cb_colors.purple5, cb_colors.purple6, cb_colors.purple7])

#################################### Basemap Variables:  #####################################################

resolution 	= 'h'
area_thresh 	= 1000.

damage_files = '' #['/Volumes/fast_scr/pythonletkf/vortex_se/2013-11-17/shapefiles/extractDamage_11-17/extractDamagePaths']

#################################### Contour Levels:  #####################################################

prob_levels             = np.arange(0.1,1.1,0.1)                #(%)
wz_levels               = np.arange(0.002,0.01175,0.00075)      #(s^-1)
uh2to5_levels           = np.arange(40.,560.,40.)               #(m^2 s^-2)
uh0to2_levels           = np.arange(15.,210.,15.)               #(m^2 s^-2)
cape_levels             = np.arange(250.,4000.,250.)            #(J Kg^-1)
cin_levels              = np.arange(-325.,0.,25.)               #(J Kg^-1)
temp_levels             = np.arange(40., 103., 3.)              #(deg F)
td_levels               = np.arange(30., 80., 2.)               #(deg F)
ws_levels_low           = np.arange(5.,35.,3.)                  #(m s^-1)
ws_levels_high          = np.arange(15.,45.,3.)                 #(m s^-1)
srh_levels_high         = np.arange(50.,550.,50.)               #(m^2 s^-2)
srh_levels_low          = np.arange(25.,275.,25.)               #(m^2 s^-2)
stp_levels              = np.arange(0.25,7.75,0.5)              #(unitless)
rain_levels             = np.arange(0.0,1.95,0.15)              #(in)
dz_levels               = np.arange(5.0,75.,5.)                 #(dBZ)
dz_levels2              = np.arange(20.0,75.,5.)                #(dBZ)
wup_levels              = np.arange(3.,42.,3.)
graup_levels            = np.arange(3.,42.,3.)

pmm_dz_levels           = [35., 50.]                            #(dBZ) 
pmm_dz_colors_gray      = [cb_colors.gray8, cb_colors.gray8]    #gray contours
#pmm_dz_colors_blue      = [cb_colors.blue6, cb_colors.blue8]    #blue contours
#pmm_dz_colors_red       = [cb_colors.red6, cb_colors.red8]      #red contours
#pmm_dz_colors_green     = [cb_colors.green6, cb_colors.green8]  #red contours

td_spec_levels          = [60.]                                 #extra contours for important Td levels
td_spec_colors          = [cb_colors.purple6]                   #colors for important Td levels

#################################### Initialize plot attributes using 'web plot' objects:  #####################################################

base_plot = web_plot('',									\
                     '', 									\
                     'Probability Matched Mean - Composite Reflectivity (dBZ)',                 \
                     cb_colors.gray6,    							\
                     '',           						         	\
                     pmm_dz_levels,       							\
                     '',                  							\
                     '',                  							\
                     pmm_dz_colors_gray,     							\
                     cb_colors.purple4,              						\
                     'none',              							\
                     wz_cmap_extend,            							\
                     'max',               							\
                   plot_alpha,               \
                   neighborhood)

if (name[0:2] == 'wz'):
   base_plot.var1_levels = wz_levels
elif (name == 'uh0to2'):
   base_plot.var1_levels = uh0to2_levels
elif (name == 'uh2to5'):
   base_plot.var1_levels = uh2to5_levels
elif (name == 'rain'):
   base_plot.var1_levels = rain_levels
elif (name == 'graupmax'):
   base_plot.var1_levels = graup_levels
elif (name == 'wup'):
   base_plot.var1_levels = wup_levels
elif (name == 'ws'):
   base_plot.var1_levels = ws_levels_low
   base_plot.cmap = wind_cmap  
   base_plot.over_color = cb_colors.orange8 
elif (name[0:2] == 'dz'):
   base_plot.var1_levels = dz_levels2
   base_plot.var2_tcolor = 'none'
   base_plot.var2_levels = [100., 110.]       #hack so nothing is plotted
   base_plot.var2_colors = ['none', 'none']
   base_plot.cmap = dz_cmap
   base_plot.over_color = cb_colors.purple7

prob_plot = web_plot('',                                                                     \
                   '',		                                                             \
                   'Probability Matched Mean - Composite Reflectivity (dBZ)',                \
                   cb_colors.gray6,                                                         \
                   prob_levels,                                                              \
                   pmm_dz_levels,                                                            \
                   '',                                                                       \
                   '',                                                                       \
                   pmm_dz_colors_gray,                                                          \
                   'none',	                                                             \
                   'none',		                                                     \
                   wz_cmap,                                                                 \
                   'neither',                                                                \
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
   
pmm_dz = fin.variables['PMM_DZ'][0:max_t,:,:]   
fin.close()
del fin

##################### Get list of summary files to process: ##############################

files = []
files_temp = os.listdir(exp_dir)
for f, file in enumerate(files_temp):
   if (file[0] == '2'):
      files.append(file)

files.sort()
ne = len(files)

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
      init_date = date
      init_label = 'Init: ' + date + ', ' + file[11:13] + file[14:16] + ' UTC'
      init_hr = int(file[11:13])

      time = fin.variables['TIME'][:]
      if (time is ma.masked):
         time = fin.variables['TIME'][t-1] + 300.

      valid_hour = np.floor(time[(max_t-1)] / 3600.)
      valid_min = np.floor((time[(max_t-1)] - valid_hour * 3600.) / 60.)


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

      var = np.zeros((ne, len(time), xlat.shape[0], xlat.shape[1]))
      dz = np.zeros((ne, len(time), xlat.shape[0], xlat.shape[1]))

######################## Read in summary file variables: ################################

   var[f,:,:] = fin.variables[var_name][:,edge:-edge,edge:-edge]
   dz[f,:,:] = fin.variables['DZ_COMP'][:,edge:-edge,edge:-edge]

   fin.close()
   del fin

######################## check for missing/unrealistic values and replace with zeros: #############################################

var = np.where(var > 100000., 0., var)
dz = np.where(dz > 100000., 0., dz)

if (name == 'rain'): #hack so only rain values > trace are plotted 
   var = np.where(var <= 0.01, -1., var)

################# Process forecast initialization time: #####################################

start_hour = np.floor(time[0] / 3600.)
start_min = np.floor((time[0] - start_hour * 3600.) / 60.)

str_hour = str(int(start_hour))
str_min = str(int(start_min))

if (len(str_hour) == 1):
   str_hour = '0' + str_hour
if (len(str_min) == 1):
   str_min = '0' + str_min

##################### Calculate ensemble mean values: ###################################

mean_var = np.mean(var, axis=0)
mean_dz = np.mean(dz, axis=0)

#################################### Calc Swaths:  #####################################################

print('swath part')

########################################################################################################
### If updraft helicity, vertical vorticity, or updraft plot, run maximum value and convolution filters
### over the raw data to spread and smooth the data
########################################################################################################

if ((name[0:2] == 'wz') or (name[0:2] == 'uh') or (name == 'w1km')):
   var_convolve_temp = var * 0.
   var_convolve = var * 0.

   kernel = gauss_kern(radius_gauss)

   for n in range(0, var.shape[0]):
      for t in range(0, max_t):
         var_convolve_temp[n,t,:,:] = get_local_maxima2d(var[n,t,:,:], radius_max)
         var_convolve[n,t,:,:] = signal.convolve2d(var_convolve_temp[n,t,:,:], kernel, 'same')

#################################### Calc Prob Matched Mean:  #############################################

print('prob match mean part') 

if ((name[0:2] == 'wz') or (name[0:2] == 'uh') or (name == 'w1km')):
   vartempswath = np.max(var_convolve, axis=1)
   vartempswath_90 = np.max(var_convolve[:,0:19,:,:], axis=1)
else:
   vartempswath = np.max(var, axis=1)
   vartempswath_90 = np.max(var[:,0:19,:,:], axis=1)

varpmmswath = mean_var[0,:,:] * 0. 
varpmmswath_90 = mean_var[0,:,:] * 0.

varpmmswath = prob_match_mean(vartempswath, np.mean(vartempswath, axis=0), neighborhood)
varpmmswath_90 = prob_match_mean(vartempswath_90, np.mean(vartempswath_90, axis=0), neighborhood)

################################# Make Figure Template: ###################################################

print('basemap part')

map, fig, ax1, ax2, ax3 = create_fig(sw_lat_full, sw_lon_full, ne_lat_full, ne_lon_full, true_lat1, true_lat2, cen_lat, stand_lon, damage_files, resolution, area_thresh)

x, y = list(map(xlon[:], xlat[:]))
xx, yy = map.makegrid(xlat.shape[1], xlat.shape[0], returnxy=True)[2:4]   #equidistant x/y grid for streamline plots

###################################### Calc Percentiles: ###############################################

varperc = np.zeros((len(perc), var.shape[1], var.shape[2], var.shape[3]))

for i in range(0, len(perc)):
   if ((name[0:2] == 'wz') or (name[0:2] == 'uh')):
      varperc[i,:,:,:] = np.percentile(var_convolve, perc[i], axis=0)
   else: 
      varperc[i,:,:,:] = np.percentile(var, perc[i], axis=0)

##################################### Calc Probabilities: #############################################

###NOTE -> requires all thresh lists to be same length

varprob = np.zeros((len(var_thresh), var.shape[1], var.shape[2], var.shape[3]))

for i in range(0, len(var_thresh)):
   if ((name[0:2] == 'wz') or (name[0:2] == 'uh')):
      varprob[i,:,:,:] = calc_prob(var_convolve, var_thresh[i])
   else: 
      varprob[i,:,:,:] = calc_prob(var, var_thresh[i])

################################################################################################################################
############## If forecast length > 90 minutes, calculate second set of prob/perc swaths for 90-min forecast: ##################
################################################################################################################################

varperc_90 = np.zeros((len(perc), 19, var.shape[2], var.shape[3]))
varprob_90 = np.zeros((len(var_thresh), 19, var.shape[2], var.shape[3]))

if (max_t > 18): 

###################################### Calc Percentiles: ###############################################

   for i in range(0, len(perc)):
      if ((name[0:2] == 'wz') or (name[0:2] == 'uh')):
         varperc_90[i,:,:,:] = np.percentile(var_convolve[:,0:19,:,:], perc[i], axis=0)
      else:
         varperc_90[i,:,:,:] = np.percentile(var[:,0:19,:,:], perc[i], axis=0)

##################################### Calc Probabilities: #############################################

###NOTE -> requires all thresh lists to be same length

   for i in range(0, len(var_thresh)):
      if ((name[0:2] == 'wz') or (name[0:2] == 'uh')):
         varprob_90[i,:,:,:] = calc_prob(var_convolve[:,0:19,:,:], var_thresh[i])
      else:
         varprob_90[i,:,:,:] = calc_prob(var[:,0:19,:,:], var_thresh[i])

##########################################################################################################
###################################### Make Plots: #######################################################
##########################################################################################################

print('plot part')

####### Summary Swath Plots: ########

if (max_t > 18): #plot separate summary swaths for 90- and 180-min forecasts

   varswath_90 = np.max(varprob_90, axis=1)
   varswath = np.max(varprob, axis=1)

   for i in range(0, len(var_thresh)):
      thresh = str(var_thresh[i])
      prob_plot.name = name + 'probswath3'
      prob_plot.var1_title = 'Probability of ' + var_label + ' > %s ' % thresh
      prob_plot.var1_title = prob_plot.var1_title + var_units 
      env_plot(map, fig, ax1, ax2, ax3, x, y, prob_plot, varswath[i,:,:], pmm_dz[0,:,:], i, init_label, valid_label, domain, outdir, '', '', '', 1, 1, spec='False', quiv='False')

   for i in range(0, len(var_thresh)):
      thresh = str(var_thresh[i])
      prob_plot.name = name + 'probswath'
      prob_plot.var1_title = 'Probability of ' + var_label + ' > %s ' % thresh
      prob_plot.var1_title = prob_plot.var1_title + var_units 
      env_plot(map, fig, ax1, ax2, ax3, x, y, prob_plot, varswath_90[i,:,:], pmm_dz[0,:,:], i, init_label, valid_label, domain, outdir, '', '', '', 1, 1, spec='False', quiv='False')

   varperswath_90 = np.max(varperc_90, axis=1)
   varperswath = np.max(varperc, axis=1)

   for i in range(0, len(perc)):
      temp = str(perc[i])
      base_plot.name = name + 'percswath3'
      base_plot.var1_title = 'Ens. %sth Percentile Value of ' % temp
      base_plot.var1_title = base_plot.var1_title + var_label + ' (' + var_units + ')' 
      env_plot(map, fig, ax1, ax2, ax3, x, y, base_plot, varperswath[i,:,:], pmm_dz[0,:,:], i, init_label, valid_label, domain, outdir, '', '', '', 10, 0, spec='False', quiv='False', showmax='True')

   for i in range(0, len(perc)):
      temp = str(perc[i])
      base_plot.name = name + 'percswath'
      base_plot.var1_title = 'Ens. %sth Percentile Value of ' % temp
      base_plot.var1_title = base_plot.var1_title + var_label + ' (' + var_units + ')'
      env_plot(map, fig, ax1, ax2, ax3, x, y, base_plot, varperswath_90[i,:,:], pmm_dz[0,:,:], i, init_label, valid_label, domain, outdir, '', '', '', 10, 0, spec='False', quiv='False', showmax='True')

   base_plot.name = name + 'pmmswath3'
   base_plot.var1_title = 'Prob. Matched Mean Value of ' + var_label + ' (' + var_units + ')'
   env_plot(map, fig, ax1, ax2, ax3, x, y, base_plot, varpmmswath, pmm_dz[0,:,:], 0, init_label, valid_label, domain, outdir, '', '', '', 0, 0, spec='False', quiv='False', showmax='True')

   base_plot.name = name + 'pmmswath'
   base_plot.var1_title = 'Prob. Matched Mean Value of ' + var_label + ' (' + var_units + ')'
   env_plot(map, fig, ax1, ax2, ax3, x, y, base_plot, varpmmswath_90, pmm_dz[0,:,:], 0, init_label, valid_label, domain, outdir, '', '', '', 0, 0, spec='False', quiv='False', showmax='True')

else: #just plot what you've got if 90-min forecast

   varswath = np.max(varprob, axis=1)

   for i in range(0, len(var_thresh)):
      thresh = str(var_thresh[i])
      prob_plot.name = name + 'probswath'
      prob_plot.var1_title = 'Probability of ' + var_label + ' > %s ' % thresh
      prob_plot.var1_title = prob_plot.var1_title + var_units 
      env_plot(map, fig, ax1, ax2, ax3, x, y, prob_plot, varswath[i,:,:], pmm_dz[0,:,:], i, init_label, valid_label, domain, outdir, '', '', '', 1, 1, spec='False', quiv='False')

   varperswath = np.max(varperc, axis=1)

   for i in range(0, len(perc)):
      temp = str(perc[i])
      base_plot.name = name + 'percswath'
      base_plot.var1_title = 'Ens. %sth Percentile Value of ' % temp
      base_plot.var1_title = base_plot.var1_title + var_label + ' (' + var_units + ')'
      env_plot(map, fig, ax1, ax2, ax3, x, y, base_plot, varperswath[i,:,:], pmm_dz[0,:,:], i, init_label, valid_label, domain, outdir, '', '', '', 10, 0, spec='False', quiv='False', showmax='True')

   base_plot.name = name + 'pmmswath3'
   base_plot.var1_title = 'Prob. Matched Mean Value of ' + var_label + ' (' + var_units + ')'
   env_plot(map, fig, ax1, ax2, ax3, x, y, base_plot, varpmmswath, pmm_dz[0,:,:], 0, init_label, valid_label, domain, outdir, '', '', '', 0, 0, spec='False', quiv='False', showmax='True')


######## Swath Accumulation Plots: ########

for t in range(start_t, max_t):
   temp_hour = int(np.floor(time[t] / 3600.))
   temp_min = int(np.floor((time[t] - temp_hour * 3600.) / 60.))

   if (temp_hour > 23):
      temp_hour = temp_hour - 24
      if (init_hr > 20):
         temp_day = int(init_date[-2:])+1
         temp_day = str(temp_day)
         if (len(temp_day) == 1):
            temp_day = '0' + temp_day
         date = date[:-2] + temp_day
   else:
      date = init_date

   temp_hour = str(temp_hour)
   temp_min = str(temp_min)

   if (len(temp_hour) == 1):
      temp_hour = '0' + temp_hour
   if (len(temp_min) == 1):
      temp_min = '0' + temp_min

   valid_label = 'Valid: ' + date + ', ' + temp_hour + temp_min + ' UTC'

######## Rotation Percentile Plots: ########

   if (t == 0):
      var50 = varperc[5,t,:,:]
      var90 = varperc[9,t,:,:]

   elif (t == len(time)):
      var_temp = np.max(varperc, axis=1)

      var50 = var_temp[5,:,:]
      var90 = var_temp[9,:,:]

   else:
      var50 = np.max(varperc[5,:(t+1),:,:], axis=0)
      var90 = np.max(varperc[9,:(t+1),:,:], axis=0)

   base_plot.name = name + '50'
   base_plot.var1_title = 'Ens. 50th Percentile Value of ' + var_label + ' (' + var_units + ')'
   env_plot(map, fig, ax1, ax2, ax3, x, y, base_plot, var50, pmm_dz[t,:,:], t, init_label, valid_label, domain, outdir, '', '', '', 5, 0, spec='False', quiv='False', showmax='True') 

   base_plot.name = name + '90'
   base_plot.var1_title = 'Ens. 90th Percentile Value of ' + var_label + ' (' + var_units + ')'
   env_plot(map, fig, ax1, ax2, ax3, x, y, base_plot, var90, pmm_dz[t,:,:], t, init_label, valid_label, domain, outdir, '', '', '', 5, 0, spec='False', quiv='False', showmax='True') 

######## Probability Plots: ########

   if (t == 0):
      if ((name[0:2] == 'wz') or (name[0:2] == 'uh')):
         varproblow = varprob[2,t,:,:]
         varprobhigh = varprob[4,t,:,:]
         temp_thresh_low = str(var_thresh[2])
         temp_thresh_high = str(var_thresh[4])
      elif (name == 'rain'):
         varproblow = varprob[1,t,:,:]
         varprobhigh = varprob[2,t,:,:]
         varprobinch = varprob[4,t,:,:]
         temp_thresh_low = str(var_thresh[1])
         temp_thresh_high = str(var_thresh[2])
         temp_thresh_inch = str(var_thresh[4])
      elif ((name == 'ws') or (name[0:2] == 'dz') or (name == 'wup') or (name == 'wdn') or (name == 'w1km') or (name == 'graupmax')):
         varproblow = varprob[1,t,:,:]
         varprobhigh = varprob[3,t,:,:]
         temp_thresh_low = str(var_thresh[1])
         temp_thresh_high = str(var_thresh[3])

   elif (t == len(time)):
      var_temp = np.max(varprob, axis=1)

      if ((name[0:2] == 'wz') or (name[0:2] == 'uh')):
         varproblow = var_temp[2,t,:,:]
         varprobhigh = var_temp[4,t,:,:]
         temp_thresh_low = str(var_thresh[2])
         temp_thresh_high = str(var_thresh[4])
      elif (name == 'rain'):
         varproblow = var_temp[1,t,:,:]
         varprobhigh = var_temp[2,t,:,:]
         varprobinch = var_temp[4,t,:,:]
         temp_thresh_low = str(var_thresh[1])
         temp_thresh_high = str(var_thresh[2])
         temp_thresh_inch = str(var_thresh[4])
      elif ((name == 'ws') or (name[0:2] == 'dz') or (name == 'wup') or (name == 'wdn') or (name == 'w1km') or (name == 'graupmax')):
         varproblow = var_temp[1,t,:,:]
         varprobhigh = var_temp[3,t,:,:]
         temp_thresh_low = str(var_thresh[1])
         temp_thresh_high = str(var_thresh[3])

   else:
      if ((name[0:2] == 'wz') or (name[0:2] == 'uh')):
         varproblow = np.max(varprob[2,:(t+1),:,:], axis=0)
         varprobhigh = np.max(varprob[4,:(t+1),:,:], axis=0)
         temp_thresh_low = str(var_thresh[2])
         temp_thresh_high = str(var_thresh[4])
      elif (name == 'rain'):
         varproblow = np.max(varprob[1,:(t+1),:,:], axis=0)
         varprobhigh = np.max(varprob[2,:(t+1),:,:], axis=0)
         varprobinch = np.max(varprob[4,:(t+1),:,:], axis=0)
         temp_thresh_low = str(var_thresh[1])
         temp_thresh_high = str(var_thresh[2])
         temp_thresh_inch = str(var_thresh[4])
      elif ((name == 'ws') or (name[0:2] == 'dz') or (name == 'wup') or (name == 'wdn') or (name == 'w1km') or (name == 'graupmax')):
         varproblow = np.max(varprob[1,:(t+1),:,:], axis=0)
         varprobhigh = np.max(varprob[3,:(t+1),:,:], axis=0)
         temp_thresh_low = str(var_thresh[1])
         temp_thresh_high = str(var_thresh[3])

   prob_plot.name = name + 'prob'
   prob_plot.var1_title = 'Probability of ' + var_label + ' > ' + temp_thresh_low + ' ' + var_units
   env_plot(map, fig, ax1, ax2, ax3, x, y, prob_plot, varproblow, pmm_dz[t,:,:], t, init_label, valid_label, domain, outdir, '', '', '', 5, 0, spec='False', quiv='False') 

   if ((name == 'rain') or (name == 'ws')):
      prob_plot.name = name + 'probhigh'
      prob_plot.var1_title = 'Probability of ' + var_label + ' > ' + temp_thresh_high + ' ' + var_units
      env_plot(map, fig, ax1, ax2, ax3, x, y, prob_plot, varprobhigh, pmm_dz[t,:,:], t, init_label, valid_label, domain, outdir, '', '', '', 5, 0, spec='False', quiv='False') 

   if (name == 'rain'):
      prob_plot.name = name + 'probinch'
      prob_plot.var1_title = 'Probability of ' + var_label + ' > ' + temp_thresh_inch + ' ' + var_units
      env_plot(map, fig, ax1, ax2, ax3, x, y, prob_plot, varprobinch, pmm_dz[t,:,:], t, init_label, valid_label, domain, outdir, '', '', '', 5, 0, spec='False', quiv='False') 


