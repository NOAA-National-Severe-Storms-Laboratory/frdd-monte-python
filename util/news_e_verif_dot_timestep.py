#!/usr/bin/env python
import matplotlib
matplotlib.use('Agg')
from mpl_toolkits.basemap import Basemap
import scipy
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

####################################### KD tree for finding nearest point: ######################################################

def do_kdtree(combined_x_y_arrays,points):
    mytree = scipy.spatial.cKDTree(combined_x_y_arrays)
    dist, indexes = mytree.query(points)
    return indexes

####################################### Parse Options: ######################################################

parser = OptionParser()
parser.add_option("-d", dest="exp_dir", type="string", default= None, help="Input Directory (of summary files)")
parser.add_option("-o", dest="outdir", type="string", help = "Output Directory (for images)")
parser.add_option("-m", dest="meso_path", type="string", help = "Path to mesonet files")
parser.add_option("-t", dest="t", type="int", help = "Timestep to process")

(options, args) = parser.parse_args()

if ((options.exp_dir == None) or (options.outdir == None) or (options.meso_path == None) or (options.t == None)):
    print()
    parser.print_help()
    print()
    sys.exit(1)
else:
    exp_dir = options.exp_dir
    outdir = options.outdir
    meso_path = options.meso_path
    t = options.t

domain = 'full'

geo_path = '/lus/scratch/skinnerp/news_e_post/mesonet/geoinfo.csv'  ### hardcode path to mesonet site info

#################################### User-Defined Variables:  #####################################################

edge            = 7 		#number of grid points to remove from near domain boundaries
thin		= 8		#thinning factor for quiver values (e.g. 6 means slice every 6th grid point)

fcst_len 	= 5400. 	#length of forecast to consider (s)
radius_max      = 3             #grid point radius for maximum value filter
radius_gauss    = 2             #grid point radius of convolution operator
neighborhood 	= 15		#grid point radius of prob matched mean neighborhood

plot_alpha 	= 0.4		#transparency value for filled contour plots

#################################### Threshold Values:  #####################################################

wz_thresh 	= [0.001, 0.002, 0.003, 0.004, 0.005, 0.006]		#vertical vorticity thresh (s^-1)
uh2to5_thresh 	= [20., 40., 60., 80., 100., 120.]		#updraft helicity thresh (2-5 km) (m^2 s^-2)
uh0to2_thresh 	= [10., 20., 30., 40., 50., 60.]		#updraft helicity thresh (0-2 km) (m^2 s^-2)
ws_thresh 	= [10., 15., 20., 25., 30., 35.]		#wind speed thresholds (m s^-1)
#rain_thresh     = [1., 7., 13., 19., 25., 31.] 			#qpf thresholds (mm)
rain_thresh     = [0.01, 0.25, 0.5, 0.75, 1., 1.25]                        #qpf thresholds (in)
dz_thresh       = [35., 40., 45., 50., 55., 60.]		#radar reflectivity thresholds (dBZ)

perc            = [10, 20, 30, 40, 50, 60, 70, 80, 90]		#Ens. percentiles to calc/plot

#################################### Colormap Names:  #####################################################

td_cmap_ncar = matplotlib.colors.ListedColormap(['#ad598a', '#c589ac','#dcb8cd','#e7cfd1','#d0a0a4','#ad5960', '#8b131d', '#8b4513','#ad7c59', '#c5a289','#dcc7b8','#eeeeee', '#dddddd', '#bbbbbb', '#e1e1d3', '#e1d5b1','#ccb77a','#ffffe5','#f7fcb9', '#addd8e', '#41ab5d', '#006837', '#004529', '#195257', '#4c787c'])

temp_cmap = matplotlib.colors.ListedColormap([cb_colors.purple4, cb_colors.purple5, cb_colors.purple6, cb_colors.purple7, cb_colors.blue8, cb_colors.blue7, cb_colors.blue6, cb_colors.blue5, cb_colors.blue4, cb_colors.blue3, cb_colors.green7, cb_colors.green6, cb_colors.green5, cb_colors.green4, cb_colors.green3, cb_colors.green2, cb_colors.orange2, cb_colors.orange3, cb_colors.orange4, cb_colors.orange5, cb_colors.red5, cb_colors.red6, cb_colors.red7, cb_colors.red8, cb_colors.purple3, cb_colors.purple4, cb_colors.purple5, cb_colors.purple6])  

uv_cmap = matplotlib.colors.ListedColormap([cb_colors.purple5, cb_colors.purple4, cb_colors.purple3, cb_colors.purple2, cb_colors.purple1, cb_colors.orange1, cb_colors.orange2, cb_colors.orange3, cb_colors.orange4, cb_colors.orange5])

diff_cmap = matplotlib.colors.ListedColormap([cb_colors.blue7, cb_colors.blue6, cb_colors.blue5, cb_colors.blue4, cb_colors.blue3, cb_colors.blue2, cb_colors.blue1, cb_colors.red1, cb_colors.red2, cb_colors.red3, cb_colors.red4, cb_colors.red5, cb_colors.red6, cb_colors.red7])  

td_dep_cmap = matplotlib.colors.ListedColormap([cb_colors.purple1, cb_colors.purple2, cb_colors.purple3, cb_colors.purple4, cb_colors.purple5, cb_colors.purple6, cb_colors.purple7])

#################################### Basemap Variables:  #####################################################

resolution 	= 'h'
area_thresh 	= 1000.

damage_files = '' #['/Volumes/fast_scr/pythonletkf/vortex_se/2013-11-17/shapefiles/extractDamage_11-17/extractDamagePaths']

#################################### Contour Levels:  #####################################################

prob_levels    		= np.arange(0.1,1.1,0.1)		#(%)

temp_levels             = np.arange(-20., 125., 5.)
td_levels		= np.arange(-16., 88., 4.)		#(deg F)
uv_levels              = np.arange(-15.,18.,3.)
td_dep_levels          = np.arange(0., 40., 5.)

tdd_diff_levels          = np.arange(-10.5,12.,1.5)
t_diff_levels          = np.arange(-7.,8.,1.)
ws_diff_levels         = np.arange(-3.,3.25,0.5)

pmm_dz_levels 		= [35., 50.]				#(dBZ) 
pmm_dz_colors_gray		= [cb_colors.gray6, cb_colors.gray8]	#gray contours
pmm_dz_colors_blue		= [cb_colors.blue6, cb_colors.blue8]	#blue contours
pmm_dz_colors_red		= [cb_colors.red6, cb_colors.red8]	#red contours
pmm_dz_colors_green		= [cb_colors.green6, cb_colors.green8]	#red contours

td_spec_levels          = [60.]					#extra contours for important Td levels
td_spec_colors          = [cb_colors.purple6]			#colors for important Td levels

cape_spec_levels        = [1000., 2000., 3000.]
cape_spec_colors        = [cb_colors.red4, cb_colors.red6, cb_colors.red8]

#################################### Initialize plot attributes:  #####################################################

temp_plot = web_plot('tempdot',                 \
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

td_plot = web_plot('tddot',                   \
                   'Ens. Mean 2 m Dewpoint Temp ($^{\circ}$F)',			\
                   'Probability Matched Mean - Composite Reflectivity (dBZ)',                   \
                   cb_colors.red6,      \
                   td_levels,            \
                   pmm_dz_levels,        \
                   td_spec_levels,                   \
                   td_spec_colors,                   \
                   pmm_dz_colors_red,        \
                   'none',		\
                   'none',		\
                   td_cmap_ncar,              \
                   'neither',               \
                   plot_alpha,               \
                   neighborhood)  

u_plot = web_plot('udot',                   \
                   'Ens. Mean 10 m U Component of Wind (m s$^{-1}$)',                 \
                   'Probability Matched Mean - Composite Reflectivity (dBZ)',                   \
                   cb_colors.green6,      \
                   uv_levels,            \
                   pmm_dz_levels,        \
                   'none',                   \
                   'none',                   \
                   pmm_dz_colors_green,        \
                   cb_colors.orange7,              \
                   cb_colors.purple7,              \
                   uv_cmap,              \
                   'both',               \
                   plot_alpha,               \
                   neighborhood)

v_plot = web_plot('vdot',                   \
                   'Ens. Mean 10 m V Component of Wind (m s$^{-1}$)',                 \
                   'Probability Matched Mean - Composite Reflectivity (dBZ)',                   \
                   cb_colors.green6,      \
                   uv_levels,            \
                   pmm_dz_levels,        \
                   'none',                   \
                   'none',                   \
                   pmm_dz_colors_green,        \
                   cb_colors.orange7,              \
                   cb_colors.purple7,              \
                   uv_cmap,              \
                   'both',               \
                   plot_alpha,               \
                   neighborhood)

td_dep_plot = web_plot('tdddot',                   \
                   'Ens. Mean 2 m Dewpoint Depression ($^{\circ}$F)',                 \
                   'Probability Matched Mean - Composite Reflectivity (dBZ)',                   \
                   cb_colors.red6,      \
                   td_dep_levels,            \
                   pmm_dz_levels,        \
                   td_spec_levels,                   \
                   td_spec_colors,                   \
                   pmm_dz_colors_red,        \
                   'none',              \
                   cb_colors.purple8,              \
                   td_dep_cmap,              \
                   'max',               \
                   plot_alpha,               \
                   neighborhood)

#############################################################################################

#################################### Read Data:  #####################################################

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

files = []
files_temp = os.listdir(exp_dir)
for f, file in enumerate(files_temp):
   if (file[0] == '2'):
      files.append(file)

files.sort()
ne = len(files)

for f, file in enumerate(files):
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
      init_hr = int(file[11:13])

      time = fin.variables['TIME'][t]

      if (time < 25000.):
         time_correct = time + 86400.
      else:
         time_correct = time

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

      xlat = fin.variables['XLAT'][:]
      xlon = fin.variables['XLON'][:]
      xlat = xlat[edge:-edge,edge:-edge]
      xlon = xlon[edge:-edge,edge:-edge]

      ### make lise of combined xlat/xlon values for kdtree: ###

      combo_lat_lon = np.dstack([xlat.ravel(),xlon.ravel()])[0]

      sw_lat_full = xlat[0,0]
      sw_lon_full = xlon[0,0]
      ne_lat_full = xlat[-1,-1]
      ne_lon_full = xlon[-1,-1]

      cen_lat = fin.CEN_LAT
      cen_lon = fin.CEN_LON
      stand_lon = fin.STAND_LON
      true_lat1 = fin.TRUE_LAT1
      true_lat2 = fin.TRUE_LAT2

      u_10 = np.zeros((ne, xlat.shape[0], xlat.shape[1]))
      v_10 = np.zeros((ne, xlat.shape[0], xlat.shape[1]))
      t_2 = np.zeros((ne, xlat.shape[0], xlat.shape[1]))
      p_2 = np.zeros((ne, xlat.shape[0], xlat.shape[1]))
      qv_2 = np.zeros((ne, xlat.shape[0], xlat.shape[1]))

   u_10[f,:,:] = fin.variables['U_10'][t,edge:-edge,edge:-edge]
   v_10[f,:,:] = fin.variables['V_10'][t,edge:-edge,edge:-edge]
   t_2[f,:,:] = fin.variables['T_2'][t,edge:-edge,edge:-edge]
   p_2[f,:,:] = fin.variables['P_SFC'][t,edge:-edge,edge:-edge]
   qv_2[f,:,:] = fin.variables['QV_2'][t,edge:-edge,edge:-edge]

   fin.close()
   del fin

u_10 = np.where(u_10 > 100000., 0., u_10)
v_10 = np.where(v_10 > 100000., 0., v_10)

t_2 = np.where(t_2 > 100000., 0., t_2)
p_2 = np.where(p_2 > 100000., 0., p_2)
qv_2 = np.where(qv_2 > 100000., 0., qv_2)

qv_2 = np.where(qv_2 < 0., 0., qv_2)
td_2 = calc_td(t_2, p_2, qv_2)

mean_u_10 = np.mean(u_10, axis=0)
mean_v_10 = np.mean(v_10, axis=0)

mean_t_2 = np.mean(t_2, axis=0)
mean_td_2 = np.mean(td_2, axis=0)

mean_tf_2 = (mean_t_2 - 273.15) * 1.8 + 32.
mean_tdf_2 = (mean_td_2 - 273.15) * 1.8 + 32.

mean_td_dep_2 = mean_tf_2 - mean_tdf_2

mean_tf_2_ravel = mean_tf_2.ravel()
mean_tdf_2_ravel = mean_tdf_2.ravel()
mean_td_dep_2_ravel = mean_td_dep_2.ravel()
mean_u_10_ravel = mean_u_10.ravel()
mean_v_10_ravel = mean_v_10.ravel()

quiv_xlon = xlon[0:-1:thin,0:-1:thin]
quiv_xlat = xlat[0:-1:thin,0:-1:thin]
quiv_u_10 = mean_u_10[0:-1:thin,0:-1:thin]
quiv_v_10 = mean_v_10[0:-1:thin,0:-1:thin]

################################# Read Matching Mesonet Obs: ###################################################

station_info = np.genfromtxt(geo_path, dtype=None, skip_header=1, delimiter=',')

stn_id = np.zeros((len(station_info)))
stn_lat = np.zeros((len(station_info)))
stn_lon = np.zeros((len(station_info)))

for i in range(0, len(station_info)):
   stn_id[i] = station_info[i][0]
   stn_lat[i] = station_info[i][7]
   stn_lon[i] = station_info[i][8]

meso_files = []
meso_files_temp = os.listdir(meso_path)
for f, file in enumerate(meso_files_temp):
   if (file[0] == '2'):
      meso_files.append(file)

meso_files.sort()

meso_times = np.zeros((len(meso_files)))

for f, file in enumerate(meso_files):
   mesodate = file[0:8]
   mesohour = file[8:10]
   mesominute = file[10:12]

   meso_time_in_seconds = double(mesohour) * 3600. + double(mesominute) * 60.

   if (double(mesohour) < 10):
      meso_time_in_seconds = meso_time_in_seconds + 86400.

   meso_times[f] = meso_time_in_seconds

meso_t = (np.abs(time_correct - meso_times)).argmin()
meso_file_temp = meso_files[meso_t]
meso_file = os.path.join(meso_path, meso_file_temp)

obs = np.genfromtxt(meso_file, dtype=None, skip_header=3, skip_footer=1)

ob_lat = np.zeros((len(obs)))
ob_lon = np.zeros((len(obs)))

ob_t = np.zeros((len(obs)))
ob_td = np.zeros((len(obs)))
ob_td_dep = np.zeros((len(obs)))
ob_u = np.zeros((len(obs)))
ob_v = np.zeros((len(obs)))

ob_spd = np.zeros((len(obs)))
ob_dir = np.zeros((len(obs)))

for i in range(0, len(obs)):
   ob_stn = obs[i][1]
   stn_index = (np.abs(ob_stn - stn_id).argmin())

   ob_lat[i] = stn_lat[stn_index]
   ob_lon[i] = stn_lon[stn_index]

   ob_lat_lon = list([ob_lat[i], ob_lon[i]])

   newse_indices = do_kdtree(combo_lat_lon,ob_lat_lon)

   ob_t[i] = mean_tf_2_ravel[newse_indices] - (obs[i][4] * 1.8 + 32) ## convert to deg F

   ob_t_k = obs[i][4] + 273.16
   ob_rh = obs[i][3]
   ob_pres = obs[i][12]

   ob_theta = ob_t_k * (1000. / ob_pres) ** (287. / 1004.)
   ob_sat_vap_pres = 6.11 * np.exp(2500000. / 461. * (1 / 273.15 - 1/ob_t_k))
   ob_vap_pres = ob_sat_vap_pres * (ob_rh / 100.)
   ob_qv = 0.622 * ob_vap_pres / (ob_pres - ob_vap_pres)

   ob_td_k = 1. / ((1. / 273.16) - ((461.5 / 2501000.) * np.log(ob_vap_pres / 6.1173)))
   ob_td[i] = mean_tdf_2_ravel[newse_indices] - ((ob_td_k - 273.16) * 1.8 + 32) 

   ob_td_dep[i] = mean_td_dep_2_ravel[newse_indices] - ((obs[i][4] * 1.8 + 32) - ((ob_td_k - 273.16) * 1.8 + 32))

   ob_spd = obs[i][5]
   ob_dir = obs[i][7]

   ob_u[i] = mean_u_10_ravel[newse_indices] - ob_spd * np.cos(np.pi / 180. * (270 - ob_dir))
   ob_v[i] = mean_v_10_ravel[newse_indices] - ob_spd * np.sin(np.pi / 180. * (270 - ob_dir))

################################# Make Figure Template: ###################################################

print('basemap part')

map, fig, ax1, ax2, ax3 = create_fig(sw_lat_full, sw_lon_full, ne_lat_full, ne_lon_full, true_lat1, true_lat2, cen_lat, stand_lon, damage_files, resolution, area_thresh, object='True')

x, y = list(map(xlon[:], xlat[:]))
xx, yy = map.makegrid(xlat.shape[1], xlat.shape[0], returnxy=True)[2:4]   #equidistant x/y grid for streamline plots

ob_x, ob_y = list(map(ob_lon[:], ob_lat[:]))

######## Rotate quiver values according to map projection: ########

q_u, q_v = map.rotate_vector(quiv_u_10[:,:], quiv_v_10[:,:], quiv_xlon, quiv_xlat, returnxy=False)

###################################### Make Plots: #######################################################

print('plot part')

######## Dot Plots: ########

t_label = 'Temperature Error (NEWS-e - Ob; $^{\circ}$F)'
dot_plot(map, fig, ax1, ax2, ax3, x, y, ob_x, ob_y, ob_t, t_diff_levels, diff_cmap, t_label, temp_plot, mean_tf_2[:,:], pmm_dz[:,:], t, init_label, valid_label, domain, outdir, q_u, q_v, 500, 5, 0, spec='False', quiv='True') 

td_label = 'Dewpoint Error (NEWS-e - Ob; $^{\circ}$F)'
dot_plot(map, fig, ax1, ax2, ax3, x, y, ob_x, ob_y, ob_td, t_diff_levels, diff_cmap, td_label, td_plot, mean_tdf_2[:,:], pmm_dz[:,:], t, init_label, valid_label, domain, outdir, q_u, q_v, 500, 5, 0, spec='False', quiv='True') 

td_dep_label = 'Dewpoint Depression Error (NEWS-e - Ob; $^{\circ}$F)'
dot_plot(map, fig, ax1, ax2, ax3, x, y, ob_x, ob_y, ob_td_dep, tdd_diff_levels, diff_cmap, td_dep_label, td_dep_plot, mean_td_dep_2[:,:], pmm_dz[:,:], t, init_label, valid_label, domain, outdir, q_u, q_v, 500, 5, 0, spec='False', quiv='True')

u_label = 'U Component of Wind Error (NEWS-e - Ob; m s$^{-1}$)'
dot_plot(map, fig, ax1, ax2, ax3, x, y, ob_x, ob_y, ob_u, ws_diff_levels, diff_cmap, u_label, u_plot, mean_u_10[:,:], pmm_dz[:,:], t, init_label, valid_label, domain, outdir, q_u, q_v, 500, 5, 0, spec='False', quiv='True') 

v_label = 'V Component of Wind Error (NEWS-e - Ob; m s$^{-1}$)'
dot_plot(map, fig, ax1, ax2, ax3, x, y, ob_x, ob_y, ob_v, ws_diff_levels, diff_cmap, v_label, v_plot, mean_v_10[:,:], pmm_dz[:,:], t, init_label, valid_label, domain, outdir, q_u, q_v, 500, 5, 0, spec='False', quiv='True') 

################

