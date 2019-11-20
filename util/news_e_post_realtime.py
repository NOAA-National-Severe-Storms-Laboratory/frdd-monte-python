###################################################################################################

from mpl_toolkits.basemap import Basemap
import matplotlib
import math
from scipy import *
import pylab as P
import numpy as np
import sys
import os
import netCDF4
from optparse import OptionParser
from news_e_post_cbook import *

###################################################################################################

parser = OptionParser()
parser.add_option("-d", dest="dir", type="string", default= None, help="Input Directory (of wrfouts)")
parser.add_option("-o", dest="outdir", type="string", help = "Output Directory (for summary file)")
parser.add_option("-s", dest="start_time", type="int", help = "Starting time index to process")
parser.add_option("-e", dest="end_time", type="int", help = "Ending time index to process")
parser.add_option("-t", dest="fcst_nt", type="int", help = "Total number of timesteps in forecast")

(options, args) = parser.parse_args()

if ((options.dir == None) or (options.outdir == None) or (options.start_time == None) or (options.end_time == None) or (options.fcst_nt == None)):
   print()
   parser.print_help()
   print()
   sys.exit(1)
else:
   dir = options.dir
   outdir = options.outdir
   start_time = options.start_time
   end_time = options.end_time
   fcst_nt = options.fcst_nt    

times = np.arange(start_time,end_time)
nt = len(times)

member_files = []
member_files_temp = os.listdir(dir)

for f, file in enumerate(member_files_temp):
   if (file[0:6] == 'wrfout'):         	        	#assumes filename format of: "wrfout_d02_yyyy-mm-dd_hh:mm:ss
      member_files.append(file)

member_files.sort()
first_file = member_files[0]

member_files = member_files[(start_time):(end_time)]
member = dir[-2:]

if (member[0] == '_'):
   member = '0' + member[1]

######### Get dimensions, (t, z, y, x), attributes, and initialize variables for forecast #########

file = member_files[0]
start_datetime = first_file[11:30]        		#date/time string for forecast initialization
start_day = double(first_file[19:21])          	#day of forecast initialization - used to test if fcst crosses 00 UTC

temp_path = os.path.join(dir, file)
outname = start_datetime + "_" + member + ".nc" #output file
output_path = outdir + outname

try:
   fin = netCDF4.Dataset(temp_path, "r")
   print("Opening %s \n" % temp_path)
except:
   print("%s does not exist! \n" %temp_path)
   sys.exit(1)

dx = fin.DX                    	          	#east-west grid spacing (m)
dy = fin.DY                      	        #north-south grid spacing (m)
cen_lat = fin.CEN_LAT                    	#center of domain latitude (dec deg)
cen_lon = fin.CEN_LON                    	#center of domain longitude (dec deg)
stand_lon = fin.STAND_LON			#center lon of Lambert conformal projection
true_lat_1 = fin.TRUELAT1                	#true lat value 1 for Lambert conformal conversion (dec deg)
true_lat_2 = fin.TRUELAT2                	#true lat value 2 for Lambert conformal conversion (dec deg)

xlat = fin.variables["XLAT"][0,:,:]              #latitude (dec deg; Lambert conformal)
xlon = fin.variables["XLONG"][0,:,:]     	#longitude (dec deg; Lambert conformal)
hgt = fin.variables["HGT"][0,:,:]                #terrain height above MSL (m)

ny = xlat.shape[0]
nx = xlat.shape[1]

fin.close()
del fin

######### Initialize output variables #########

#time_in_seconds = np.zeros((nt))				#forecast output times (s; s+86400 if forecast crosses 00 UTC)
u_10 = np.zeros((ny,nx))                       	#10m U component of wind (m/s)
v_10 = np.zeros((ny,nx))                       	#10m V component of wind (m/s)
ws_max_10 = np.zeros((ny,nx))

p_sfc = np.zeros((ny,nx))                      	#surface pressure (Pa)
t_2 = np.zeros((ny,nx))                       	#2m temp (K)
th_2 = np.zeros((ny,nx))                      	#2m potential temp (K)
qv_2 = np.zeros((ny,nx))                      	#2m water vapor mixing ratio (g/kg)

dbz_1km = np.zeros((ny,nx))           	        #Simulated reflectivity (dBZ) at 1 km AGL     
dbz_composite = np.zeros((ny,nx))                   #Simulated reflectivity (dBZ) averaged between 0 and 7000 m AGL     
 
t_75mb = np.zeros((ny,nx))                      	#lowest 75mb mixed layer temperature (K)
th_75mb = np.zeros((ny,nx))                      	#lowest 75mb mixed layer potential temperature (K)
td_75mb = np.zeros((ny,nx))                      	#lowest 75mb mixed layer dewpoint temperature (K)
th_v_75mb = np.zeros((ny,nx))                      #lowest 75mb mixed layer virtual potential temperature (K)
th_e_75mb = np.zeros((ny,nx))                      #lowest 75mb mixed layer equivalent potential temperature (K)
qv_75mb = np.zeros((ny,nx))                        #lowest 75mb mixed layer water vapor mixing ratio (g/kg)
p_75mb = np.zeros((ny,nx))                         #lowest 75mb mixed layer pressure (hPa)

hailnc = np.zeros((ny,nx))				#accumulated hail (mm)
graupelnc = np.zeros((ny,nx))			#accumulated graupel (mm)
snownc = np.zeros((ny,nx))				#accumulated snow/ice (mm)
rainnc = np.zeros((ny,nx))				#accumulated rain (mm)

graupel_max = np.zeros((ny,nx)) 

lcl_75mb = np.zeros((ny,nx))                       #lowest 75mb mixed layer lifted condensation level (m AGL)
lfc_75mb = np.zeros((ny,nx))                       #lowest 75mb mixed layer level of free convection (m AGL)
el_75mb = np.zeros((ny,nx))                        #lowest 75mb mixed layer equilibrium level (m AGL)
cape_75mb = np.zeros((ny,nx))                      #lowest 75mb mixed layer CAPE (J/kg)
cin_75mb = np.zeros((ny,nx))                       #lowest 75mb mixed layer CIN (J/kg)
cape_0to3_75mb = np.zeros((ny,nx))                  #lowest 75mb mixed layer 0-3 km AGL CAPE (J/kg)

shear_u_0to1 = np.zeros((ny,nx))                  	#u-component of the 0-1 km AGL wind shear (m/s)
shear_v_0to1 = np.zeros((ny,nx))                  	#v-component of the 0-1 km AGL wind shear (m/s)
shear_u_0to6 = np.zeros((ny,nx))                  	#u-component of the 0-6 km AGL wind shear (m/s)
shear_v_0to6 = np.zeros((ny,nx))                  	#v-component of the 0-6 km AGL wind shear (m/s)
bunk_r_u = np.zeros((ny,nx))                  	#u-component of the Bunkers storm motion (right mover) (m/s)
bunk_r_v = np.zeros((ny,nx))                  	#v-component of the Bunkers storm motion (right mover) (m/s)
bunk_l_u = np.zeros((ny,nx))                  	#u-component of the Bunkers storm motion (left mover) (m/s)
bunk_l_v = np.zeros((ny,nx))                  	#v-component of the Bunkers storm motion (left mover) (m/s)
srh_0to1 = np.zeros((ny,nx))                  	#0-1 km AGL storm-relative helicity (m^2/s^2)
srh_0to3 = np.zeros((ny,nx))                  	#0-3 km AGL storm-relative helicity (m^2/s^2)
srh_0to6 = np.zeros((ny,nx))                  	#0-6 km AGL storm-relative helicity (m^2/s^2)
u_500 = np.zeros((ny,nx))                        	#u-component of the wind at 500 m AGL (m/s)      
v_500 = np.zeros((ny,nx))                        	#v-component of the wind at 500 m AGL (m/s)      

stp_75mb = np.zeros((ny,nx))                  	#Significant Tornado Parameter for 75mb mixed layer

wz_0to2 = np.zeros((ny,nx))                  	#Average 0-2 km AGL vertical vorticity (s^-1)
wz_2to5 = np.zeros((ny,nx))                  	#Average 2-5 km AGL vertical vorticity (s^-1)
wz_0to6 = np.zeros((ny,nx))                  	#Average 0-6 km AGL vertical vorticity (s^-1)
uh_0to2 = np.zeros((ny,nx))                  	#0-2 km Updraft Helicity (m^2/s^2)
uh_2to5 = np.zeros((ny,nx))                  	#0-2 km Updraft Helicity (m^2/s^2)
uh_0to6 = np.zeros((ny,nx))                  	#0-2 km Updraft Helicity (m^2/s^2)
      
w_up_max = np.zeros((ny,nx))
w_down_max = np.zeros((ny,nx))
w_1km = np.zeros((ny,nx))

######### Read/concatenate state variable arrays #########

for t, file in enumerate(member_files):
   temp_path = os.path.join(dir, file)
  
   day = double(file[19:21])                              #parse time from filename and convert to seconds
   hour = double(file[22:24])
   min = double(file[25:27])
   sec = double(file[28:30])

   time_in_seconds = hour * 3600. + min * 60. + sec
   if (time_in_seconds < 10000.):
      time_in_seconds = time_in_seconds + 86400.
#   time_in_seconds = (day - start_day) * 86400. + hour * 3600. + min * 60. + sec

   try:
      fin = netCDF4.Dataset(temp_path, "r")
   except:
      print("%s does not exist! \n" %temp_path)
      sys.exit(1)

   u_10 = fin.variables["U10"][0,:,:]            #expects var dimensions of (nt, ny, nx) with nt = 1
   v_10 = fin.variables["V10"][0,:,:]
   ws_max_10 = fin.variables["WSPD10MAX"][0,:,:]
   p_sfc = fin.variables["PSFC"][0,:,:]
   p_sfc = p_sfc / 100.			#convert to hPa

   w_up_max = fin.variables["W_UP_MAX"][0,:,:]
   w_down_max = fin.variables["W_DN_MAX"][0,:,:]
   w_down_max = np.abs(w_down_max) 

#   hailnc[t,:,:] = fin.variables["HAILNC"][0,:,:]
   graupelnc = fin.variables["GRAUPELNC"][0,:,:]
   snownc = fin.variables["SNOWNC"][0,:,:]
   rainnc = fin.variables["RAINNC"][0,:,:]
   rainnc = rainnc / 25.4

   graupel_max = fin.variables["GRPL_MAX"][0,:,:]

   t_2 = fin.variables["T2"][0,:,:]
   th_2 = fin.variables["TH2"][0,:,:]
   qv_2 = fin.variables["Q2"][0,:,:]

   qv_2 = np.where(qv_2 < 0., 0.0001, qv_2)  #force qv to be positive definite
   td_2 = calc_td(t_2, p_sfc, qv_2) 

   u = fin.variables["U"][0,:,:,:]        	    	#expects var dimensions of (nt, nz, ny, nx) with nt = 1
   v = fin.variables["V"][0,:,:,:]
   w = fin.variables["W"][0,:,:,:]
   ph = fin.variables["PH"][0,:,:,:]
   phb = fin.variables["PHB"][0,:,:,:]
   p = fin.variables["P"][0,:,:,:]
   pb = fin.variables["PB"][0,:,:,:]

   uc = (u[:,:,:-1]+u[:,:,1:])/2.      		#convert staggered grids to centered
   vc = (v[:,:-1,:]+v[:,1:,:])/2.
   wc = (w[:-1,:,:]+w[1:,:,:])/2.

   qv = fin.variables["QVAPOR"][0,:,:,:]
   qc = fin.variables["QCLOUD"][0,:,:,:]
   qr = fin.variables["QRAIN"][0,:,:,:]
   qi = fin.variables["QICE"][0,:,:,:]
   qs = fin.variables["QSNOW"][0,:,:,:]
   qh = fin.variables["QGRAUP"][0,:,:,:]

   qt = qc + qr + qi + qs + qh 				

   qv = np.where(qv < 0., 0.0001, qv)  #force qv to be positive definite

   th = fin.variables["T"][0,:,:,:]
   th = th + 300. 					#add base state temp (290 K) to potential temp

   dbz = fin.variables["REFL_10CM"][0,:,:,:]
   wz = fin.variables["REL_VORT"][0,:,:,:]

   fin.close()
   del fin

######### Calculate summary values #########

   z, dz = calc_height(ph, phb)				#height and layer thickness (m)
   z_agl = z - hgt

   p = (p + pb) / 100. 					#pressure (hPa)
   temp = calc_t(th, p)

######### Sfc/2m layer values #########

   t_v = calc_thv(temp, qv, qt)
   td = calc_td(temp, p, qv)
   th_v = calc_thv(th, qv, qt)
   temp_0to3 = np.ma.masked_where((z_agl > 3000.), (t_v))			#Set values above 3 km AGL to zero for calculating 0-3 km CAPE

######### 75mb mixed-layer values #########
              
   masked_temp = np.ma.masked_where((p_sfc - p) > 75., (temp))
   masked_th = np.ma.masked_where((p_sfc - p) > 75., (th))
   masked_td = np.ma.masked_where((p_sfc - p) > 75., (td))
   masked_th_v = np.ma.masked_where((p_sfc - p) > 75., (th_v))
   masked_t_v = np.ma.masked_where((p_sfc - p) > 75., (t_v))
   masked_qv = np.ma.masked_where((p_sfc - p) > 75., (qv))
   masked_p = np.ma.masked_where((p_sfc - p) > 75., (p))
          
   t_75mb = np.ma.average(masked_temp, axis=0, weights=dz)        
   th_75mb = np.ma.average(masked_th, axis=0, weights=dz)        
   td_75mb = np.ma.average(masked_td, axis=0, weights=dz)        
   th_v_75mb = np.ma.average(masked_th_v, axis=0, weights=dz)        
   t_v_75mb = np.ma.average(masked_t_v, axis=0, weights=dz)        
   qv_75mb = np.ma.average(masked_qv, axis=0, weights=dz)        
   p_75mb = np.ma.average(masked_p, axis=0, weights=dz)        

   lcl_t_75mb, lcl_p_75mb = calc_lcl(t_v_75mb, td_75mb, th_v_75mb, p_75mb)
   lcl_up_index, lcl_low_index, lcl_interp = find_upper_lower(lcl_p_75mb, p)
   lcl_75mb = calc_interp(z_agl, lcl_up_index, lcl_low_index, lcl_interp)
         
#   th_e_75mb = calc_the(lcl_t_75mb, lcl_p_75mb)

   t_75mb_parcel, th_75mb_parcel = calc_parcel(t_v, th_v, p, t_v_75mb, p_75mb, lcl_t_75mb, lcl_p_75mb)

   lfc_p_75mb, el_p_75mb = calc_lfc_el(t_v, t_75mb_parcel, p, lcl_t_75mb, lcl_p_75mb)
   lfc_up_index, lfc_low_index, lfc_interp = find_upper_lower(lfc_p_75mb, p)
   lfc_75mb = calc_interp(z_agl, lfc_up_index, lfc_low_index, lfc_interp)

   el_up_index, el_low_index, el_interp = find_upper_lower(el_p_75mb, p)
   el_75mb = calc_interp(z_agl, el_up_index, el_low_index, el_interp)

   lfc_p_75mb = np.where(lfc_75mb > 1000000., p_sfc, lfc_p_75mb)
   el_p_75mb = np.where(lfc_75mb > 1000000., p_sfc, el_p_75mb)
   el_75mb = np.where(lfc_75mb > 1000000., 0., el_75mb)
   lfc_75mb = np.where(lfc_75mb > 1000000., 0., lfc_75mb)

   cape_75mb = calc_cape(t_v, t_75mb_parcel, p, lcl_p_75mb, dz)
   cin_75mb = calc_cin(t_v, t_75mb_parcel, p, lfc_p_75mb, dz)

   t_75mb_parcel_0to3 = np.ma.masked_where((z_agl > 3000.), (t_75mb_parcel))
  
   cape_0to3_75mb = calc_cape(temp_0to3, t_75mb_parcel_0to3, p, lcl_p_75mb, dz)

######### Wind values #########

   temp = np.zeros((z_agl.shape[1], z_agl.shape[2])) + 500.
   agl500_upper, agl500_lower, agl500_interp = find_upper_lower(temp, z_agl)
   u_500 = calc_interp(uc, agl500_upper, agl500_lower, agl500_interp)
   v_500 = calc_interp(vc, agl500_upper, agl500_lower, agl500_interp)

   shear_u_0to1, shear_v_0to1 = calc_wind_shear(z_agl, uc, vc, 0., 1000.)
   shear_u_0to6, shear_v_0to6 = calc_wind_shear(z_agl, uc, vc, 0., 6000.)

   bunk_r_u, bunk_r_v, bunk_l_u, bunk_l_v = calc_bunkers(p, z_agl, uc, vc)

   srh_0to1 = calc_srh(z_agl, uc, vc, dz, 0., 1000., bunk_r_u, bunk_r_v)
   srh_0to3 = calc_srh(z_agl, uc, vc, dz, 0., 3000., bunk_r_u, bunk_r_v)
   srh_0to6 = calc_srh(z_agl, uc, vc, dz, 0., 6000., bunk_r_u, bunk_r_v)

   stp_75mb = calc_stp(cape_75mb, lcl_75mb, srh_0to1, shear_u_0to6, shear_v_0to6)

######### Storm-scale values #########

   masked_dbz = np.ma.masked_where((z_agl > 7000.), (dbz))
   dbz_composite = np.ma.average(masked_dbz, axis=0, weights=dz)

   temp = np.zeros((dbz.shape[1], dbz.shape[2])) + 1000.
   dbz_upper, dbz_lower, dbz_interp = find_upper_lower(temp, z_agl)         
   dbz_1km = calc_interp(dbz, dbz_upper, dbz_lower, dbz_interp)

   temp = np.zeros((wc.shape[1], wc.shape[2])) + 1000.
   w_upper, w_lower, w_interp = find_upper_lower(temp, z_agl)         
   w_1km = calc_interp(wc, w_upper, w_lower, w_interp)
      
   wz_0to2 = calc_avg_vort(wz, z_agl, dz, 0., 2000.)
   wz_2to5 = calc_avg_vort(wz, z_agl, dz, 2000., 5000.)
   wz_0to6 = calc_avg_vort(wz, z_agl, dz, 0., 6000.)
   uh_0to2 = calc_uh(wc, wz, z_agl, dz, 0., 2000.)
   uh_2to5 = calc_uh(wc, wz, z_agl, dz, 2000., 5000.)
   uh_0to6 = calc_uh(wc, wz, z_agl, dz, 0., 6000.)


########## Save output as netcdf file: #################

   if ((t == 0) and (start_time == 0)):
      try:
         fout = netCDF4.Dataset(output_path, "w")
      except:
         print("Could not create %s!\n" % output_path)

      fout.createDimension('NX', nx)
      fout.createDimension('NY', ny)
      fout.createDimension('NT', fcst_nt)

      setattr(fout,'DX',dx)
      setattr(fout,'DY',dy)
      setattr(fout,'CEN_LAT',cen_lat)
      setattr(fout,'CEN_LON',cen_lon)
      setattr(fout,'STAND_LON',stand_lon)
      setattr(fout,'TRUE_LAT1',true_lat_1)
      setattr(fout,'TRUE_LAT2',true_lat_2)

      fout.createVariable('TIME', 'f4', ('NT',))
      fout.createVariable('XLAT', 'f4', ('NY','NX',))
      fout.createVariable('XLON', 'f4', ('NY','NX',))
      fout.createVariable('HGT', 'f4', ('NY','NX',))

      fout.createVariable('U_10', 'f4', ('NT','NY','NX',))
      fout.createVariable('V_10', 'f4', ('NT','NY','NX',))
      fout.createVariable('WS_MAX_10', 'f4', ('NT','NY','NX',))
      fout.createVariable('P_SFC', 'f4', ('NT','NY','NX',))
      fout.createVariable('T_2', 'f4', ('NT','NY','NX',))
      fout.createVariable('TH_2', 'f4', ('NT','NY','NX',))
      fout.createVariable('TD_2', 'f4', ('NT','NY','NX',))
      fout.createVariable('QV_2', 'f4', ('NT','NY','NX',))

      fout.createVariable('HAIL', 'f4', ('NT','NY','NX',))
      fout.createVariable('GRAUPEL', 'f4', ('NT','NY','NX',))
      fout.createVariable('SNOW', 'f4', ('NT','NY','NX',))
      fout.createVariable('RAIN', 'f4', ('NT','NY','NX',))

      fout.createVariable('GRAUPEL_MAX', 'f4', ('NT','NY','NX',))

      fout.createVariable('T_ML', 'f4', ('NT','NY','NX',))
      fout.createVariable('TH_ML', 'f4', ('NT','NY','NX',))
      fout.createVariable('TD_ML', 'f4', ('NT','NY','NX',))
      fout.createVariable('TH_V_ML', 'f4', ('NT','NY','NX',))
      fout.createVariable('TH_E_ML', 'f4', ('NT','NY','NX',))
      fout.createVariable('QV_ML', 'f4', ('NT','NY','NX',))
      fout.createVariable('P_ML', 'f4', ('NT','NY','NX',))

      fout.createVariable('LCL_ML', 'f4', ('NT','NY','NX',))
      fout.createVariable('LFC_ML', 'f4', ('NT','NY','NX',))
      fout.createVariable('EL_ML', 'f4', ('NT','NY','NX',))
      fout.createVariable('CAPE_ML', 'f4', ('NT','NY','NX',))
      fout.createVariable('CIN_ML', 'f4', ('NT','NY','NX',))
      fout.createVariable('CAPE_0TO3_ML', 'f4', ('NT','NY','NX',))

      fout.createVariable('U_500', 'f4', ('NT','NY','NX',))
      fout.createVariable('V_500', 'f4', ('NT','NY','NX',))
      fout.createVariable('SHEAR_U_0TO1', 'f4', ('NT','NY','NX',))
      fout.createVariable('SHEAR_V_0TO1', 'f4', ('NT','NY','NX',))
      fout.createVariable('SHEAR_U_0TO6', 'f4', ('NT','NY','NX',))
      fout.createVariable('SHEAR_V_0TO6', 'f4', ('NT','NY','NX',))
      fout.createVariable('BUNK_R_U', 'f4', ('NT','NY','NX',))
      fout.createVariable('BUNK_R_V', 'f4', ('NT','NY','NX',))
      fout.createVariable('BUNK_L_U', 'f4', ('NT','NY','NX',))
      fout.createVariable('BUNK_L_V', 'f4', ('NT','NY','NX',))
      fout.createVariable('SRH_0TO1', 'f4', ('NT','NY','NX',))
      fout.createVariable('SRH_0TO3', 'f4', ('NT','NY','NX',))
      fout.createVariable('SRH_0TO6', 'f4', ('NT','NY','NX',))
      fout.createVariable('STP_ML', 'f4', ('NT','NY','NX',))

      fout.createVariable('DZ_1KM', 'f4', ('NT','NY','NX',))
      fout.createVariable('DZ_COMP', 'f4', ('NT','NY','NX',))
      fout.createVariable('WZ_0TO2', 'f4', ('NT','NY','NX',))
      fout.createVariable('WZ_2TO5', 'f4', ('NT','NY','NX',))
      fout.createVariable('WZ_0TO6', 'f4', ('NT','NY','NX',))
      fout.createVariable('UH_0TO2', 'f4', ('NT','NY','NX',))
      fout.createVariable('UH_2TO5', 'f4', ('NT','NY','NX',))
      fout.createVariable('UH_0TO6', 'f4', ('NT','NY','NX',))

      fout.createVariable('W_UP_MAX', 'f4', ('NT','NY','NX',))
      fout.createVariable('W_DN_MAX', 'f4', ('NT','NY','NX',))
      fout.createVariable('W_1KM', 'f4', ('NT','NY','NX',))

      fout.variables['XLAT'][:] = xlat
      fout.variables['XLON'][:] = xlon
      fout.variables['HGT'][:] = hgt

   else:
      try:
         fout = netCDF4.Dataset(output_path, "a")
      except:
         print("Could not create %s!\n" % output_path)

   fout.variables['TIME'][times[t]] = time_in_seconds
   
   fout.variables['U_10'][times[t],:,:] = u_10
   fout.variables['V_10'][times[t],:,:] = v_10
   fout.variables['WS_MAX_10'][times[t],:,:] = ws_max_10
   fout.variables['P_SFC'][times[t],:,:] = p_sfc
   fout.variables['T_2'][times[t],:,:] = t_2
   fout.variables['TD_2'][times[t],:,:] = td_2
   fout.variables['TH_2'][times[t],:,:] = th_2
   fout.variables['QV_2'][times[t],:,:] = qv_2

   fout.variables['HAIL'][times[t],:,:] = hailnc
   fout.variables['GRAUPEL'][times[t],:,:] = graupelnc
   fout.variables['SNOW'][times[t],:,:] = snownc
   fout.variables['RAIN'][times[t],:,:] = rainnc

   fout.variables['GRAUPEL_MAX'][times[t],:,:] = graupel_max

   fout.variables['T_ML'][times[t],:,:] = t_75mb
   fout.variables['TH_ML'][times[t],:,:] = th_75mb
   fout.variables['TD_ML'][times[t],:,:] = td_75mb
   fout.variables['TH_V_ML'][times[t],:,:] = th_v_75mb
   fout.variables['TH_E_ML'][times[t],:,:] = th_e_75mb
   fout.variables['QV_ML'][times[t],:,:] = qv_75mb
   fout.variables['P_ML'][times[t],:,:] = p_75mb

   fout.variables['LCL_ML'][times[t],:,:] = lcl_75mb
   fout.variables['LFC_ML'][times[t],:,:] = lfc_75mb
   fout.variables['EL_ML'][times[t],:,:] = el_75mb
   fout.variables['CAPE_ML'][times[t],:,:] = cape_75mb
   fout.variables['CIN_ML'][times[t],:,:] = cin_75mb
   fout.variables['CAPE_0TO3_ML'][times[t],:,:] = cape_0to3_75mb

   fout.variables['U_500'][times[t],:,:] = u_500
   fout.variables['V_500'][times[t],:,:] = v_500
   fout.variables['SHEAR_U_0TO1'][times[t],:,:] = shear_u_0to1
   fout.variables['SHEAR_V_0TO1'][times[t],:,:] = shear_v_0to1
   fout.variables['SHEAR_U_0TO6'][times[t],:,:] = shear_u_0to6
   fout.variables['SHEAR_V_0TO6'][times[t],:,:] = shear_v_0to6
   fout.variables['BUNK_R_U'][times[t],:,:] = bunk_r_u
   fout.variables['BUNK_R_V'][times[t],:,:] = bunk_r_v
   fout.variables['BUNK_L_U'][times[t],:,:] = bunk_l_u
   fout.variables['BUNK_L_V'][times[t],:,:] = bunk_l_v
   fout.variables['SRH_0TO1'][times[t],:,:] = srh_0to1
   fout.variables['SRH_0TO3'][times[t],:,:] = srh_0to3
   fout.variables['SRH_0TO6'][times[t],:,:] = srh_0to6
   fout.variables['STP_ML'][times[t],:,:] = stp_75mb

   fout.variables['DZ_1KM'][times[t],:,:] = dbz_1km
   fout.variables['DZ_COMP'][times[t],:,:] = dbz_composite
   fout.variables['WZ_0TO2'][times[t],:,:] = wz_0to2
   fout.variables['WZ_2TO5'][times[t],:,:] = wz_2to5
   fout.variables['WZ_0TO6'][times[t],:,:] = wz_0to6
   fout.variables['UH_0TO2'][times[t],:,:] = uh_0to2
   fout.variables['UH_2TO5'][times[t],:,:] = uh_2to5
   fout.variables['UH_0TO6'][times[t],:,:] = uh_0to6

   fout.variables['W_UP_MAX'][times[t],:,:] = w_up_max
   fout.variables['W_DN_MAX'][times[t],:,:] = w_down_max
   fout.variables['W_1KM'][times[t],:,:] = w_1km

   fout.close()
   del fout



 
