
#######################################################################################
#news_e_post_cbook.py - written 11/2015 by Pat Skinner
#
#This code is a collection of subroutines used to calculate summary quantities from 
#WRFOUT files.  
#
#Much of the code has been adapted from the SHARPpy library (https://github.com/sharppy/)
#Copyright (c) 2015, Kelton Halbert, Greg Blumberg, Tim Supinie, and Patrick Marsh
#
#Halbert, K. T., W. G. Blumberg, and P. T. Marsh, 2015: "SHARPpy: Fueling the Python Cult". 
#Preprints, 5th Symposium on Advances in Modeling and Analysis Using Python, Phoenix AZ.
#
#
#Input formats are generally assumed to be numpy arrays pulled from WRFOUT files, 
#so 3d arrays with a format [z,y,x]
#
#Output is typically 2D numpy arrays of format [y,x]
#
#
#Subroutines available are: 
#
#calc_td -> calculate dewpoint temperature
#calc_thv -> calculate virtual temperature (or virtual potential temperature)
#calc_th -> calculate potential temperature
#calc_t -> calculate temperature
#calc_the -> calculate equivalent potential temperature
#calc_lcl -> calculate lifted condensation level
#calc_lfc_el -> calculate level of free convection and equilibrium level 
#wetlift_2d -> calculates moist adiabatic lapse rate (used to create a parcel temperature profile) -> 2D ARRAY
#wetlift -> calculates moist adiabatic lapse rate (used to create a parcel temperature profile) -> SINGLE VALUE
#calc_wobus_2d -> Wobus correction for calculating moist adiabatic lapse rate -> 2D ARRAY
#calc_wobus -> Wobus correction for calculating moist adiabatic lapse rate -> SINGLE VALUE
#calc_parcel -> calculates the temp and potential temp for a lifted parcel 
#calc_cape -> calculates convective available potential energy
#calc_cin -> calculates convective inhibition
#calc_height -> calculates height (MSL) and vertical grid spacing 
#find_upper_lower -> finds indices above and below specified values
#calc_interp -> interpolates values between upper and lower grid points
#calc_mean_wind -> calculates mean u and v components of the wind in specified layer 
#calc_wind_shear -> calculates u and v components of the wind shear in specified layer 
#calc_bunkers -> calculates Bunkers storm motion (u and v components for right and left movers)
#calc_srh -> calculates storm-relative helicity in a layer
#calc_stp -> calculates significant tornado parameter
#calc_avg_wz -> calculates average vertical vorticity within layer
#calc_uh -> calculates updraft helicity within a layer
#calc_pmm -> calculates neighborhood probability matched mean for a variable
#calc_prob -> calculates gridpoint probability of exceedance for a 3d variable
#get_local_maxima2d -> runs a maximum value filter over a 2d array
#gauss_kern -> creates a Gaussian convolution kernel for smoothing
#gridpoint_interp -> interpolates a 2d array to a specific point within grid 
#
#Future needs: 
#
#Most layer-dependent values are calculated for levels within the layer, include interpolation 
#to upper and lower bounds in the future. 
#
#Calculate effective inflow layer (e.g. Thompson et al 2012) for calculation of summary values
#
#Try to make probability matched mean and wetlift more efficient (using KDTrees)
#
#######################################################################################

import numpy as np

###### below only needed for get_local_maxima2d and gauss_kern

from scipy import signal
from scipy import *
from scipy import ndimage

################  Constants ################

e_0 = 6.1173 			#std atm vapor pressure (hPa)
t_0 = 273.16			#std atm surface temp (K)
p_0 = 1000.			#std atm surface pressure (hPa)
Rd = 287.0			#gas constant for dry air (J/K Kg)
Rv = 461.5			#gas constant for water vapor (J/K Kg)
Cp = 1004.			#heat capacity at constant pressure (J/K)
Lv = 2501000.			#latent heat of vaporization (J/Kg)
g = 9.81			#gravity (m/s^2)

#######################################################################################

def calc_td(t, p, qv):

   #Calculates dewpoint temp for a np.ndarray

   #Adapted from similar wrftools routine (https:https://github.com/keltonhalbert/wrftools/)
   #Copyright (c) 2015, Kelton Halbert

   #Input:  

   #t - np.ndarray of temperature (K)
   #p - np.ndarray of pressure (hPa)
   #qv - np.ndarray of water vapor mixing ratio (g/Kg)

   #Returns: 

   #td - np.ndarray of dewpoint temperature (K)

#######################

   e_s = e_0 * np.exp((Lv / Rv) * ((1. / t_0) - (1. / t)))	#sat vapor pressure via Clasius Clapeyron equation (hPa)
   w_s = (0.622 * e_s) / (p - e_s) 				#sat mixing ratio (g/Kg)
   rh = (qv / w_s) * 100. 					#relative humidity
   e = (rh / 100.) * e_s					#vapor pressure (hPa)

   td = 1. / ((1. / t_0) - ((Rv / Lv) * np.log(e / e_0)))

   nans = np.isnan(td)						#handle NaNs
   td[nans] = 0.

   return td


#######################################################################################

def calc_thv(th, qv, qt): 

   #Calculates virtual temp (or virtual potential temp) for a np.ndarray
   
   #Adapted from similar wrftools routines (https:https://github.com/keltonhalbert/wrftools/)
   #Copyright (c) 2015, Kelton Halbert

   #Input:  

   #th - np.ndarray of temperature or potential temperature (K)
   #qv - np.ndarray of water vapor mixing ratio (g/Kg)
   #qt - np.ndarray of total water mixing ratio (g/Kg) (typically sum of cloud, rain, snow, ice, and graupel mixing ratios)

   #Returns: 

   #thv - np.ndarray of virtual (potential) temperature (K)

#######################

   thv = th * (1. + 0.61 * qv - qt)

   nans = np.isnan(thv)						#handle NaNs
   thv[nans] = 0.

   return thv


#######################################################################################

def calc_th(t, p): 

   #Calculates potential temperature for a np.ndarray
   
   #Adapted from similar wrftools routines (https:https://github.com/keltonhalbert/wrftools/)
   #Copyright (c) 2015, Kelton Halbert

   #Input:  

   #t - np.ndarray of temperature (K)
   #p - np.ndarray of pressure (hPa)

   #Returns: 

   #th - np.ndarray of potential temperature (K)

#######################

   th = t * (p_0 / p)**(Rd / Cp) 
 
   nans = np.isnan(th)						#handle NaNs
   th[nans] = 0.

   return th

#######################################################################################

def calc_t(th, p):

   #Calculates temperature for a np.ndarray

   #Adapted from similar wrftools routines (https:https://github.com/keltonhalbert/wrftools/)
   #Copyright (c) 2015, Kelton Halbert

   #Input:

   #th - np.ndarray of potential temperature (K)
   #p - np.ndarray of pressure (hPa)

   #Returns:

   #th - np.ndarray of temperature (K)

#######################

   t = th * (p / p_0)**(Rd / Cp)

   nans = np.isnan(t)						#handle NaNs
   t[nans] = 0.

   return t


#######################################################################################

def calc_lcl(t, td, th, p):

   #Calculates the lifted condensation level pressure for a 1 or 2d numpy array

   #Adapted from similar sharppy routine (https://github.com/sharppy/)
   #Copyright (c) 2015, Kelton Halbert, Greg Blumberg, Tim Supinie, and Patrick Marsh

   #Halbert, K. T., W. G. Blumberg, and P. T. Marsh, 2015: "SHARPpy: Fueling the Python Cult". 
   #Preprints, 5th Symposium on Advances in Modeling and Analysis Using Python, Phoenix AZ.

   #Input:

   #t - 1, or 2d numpy array of temperature (K)
   #td - 1 or 2d numpy array of dewpoint temperature (K)
   #th - 1 or 2d numpy array of potential temperature (K)
   #p - 1 or 2d numpy array of pressure (hPa)

   #Returns:

   #lcl_t - 1 or 2d numpy array of LCL temperature (K)
   #lcl_p - 1 or 2d numpy array of LCL pressure (hPa)

#######################

   ### Calc change in temp given dewpoint depression and dry adiabatic lapse rate as defined in Halpert et al. (2015):

   dt_dz = (t - td) * (1.2185 + 0.001278 * t + (t - td) * (-0.00219 + 1.173e-5 * (t - td) - 0.0000052 * t))  
   dt_dz = np.where(dt_dz < 0., 0., dt_dz)			#cannot have an LCL underground ...

   lcl_t = t - dt_dz
   lcl_p = p_0 / (th / lcl_t)**(Cp / Rd)

   nans = np.isnan(lcl_t)					#handle NaNs
   lcl_t[nans] = 0.
   lcl_p[nans] = 0.

   return lcl_t, lcl_p


#######################################################################################

def calc_wobus_2d(t):

   #Polynomial approximation for the saturation vapor pressure profile of a parcel lifted moist 
   #adiabatically using the technique developed by Herman Wobus

   #Adapted from similar sharppy routine (https://github.com/sharppy/)
   #Copyright (c) 2015, Kelton Halbert, Greg Blumberg, Tim Supinie, and Patrick Marsh

   #Halbert, K. T., W. G. Blumberg, and P. T. Marsh, 2015: "SHARPpy: Fueling the Python Cult".
   #Preprints, 5th Symposium on Advances in Modeling and Analysis Using Python, Phoenix AZ.

   #Input:

   #t -  2d numpy array of temperature IN DEGREES C!

   #Returns:

   #2d numpy array of the correction to theta (K) for calculation of saturated potential temperature

#######################

   t = t - 20. 	#subtract 20 for some reason

   pol = t * 0.

   pol = np.where(t <= 0., (15.13 / (np.power(1. + t * (-8.841660499999999e-3 + t * ( 1.4714143e-4 + t * (-9.671989000000001e-7 + t * (-3.2607217e-8 + t * (-3.8598073e-10))))),4))), pol)

   pol = np.where(t > 0., ((29.93 / np.power(1 + t * (3.6182989e-03 + t * (-1.3603273e-05 + t * (4.9618922e-07 + t * (-6.1059365e-09 + t * (3.9401551e-11 + t * (-1.2588129e-13 + t * (1.6688280e-16))))))),4)) + (0.96 * t) - 14.8), pol)

   return pol

#######################################################################################

def calc_wobus(t):

   #Polynomial approximation for the saturation vapor pressure profile of a parcel lifted moist
   #adiabatically using the technique developed by Herman Wobus

   #Adapted from similar sharppy routine (https://github.com/sharppy/)
   #Copyright (c) 2015, Kelton Halbert, Greg Blumberg, Tim Supinie, and Patrick Marsh

   #Halbert, K. T., W. G. Blumberg, and P. T. Marsh, 2015: "SHARPpy: Fueling the Python Cult".
   #Preprints, 5th Symposium on Advances in Modeling and Analysis Using Python, Phoenix AZ.

   #Input:

   #t - Float value of temperature IN DEGREES C!

   #Returns:

   #Float value of the correction to theta (K) for calculation of saturated potential temperature

#######################

   t = t - 20.  #subtract 20 for some reason

   pol = t * 0.

   if t <= 0:
      npol = 1. + t * (-8.841660499999999e-3 + t * ( 1.4714143e-4 + t * (-9.671989000000001e-7 + t * (-3.2607217e-8 + t * (-3.8598073e-10)))))
      npol = 15.13 / (np.power(npol,4))
      return npol
   else:
      ppol = t * (4.9618922e-07 + t * (-6.1059365e-09 + t * (3.9401551e-11 + t * (-1.2588129e-13 + t * (1.6688280e-16)))))
      ppol = 1 + t * (3.6182989e-03 + t * (-1.3603273e-05 + ppol))
      ppol = (29.93 / np.power(ppol,4)) + (0.96 * t) - 14.8
      return ppol

#######################################################################################

def wetlift_2d(t, th, p2):

   #Lifts a parcel to a new level

   #Adapted from similar sharppy routine (https://github.com/sharppy/)
   #Copyright (c) 2015, Kelton Halbert, Greg Blumberg, Tim Supinie, and Patrick Marsh

   #Halbert, K. T., W. G. Blumberg, and P. T. Marsh, 2015: "SHARPpy: Fueling the Python Cult".
   #Preprints, 5th Symposium on Advances in Modeling and Analysis Using Python, Phoenix AZ.

   #Input:

   #t - 2d numpy array of temperature (K)
   #th - 2d numpy array of potential temperature (K)
   #p2 - 2d numpy array of pressure to lift parcel(s) to (hPa)

   #Dependencies:

   #calc_wobus_2d

   #Returns:

   #numpy 2d array of the temperature (K) of saturated parcel(s) at p2

#######################

   th = th - t_0                #convert temps to degrees C
   t = t - t_0
   
   thetam = th - calc_wobus_2d(th) + calc_wobus_2d(t)
   thetam_wobus2d = calc_wobus_2d(thetam)
   eor = th * 0 + 999
   pwrp = th * 0.
   t1 = th * 0.
   e1 = th * 0.
   rate = th * 0.
   iter = 0

   while (np.any((np.fabs(eor) - 0.1) > 0) and (iter < 20)):
      if (iter == 0):
         pwrp = (p2 / p_0)**(Rd / Cp)
         t1   = (thetam + t_0) * pwrp - t_0
         
         ts  = t1 - 20. 	#subtract 20 for some reason

         pol = ts * 0.

         pol = np.where(ts <= 0., (15.13 / (np.power(1. + ts * (-8.841660499999999e-3 + ts * ( 1.4714143e-4 + ts * (-9.671989000000001e-7 + ts * (-3.2607217e-8 + ts * (-3.8598073e-10))))),4))), pol)

         pol = np.where(ts > 0., ((29.93 / np.power(1 + ts * (3.6182989e-03 + ts * (-1.3603273e-05 + ts * (4.9618922e-07 + ts * (-6.1059365e-09 + ts * (3.9401551e-11 + ts * (-1.2588129e-13 + ts * (1.6688280e-16))))))),4)) + (0.96 * ts) - 14.8), pol)

         e1  = pol - thetam_wobus2d
         
         rate = 1.
      else:
         temp = e2 - e1
         temp = np.where(temp == 0., 0.000001, temp)
         rate = (t2 - t1) / temp
         t1 = t2
         e1 = e2

      t2 = t1 - (e1 * rate)
      e2 = (t2 + t_0) / pwrp - t_0
 
      ts     = t2 - 20. 	#subtract 20 for some reason
      t2_wob = ts * 0.
      t2_wob = np.where(ts <= 0., (15.13 / (np.power(1. + ts * (-8.841660499999999e-3 + ts * ( 1.4714143e-4 + ts * (-9.671989000000001e-7 + ts * (-3.2607217e-8 + ts * (-3.8598073e-10))))),4))), t2_wob)
      t2_wob = np.where(ts > 0., ((29.93 / np.power(1 + ts * (3.6182989e-03 + ts * (-1.3603273e-05 + ts * (4.9618922e-07 + ts * (-6.1059365e-09 + ts * (3.9401551e-11 + ts * (-1.2588129e-13 + ts * (1.6688280e-16))))))),4)) + (0.96 * ts) - 14.8), t2_wob)

      ts     = e2 - 20. 	#subtract 20 for some reason
      e2_wob = ts * 0.
      e2_wob = np.where(ts <= 0., (15.13 / (np.power(1. + ts * (-8.841660499999999e-3 + ts * ( 1.4714143e-4 + ts * (-9.671989000000001e-7 + ts * (-3.2607217e-8 + ts * (-3.8598073e-10))))),4))), t2_wob)
      e2_wob = np.where(ts > 0., ((29.93 / np.power(1 + ts * (3.6182989e-03 + ts * (-1.3603273e-05 + ts * (4.9618922e-07 + ts * (-6.1059365e-09 + ts * (3.9401551e-11 + ts * (-1.2588129e-13 + ts * (1.6688280e-16))))))),4)) + (0.96 * ts) - 14.8), t2_wob)
      
      e2 += t2_wob - e2_wob - thetam

      eor = e2 * rate
      iter = iter + 1
   if (iter == 20):			#set max number of iterations to reduce slowdowns (rarely hit for 2016) 
      print('iter hit max: ', iter)

   return t2 - eor + t_0

#######################################################################################

def wetlift(t, th, p2):

   #Lifts a parcel to a new level 

   #Adapted from similar sharppy routine (https://github.com/sharppy/)
   #Copyright (c) 2015, Kelton Halbert, Greg Blumberg, Tim Supinie, and Patrick Marsh

   #Halbert, K. T., W. G. Blumberg, and P. T. Marsh, 2015: "SHARPpy: Fueling the Python Cult".
   #Preprints, 5th Symposium on Advances in Modeling and Analysis Using Python, Phoenix AZ.

   #Input:

   #t - float value of temperature (K)
   #th - float value of potential temperature (K)
   #p2 - float value of pressure to lift parcel(s) to (hPa)

   #Dependencies: 

   #calc_wobus

   #Returns:

   #float of the temperature (K) of saturated parcel(s) at p2

#######################

   th = th - t_0		#convert temps to degrees C
   t = t - t_0

   thetam = th - calc_wobus(th) + calc_wobus(t) 
   eor = 999
   while ((np.fabs(eor) - 0.1) > 0):
      if eor == 999: 				#if first pass
         pwrp = (p2 / p_0)**(Rd / Cp)
         t1 = (thetam + t_0) * pwrp - t_0
         e1 = calc_wobus(t1) - calc_wobus(thetam)
         rate = 1
      else:
         rate = (t2 - t1) / (e2 - e1)
         t1 = t2
         e1 = e2
         

      t2 = t1 - (e1 * rate)
      e2 = (t2 + t_0) / pwrp - t_0
      e2 += calc_wobus(t2) - calc_wobus(e2) - thetam
      eor = e2 * rate

   return t2 - eor + t_0


#######################################################################################

def calc_the(lcl_t, lcl_p):

   #Calculates equivalent potential temperature for a 1 or 2d numpy array as long as the LCL temp 
   #and pressure are known (equivalent to skew-T method for calculation)

   #Note:  Assumes 100 hPa is a sufficient upper level 

   #Adapted from similar sharppy routine (https://github.com/sharppy/)
   #Copyright (c) 2015, Kelton Halbert, Greg Blumberg, Tim Supinie, and Patrick Marsh

   #Halbert, K. T., W. G. Blumberg, and P. T. Marsh, 2015: "SHARPpy: Fueling the Python Cult".
   #Preprints, 5th Symposium on Advances in Modeling and Analysis Using Python, Phoenix AZ.

   #Input:

   #lcl_t - 1 or 2d numpy array of LCL temperature (K)
   #lcl_p - 1 or 2d numpy array of LCL pressure (hPa)

   #Dependencies:

   #calc_th, calc_lcl, calc_wobus, wetlift

   #Returns:

   #1 or 2d numpy array of equivalent potential temperature (K) 

#######################

   lcl_th = calc_th(lcl_t, lcl_p)
   p_top = np.zeros((lcl_th.shape)) + 100.			#specify numpy array for upper bound (100 hPa)
   t_top = wetlift_2d(lcl_t, lcl_th, p_top)
  
   the = calc_th(t_top, p_top)

   return the


#######################################################################################

def calc_parcel(t, th, p, base_t, base_p, lcl_t, lcl_p):

   #Creates lifted parcel profiled for the domain

   #Dependencies:  wetlift, calc_wobus, calc_th

   #Input:

   #t - 3d numpy array of environmental temperature (K)
   #th - 3d numpy array of environmental potential temperature (K)
   #p - 3d numpy array of pressure (hPa)
   #base_t - 2d numpy array of temperature (K) for layer to be lifted
   #base_p - 2d numpy array of pressure (hPa) for layer to be lifted
   #lcl_t - 2d numpy array of LCL temperature (K)
   #lcl_p - 2d numpy array of LCL pressure (hPa)

   #Returns:

   #t_parcel - 3d numpy array of lifted parcel temperature (K)
   #th_parcel - 3d numpy array of lifted parcel potential temperature (K)

#######################

   base_th = calc_th(base_t, base_p)

   t_parcel = t * 0.								#initialize t/th_parcel with zeroes
   th_parcel = th * 0.
   t_parcel[0,:,:] = base_t							#set lowest level to base_t/th 
   th_parcel[0,:,:] = base_th 

   lcl_diff = lcl_p - p
   masked_lcl_diff = np.ma.masked_where((lcl_diff) > 0, (lcl_diff))		#mask indices above LCL
   lower = np.abs(masked_lcl_diff).argmin(axis=0)               		#find indices of nearest values in t below lcl

   p_diff = base_p - p
   masked_p_diff = np.ma.masked_where((p_diff) > 0, (p_diff))			#mask indices above p_base
   base_index = np.abs(masked_p_diff).argmin(axis=0)                  		#find indices of nearest values in p below p_base

   for k in range(1, t.shape[0]):
      #### Handle 1st layer above base

      t_parcel[k,:,:] = np.where((k == base_index + 1) & (k <= lower), calc_t(base_th, p[k,:,:]), t_parcel[k,:,:])
      t_parcel[k,:,:] = np.where((k == base_index + 1) & (k > lower), wetlift_2d(base_t, base_th, p[k,:,:]), t_parcel[k,:,:])

      #### Lift with constant potential temperature below the LCL:

      t_parcel[k,:,:] = np.where((k > base_index + 1) & (k <= lower), calc_t(th_parcel[k-1,:,:], p[k,:,:]), t_parcel[k,:,:])

      #### For layer that contains LCL:  Lift with constant potential temperature to LCL, then moist adiabatically 
      ####to next pressure level above LCL: 

      t_temp = np.where((k > base_index + 1) & (k == (lower + 1)), calc_t(th_parcel[k-1,:,:], lcl_p), 0.)
      th_temp = calc_th(t_temp, lcl_p)
      t_parcel[k,:,:] = np.where((k > base_index + 1) & (k == (lower + 1)), wetlift_2d(t_temp, th_temp, p[k,:,:]), t_parcel[k,:,:])

      #### Lift moist adiabatically above LCL: 

      t_parcel[k,:,:] = np.where((k > base_index + 1) & (k > (lower + 1)), wetlift_2d(t_parcel[k-1,:,:], th_parcel[k-1,:,:], p[k,:,:]), t_parcel[k,:,:])
      th_parcel[k,:,:] = calc_th(t_parcel[k,:,:], p[k,:,:])

   return t_parcel, th_parcel

#######################################################################################

def calc_lfc_el(t_env, t_parc, p, lcl_t, lcl_p):

   #Calculates the temp of the LFC and EL given a 3d numpy array for the environmental temperature (K), 
   #lifted parcel temperature (K), and vertical depth of each grid point (m)

   #Code will integrate (t_env - t_parc) via the trapezoidal rule for any layers
   #where t_env > t_parc (i.e. it does not care about multiple LFCs/ELs)

   #Input:

   #t_env - 3d numpy array [z,y,x] of environmental temperature (K)
   #t_parc - 3d numpy array [z,y,x] of lifted parcel temperature (K)
   #p - 3d numpy array [z,y,x] of environmental pressure (hPa)
   #lcl_t - 2d numpy array [y,x] of LCL temperature (K)
   #lcl_p - 2d numpy array [y,x] of LCL pressure (hPa)

   #Returns:

   #lfc_p - masked 2d numpy array [y,x] of LFC pressure (hPa) -> First LFC if multiple are present
   #el_p - masked 2d numpy array [y,x] of EL pressure (hPa) -> Last EL if multiple are present

   #NOTE:  Does NOT interpolate between levels so LFC (EL) will normally be a slight over (under) estimate

#######################

   p_base = p[0,:,:]
   masked_p = np.ma.masked_where((p_base - p) < 75., (p))		#Mask values below top of mixed layer

   t_diff = t_parc - t_env

   t_diff = np.where(t_diff > 0.5, t_diff, 0.)				#set all regions where t_env > t_parc to 0. (don't consider CIN) (0.5 is buffer for shallow regions of potential instability in the boundary layer)

   for k in range(1, t_diff.shape[0]):					#set all regions below LCL to 0. (LFC cannot occur below LCL)
      t_diff[k,:,:] = np.where(masked_p[k,:,:] > lcl_p, 0., t_diff[k,:,:])

   masked_p2 = np.ma.masked_where(t_diff <= 0., (masked_p)) 	        #mask all pressures where t_env > t_parc 

   lfc_p = np.max(masked_p2, axis=0)
   el_p = np.min(masked_p2, axis=0)

   np.ma.set_fill_value(lfc_p, 0.)
   np.ma.set_fill_value(el_p, 0.)

   return lfc_p, el_p 


#######################################################################################

def calc_cape(t_env, t_parc, p, lcl_p, dz):

   #Calculates CAPE given a 3d numpy array for the environmental temperature (K), lifted
   #parcel temperature (K), and vertical depth of each grid point (m)

   #Code will integrate (t_env - t_parc) via the trapezoidal rule for any layers
   #where t_env > t_parc (i.e. it does not care about multiple LFCs/ELs)

   #Input:

   #t_env - 3d numpy array [z,y,x] of environmental temperature (K)
   #t_parc - 3d numpy array [z,y,x] of lifted parcel temperature (K)
   #p - 3d numpy array [z,y,x] of environmental pressure (hPa)
   #lcl_p - 2d numpy array [y,x] of LCL pressure (hPa)
   #dz - 3d numpy array [y,x] of vertical grid spacing for each grid point (m)

   #Returns:

   #2d numpy array of CAPE (J/Kg)

#######################

   t_diff = t_parc - t_env
   t_diff = np.where(t_diff > 0., t_diff, 0.)				#set all regions where t_env > t_parc to 0. (don't consider CIN)

   for k in range(1, t_diff.shape[0]):                                  #set all regions below LCL to 0. (LFC cannot occur below LCL)
      t_diff[k,:,:] = np.where(p[k,:,:] > lcl_p, 0., t_diff[k,:,:])

   cape = g * np.trapz((t_diff / t_env), dx = dz[:-1,:,:], axis=0)

   return cape

#######################################################################################

def calc_cin(t_env, t_parc, p, lfc_p, dz):

   #Calculates CIN given a 3d numpy array for the environmental temperature (K), lifted
   #parcel temperature (K), and vertical depth of each grid point (m)

   #Code will integrate (t_env - t_parc) via the trapezoidal rule for any layers
   #where t_env < t_parc (i.e. it does not care about multiple LFCs/ELs)

   #Input:

   #t_env - 3d numpy array [z,y,x] of environmental temperature (K)
   #t_parc - 3d numpy array [z,y,x] of lifted parcel temperature (K)
   #p - 3d numpy array [z,y,x] of environmental pressure (hPa)
   #lfc_p - 2d numpy array [z,y,x] of LFC pressure (hPa)
   #dz - 3d numpy array [z,y,x] of vertical grid spacing for each grid point (m)

   #Returns:

   #2d numpy array of CIN (J/Kg)

#######################

   t_parc = np.where(t_parc == 0., t_env, t_parc)			#replace 0's in parcel with environmental temps (sets CIN to 0 where t_parc = 0)

   t_diff = t_parc - t_env
   t_diff = np.where(t_diff < 0., t_diff, 0.)          			#set all regions where t_env > t_parc to 0. (don't consider CIN)

   for k in range(1, t_diff.shape[0]):                                  #set all regions above LFC to 0. (CIN only calculated below LFC)
      t_diff[k,:,:] = np.where(p[k,:,:] < lfc_p, 0., t_diff[k,:,:])

   cin = g * np.trapz((t_diff / t_env), dx = dz[:-1,:,:], axis=0)

   return cin

#######################################################################################

def calc_height(ph, phb):

   #Calculates the height (MSL) of the domain and vertical grid spacing (m) 

   #Input:

   #ph - 3d numpy array (z-axis must be axis 0) of perturbation Geopotential height (m)
   #phb - 3d numpy array (z-axis must be axis 0) of base state Geopotential height (m)

   #Returns:

   #z -  3d numpy array of centered grid point height (m; MSL)
   #dz -  3d numpy array of vertical grid spacing (m)

#######################

   z_stag = (ph + phb)/ g 				#convert Geopotential to height MSL on staggered grid
   z = (z_stag[:-1,:,:]+z_stag[1:,:,:]) / 2. 		#convert to centered grid
   dz = z_stag[1:,:,:] - z_stag[:-1,:,:]		#calculate vertical grid spacing

   return z, dz

#######################################################################################

def find_upper_lower(value, field):

   #Finds nearest indices from a 3d numpy array (axis=0) of values above and below a 2d 
   #array of values 

   #NOTE:  Assumes all points in values are between the min and max values in field 

   #Input:

   #value - Scalar or 2d numpy array of with dimensions equal to field[0,:,:]
   #field - 3d numpy array of dimensions [k, value[:,0], value[0,:]]

   #Returns:

   #upper - 2d numpy array of the nearest k index in field above corresponding grid point in value
   #lower - 2d numpy array of the nearest k index in field below corresponding grid point in value
   #interp - 2d numpy array of interpolation coefficients for value in field

#######################

   diff = value - field						#calculate offset in field for each grid point in values
   masked_diff = np.ma.masked_where((diff) > 0, (diff))		#mask differences greater than 0 (only consider values where field < value)
   lower = np.abs(masked_diff).argmin(axis=0)			#find indices of nearest values in field below corresponding grid point in values
   upper = lower + 1						#next index up will be closest index > value[grid point]
   upper = np.where(upper >= field.shape[0], lower, upper)
   
   i,j = np.meshgrid(np.arange(value.shape[1]), np.arange(value.shape[0])) 	#2-d arrays of size value.shape
   upper_values = field[upper,j,i]						#2-d array of field[upper,:,:]
   lower_values = field[lower,j,i]						#2-d array of field[lower,:,:]

   interp = (value - lower_values) / (upper_values - lower_values)

   return upper, lower, interp

#######################################################################################

def calc_interp(field, upper, lower, interp):

   #Interpolates 3d numpy array (along axis 0) according to upper and lower indices and an interp coefficient

   #Input:

   #field - 3d numpy array of dimensions [k, upper[:,0], upper[0,:]]
   #upper - 2d numpy array of the nearest k index in field above corresponding grid point in value
   #lower - 2d numpy array of the nearest k index in field below corresponding grid point in value
   #interp - 2d numpy array of interpolation coefficients for value in field

   #Returns:

   #interp_values - 2d numpy array of interpolated values

#######################

   i,j = np.meshgrid(np.arange(field.shape[2]), np.arange(field.shape[1]))      #2-d arrays of size value.shape 
   upper_values = field[upper,j,i]                                              #2-d array of field[upper,:,:]
   lower_values = field[lower,j,i]                                              #2-d array of field[lower,:,:]

   interp_values = interp * (upper_values - lower_values) + lower_values

   interp_values = np.where(upper_values == lower_values, upper_values, interp_values)

   return interp_values


#######################################################################################

def calc_mean_wind(p, z, dz, u, v, lower, upper):

   #Calculates pressure-weighted mean wind through a layer

   #Adapted from similar sharppy routine (https://github.com/sharppy/)
   #Copyright (c) 2015, Kelton Halbert, Greg Blumberg, Tim Supinie, and Patrick Marsh

   #Halbert, K. T., W. G. Blumberg, and P. T. Marsh, 2015: "SHARPpy: Fueling the Python Cult".
   #Preprints, 5th Symposium on Advances in Modeling and Analysis Using Python, Phoenix AZ.

   #Input:

   #p - 3d numpy array of pressure (hPa) 
   #z - 3d numpy array of height AGL (m) 
   #dz - 3d numpy array of vertical grid spacing depth (m)
   #u - 3d numpy array of the u component of the wind (m/s) 
   #v - 3d numpy array of the v component of the wind (m/s) 
   #lower - float of lower height (AGL) of layer
   #upper - float of upper height (AGL) of layer

   #Returns:

   #mean_u - 2d numpy array of pressure-weighted mean u component of the wind in the layer (m/s)
   #mean_v - 2d numpy array of pressure-weighted mean v component of the wind in the layer (m/s)

   #NOTE: Does NOT currently interpolate to lower and upper values (i.e. mean wind is calculated for grid points between 
   #lower and upper not the actual values)

#######################

   p = np.ma.masked_where(z > upper, (p))
   p = np.ma.masked_where(z < lower, (p))
   u = np.ma.masked_where(z > upper, (u))
   u = np.ma.masked_where(z < lower, (u))
   v = np.ma.masked_where(z > upper, (v))
   v = np.ma.masked_where(z < lower, (v))
   dz = np.ma.masked_where(z > upper, (dz))
   dz = np.ma.masked_where(z < lower, (dz))

   mean_u = np.ma.average(u, axis=0, weights=(dz*p))
   mean_v = np.ma.average(v, axis=0, weights=(dz*p)) 
   return mean_u, mean_v 


#######################################################################################

def calc_wind_shear(z, u, v, lower, upper):

   #Calculates wind shear through a layer using pressure-weighted mean wind 500 m above(below)
   #the lower(upper) boundary

   #Adapted from similar sharppy routine (https://github.com/sharppy/)
   #Copyright (c) 2015, Kelton Halbert, Greg Blumberg, Tim Supinie, and Patrick Marsh

   #Halbert, K. T., W. G. Blumberg, and P. T. Marsh, 2015: "SHARPpy: Fueling the Python Cult".
   #Preprints, 5th Symposium on Advances in Modeling and Analysis Using Python, Phoenix AZ.

   #Input:

   #z - 3d numpy array of height AGL (m)
#   #p - 3d numpy array of pressure (hPa) 
#   #dz - 3d numpy array of vertical grid spacing depth (m)
   #u - 3d numpy array of the u component of the wind (m/s)
   #v - 3d numpy array of the v component of the wind (m/s)
   #lower - float of lower height (AGL) of layer
   #upper - float of upper height (AGL) of layer

   #Dependencies: 

   #calc_mean_wind

   #Returns:

   #shear_u - 2d numpy array of the u component of the wind shear in layer (m/s)
   #shear_v - 2d numpy array of the v component of the wind shear in layer (m/s)

#   #NOTE: Does NOT currently interpolate to lower and upper values (i.e. wind shear is calculated between the nearest vertical 
#   #layers to the lower and upper values, not the actual values)

#######################

   u = np.ma.masked_where(z > upper, (u))
   u = np.ma.masked_where(z < lower, (u))
   v = np.ma.masked_where(z > upper, (v))
   v = np.ma.masked_where(z < lower, (v))
#
   lower_indices, upper_indices = np.ma.notmasked_edges(u,axis=0)
#
   u_upper = u[upper_indices].reshape(u.shape[1],u.shape[2])
   u_lower = u[lower_indices].reshape(u.shape[1],u.shape[2])
   v_upper = v[upper_indices].reshape(v.shape[1],v.shape[2])
   v_lower = v[lower_indices].reshape(v.shape[1],v.shape[2])

#   u_lower, v_lower = calc_mean_wind(p, z, dz, u, v, lower, lower+500.)
#   u_upper, v_upper = calc_mean_wind(p, z, dz, u, v, upper-500., upper)

   shear_u = u_upper - u_lower
   shear_v = v_upper - v_lower

   return shear_u, shear_v


#######################################################################################

def calc_bunkers(p, z, dz, u, v):

   #Calculates Bunkers storm motion provided input arrays of pressure, height, u, and v

   #Adapted from similar sharppy routine (https://github.com/sharppy/)
   #Copyright (c) 2015, Kelton Halbert, Greg Blumberg, Tim Supinie, and Patrick Marsh

   #Halbert, K. T., W. G. Blumberg, and P. T. Marsh, 2015: "SHARPpy: Fueling the Python Cult".
   #Preprints, 5th Symposium on Advances in Modeling and Analysis Using Python, Phoenix AZ.

   #NOTE:  This is the old method for calculating Bunkers storm motion ... should be upgraded to the 
   #effective inflow layer (Bunkers et al. 2014; JOM) in the future

   #Dependencies: calc_mean_wind, calc_wind_shear

   #Input:

   #p - 3d numpy array of pressure (hPa) 
   #z - 3d numpy array of height AGL (m) 
   #dz - 3d numpy array of vertical grid spacing depth (m)
   #u - 3d numpy array of the u component of the wind (m/s)
   #v - 3d numpy array of the v component of the wind (m/s)

   #Returns:

   #bunk_r_u - 2d numpy array of the u component of Bunkers motion for a right-moving supercell (m/s)
   #bunk_r_v - 2d numpy array of the v component of Bunkers motion for a right-moving supercell (m/s)
   #bunk_l_u - 2d numpy array of the u component of Bunkers motion for a left-moving supercell (m/s)
   #bunk_l_v - 2d numpy array of the v component of Bunkers motion for a left-moving supercell (m/s)

#######################

   d = 7.5 						#Empirically-derived deviation value from Bunkers et al. (2014; JOM) 

   upper = 6000.  					#Upper-limit to storm motion layer (m AGL)
   lower = 0. 						#Lower-limit to storm motion layer (m AGL)

   mean_u, mean_v = calc_mean_wind(p, z, dz, u, v, lower, upper)
   shear_u, shear_v = calc_wind_shear(z, u, v, lower, upper)

   modifier = d / np.sqrt(shear_u**2 + shear_v**2) 
   bunk_r_u = mean_u + (modifier * shear_v)
   bunk_r_v = mean_v - (modifier * shear_u)
   bunk_l_u = mean_u - (modifier * shear_v)
   bunk_l_v = mean_v + (modifier * shear_u)

   return bunk_r_u, bunk_r_v, bunk_l_u, bunk_l_v


#######################################################################################

def calc_srh(z, u, v, dz, lower, upper, sm_u, sm_v):

   #Calculates storm relative helicity provided input arrays of height, u, and v; as well as upper and
   #lower limits to the layer and storm motion vectors

   #Adapted from similar sharppy routine (https://github.com/sharppy/)
   #Copyright (c) 2015, Kelton Halbert, Greg Blumberg, Tim Supinie, and Patrick Marsh

   #Halbert, K. T., W. G. Blumberg, and P. T. Marsh, 2015: "SHARPpy: Fueling the Python Cult".
   #Preprints, 5th Symposium on Advances in Modeling and Analysis Using Python, Phoenix AZ.

   #Input:

   #z - 3d numpy array of height AGL (m)
   #u - 3d numpy array of the u component of the wind (m/s)
   #v - 3d numpy array of the v component of the wind (m/s)
   #dz - 3d numpy array [z,y,x] of vertical grid spacing for each grid point (m)
   #lower - float value of the lower boundary of the layer (m AGL)
   #upper - float value of the upper boundary of the layer (m AGL)
   #sm_u - float value of the u component of the storm motion (m/s)
   #sm_v - float value of the v component of the storm motion (m/s)

   #Returns:

   #srh - 2d numpy array of storm-relative helicity for the layer

   #NOTE: Does NOT currently interpolate to lower and upper values (i.e. SRH is calculated between the nearest vertical 
   #layers to the lower and upper values, not the actual values)

#######################

   sr_u = u - sm_u
   sr_v = v - sm_v

   du = sr_u[1:,:,:]-sr_u[:-1,:,:]			#du/dz
   dv = sr_v[1:,:,:]-sr_v[:-1,:,:]			#dv/dz

   layers = (sr_v[:-1,:,:]*(du/dz[:-1,:,:]) - sr_u[:-1,:,:]*(dv/dz[:-1,:,:])) * dz[:-1,:,:] 
   masked_layers = np.ma.masked_where((z[:-1,:,:]) > upper, (layers))
   masked_layers = np.ma.masked_where((z[:-1,:,:]) < lower, (masked_layers))

   srh = np.sum(masked_layers, axis=0)

   return srh


#######################################################################################

def calc_stp(cape, lcl, srh, shear_u, shear_v):

   #Calculates significant tornado parameter for a fixed layer according to Thompson et al. (2003)

   #Adapted from similar sharppy routine (https://github.com/sharppy/)
   #Copyright (c) 2015, Kelton Halbert, Greg Blumberg, Tim Supinie, and Patrick Marsh

   #Halbert, K. T., W. G. Blumberg, and P. T. Marsh, 2015: "SHARPpy: Fueling the Python Cult".
   #Preprints, 5th Symposium on Advances in Modeling and Analysis Using Python, Phoenix AZ.

   #Input:

   #cape - 2d numpy array of CAPE (J/Kg) -> Typically SBCAPE
   #lcl - 2d numpy array of LCL heights (m AGL)
   #srh - 2d numpy array of SRH (J/Kg)  -> Typically 0-1 km SRH
   #shear_u - 2d numpy array of the u component of wind shear (m/s)  -> Typically 0-6 km shear
   #shear_v - 2d numpy array of the v component of wind shear (m/s)  -> Typically 0-6 km shear

   #Returns:

   #stp - 2d numpy array of significant tornado parameter

#######################

   lcl_t = ((2000. - lcl) / 1000.)
   lcl_t = np.where(lcl < 1000., 1., lcl_t)
   lcl_t = np.where(lcl > 2000., 0., lcl_t)

   bulk_shear = np.sqrt(shear_u**2 + shear_v**2)
   bulk_shear = np.where(bulk_shear > 30., 30., bulk_shear)
   bulk_shear = np.where(bulk_shear < 12.5, 0., bulk_shear)

   bulk_t = bulk_shear / 20.

   cape_t = cape / 1500.
   srh_t = srh / 150. 

   stp = cape_t * lcl_t * srh_t * bulk_t

   return stp

#######################################################################################

def calc_avg_vort(wz, z, dz, lower, upper):

   #Calculates vertical grid spacing-weighted vertical vorticity between lower and upper

   #Input:

   #wz - 3d numpy array of relative vertical vorticity (s^-1)
   #z - 3d numpy array of height AGL (m)
   #dz - 3d numpy array of vertical grid spacing depth (m)
   #lower - float of lower height of layer (m)
   #upper - float of upper height of layer (m)

   #Returns:

   #avg_wz - 2d numpy array of vertical grid spacing-weighted mean relative vertical vorticity

#######################

   wz = np.ma.masked_where((z > upper), wz)			#set wz to 0. outside of lower/upper bounds
   wz = np.ma.masked_where((z < lower), wz)

   avg_wz = np.ma.average(wz, axis=0, weights=dz) 	#calc weighted average for remaining values

   return avg_wz

#######################################################################################

def calc_uh(w, wz, z, dz, lower, upper):

   #Calculates updraft helicity between lower and upper

   #Input:

   #w - 3d numpy array of vertical velocity (centered grid) (s^-1)
   #wz - 3d numpy array of relative vertical vorticity (s^-1)
   #z - 3d numpy array of height AGL (m)
   #dz - 3d numpy array of vertical grid spacing depth (m)
   #lower - float of lower height of layer (m)
   #upper - float of upper height of layer (m)

   #Returns:

   #uh - 2d numpy array of layer-integrated (via trapezoidal rule) updraft helicity

#######################

   hel = w * wz					#calculate helicity

   hel = np.where(z > upper, 0., hel)		#set helicity to 0. outside upper/lower bounds
   hel = np.where(z < lower, 0., hel)

   uh = np.trapz(hel, dx = dz[:-1,:,:], axis=0)		#calc updraft helicity for layer

   return uh 

###########################################################################################################

def prob_match_mean(var, mean_var, neighborhood):

   #Calculate a 2d array of the neighborhood probability matched mean when provided with an ensemble mean and raw data

   #Dependencies:

   #Input:

   #var - 3d array of raw data (e.g. var[member, y, x]) 
   #mean_var - 2d array of ensemble mean values
   #neighborhood - grid point radius to perform prob matching within (e.g. 15 corresponds to 30x30 grid point region) 

   #Returns:
   #pmm_var - 2d array of probability matched mean 

#######################

   ne = var.shape[0] #ensemble size
   pmm_var = mean_var * 0.

   for i in range(neighborhood, (mean_var.shape[0] - neighborhood)):

      if (neighborhood > 0):
         i_min = i - neighborhood
         i_max = i + neighborhood
      else:
         i_min = 0
         i_max = mean_var.shape[0]
 
      for j in range(neighborhood, (mean_var.shape[1] - neighborhood)):

         if (neighborhood > 0):
            j_min = j - neighborhood
            j_max = j + neighborhood
         else:  
            j_min = 0
            j_max = mean_var.shape[1]
 
         mean_temp = mean_var[i_min:i_max,j_min:j_max]
         ref_index = (i - i_min) * mean_temp.shape[1] + (j - j_min) #equivalent index to mean_swath[i,j] in a raveled array

         mean_var_ravel = np.ravel(mean_var[i_min:i_max,j_min:j_max])

         pool_var = np.ravel(var[:,i_min:i_max,j_min:j_max])
         pool_var_sort = np.sort(pool_var[::ne])  #sorted array of pool values with size equal to raveled mean_var

         mean_var_indices = np.argsort(mean_var_ravel)
         var_index = np.where(mean_var_indices == ref_index)

         pmm_var[i,j] = pool_var_sort[var_index]

   return pmm_var

###########################################################################################################

#######################

def calc_prob(var, thresh):

   #Calculate a 2d array of the probability a 3d variable exceeds a given threshold

   #Dependencies:

   #numpy

   #Input:

   #var - 3d array of data (e.g. var[member, y, x])
   #thresh - float of value to calculate probability of exceedance

   #Returns:
   #prob_var - nd array of probability of exceedance

#######################

######### Mask values < thresh, count occurences of exceedance at each gridpoint, convert to probability: ############

   masked_var = np.ma.masked_array(var, var < thresh)
   count_var = double(np.ma.count(masked_var, axis=0))
   prob_var = count_var / var.shape[0]

   return prob_var

#######################

###########################################################################################################

###########################################################################################################

def get_local_maxima2d(array2d, radius):

   #Run a max value filter of radius 'radius' over a 2d

   #Dependencies:

   #numpy, ndimage

   #Input:

   #array2d - 2d array of data to filter
   #radius - int of filter gridpoint radius

   #Returns:
   #array2d after application of gridpoint max value filter

#######################

    return ndimage.filters.maximum_filter(array2d, size=(radius,radius))

###########################################################################################################

def gauss_kern(size, sizey=None):

   #Create a normalized, 2d, Gaussian convolution filter 

   #Dependencies:
 
   #numpy

   #Input:

   #size - gridpoint radius of filter
   #sizey - if filter is elliptical, y-axis gridpoint radius

   #Returns:
   #2d Gaussian convolution filter

#######################

   size = int(size)
   if not sizey:
      sizey = size
   else:
      sizey = int(sizey)

   xx, yy = mgrid[-size:size+1, -sizey:sizey+1]
   g = exp(-(xx**2/float(size)+yy**2/float(sizey)))
   return g / g.sum()

#############################################################################################

#######################

###########################################################################################################

def gridpoint_interp(field, x_index, y_index):

   #interpolate 2-d array (field) to a specific point (x_index, y_index)

   #Dependencies:
 
   #numpy

   #Input:
   #
   #field       - 2d array (e.g. lat, lon)
   #x_index     - x index value of interpolation point (must be within field[:,:])
   #y_index     - y index value of interpolation point (must be within field[:,:])

   #Returns
   #
   #interp      - field interpolated to x_index, y_index

#######################

   interp_x = x_index - floor(x_index)
   interp_y = y_index - floor(y_index)

   lowlow = field[floor(y_index),floor(x_index)]
   lowup = field[floor(y_index),ceil(x_index)]
   upup = field[ceil(y_index),ceil(x_index)]
   uplow = field[ceil(y_index),floor(x_index)]

   low = interp_y * (lowup - lowlow) + lowlow
   up = interp_y * (upup - uplow) + uplow

   interp = interp_x * (up - low) + low

   return interp

