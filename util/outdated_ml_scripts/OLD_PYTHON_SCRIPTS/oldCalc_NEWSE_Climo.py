import pickle
import os, sys
from PHD_cookbook import *
import timeit
from news_e_post_cbook import *
from scipy import signal

# User-defined parameters 
year             = 2017  # Year 
thresh           = 0.01  # Threshold post-convolution and maximing to determine binary output 
radius_max       = 2     # Gridpoint Radius of maximum 
radius_gauss     = 6     # Gridpoint radius of convolution 
area_thresh      = 70.   # Minimum area of rotation object
kernel           = gauss_kern(radius_gauss)

dateSet, timeSet, SummaryPath, path, mrmsVerPath, BinaryPath = allDates_and_Times( year )
DateSet = findDateSet( dateSet, timeSet, mrmsVerPath )

count = 0
time  = '1900' 
for date in DateSet:
        Path1 = BinaryPath + str(date) +'/'
	if not os.path.exists(Path1):
                os.makedirs(Path1)
        filename          = Path1 + 'ti_%s_radMax=%s_radGauss=%s_thr=%s' % ( time, str(radius_max), str(radius_gauss), str(thresh ) )
	mrms_             = MRMS(mrmsVerPath, str(date), time, nt=96)
	var               = 2. * mrms_.calc_track_for_var(  var='LOW_CRESSMAN' )
	var_convolve_temp = get_local_maxima2d(var, radius_max)
	var_convolve      = signal.convolve2d(var_convolve_temp, kernel, 'same')
	radmask           = mrms_.radarMask( )	
	objects           = find_objects_timestep(var_convolve, radmask, thresh, area_thresh)
	binary            = np.where(( objects >= thresh), 1, 0)

	# Count the non-zero elements of binary 
	count		 += np.count_nonzero( binary ) 

climo = float(count) / ( len(DateSet) * 230. * 230. )  

print "  Year:  %s  " % ( year ) 
print " Climo:  %s  " % ( climo )  



