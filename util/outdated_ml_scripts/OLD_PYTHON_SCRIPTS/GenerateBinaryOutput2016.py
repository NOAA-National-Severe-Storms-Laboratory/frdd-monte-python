import pickle
import os, sys
from PHD_cookbook import *
import timeit
from news_e_post_cbook import *
from scipy import signal

# User-defined parameters 
year             = 2016
nghbrd           = 0      
threshold        = 0.0035 

ConvThres        = 0.004 # Threshold post-convolution and maximing to determine binary output 
radius_max       = 3     # Gridpoint Radius of maximum 
radius_gauss     = 1     # Gridpoint radius of convolution 
kernel           = gauss_kern(radius_gauss)

dateSet, timeSet, SummaryPath, path, mrmsVerPath, BinaryPath = allDates_and_Times( year )
DateSet = findDateSet( dateSet, timeSet, mrmsVerPath, nghbrd, threshold=0.003 )

for date in DateSet: 
        Path1 = BinaryPath + str(date) +'/'
	if not os.path.exists(Path1):
                os.makedirs(Path1)
        for time in timeSet:
                filename          = Path1 + 'ti_%s_radiusMax=%s_radiusGauss=%s_ConvThres=%s' % ( time, str(radius_max), str(radius_gauss), str(ConvThres ) )
		mrms_             = MRMS(mrmsVerPath, str(date), time, nghbrd, threshold)
		var               = mrms_.calc_rotationTrack( ) # dim: ny, nx
		var_convolve_temp = get_local_maxima2d(var, radius_max)
		var_convolve      = signal.convolve2d(var_convolve_temp, kernel, 'same')
		binary            = np.where(( var_convolve >= ConvThres), 1, 0)
		#binary = np.where(( var >= ConvThres), 1, 0)
                np.save( filename , binary)
	



