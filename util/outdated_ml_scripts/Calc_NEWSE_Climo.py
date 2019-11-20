import pickle
import os, sys
from PHD_cookbook import *
import timeit
from news_e_post_cbook import *
from scipy import signal

# User-defined parameters 
year             = 2016  # Year 
thresh           = 0.01  # Threshold post-convolution and maximing to determine binary output 
radius_max       = 2     # Gridpoint Radius of maximum 
radius_gauss     = 6     # Gridpoint radius of convolution 

dateSet, timeSet, SummaryPath, path, mrmsVerPath, BinaryPath = allDates_and_Times( year )
DateSet = findDateSet( dateSet, timeSet, mrmsVerPath )

count = 0
for date in DateSet:
        Path1 = BinaryPath + str(date) +'/'
	for time in timeSet: 	
        	filename          = Path1 + 'ti_%s_radMax=%s_radGauss=%s_thr=%s.npy' % ( time, str(radius_max), str(radius_gauss), str(thresh ) )
		mrmsAziShear      = np.load( filename )	
		# Count the non-zero elements of binary 
		count		 += np.count_nonzero( mrmsAziShear ) 

climo = float(count) / ( len(DateSet) * 230. * 230. )  

print "  Year:  %s  " % ( year ) 
print " Climo:  %s  " % ( climo )  



