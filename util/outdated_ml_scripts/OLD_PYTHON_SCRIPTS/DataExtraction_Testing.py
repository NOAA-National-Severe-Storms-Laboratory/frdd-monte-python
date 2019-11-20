#===================================================================================================
# MODULE IMPORTS
#===================================================================================================
import matplotlib
matplotlib.use('Agg')
import math
import numpy as np
import sys
import netCDF4
from   netcdftime import utime
import os
import datetime
import random
import news_e_post_cbook as Newse_CBOOK

from PHD_cookbook import * 

# USER-DEFINED PARAMETERS 
n            = 18 # Number of ensemble members  
nghbrd       = 0  # Neighborhood size for maximum filter (number of grid points) 
numStats     = 6  # Number of Ensemble statistics used
radius_max   = 1  # Radius for maximum filter applied to Azimuthal Shear  
radius_gauss = 1  # Radius used for convolutions 
ConvThres    = 0.004
year         = 2017
varSet       = VariableSet( year )
dateSet, timeSet, SummaryPath, path, mrmsVerPath, BinaryPath = allDates_and_Times( year )


def determineLabel( y,  x, evaluateSet ): 
	pair = (y, x) 
	if pair in evaluateSet: 
		return 1.
	else: 
		return 0.

def main( path, dateSet, timeSet, varSet ): 
	mainData = [ ] ; numDate = [ ] ; BinaryData = [ ] ; numTime = [ ]
	print "Finding unique dates..."
	newDateSet = findDateSet( dateSet, timeSet, mrmsVerPath, nghbrd, threshold=0.003)
	print "Finished!" 
	for date in newDateSet:
		print "Running the following date: ", str(date)
		_mrms = MRMS(mrmsVerPath, str(date), time='1900', nghbrd= 0, threshold=15., nt=97 )	
		dbzj, dbzi  = _mrms.locateDBZ( ) 
		vortj,vorti = _mrms.locateAziShear( )
		evaluateSet = zip( vortj, vorti ) 
		for time in timeSet: 
			ML               = MachineLearningProject(n, nghbrd, path, varSet, numStats, str(date), time )
			ensMemSet        = ML.ensembleStringSet( ) 
			if len(ensMemSet) == 0:
				print time , "was empty!" 
				continue 
			else:
				allVar_allEnsMem_raw , allVar_allEnsMem_grad = ML.append_stats_and_ensMem( )
                                allVar_raw           = ML.CalcStats( allVar_allEnsMem_raw )  #dim: numStats, num of vars, ny, nx  
                                allVar_grad          = ML.CalcStats( allVar_allEnsMem_grad )
                                allVar               = np.concatenate(( allVar_raw, allVar_grad ), axis = 0 )
				for row in range( len(dbzj) ):
					mainData.append(  np.ravel( allVar[ :, :, dbzj[row] , dbzi[row] ] ) ) 
					numDate.append( int(date) ) 
					BinaryData.append(  determineLabel( dbzj[row], dbzi[row], evaluateSet )  )  
					numTime.append( int(time) ) 

	return mainData, numDate, BinaryData, numTime  

if __name__ == "__main__":
        mainData, numDate, BinaryData, numTime =  main(path, dateSet, timeSet, varSet)  
	np.savez_compressed( 'TestDateExtraction2017_nbrd=%s_rmax=%s_rgauss=%s' % ( nghbrd, radius_max, radius_gauss ), mainData, numDate, BinaryData, numTime) 





