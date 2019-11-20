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
nghbrd       = 1  # Neighborhood size for maximum filter (number of grid points) 
numStats     = 6  # Number of Ensemble statistics used
radius_max   = 1  # Radius for maximum filter applied to Azimuthal Shear  
radius_gauss = 1  # Radius used for convolutions 
ConvThres    = 0.004 
year         = 2017
varSet       = VariableSet( year )
dateSet, timeSet, SummaryPath, path, mrmsVerPath, BinaryPath = allDates_and_Times( year ) 

def locateAziShear( filePath):
                """
                Locates Azimuthal Shear based on the date, time, and threshold given
                Returns set of Y and X coordinates 
                """
                mrmsAziShear = np.load( filePath ) 
                j1, i1       = np.where(( mrmsAziShear == 1 ))
                j0, i0       = np.where(( mrmsAziShear == 0 ))

		Jindices      = np.where( ( j0 > 10) & (j0 < 240 ) ) 
		Iindices      = np.where( ( i0 > 10) & (i0 < 240 ) )		

		return j1 , i1 , j0[Jindices], i0[Iindices] 

def main( path, dateSet, timeSet, varSet ): 
	mainData = [ ] ; numDate = [ ] ; BinaryData = [ ]
	print "Finding unique dates..."
	newDateSet = findDateSet( dateSet, timeSet, mrmsVerPath, nghbrd, threshold=0.003)
	print "Finished!" 
	for date in newDateSet:
		print "Running the following date: ", str(date)
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
				fullMRMSPath         = BinaryPath + str(date) +'/' + 'ti_%s_radiusMax=%s_radiusGauss=%s_ConvThres=%s.npy' % ( time, radius_max, radius_gauss, ConvThres )
                        	y1, x1, y0, x0       = locateAziShear( fullMRMSPath ) 
				if len(y1) == 0: 
					print "No LLMs at time:", time 
					continue
				else: 
					x_sub, y_sub     = zip(*random.sample(list(zip(x0, y0)), len(y1)))
					for row in range( len(y1) ):
						mainData.append(  np.ravel( allVar[ :, :, y1[row] , x1[row] ] ) ) 
						numDate.append( int(date) ) 
						BinaryData.append( 1 )  
					for row in range( len(y_sub) ):
						mainData.append(  np.ravel( allVar[ :, :, y_sub[row] , x_sub[row] ] ) )
                                		numDate.append( int(date) )
                                		BinaryData.append( 0 ) 
	return mainData, numDate, BinaryData  

if __name__ == "__main__":
        mainData, numDate, BinaryData =  main(path, dateSet, timeSet, varSet)  
	np.savez_compressed( 'DateExtraction2016_nghbrd=%s'%(nghbrd), mainData, numDate, BinaryData) 
	print np.shape( mainData )  




