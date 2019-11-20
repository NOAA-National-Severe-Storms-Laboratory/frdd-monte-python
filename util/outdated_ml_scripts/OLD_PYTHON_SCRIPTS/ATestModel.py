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
import multiprocessing as mp

from PHD_cookbook import * 
import pickle 

# PARAMETERS 
n            = 18    # Number of ensemble members  
numStats     = 6     # Number of Ensemble statistics used

# USER-DEFINED PARAMETERS
nghbrd       = 0     # Neighborhood size for maximum filter on NEWS-e data (number of grid points)
year         = 2016  # Year
thresh	     = 0.01
radius_max   = 3 
radius_gauss = 2 

clfName            = 'RandomForest'
filename2          = '/home/monte.flora/PHD_RESEARCH/SAV_FILES/%s_IsotonicRegressionModel_nghbrd=%s.sav' % ( clfName, nghbrd )
loaded_isotonic    = pickle.load(open(filename2, 'rb'))
varSet       	   = VariableSet( year )
dateSet, \
timeSet, \
SummaryPath, \
path, \
mrmsVerPath, \
BinaryPath         = allDates_and_Times( year ) 

npzfilename = '/home/monte.flora/PHD_RESEARCH/NPZ_FILES/%s_ForecastStats_nghbrd=%s' % ( year, nghbrd )

print "Filename of the NPZ file: " , npzfilename

def main( path, timeSet, varSet ): 
	cr_machine      = [ ] 
	cr_nmep		= [ ] 
	timeList	= [ ]   
	dateList	= [ ] 

	filename2016    = '/home/monte.flora/PHD_RESEARCH/NPZ_FILES/ClimoData2016_nbrd=%s.npz' % ( nghbrd )
	examples, \
	binary,   \
	dateColumn, \
	timeColumn, \
	Y, X 		= readNPZ_Files( filename2016  )
	
	uniqueDates     = np.unique( dateColumn )
	count           = int( len( uniqueDates ) / 2. )

	for date in uniqueDates[count:]: 
		print "Running the following date: ", str(date)
		for time in timeSet:
			ML               = MachineLearningProject(n, nghbrd, path, varSet, numStats, str(date), time )
			ensMemSet        = ML.ensembleStringSet( ) 
			fullMRMSPath     = rot_track_file( date, time, radius_max, radius_gauss, thresh, BinaryPath )
			
			mrmsAziShear     = np.load( fullMRMSPath )
			X_caseStudy          = ML.convert2Ddata_to_MLExamples( )
        		
			clf_probs_calibrated = loaded_isotonic.predict_proba( X_caseStudy )[:,1]
        		probs_2D             = ML.map2D( clf_probs_calibrated ) 				
			NMEP   		     = ML.Prob_of_Exceed( )
			
			cr_machine.append( CorrespondenceRatio( probs_2D , mrmsAziShear ) )
			cr_nmep.append   ( CorrespondenceRatio( NMEP     , mrmsAziShear ) ) 
			timeList.append  ( time ) 
			dateList.append  ( date ) 

	return cr_machine , cr_nmep, timeList , dateList 

if __name__ == "__main__":
        cr_machine, cr_nmep, timeList, dateList   =  main(path, timeSet, varSet)  
	np.savez_compressed( npzfilename, cr_machine, cr_nmep, timeList, dateList) 



