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
from itertools import product

# USER-DEFINED PARAMETERS 
SummaryPath  = '/work1/skinnerp/2016_news_e_post/summary_files/'
path         = '/work1/skinnerp/2016_news_e_post/summary_files/'
MRMSPath     = '/work1/skinnerp/2016_MRMS_verif/mrms_cressman/' 

n            = 18  # Number of ensemble members  
nghbrd       = 3   # Neighborhood size (number of grid points) 
numStats     = 6   # Number of Ensemble statistics used
numPoints    = 2650 # Number of randomly sampled points in the 2D domain 
threshold    = 0.0038
varSet       = [ 'LCL_ML' , 'SRH_0TO1' , 'SRH_0TO3' , 'UH_0TO2' , 'UH_2TO5' , 
'WZ_0TO2', 'CAPE_ML', 'CIN_ML', 'DZ_COMP', 'P_SFC', 'SHEAR_U_0TO1', 'SHEAR_U_0TO6', 'SHEAR_V_0TO1',  
'STP_ML', 'TD_2', 'TH_2', 'TH_E_ML', 'T_2', 'U_10', 'QV_2', 'QV_ML' ]

dateSet      = os.listdir(MRMSPath) 
dateSet.remove('20160429_ok')
dateSet.remove('20160429') 
dateSet.remove('20151223') 
dateSet.remove('20160331') 
dateSet.remove('20160410') 
SummaryPath += dateSet[0] + '/'
timeSet      = [f for f in os.listdir( SummaryPath ) if not f.endswith('.nc')]
if '0330' in timeSet: 
	timeSet.remove('0330') 

def main( path, dateSet, timeSet, varSet ):
        mainData  = [ ] ; outputData = [ ]
	j = range(20, 230) 
	i = range(20, 230)  
        for date in dateSet: 
                print date
		originalDate = date
                for time in timeSet: 
                        ML               = MachineLearningProject(n, nghbrd, path, varSet, numStats, date, time )
			ensMemSet        = ML.ensembleStringSet(  )
                        if len (ensMemSet ) == 0: 
                                continue
                        nt, ny, nx       = ML.dimensions( )
			mrms_            = MRMS(MRMSPath, originalDate, time, nghbrd, threshold, ny, nx)
			mrms_y , mrms_x  = mrms_.locateAziShear(  )	
			mrmsPair         = zip(mrms_x, mrms_y)
			y, x             = RandomPoints( j, i, numPoints  )
			allVar_allEnsMem = ML.append_stats_and_ensMem( )
                        allVar           = ML.CalcStats( allVar_allEnsMem) # Dim: Num of Var, ny, nx 
                        for row in range( len(y) ):
                                mainData.append( np.ravel( allVar[ :, :, int(y[row]) , int(x[row]) ] ) )
				outputData.append( DetermineBinaryOutput( mrmsPair , int(y[row]), int(x[row])  ) ) 	

        return mainData, outputData

if __name__ == "__main__":
        mainData, outputData =  main(path, dateSet, timeSet, varSet)
        np.savez( 'DataTable_Random', mainData)
	np.savez( 'Output_Random', outputData )



	





