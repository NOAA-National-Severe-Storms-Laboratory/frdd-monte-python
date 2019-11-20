import pickle 
import os, sys  
from PHD_cookbook import *
import timeit

# User-defined parameters 
n         = 18 
nghbrd    = 3
numStats  = 6
threshold = 0.0038
varSet    = [ 'LCL_ML' , 'SRH_0TO1' , 'SRH_0TO3' , 'UH_0TO2' , 'UH_2TO5' ,
'WZ_0TO2', 'CAPE_ML', 'CIN_ML', 'DZ_MAX', 'P_SFC', 'SHEAR_U_0TO1', 'SHEAR_U_0TO6', 'SHEAR_V_0TO1',
'STP_ML', 'TD_2', 'TH_2', 'TH_E_ML', 'T_2', 'U_10', 'QV_2', 'QV_ML' ]

# Load the classifier model 
filename          = 'RandomForest_ClassifierModel.sav'
loaded_classifier = pickle.load(open(filename, 'rb')) 

SummaryPath  = '/work1/skinnerp/2017_newse_post/summary_files/'
path         = '/work1/skinnerp/2017_newse_post/summary_files/'
mrmsPath     = '/work1/skinnerp/MRMS_verif/mrms_cressman/'

dateSet      = os.listdir( mrmsPath )
SummaryPath += dateSet[0] + '/'
timeSet      = os.listdir ( SummaryPath )

startPath = '/home/monte.flora/PHD_RESEARCH/ClassiferProbability/RandomForest/'
 
for date in dateSet:
	start = timeit.default_timer()
	Path1 = startPath + date +'/'
	if not os.path.exists(Path1):
    		os.makedirs(Path1)
	for time in timeSet:
		filename    = Path1 + 'ti_%s' % ( time ) 
		print filename 
		ML          = MachineLearningProject(n , nghbrd , path, varSet , numStats, date, time )
		X_case      = ML.convert2Ddata_to_MLExamples( )
	        clf_probs   = loaded_classifier.predict_proba(X_case)[:,1]	#1D probabilities 
	  	np.save( filename , clf_probs ) 	
		
	stop = timeit.default_timer()
        total_time = stop - start
        mins, secs = divmod(total_time, 60)
        hours, mins = divmod(mins, 60)
        sys.stdout.write("Total running time: %d:%d:%d.\n"  % (hours, mins, secs))
		
"""
# Output binary output into 1D array to match the classifier probability array 
# np.where(( data > threshold ))  
# Generic Filename:  ClassifierProbability/RandomForest/20160516/ti=0000.npz
startPath = '/home/monte.flora/PHD_RESEARCH/MRMS_LOWLEVELROTATION/'
for date in dateSet: 
	Path1 = startPath + date +'/'
        for time in timeSet:
                filename    = Path1 + 'ti_%s' % ( time )
                print filename
		ML           = MachineLearningProject(n, nghbrd, path, varSet, numStats, date, time )
                ensMemSet    = ML.ensembleStringSet(  )
                nt, ny, nx   = ML.dimensions(  )
		mrms_        = MRMS(mrmsPath, date, time, nghbrd, threshold, ny, nx)                 
		binaryOutput = np.where(( mrms_.calc_rotationTrack(  ) >= threshold, 1, 0 )) 
                np.savez_compressed( filename , clf_probs )
"""

# Read in classifier probabilities and binary output from azimuthal shear 

# Calculate ROC and Reliability Curve, AUC, BSS for each date and initialization time 

# E.g., plot BSS as function of initialization time or date 
# E.g., average BSS over the initialization times and dates for overall skill   




