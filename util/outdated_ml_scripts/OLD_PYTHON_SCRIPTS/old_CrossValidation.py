#!/usr/bin/env python
#===================================================================================================
# MODULE IMPORTS
#===================================================================================================
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model.stochastic_gradient import SGDClassifier

from sklearn.calibration import calibration_curve
import modelEvaluationFunctions as model

# Personal Modules 
from PHD_cookbook import *

filename1 = 'DateExtraction2017_nbrd=0_rmax=1_rgauss=1.npz'
filename2 = 'TestDateExtraction2017_nbrd=0_rmax=1_rgauss=1.npz'

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

clfName   = 'RF'
year      = '2017IndTimes'
filename  = '%s%s_ForecastVerificationStats' % ( clfName, year )
print " Classifier:    ", clfName 
print " Year:          ", year
print ""

AUCs  = [ ]
POFDs = [ ] 
PODs  = [ ] 
RelX  = [ ] 
RelY  = [ ]
bss   = [ ] 
SRs   = [ ] 

year = 2017 
print "Reading in the datasets..." 
examples2017,     binary2017,     dateColumn2017,     timeColumn2017, Y, X = readNPZ_Files( filename1  )
TestExamples2017, TestBinary2017, TestDateColumn2017, TestTimeColumn2017   = readNPZ_Files( filename2  )
dateSet, timeSet, SummaryPath, path, mrmsVerPath, BinaryPath               = allDates_and_Times( year )

for date in np.unique( dateColumn2017 ): 
	print date 
	print "\n Splitting up training dataset" 
	X_train, y_train = cross_validation_split( examples2017, binary2017, dateColumn2017, date )

	print "Training..." 
	clf = RandomForestClassifier( n_jobs = -1, n_estimators = 200, criterion = 'entropy', min_samples_leaf = 20)
		#clf  = LogisticRegression(n_jobs=-1, solver='lbfgs')
		#clf  = SGDClassifier(loss='log', penalty='elasticnet') 
		#clf  = GradientBoostingClassifier( loss = 'exponential', n_estimators = 500  ) 
		#clf  = MLPClassifier( hidden_layer_sizes = (20,) , activation ='tanh' , alpha =1  )  
	clf.fit( X_train, y_train )
	
	for time in timeSet:
		print time 
		X_test, y_test   = cross_validation_time_split( TestExamples2017, TestBinary2017, TestTimeColumn2017, TestDateColumn2017, int(time), date )
		if X_test == [ ] : 
			continue 
		else:

		#nghbrd = 0 
		#ML         = MachineLearningProject( n, nghbrd, path, varSet, numStats, str(date), time )
		#_mrms      = MRMS(mrmsVerPath, str(date), time='1900', nghbrd= 0, threshold=15., nt=97 )
		#X_test     = ML.convert2Ddata_to_MLExamples( mrmsVerPath, _mrms )
		#y_test, _  = _mrms.determineLabel( BinaryPath, radius_max, radius_gauss, ConvThres, time  )
	
			clf_probs = clf.predict_proba(X_test)[:,1]
	
			pofd, pod, SR                   = model.rocCurve( clf_probs, y_test )
			meanForecastProb, condEventFreq = model.RelabilityCurve( clf_probs, y_test)

			bss.append( model.brier_skill_score( y_test, y_train, clf_probs ) )
			AUCs.append( roc_auc_score( y_test, clf_probs ) )
			PODs.append( pod ) ; POFDs.append( pofd )
			RelX.append( meanForecastProb ) ; RelY.append( condEventFreq )
			SRs.append( SR ) 

np.savez_compressed( filename, AUCs, bss, POFDs, PODs, RelX, RelY, np.unique(dateColumn2017), SRs )





