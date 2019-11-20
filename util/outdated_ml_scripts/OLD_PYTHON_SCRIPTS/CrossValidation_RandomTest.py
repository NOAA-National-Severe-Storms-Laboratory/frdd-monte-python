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

# USER-DEFINED PARAMETERS 
n            = 18 # Number of ensemble members  
nghbrd       = 1  # Neighborhood size for maximum filter (number of grid points) 
numStats     = 6  # Number of Ensemble statistics used
radius_max   = 1  # Radius for maximum filter applied to Azimuthal Shear  
radius_gauss = 1  # Radius used for convolutions 
ConvThres    = 0.004
year         = 'Random'
#varSet       = VariableSet( year )
#dateSet, timeSet, SummaryPath, path, mrmsVerPath, BinaryPath = allDates_and_Times( year )

clfName   = 'GradientBoosted'
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

print "Reading in the datasets..." 
filename2016 = 'RandomDateExtraction2016_nbrd=%s_rmax=%s_rgauss=%s.npz' % ( nghbrd , radius_max , radius_gauss ) 
filename2017 = 'DateExtraction2017_nbrd=%s_rmax=%s_rgauss=%s.npz' % ( nghbrd , radius_max , radius_gauss )
#examples , binary , dateColumn = combine( filename2017, filename2016 )

examples1, binary1, dateColumn1, timeColumn1, Y1, X1 = readNPZ_Files( filename2016  )
examples2, binary2, dateColumn2, timeColumn2, Y2, X2 = readNPZ_Files( filename2017  )

#clf = RandomForestClassifier( n_jobs = -1, n_estimators = 200, criterion = 'entropy', min_samples_leaf = 20)
#clf  = LogisticRegression(n_jobs=-1, solver='lbfgs')
#clf  = SGDClassifier(loss='log', penalty='elasticnet') 
clf  = GradientBoostingClassifier( loss = 'exponential', n_estimators = 500  ) 
#clf  = MLPClassifier( hidden_layer_sizes = (20,) , activation ='tanh' , alpha =1  )

X_train, y_train = examples2, binary2

clf.fit( X_train, y_train )

for date in np.unique( dateColumn1 ): 
	print date 
	_, _ , X_test, y_test = cross_validation_split( examples1, binary1, dateColumn1, date )

	clf_probs        = clf.predict_proba(X_test)[:,1]
	
	pofd, pod, SR                   = model.rocCurve( clf_probs, y_test )
	meanForecastProb, condEventFreq = model.RelabilityCurve( clf_probs, y_test)

	bss.append( model.brier_skill_score( y_test, y_train, clf_probs ) )
	AUCs.append( roc_auc_score( y_test, clf_probs ) )
	PODs.append( pod ) ; POFDs.append( pofd )
	RelX.append( meanForecastProb ) ; RelY.append( condEventFreq )
	SRs.append( SR ) 

np.savez_compressed( filename, AUCs, bss, POFDs, PODs, RelX, RelY, np.unique(dateColumn1), SRs )





