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
import pickle 

# Personal Modules 
from PHD_cookbook import *

# USER-DEFINED PARAMETERS 
nghbrd       = 0    # Neighborhood size for maximum filter (number of grid points) 
year         = '2017'

clfName   = 'ElasticNets'
print " Classifier:    ", clfName 
print " Year:          ", year
print ""

AUCs  = [ ]

print "Reading in the datasets..." 
filename2017 = '/home/monte.flora/PHD_RESEARCH/NPZ_FILES/SmartSample2017_nbrd=%s.npz' % ( nghbrd )
examples, binary, dateColumn, timeColumn, Y, X = readNPZ_Files( filename2017  )

for date in np.unique( dateColumn ): 
	print date 
	X_train, y_train , X_test, y_test = cross_validation_split( examples, binary, dateColumn, date )

	#clf  = RandomForestClassifier( n_jobs = -1, n_estimators = 200, criterion = 'entropy', min_samples_leaf = 20)
	#clf  = LogisticRegression(n_jobs=-1, solver='lbfgs')
	clf  = SGDClassifier(loss='log', penalty='elasticnet') 
	#clf  = GradientBoostingClassifier( loss = 'exponential', n_estimators = 100  ) 
	#clf  = MLPClassifier( hidden_layer_sizes = (5,)  )  
	clf.fit( X_train, y_train )
	clf_probs  = clf.predict_proba(X_test)[:,1]
	
	AUCs.append( roc_auc_score( y_test, clf_probs ) )

index = np.argmax( AUCs ) 
uniqueDates = np.unique( dateColumn ) 
bestDate = uniqueDates[index] 

print AUCs[index], bestDate 

X_train, y_train , X_test, y_test = cross_validation_split( examples, binary, dateColumn, bestDate )

print "Training the model..."
clf.fit( X_train, y_train )
filename = '/home/monte.flora/PHD_RESEARCH/SAV_FILES/%s_ClassifierModel_nghbrd=%s.sav' % ( clfName, nghbrd ) 
print "Finished and saving the model..."
pickle.dump( clf, open(filename, 'wb') )







