#!/usr/bin/env python
#===================================================================================================
# MODULE IMPORTS
#===================================================================================================
from sklearn.isotonic import IsotonicRegression
import pickle
from sklearn.metrics import brier_score_loss
from PHD_cookbook import *

# USER-DEFINED PARAMETERS 
nghbrd       = 0     # Neighborhood size for maximum filter (number of grid points) 

clfName      = 'NeuralNet' 
bs   	     = [ ] 

print "Reading in the datasets..." 
filename2016 = '/home/monte.flora/PHD_RESEARCH/NPZ_FILES/ClimoData2016_nbrd=%s.npz' % ( nghbrd )
examples, binary, dateColumn, timeColumn, Y, X = readNPZ_Files( filename2016  )

uniqueDates = np.unique( dateColumn ) 
count = int( len( uniqueDates ) / 2. ) 

filename = '/home/monte.flora/PHD_RESEARCH/SAV_FILES/%s_ClassifierModel_nghbrd=%s.sav' % ( clfName, nghbrd )
loaded_classifier = pickle.load(open(filename, 'rb'))  

for date in uniqueDates[:count] :  
	print date 
	X_train, y_train , X_test, y_test = cross_validation_split( examples, binary, dateColumn, date )
	
	clf_isotonic = CalibratedClassifierCV(loaded_classifier, cv='prefit', method='isotonic')
	clf_isotonic.fit( X_train, y_train ) 
	iso_probs = clf_isotonic.predict_proba(X_test)[:,1]

	bs.append( brier_score_loss( y_test , iso_probs ) )  
		
index = np.argmax( bs )
uniqueDates = np.unique( dateColumn )
bestDate = uniqueDates[index]

print bs[index], bestDate

X_train, y_train , X_test, y_test = cross_validation_split( examples, binary, dateColumn, bestDate )

print "Training the model..."
clf_isotonic = CalibratedClassifierCV(loaded_classifier, cv='prefit', method='isotonic')
clf_isotonic.fit( X_train, y_train )
filename = '/home/monte.flora/PHD_RESEARCH/SAV_FILES/%s_IsotonicRegressionModel_nghbrd=%s.sav' % ( clfName, nghbrd )
print "Finished and saving the model..."
pickle.dump( clf_isotonic, open(filename, 'wb') )





