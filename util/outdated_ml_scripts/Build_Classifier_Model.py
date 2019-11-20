#!/usr/bin/env python
#===================================================================================================
# MODULE IMPORTS
#===================================================================================================
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import pickle 

# Personal Modules 
from PHD_cookbook import *

clfName = 'RandomForest' 
print "\n Classifier: %s" % ( clfName )  

def cross_validation_split(  Examples, BinaryOutput, dateColumn, date ):
        """
        Leave a day out as testing, use the rest as training. 
        """
        rowIndices = np.where(( dateColumn == date ))

        rowIndicesMin = np.amin( rowIndices[0] )
        rowIndicesMax = np.amax( rowIndices[0] )

        X_test    = Examples[rowIndicesMin:rowIndicesMax+1, : ]
        y_test    = BinaryOutput[rowIndicesMin:rowIndicesMax+1]

        X_train = np.delete( Examples    , rowIndices, axis = 0)
        y_train = np.delete( BinaryOutput, rowIndices)

        return X_test, X_train, y_test, y_train

def combine(  ):
        examples2016, binary2016, dateColumn2016 = readNPZ_Files( 'DateExtraction2016.npz'   )
        examples2017, binary2017, dateColumn2017 = readNPZ_Files( 'DateExtraction2017.npz'  )

        examplesCombined = np.concatenate((examples2016, examples2017), axis = 0 )
        binaryCombined   = np.concatenate((binary2016, binary2017), axis = 0 )
        dateCombined     = np.concatenate((dateColumn2016, dateColumn2017), axis = 0 )

        return examplesCombined , binaryCombined , dateCombined


"""
FIT MODEL TO THE TRAINING DATASET AND THEN OUTPUT FOR USE LATER
"""
examplesCombined , binaryCombined , dateCombined = combine( )
date = 20170516 
X_test, X_train, y_test, y_train = cross_validation_split( examplesCombined, binaryCombined, dateCombined, date )

if clfName == 'RandomForest':
                clf = RandomForestClassifier( n_jobs = -1, n_estimators = 100, criterion = 'entropy', max_depth = 10, min_samples_leaf = 20)
elif clfName == 'Logistic':
                clf  = LogisticRegression(C=1., solver='lbfgs')

print "Training the model..." 
clf.fit( X_train, y_train )
filename = 'RandomForest_ClassifierModel.sav' 
print "Finished and saving the model..." 
pickle.dump( clf, open(filename, 'wb') )


