#===============================
# Test the storm model classification
#================================

import unittest 
import os, sys
from os.path import join
import numpy as np
import xarray as xr 

# Adding the parent directory to path so that 
# skexplain can be imported without being explicitly
path = os.path.dirname(os.getcwd())
sys.path.insert(0, path)

import monte_python

class TestGetData(unittest.TestCase):
    def setUp(self):
        """ Get a real-case """
        TESTDATA_FILENAME = os.path.join(os.path.dirname(__file__), 'test_storm_mode.nc')
        ds = xr.open_dataset(TESTDATA_FILENAME)
        
        self.dbz_vals = ds['DBZ'].values
        self.rot_vals = ds['ROT'].values
        self.true_modes = ds['Storm Modes'].values
        self.true_labels = ds['Storm Labels'].values
        
        ds.close()

class TestStormModeClassifier(TestGetData):
    """ Test the object ID code """
    def test_run(self):
        """ Test that the code works and gives the right answer. """
        clf = monte_python.StormModeClassifier()
        storm_modes, labels, dbz_props = clf.classify(self.dbz_vals, self.rot_vals)       
            
        diff = np.max(self.true_labels - labels)
        self.assertEqual(diff, 0)
    
        diff = np.max(self.true_modes - storm_modes)
        self.assertEqual(diff, 0)

         
        