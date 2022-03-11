#===============================
# Test the object ID code 
#================================

import unittest 
import os, sys
import numpy as np

# Adding the parent directory to path so that 
# skexplain can be imported without being explicitly
path = os.path.dirname(os.getcwd())
sys.path.append(path)

import monte_python

class TestGetData(unittest.TestCase):
    def setUp(self):
        """ Create some fake storms for testing"""
        centers = [(40, 45), (40, 58), (65, 90), (90, 55), (40,20)]

        self.storms,_,_ = monte_python.create_fake_storms(centers)
    