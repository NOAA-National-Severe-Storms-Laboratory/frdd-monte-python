#===============================
# Test the object QC code
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

        self.storms,_,_ = monte_python.create_fake_storms(centers, add_small_area=True)
        self.qcer = monte_python.QualityControler()
        self.storm_labels, self.object_props = monte_python.label( self.storms, 
                      method ='single_threshold', 
                      return_object_properties=True, 
                      params = {'bdry_thresh':25} )
        
        
class TestObjectQC(TestGetData):
    """ Test the object ID code """
    def test_qc(self):
        """ Test removing based on small area """
        qc_params = [('min_area', 35)]
        new_labels, new_props = self.qcer.quality_control(self.storms, 
                                                          self.storm_labels, 
                                                          self.object_props, qc_params)

        # One object was removed due to its small area. 
        self.assertEqual(len(np.unique(self.storm_labels)[1:])-1, len(np.unique(new_labels)[1:]))
    
    def test_complex_qc(self):
        """ Test removing based on area and intensity """
        qc_params = [('min_area', 35), ('max_thresh', (60, 100))]
        new_labels, new_props = self.qcer.quality_control(self.storms, 
                                                     self.storm_labels, 
                                                     self.object_props, qc_params)
        
        # One object was removed due to its small area and two for their intensity 
        self.assertEqual(len(np.unique(self.storm_labels)[1:])-3, len(np.unique(new_labels)[1:]))
        
        
    
    
    
    
    