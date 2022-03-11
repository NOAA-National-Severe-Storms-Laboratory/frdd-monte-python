#===============================
# Test the object matching code. 
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
        forecast_centers = [(40, 45), (40, 58), (65, 90), (90, 55), (40,20)]
        obs_centers = [(50, 55), (45, 70), (75, 110), (95, 65), (50,30)]

        self.forecast_storms,x,y = monte_python.create_fake_storms(forecast_centers)
        self.obs_storms,x,y = monte_python.create_fake_storms(obs_centers)
    
        param_set = [ {'min_thresh':10,
                                 'max_thresh':80,
                                 'data_increment':20,
                                 'area_threshold': 200,
                                 'dist_btw_objects': 50} , 
            
              {'min_thresh':25,
                                 'max_thresh':80,
                                 'data_increment':20,
                                 'area_threshold': 50,
                                 'dist_btw_objects': 10} 
            ]

        params = {'params': param_set }

        self.labels = []
        for storms in [self.forecast_storms, self.obs_storms]:
            input_data = np.where(storms > 10, storms, 0)
            storm_labels, object_props = monte_python.label(  input_data = input_data, 
                       method ='iterative_watershed', 
                       return_object_properties=True, 
                       params = params,  
                       )
            self.labels.append((storm_labels, object_props))
    
    
class TestObjectMatching(TestGetData):
    """ Test the object ID code """
    def test_object_matching(self):
        matcher = monte_python.ObjectMatcher(cent_dist_max = 10, 
                                       min_dist_max = 10, time_max=0, score_thresh=0.2, 
                           one_to_one = True)

        # Match the objects 
        matched_0, matched_1, _ = matcher.match(self.labels[0][0], self.labels[1][0])
        
        self.assertEqual(len(matched_0), 3)
        self.assertEqual(len(matched_1), 3)
        
        matcher = monte_python.ObjectMatcher(cent_dist_max = 10, 
                                       min_dist_max = 10, time_max=0, score_thresh=0.2, 
                           one_to_one = False)

        # Match the objects 
        matched_0, matched_1, _ = matcher.match(self.labels[0][0], self.labels[1][0])
        
        self.assertEqual(len(matched_0), 4)
        self.assertEqual(len(matched_1), 4)
        
        
        
        
        
        
        
    
    
    