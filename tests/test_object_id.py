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
    
        
class TestObjectID(TestGetData):
    """ Test the object ID code """
    def test_base_method(self):
        """ Giving a bad method and raising the error """
        possible_methods = ['watershed', 'single_threshold', 'iterative_watershed']
        method = 'single threshold'
        with self.assertRaises(Exception) as ex:
            monte_python.label( input_data = self.storms,
                                   method =method, 
                                   return_object_properties=True, 
                                   params = {'bdry_thresh':20} )
            
        except_msg = f"{method} is not a valid method. The valid methods include {possible_methods}."
        self.assertEqual(ex.exception.args[0], except_msg)
    
    def test_missing_params_single_thresh(self):
        """ Missing bdry_thresh for single threshold """
        with self.assertRaises(Exception) as ex:
            monte_python.label( input_data = self.storms,
                                   method ='single_threshold', 
                                   return_object_properties=True, 
                                   params = {'thresh':20} )
            
        except_msg = """
                           bdry_thresh is not in params. Must provide a boundary threshold for the single
                           threshold method
                           """
        self.assertMultiLineEqual(ex.exception.args[0], except_msg)
    
    
    def test_single_threshold(self):
        """ Test the single threshold method """
        storm_labels, object_props = monte_python.label( input_data = self.storms,
                                   method ='single_threshold', 
                                   return_object_properties=True, 
                                   params = {'bdry_thresh':20} )
        true_areas = [143, 375, 177, 181]
        areas = [region.area for region in object_props]
        
        # Test that the areas are right. 
        np.testing.assert_allclose(true_areas, areas, atol=5)  
        
        # Test that the labels are right. 
        np.testing.assert_array_equal(np.unique(storm_labels), np.array([0,1,2,3,4]))

        
    def test_missing_watershed_params(self):
        """ Check that watershed does proceed if parameters are missing """
        pass
        #storm_labels, object_props = monte_python.label(  input_data = self.storms, 
        #               method ='watershed', 
        #               return_object_properties=True, 
        #               params = {'min_thresh':25,
        #                         'max_thresh':80,
        #                         'area_threshold': 150,
        #                         'dist_btw_objects': 50} 
        #               )
        
        
    def  test_watershed_method(self):
        """ Test the watershed method """
        storm_labels, object_props = monte_python.label(  input_data = self.storms, 
                       method ='watershed', 
                       return_object_properties=True, 
                       params = {'min_thresh':25,
                                 'max_thresh':80,
                                 'data_increment':20,
                                 'area_threshold': 150,
                                 'dist_btw_objects': 50} 
                       )
        true_areas = [149, 145, 151, 147, 71]
        areas = [region.area for region in object_props]
        
        # Test that the areas are right. 
        np.testing.assert_allclose(true_areas, areas, atol=5) 
        
        # Test that the labels are right. 
        np.testing.assert_array_equal(np.unique(storm_labels), np.array([0,1,2,3,4,5]))
        
    def test_iterative_watershed_method(self):
        """ Test the iterative watershed method """
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

        # This is not an important set for the watershed, but 
        # simply something to make this fake example work. 
        input_data = np.where(self.storms > 10, self.storms, 0)
        storm_labels, object_props = monte_python.label(  input_data = input_data, 
                       method ='iterative_watershed', 
                       return_object_properties=True, 
                       params = params,  
                       )

        true_areas = [261, 289, 363, 250, 312]
        areas = [region.area for region in object_props]
        
        # Test that the areas are right. 
        np.testing.assert_allclose(true_areas, areas, atol=5) 
        
        # Test that the labels are right. 
        np.testing.assert_array_equal(np.unique(storm_labels), np.array([0,1,2,3,4,5]))
        
        
        
        
        
    
    

