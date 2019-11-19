import scipy
from scipy import spatial
import numpy as np
import skimage.measure 
from skimage.measure import regionprops
import math 
import collections
from datetime import datetime
import itertools

class ObjectTracking:
    """
    ObjectMatching uses a total interest score (Davis et al. 2006) based on centroid and minimum displacement
    to match two sets of objects.
    Attributes:
        dist_max (int), maximum distance criterion for both centroid and minimum displacement ( in grid-points)
        time_max (int), maximum time displacment for matched objects (in minutes)
        score_thresh (float), minimum total interest score to be considered a match (default = 0.2)
        one_to_one (boolean), Allows for region_b (e.g., forecasts) to be matched more than once
                      = True, if matches must be one_to_one 
        only_min_dist (boolean), = True for total interest score based solely on minimum displacement  

    Example usage: 
        from ObjectMatching import ObjectMatching 
        obj_match = ObjectMatching( dist_max = 5, one_to_one = True ) # 15 km maximum distance 
        
        matched_object_set_a_labels, matched_object_set_b_labels, cent_dist_of_matched_objects = obj_match.match_objects( object_set_a, object_set_b )
    """
    def __init__( self, dist_max, score_thresh = 0.2, one_to_one = False, only_min_dist = False ):
        self.dist_max = dist_max
        self.one_to_one = one_to_one

    def track_objects( self, objects, original_data): 
        """ Tracks objects in time. 
        Args:
            objects, 3-d array or stack of 2D arrays 
            original_data, 3-d array or stack of 2D arrays 

        """
        tracked_objects = np.copy( objects ) 
        for t in np.arange(tracked_objects.shape[0]-1):
            current_objects = tracked_objects[t,:,:]
            future_objects = tracked_objects[t+1,:,:]
            matched_labels_i, matched_labels_f  = self.match_objects( current_objects, future_objects, original_data[t], original_data[t+1])
            for label_i, label_f in zip(matched_labels_i, matched_labels_f):
                tracked_objects[t+1, future_objects == label_f] = label_i

        print (self.calc_duration(time_range=np.arange(tracked_objects.shape[0]-1), objects=tracked_objects))
        return tracked_objects 

    def match_objects( self, objects_a, objects_b, original_data_a, original_data_b):
        """ Match two set of objects valid at a single or multiple times.
        Args:
            object_set_a, 2D array or list of 2D arrays, object labels at a single or multiple times
            object_set_b, 2D array or list of 2D arrays, object labels at a single or multiple times
            time_a, lists of strings of valid times (Format: '%Y%m%d %H%M') for object_set_a (default=None)
            time_b, lists of strings of valid times (Format: '%Y%m%d %H%M') for object_set_b (default=None)
        Returns:
            Lists of matched labels in set a, matched labels in set b,
            and tuples of y- and x- components of centroid displacement of matched pairs
        """
        matched_object_set_a_labels  = [ ]
        matched_object_set_b_labels  = [ ]
        
        possible_matched_pairs = self.find_possible_matches( objects_a, objects_b, original_data_a, original_data_b ) 

        sorted_possible_matched_pairs  = sorted( possible_matched_pairs, key=possible_matched_pairs.get, reverse=True ) 
        for label_a, label_b in sorted_possible_matched_pairs:
            if self.one_to_one:
                if label_a not in matched_object_set_a_labels and label_b not in matched_object_set_b_labels: #otherwise pair[0] has already been matched!
                    matched_object_set_a_labels.append( label_a )
                    matched_object_set_b_labels.append( label_b )
            else:
                if label_a not in matched_object_set_a_labels: #otherwise pair[0] has already been matched!
                    matched_object_set_a_labels.append( label_a )
                    matched_object_set_b_labels.append( label_b )
       
        return matched_object_set_a_labels, matched_object_set_b_labels
    
    def find_possible_matches( self, objects_a, objects_b, original_data_a, original_data_b ): 
        """ Finds matches that exceed the minimum total interest score criterion.
        Args: 
            regionprops_set_a, skimage.measure.regionprops for object_set_a
            regionprops_set_b, skimage.measure.regionprops for object_set_b
            times_a, 
            times_b, 
        Returns: 
            Dictionary of tuples of possible matched object pairs associated with their total interest score 
            Dictionary of y- and x-component of centroid displacement of possible matched object pairs             
        """
        # Find possible matched pairs 
        possible_matched_pairs = { }
        cent_disp_of_possible_matched_pairs = { }
        
        unique_labels_a = np.unique(objects_a)[1:]
        unique_labels_b = np.unique(objects_b)[1:]

        label_a_max_coords = {label: np.unravel_index( np.where( objects_a == label, original_data_a, 0.0 ).argmax( ), np.where( objects_a == label, original_data_a, 0.0 ).shape ) for label in unique_labels_a}
        label_b_max_coords = {label: np.unravel_index( np.where( objects_b == label, original_data_b, 0.0 ).argmax( ), np.where( objects_b == label, original_data_b, 0.0 ).shape ) for label in unique_labels_b}
        for label_a in unique_labels_a: 
            for label_b in unique_labels_b: 
                dist_btw_region_a_and_region_b = self.calc_distance( label_a_max_coords[label_a], label_b_max_coords[label_b] )
                if dist_btw_region_a_and_region_b < self.dist_max: 
                    possible_matched_pairs[(label_a, label_b)] = dist_btw_region_a_and_region_b

        return possible_matched_pairs

    def calc_distance(self, pair_a, pair_b ): 
        """ Calculates the distance between pair_a and pair_b """
        return np.sqrt( (pair_a[0] - pair_b[0])**2 + (pair_a[1] - pair_b[1])**2 )

    def calc_duration(self, time_range, objects): 
        """ Calculates the duration of storms """
        # objects (time, y, x)
        object_duration = { }
        for t in time_range: 
            for label in np.unique( objects[t] )[1:]:
                if label not in object_duration.keys( ):
                    object_duration[label] = 1 
                else:
                    object_duration[label] += 1
        return object_duration


    


