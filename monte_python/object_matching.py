import scipy 
import numpy as np
from skimage.measure import regionprops
import math 
import collections
from datetime import datetime
import itertools
import warnings 

class ObjectMatcher:
    """
    ObjectMatcher uses a total interest score based on centroid, minimum, and time 
    (optionally) displacement to match two sets of objects [1]_. Matching between
    sets can be "one-to-one" (an object in one set is matched to a single object in the second set 
    or "non-one-to-one" where multiple objects in one set can be matched to a single object in another set.
    
    Parameters
    -----------
        cent_dist_max : integer, default=None  
            maximum distance criterion for centroid displacement (in grid-point distance).
            If left as None, then the total interest score is only based on minimal distance. 
            
        min_dist_max : integer 
            maximum distance criterion for minimum displacement (in grid-point distance)
            
        time_max : integer or float, default=0
            maximum time displacment for matched objects (in minutes). 
            Only use if trying to match objects across different times. 
           
        score_thresh : float, default=0.2
            minimum total interest score to be considered a match 
            
        one_to_one : boolean , default = False
            - if True, matches must be one to one
            - if False, allows for region_b (e.g., forecasts) to be matched more than once
     
    Example usage provided in a jupyter notebook @ https://github.com/monte-flora/MontePython/
    
    References
    ----------
    .. [1] Skinner, P. S., Wheatley, D. M., Knopfmeier, K. H., Reinhart, A. E., Choate, J. J., 
           Jones, T. A., Creager, G. J., Dowell, D. C., Alexander, C. R., Ladwig, T. T., Wicker, L. J., 
           Heinselman, P. L., Minnis, P., & Palikonda, R. (2018). Object-Based Verification of a Prototype 
           Warn-on-Forecast System, Weather and Forecasting, 33(5), 1225-1250. Retrieved Mar 4, 2022, 
           from https://journals.ametsoc.org/view/journals/wefo/33/5/waf-d-18-0020_1.xml

    """
    def __init__( self, min_dist_max, cent_dist_max=None, time_max=0, score_thresh = 0.2, one_to_one = False):
        self.min_dist_max = min_dist_max
        self.cent_dist_max = cent_dist_max
        self.time_max = 1 if time_max == 0 else time_max 
        self.score_thresh = score_thresh
        self.one_to_one = one_to_one
        self.only_min_dist = True if cent_dist_max is None else False

    
    def match_objects(self, object_set_a, object_set_b, times_a=None, times_b=None):
        warnings.warn("""ObjectMatcher.match_objects is deprecated. 
                         In the future use, ObjectMatcher.match. 
                      """, DeprecationWarning) 
        
        return self.match(object_set_a, object_set_b, times_a=None, times_b=None)
        
    def match(self, object_set_a, object_set_b, times_a=None, times_b=None):
        """ 
        Match two set of objects valid at a single or multiple times.
        
        Parameters
        -----------
        
            object_set_a : {numpy.array or array-like, or list of array-like}
                Labeled data from a single or multiple times
                
            object_set_b : {numpy.array or array-like, or list of array-like}
                Labeled data from a single or multiple times. If one_to_one is
                True, then these are the labels that can be matched more than 
                once.
                
            time_a : list of strings, default is None 
                valid times (Format: '%Y%m%d %H%M') for object_set_a. 
                E.g., ['20000101 0115', '20000101 0130', '20000101 0145']
            
            time_b : list of strings, default is None 
                valid times (Format: '%Y%m%d %H%M') for object_set_a. 
                E.g., ['20000101 0115', '20000101 0130', '20000101 0145']
                
        Returns
        --------
            matched_object_set_a_labels : list of integers
                Labels in object_set_a that are matched. 
                 
            matched_object_set_b_labels : list of integers 
                Labels in object_set_b that are matched.
                Corresponding indices in matched_object_set_a_labels and
                matched_object_set_b_labels indicates which labels are 
                matched between the two sets. 
                
            cent_dist_of_matched_objects : list of 2-tuples of floats
                The (dy, dx) components of the centroid displacement between
                matched pairs. Useful for diagnosing potential biases in 
                storm motion if matching observed and forecasted objects.  
        """
        # The following code is expecting a list of 2D arrays (for the different times). However, you can provide 
        # just a single 2D array if time is not being consider for object matching
        if len(np.shape(object_set_a)) != 3:
            object_set_a = [object_set_a]
            times_a = ['20000101 0100']
            times_b = times_a
        if len(np.shape(object_set_b)) != 3:
            object_set_b = [object_set_b]
        
        regionprops_set_a = [ self._calc_object_props( set_a ) for set_a in object_set_a ]
        regionprops_set_b = [ self._calc_object_props( set_b ) for set_b in object_set_b ]
        
        all_times_a = []; all_times_b = []
        for n, set_a in enumerate(object_set_a):
            all_times_a.append([times_a[n] for m in range(len(np.unique(set_a)[1:]))])
        for n, set_b in enumerate(object_set_b):
            all_times_b.append([times_b[n] for m in range(len(np.unique(set_b)[1:]))])

        matched_object_set_a_labels  = [ ]
        matched_object_set_b_labels  = [ ]         
        cent_dist_of_matched_objects = [ ]
    
        (possible_matched_pairs, 
         cent_disp_of_possible_matched_pairs) = self._find_possible_matches( regionprops_set_a, 
                                                                             regionprops_set_b, 
                                                                             all_times_a, 
                                                                             all_times_b ) 

        sorted_possible_matched_pairs  = sorted(possible_matched_pairs, key=possible_matched_pairs.get, reverse=True)
                          
        for label_a, label_b in sorted_possible_matched_pairs:
            if self.one_to_one:
                # Check if the components of the matched pair are already matched to a better pairing.
                if label_a not in matched_object_set_a_labels and label_b not in matched_object_set_b_labels:
                    matched_object_set_a_labels.append(label_a)
                    matched_object_set_b_labels.append(label_b)
                    cent_dist_of_matched_objects.append(cent_disp_of_possible_matched_pairs[(label_a, label_b)]) 
            else:
                # For non-one-to-one matching, we are allowing labels from set b to be matched more than once.
                # Check if the components of the matched pair are already matched to a better pairing.
                if label_b not in matched_object_set_b_labels:  
                    matched_object_set_a_labels.append(label_a)
                    matched_object_set_b_labels.append(label_b)
                    cent_dist_of_matched_objects.append(cent_disp_of_possible_matched_pairs[(label_a, label_b)])
        
        if len(times_a) == 1:
            matched_object_set_a_labels = [label for (label, _) in matched_object_set_a_labels]
            matched_object_set_b_labels = [label for (label, _) in matched_object_set_b_labels]
        
        return matched_object_set_a_labels, matched_object_set_b_labels, cent_dist_of_matched_objects
    
    def _calc_object_props(self, label_image):
        """ Calculate region properties for objects.
        Parameters
        ------------
            label_image : 2D array 
                Labeled data
                
        Returns
        ---------
            skimage.measure.regionprops of the label_image
        """
        return regionprops( label_image.astype(int), label_image.astype(int) )     

    def _find_possible_matches(self, regionprops_set_a, regionprops_set_b, times_a, times_b): 
        """ Finds matches that exceed the minimum total interest score criterion.
        
        Parameters
        -----------
            regionprops_set_a : skimage.measure.regionprops object 
                Morphological properties for labeled data in object_set_a
                
            regionprops_set_b : skimage.measure.regionprops object 
                Morphological properties for labeled data in object_set_b    
            times_a, 
            times_b, 
            
        Returns
        ---------
            Dictionary of tuples of possible matched object pairs associated with their total interest score 
            Dictionary of y- and x-component of centroid displacement of possible matched object pairs             
        """
        # Find possible matched pairs 
        possible_matched_pairs = { }
        cent_disp_of_possible_matched_pairs = { }

        regionprops_set_a = list(itertools.chain.from_iterable(regionprops_set_a))
        times_a = list(itertools.chain.from_iterable(times_a))
        regionprops_set_b = list(itertools.chain.from_iterable(regionprops_set_b))
        times_b = list(itertools.chain.from_iterable(times_b))
        for region_a, time_a in zip(regionprops_set_a, times_a):
            region_a_label = (region_a.label, time_a)
            kdtree_a = scipy.spatial.cKDTree( region_a.coords )
            for region_b, time_b in zip(regionprops_set_b, times_b):
                region_b_label = (region_b.label, time_b)
                dist_btw_region_a_and_region_b, _ = kdtree_a.query( region_b.coords )
                tis, dx, dy  = self._total_interest_score( region_a.centroid, 
                                                           region_b.centroid, 
                                                           time_a, 
                                                           time_b, 
                                                           dist_btw_region_a_and_region_b )
                if round( tis, 4 ) > round( self.score_thresh, 4 ):
                    possible_matched_pairs[(region_a_label, region_b_label)] = round(tis, 4 )
                    cent_disp_of_possible_matched_pairs[(region_a_label, region_b_label)] = (dy, dx)         

        return possible_matched_pairs, cent_disp_of_possible_matched_pairs

    def _total_interest_score( self, region_a_cent, region_b_cent, time_a, time_b, dist_array, option=True ):
        """ Calculates the Total Interest Score (based on Skinner et al. 2018).
            Parameters:
                region_a_cent, centroid of region_a
                region_b_cent, centroid of region_b
                time_a, string, time valid for region_a (Format: '%Y-%m-%d %H%M')
                time_b, string, time valid for region_b (Format: '%Y-%m-%d %H%M')
                dist_array, distances between points in region_a and region_b
            Returns:
                Total Interest Score (float)
        """
        norm_min_dist = self.get_min_disp(dist_array)
        
        if self.only_min_dist:
            return norm_min_dist
        else:
            norm_cent_dist, dx, dy = self.get_cent_disp(region_a_cent, region_b_cent)
            norm_time_disp = self.get_time_difference(time_a, time_b)
            
            tis = 0.5*(norm_cent_dist + norm_min_dist)*norm_time_disp

            return tis, dx, dy if option else tis 

    def get_min_disp(self, dist_array):
        """Compute the minimal displacement between two objects"""
        if self.min_dist_max == 0:
            self.min_dist_max = 1e-8

        min_dist = np.amin(dist_array)
        min_numerator = (self.min_dist_max - min_dist)
        norm_min_dist = min_numerator / (self.min_dist_max)
        
        return norm_min_dist
        
    def get_cent_disp(self, region_a_cent, region_b_cent):
        """Compute centroid displacement betweent two objects"""
        if self.cent_dist_max == 0:
            self.cent_dist_max = 1e-8
        
        dx = region_b_cent[1] - region_a_cent[1]
        dy = region_b_cent[0] - region_a_cent[0]
        cent_dist = math.hypot(dx, dy)
        cent_numerator = (self.cent_dist_max - cent_dist )
        
        norm_cent_dist = cent_numerator / (self.cent_dist_max)
        
        return norm_cent_dist, dx, dy
        
        
    def get_time_difference(self, time_a, time_b):
        '''
        Calculates time difference between time_a and time_b in seconds.
        Assumes time_b is the most recent time. Returns the normalized 
        time difference. 
        
        Parameters
        -----------
            time_a, string
                valid time at a (Format: '%Y-%m-%d %H%M')
            time_b, string 
                valid time at b (Format: '%Y-%m-%d %H%M')
        
        Returns
        --------
            Difference of time in minutes
        '''
        datetime_format = '%Y%m%d %H%M'
        time_a = datetime.strptime( time_a, datetime_format )
        time_b = datetime.strptime( time_b, datetime_format )
        if time_b > time_a:
            diff =  time_b - time_a
        else:
            diff = time_a - time_b

        time_disp = diff.seconds / 60.
        norm_time_disp = (self.time_max - time_disp ) / (self.time_max)    
        
        # if norm_time_disp < 0:
        #   norm_time_disp = 0 
        
        return norm_time_disp

"""
Deprecated March 4, 2022.
def match_to_lsrs( object_properties, lsr_points, dist_to_lsr ):
    '''
     Match forecast objects to local storm reports.
     Parameters:
          object_properties,
          lsr_points, 
          dist_to_lsr,
     Returns:
          Dictionary where keys are the forecast label and values are binary based 
          on whether it is matched to an local storm report
    '''
    matched_fcst_objects = { }
    if len(lsr_points) == 0:
        for region in object_properties:
            matched_fcst_objects[region.label] = 0.0
        return matched_fcst_objects
    else:
        for region in object_properties:
            kdtree = scipy.spatial.cKDTree( region.coords )
            dist_btw_region_and_lsr, _ = kdtree.query( lsr_points )
            if round( np.amin( dist_btw_region_and_lsr ), 10) < dist_to_lsr:
                matched_fcst_objects[region.label] = 1.0
            else:
                matched_fcst_objects[region.label] = 0.0
        return matched_fcst_objects 
"""