import scipy
from scipy import spatial
import numpy as np
import skimage.measure 
from skimage.measure import regionprops
import math 
import collections
from datetime import datetime
import itertools

class ObjectTracker:
    """
    ObjectTracker performs simple object tracking by linking together objects from time 
    step to time step with the most overlap.

    Attributes
    -----------
        one_to_one : boolean , default = False
            - if True, matches must be one to one
            - if False, allows for region_b (e.g., forecasts) to be matched more than once
          
        percent_overlap : float, default = 0.0
            The amount of overlap to be consider a possible match for tracking. Default assumes 
            that overlap is cause for a possible match. 
  
    """
    def __init__( self, one_to_one = False, percent_overlap=0.0):
        self.one_to_one = one_to_one
        self.percent_overlap = percent_overlap

    def track_objects(self, objects): 
        """ Tracks objects in time. 
        
        Parameters:
        ----------------------
            objects, 3-d array or stack of 2D arrays 
            original_data, 3-d array or stack of 2D arrays 

        """
        objects_copy = np.copy(objects)
        # Re-label so that objects across time each have a each label. 
        tracked_objects = self.get_unique_labels(objects_copy)
        
        for t in np.arange(tracked_objects.shape[0]-1):
            current_objects = tracked_objects[t,:,:]
            future_objects = tracked_objects[t+1,:,:]
            labels_before, labels_after  = self.match_objects(current_objects, future_objects,)
            
            areas_before, areas_after = self._get_area(current_objects), self._get_area(future_objects)
            
            # Check for mergers.
            self.check_for_mergers(labels_before, labels_after, areas_before)
            
            # Re-label an object if matches (this is where the tracking is done) 
            for label_i, label_f in zip(labels_before, labels_after):
                tracked_objects[t+1, future_objects == label_f] = label_i

        # Do a final re-label so that np.max(relabel_objects) == number of tracked objects. 
        relabeled_objects = self.relabel(tracked_objects)
        return relabeled_objects
    
    def _get_area(self, arr):
        """
        Get the area of each object and return a dict.
        """
        return {label : np.count_nonzero(arr==label) for label in np.unique(arr)[1:]}
        

    def check_for_mergers(self, labels_before, labels_after, areas_before):
        """
        Mergers are cases where there are non-unique labels in the after step
        (i.e., two or more labels become one). For longer tracks, 
        the merged object label is inherited from the largest object in 
        the merger. 
    
        E.g., 
    
        labels_before = [1,2,3,4,4] - > [2,3,4,4] 
        labels_after  = [5,5,6,7,8] - > [5,6,7,8]
    
    
        Parameters
        --------------------
    
        Returns
        --------------------
        """
        # Determine if there is a merged based on non-unqique labels. 
        unique_labels_before, counts_before = np.unique(labels_before, return_counts=True)
        if any(counts_after>1):
            # Get the labels that non-unique. 
            merged_labels = unique_labels_after[counts_after>1]

            for label in merged_labels:
                # This should be 2 or more labels (which are being merged together).
                potential_label_for_merged_obj = [l for i, l in enumerate(labels_before) if labels_after[i] == label]
                print(f'{potential_label_for_merged_obj=}')
                # Sort the potential merged object labels by area. Keep the largest object and remove the 
                # others. 
                inds = np.argsort([areas_before[label] for label in potential_label_for_merged_obj])[::-1]
                labels_sorted = np.array(potential_label_for_merged_obj)[inds]
                for label in labels_sorted[1:]:
                    index = labels_before.index(label)
                    del labels_before[index]
                    del labels_after[index]
    
        return labels_before, labels_after

    def check_for_splits(labels_before, labels_after, areas_after):
        """
        Splits are cases where there are non-unique labels in the before step
        (i.e., one labels becomes two or more). For longer tracks, 
        of the split labels, the largest one inherits the before step label. 
    
    
        labels_before = [1,2,3,4,4] - > [1,2,3,4] 
        labels_after  = [5,5,6,7,8] - > [5,5,6,7]
    
    
        Parameters
        --------------------
    
        Returns
        --------------------
        """
        unique_labels_after, counts_after = np.unique(labels_after, return_counts=True)
        if any(counts_before>1): 
            split_labels = unique_labels_before[counts_before>1]
    
            for label in split_labels: 
                # This should be 2 or more labels (which are being merged together).
                potential_label_for_split_obj = [l for i, l in enumerate(labels_after) if labels_before[i] == label]  
                print(f'{potential_label_for_split_obj=}')
                # Sort the potential split object labels by area. Keep the largest object and remove the 
                # others. 
                inds = np.argsort([areas_after[label] for label in potential_label_for_split_obj])[::-1]
                labels_sorted = np.array(potential_label_for_split_obj)[inds]
        
                for label in labels_sorted[1:]:
                    index = labels_after.index(label)
                    del labels_before[index]
                    del labels_after[index]
    
        return labels_before, labels_after 
    
    
    def get_unique_labels(self, objects):
        """Ensure that initially, each object has a unique label"""
        cumulative_objects = np.cumsum([np.max(objects[i]) for i in range(len(objects))])
        
        unique_track_set = [objects[0,:,:]]
        for i in range(1, len(objects)):
            track = objects[i,:,:]
            where_zero = track==0
            unique_track = track+cumulative_objects[i-1]
            unique_track[where_zero]=0
            unique_track_set.append(unique_track)
    
        return np.array(unique_track_set)

    def relabel(self, objects):
        """Re-label objects"""
        relabelled_objects = np.copy(objects)
        #Ignore the zero label
        unique_labels = np.unique(objects)[1:]
        for i, label in enumerate(unique_labels):
            relabelled_objects[objects==label] = i+1
    
        return relabelled_objects

    def match_objects(self, objects_a, objects_b):
        """ Match two set of objects valid at a single or multiple times.
        Args:
            object_set_a, 2D array or list of 2D arrays, object labels at a single or multiple times
            object_set_b, 2D array or list of 2D arrays, object labels at a single or multiple times
        Returns:
            Lists of matched labels in set a, matched labels in set b,
            and tuples of y- and x- components of centroid displacement of matched pairs
        """
        matched_object_set_a_labels  = [ ]
        matched_object_set_b_labels  = [ ]
        
        possible_matched_pairs = self.find_possible_matches(objects_a, objects_b) 

        # Reverse means large values first! 
        sorted_possible_matched_pairs  = sorted(possible_matched_pairs, key=possible_matched_pairs.get, reverse=True) 
        for label_a, label_b in sorted_possible_matched_pairs:
            if self.one_to_one:
                if label_a not in matched_object_set_a_labels and label_b not in matched_object_set_b_labels: 
                    #otherwise pair[0] has already been matched!
                    matched_object_set_a_labels.append(label_a)
                    matched_object_set_b_labels.append(label_b)
            else:
                if label_a not in matched_object_set_a_labels: 
                    #otherwise pair[0] has already been matched!
                    matched_object_set_a_labels.append(label_a)
                    matched_object_set_b_labels.append(label_b)
       
        return matched_object_set_a_labels, matched_object_set_b_labels
    
    def percent_intersection(self, region_a, region_b):
        """
        Compute percent overlap with the region coordinates0
        """
        # Converts the input to tuples so they can be used as
        # keys (i.e., become hashable)
        region_a_coords = list(set(map(tuple, region_a.coords)))
        region_b_coords = list(set(map(tuple, region_b.coords)))
    
        denom = (len(region_a_coords)+ len(region_b_coords))
        percent_overlap_coords = float(len(list(set(region_a_coords).intersection(region_b_coords))) / denom)
    
        return percent_overlap_coords
    
    def find_possible_matches(self, objects_a, objects_b): 
        """ Finds matches based on amount of intersection between objects at time = t and time = t+1.
        Args: 
            regionprops_set_a, skimage.measure.regionprops for object_set_a
            regionprops_set_b, skimage.measure.regionprops for object_set_b
        Returns: 
            Dictionary of tuples of possible matched object pairs associated with their total interest score 
            Dictionary of y- and x-component of centroid displacement of possible matched object pairs             
        """
        
        # Re-new object 
        object_props_a, object_props_b = [regionprops(objects.astype(int)) for objects in [objects_a, objects_b]]
        
        # Find possible matched pairs 
        possible_matched_pairs = { }
        for region_a in object_props_a:
            for region_b in object_props_b:
                percent_overlap = self.percent_intersection(region_a, region_b)
                if percent_overlap > self.percent_overlap:
                    possible_matched_pairs[(region_a.label, region_b.label)] = percent_overlap
        
        return possible_matched_pairs
        
    def calc_duration(self, time_range, objects): 
        """ Calculates the duration of storms """
        
        #print (self.calc_duration(time_range=np.arange(tracked_objects.shape[0]-1), objects=tracked_objects))
        
        # objects (time, y, x)
        object_duration = { }
        for t in time_range: 
            for label in np.unique( objects[t] )[1:]:
                if label not in object_duration.keys( ):
                    object_duration[label] = 1 
                else:
                    object_duration[label] += 1
                    
        return object_duration


    


