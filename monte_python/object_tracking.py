import scipy
from scipy import spatial
import numpy as np
import pandas as pd
import skimage.measure 
from skimage.measure import regionprops, regionprops_table
import math 
import collections
from datetime import datetime
import itertools

def calc_dist(x1,x2,y1,y2):
    return (x1 - x2)**2 + (y1 - y2)**2

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

    def track_objects(self, objects, mend_tracks=False): 
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
            labels_before, labels_after = self.check_for_mergers(labels_before, labels_after, areas_before)
            
            # Check for splits. 
            labels_before, labels_after = self.check_for_mergers(labels_before, labels_after, areas_after)
            
            # Re-label an object if matches (this is where the tracking is done) 
            for label_i, label_f in zip(labels_before, labels_after):
                tracked_objects[t+1, future_objects == label_f] = label_i
  
        # Do a final re-label so that np.max(relabel_objects) == number of tracked objects. 
        tracks = self.relabel(tracked_objects)
        
        if mend_tracks:
            # Check if the track is within 9 km. 
            tracks = self.mend_broken_tracks(tracks, dist_max=3)
        
        
        return tracks
    
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
        unique_labels_after, counts_after = np.unique(labels_after, return_counts=True)
        if any(counts_after>1):
            # Get the labels that non-unique. 
            merged_labels = unique_labels_after[counts_after>1]

            for label in merged_labels:
                # This should be 2 or more labels (which are being merged together).
                potential_label_for_merged_obj = [l for i, l in enumerate(labels_before) if labels_after[i] == label]
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
        unique_labels_before, counts_before = np.unique(labels_before, return_counts=True)
        if any(counts_before>1): 
            split_labels = unique_labels_before[counts_before>1]
    
            for label in split_labels: 
                # This should be 2 or more labels (which are being merged together).
                potential_label_for_split_obj = [l for i, l in enumerate(labels_after) if labels_before[i] == label]  
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
        """Ensure that initially, each object for the different times has a unique label"""
        if not isinstance(objects, np.ndarray):
            objects = np.array(objects)
        
        unique_track_set = np.zeros(objects.shape, dtype=np.int32)
        
        num = 1
        for i in range(len(objects)):
            current_obj = objects[i,:,:]
            for label in np.unique(current_obj)[1:]:
                unique_track_set[i, current_obj==label] += num
                num+=1
                
        return unique_track_set 

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

    def get_centroid(self, df, label):
        try:
            df=df.loc[df['label'] == label]
            x_cent, y_cent = df['centroid-0'], df['centroid-1']
            x_cent=int(x_cent)
            y_cent=int(y_cent)
        except:
            return np.nan, np.nan
    
        return x_cent, y_cent 
    
    def get_track_path(self, tracked_objects):
        """ Create track path. """
        properties = ['label', 'centroid']
        object_dfs = [pd.DataFrame(regionprops_table(tracks, properties=properties)) 
              for tracks in tracked_objects]
        
        unique_labels = np.unique(tracked_objects)[1:]
        centroid_x = {l : [] for l in unique_labels}
        centroid_y = {l : [] for l in unique_labels}
    
        for df in object_dfs:
            for label in unique_labels:
                x,y = self.get_centroid(df, label)
                centroid_x[label].append(x)
                centroid_y[label].append(y)

        return centroid_x, centroid_y
    
    def find_track_start_and_end(self, data):
        """
        Based on the x-centriod or y-centroid values for a track, 
        determine when the time index when the track starts and stops. 
        """
        # If the track happens to persist for all 
        # time steps (i.e., no nan values), then 
        # the start and end indices are 0 and len(data)-1
        if not np.isnan(np.sum(data)):
            return 0, len(data)-1 
        
        # Does the track start at the first time step?
        elif not np.isnan(data[0]):
            return 0, np.where(np.isnan(data))[0][0]-1
        
        # Does the track end at the last time step? 
        elif not np.isnan(data[-1]):
            return np.where(np.isnan(data))[0][-1]+1, len(data)-1
        # Otherwise the tracks starts and stops sometime during 
        # the time period. 
        else:
            data_copy = np.copy(data)
            data_copy[np.isnan(data)] = 0
            diff = np.absolute(np.diff(data_copy))

            # This will return intersecting values in 
            # value order rather than chronological order. 
            # Need to check if the storms are moving west, 
            # in case, the start and env vals are switched. 
            vals = np.intersect1d(data, diff)
        
            is_decreasing = np.nanmean(np.diff(data)) < 0 
            start_val, end_val = vals[::-1] if is_decreasing else vals

            start_ind = np.where(data==start_val)[0]
            end_ind = np.where(data==end_val)[0]
    
            return start_ind[0], end_ind[0]
    
    
    def mend_broken_tracks(self, tracked_objects, dist_max=3):
        """
        Mend broken tracks by project track ends forward 
        in time based on estimated storm motion and 
        search for tracks that start in that projected area. 
        If close enough, assume that those two tracks 
        should be combined. Re-label that new tracks with 
        the projected tracks label. 
        """
        new_tracks = np.copy(tracked_objects)
        x_cent, y_cent = self.get_track_path(tracked_objects)
    
        # Get the start and end 
        track_start_end = {label : self.find_track_start_and_end(x_cent[label]) for label in x_cent.keys()}
    
        for label in x_cent.keys():
            # Compute the project storm position based on
            # the estimated storm motion. Since time is 
            # constant, we do not need to consider it. 
            dx = np.mean(np.diff(x_cent[label])) 
            dy = np.mean(np.diff(y_cent[label])) 

            # Get the start and end time index for this track. 
            start_ind, end_ind = track_start_end[label]
    
            x_proj = x_cent[label][end_ind] + dx
            y_proj = x_cent[label][end_ind] + dy

    
            # Given the end index of this track, we are looking for tracks that 
            # started when this tracked ended or during the next time step. 
            other_labels = [l for l in x_cent.keys() if l != label and track_start_end[l][0] in [end_ind, end_ind+1] ]
            for other_label in other_labels:
                x_val = x_cent[other_label][end_ind]
                x = x_val if x_val is not np.nan else x_cent[other_label][end_ind+1]
        
                y_val = y_cent[other_label][end_ind]
                y = y_val if y_val is not np.nan else y_cent[other_label][end_ind+1]
        
                # Is there an existing tracks start point that is within some
                # distance on this projected end of this track. If so,
                # link them together and re-label the existing track to this label. 
                dist = calc_dist(x_proj, x, y_proj, y)
                if dist <= dist_max:
                    new_tracks[tracked_objects==other_label] = label 
    
        return new_tracks