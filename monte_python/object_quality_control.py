import scipy
import numpy as np
import skimage.measure
from math import sqrt
from collections import OrderedDict
from skimage.measure import regionprops 
#import warnings
#warnings.simplefilter("ignore", UserWarning)

class QualityControler:
    '''
    QualityControl implements multiple quality control measures to post-process identified objects.
    These include: 
        - Minimum Area
        - Merging
        - Maximum Length
        - Duration
        - Maximum Value Within
        - Matched to Local Storm Reports
    '''
    qc_to_module_dict = {'min_area' : '_remove_small_objects', 
                         'merge_thresh' : '_merge', 
                         'max_length' : '_remove_long_objects',
                         'min_time' : '_remove_short_duration_objects',
                         'max_thresh' : '_remove_low_intensity_objects',
                         'match_to_lsr' : '_remove_objects_unmatched_to_lsrs'
                        }
    
    def quality_control(self, input_data, object_labels, object_properties, qc_params): 
        """
        Applies quality control to identified objects. 
        Parameters:
        -------------------------
            object_labels, 2D numpy array of labeled objects
            
            object_properties, object properties calculated by skimage.measure.regionprops for object labels
            
            input_data, original 2D numpy array of data from which object_labels was generated from
            
            qc_params,  List of tuples or ordered dictionary of the quality control option to apply 
                          and the corresponding criteria. E.g., to apply a minimum area threshold 
                          qc_params = [('min_area', 10)]. 
                Potential quality control
                 - 'min_area', removing objects below the minimum area (in number of pixels)
                 - 'merge_thresh', merging objects together closer than a minimum distance (in gridpoints)
                 - 'max_length', removing objects with a major axis length greater the given criterion (in gridpoints)
                 - 'min_time', removing objects with duration less than the given criterion (in time steps) 
                 - 'max_thresh', removing objects if the Pth percentile value is less than the given value. 
                                 For max, P=100 and min, P=0. For this quality control, use a nested tuple
                                 E.g., qc_params = [('max_thresh', (value, P)]
                 - 'match_to_lsr', remove objects not matched to an local storm report
                For additional details, read the functions below. 
 
        Returns:
        -------------------------
            2-tuple of (object_labels, object_props) 
            object_labels, quality-controlled version of the original object_labels
            object_props, object properties of the quality-controlled objects
        """
        self.input_data = input_data
        self.object_labels = object_labels
        self.object_properties = object_properties
        
        if isinstance(qc_params, list):
            qc_params = OrderedDict(qc_params)
        self.qc_params = qc_params
        
        for qc_method in self.qc_params.keys():
            getattr(self, self.qc_to_module_dict[qc_method])()
            
        return (self.object_labels, self.object_properties) 

    def _remove_small_objects(self):
        ''' 
        Removes objects with area less than min_area. Area measured by the number of
        grid cells.
        Args,
            : input_data, 2D numpy array, original data from which the object labels were generated from.
                                          Used to recalculate the regionprops after qualtiy control is applied. 
            : object_labels_and_props, 2-tuple of 2D np array of object labels and the associated regionprops object 
            : min_area, int, minimum area (in number of grid points) of an object
        '''
        qc_object_labels = np.zeros( self.object_labels.shape, dtype=int) 
        j=1
        for region in self.object_properties:
            if region.area >= self.qc_params['min_area']:
                qc_object_labels[self.object_labels == region.label] = j 
                j+=1     
        qc_object_properties = regionprops( qc_object_labels, self.input_data,  ) 
        
        self.object_labels = qc_object_labels
        self.object_properties = qc_object_properties

    def _remove_long_objects(self):
        ''' 
        Removes objects with a major axis length greater than max_length 
        Args,
            : input_data, 2D numpy array, original data from which the object labels were generated from.
                                          Used to recalculate the regionprops after qualtiy control is applied. 
            : object_labels_and_props, 2-tuple of 2D np array of object labels and the associated regionprops object 
            : max_length, int, maximum object length (in grid point distances) 
        '''
        qc_object_labels = np.zeros( self.object_labels.shape, dtype=int)
        j=1
        for region in self.object_properties:
            if round(region.major_axis_length, 2) >= self.qc_params['max_length']:
                qc_object_labels[self.object_labels == region.label] = j
                j+=1
        qc_object_properties = regionprops( qc_object_labels, self.input_data,  )

        self.object_labels = qc_object_labels
        self.object_properties = qc_object_properties

    def _remove_low_intensity_objects(self):
        ''' 
        Removes objects where the 75th percentile intensity is below the max_thresh
        Args,
            : input_data, 2D numpy array, original data from which the object labels were generated from.
                                          Used to recalculate the regionprops after qualtiy control is applied. 
            : object_labels_and_props, 2-tuple of 2D np array of object labels and the associated regionprops object 
            : max_thresh, float, intensity threshold for the 75th percentile
        '''
        qc_object_labels = np.zeros( self.object_labels.shape, dtype=int)
        j=1
        val, P = self.qc_params['max_thresh']
        for region in self.object_properties: 
            if round( np.percentile(self.input_data[self.object_labels == region.label], P), 10) \
                >= round(val,10): 
                qc_object_labels[self.object_labels == region.label] = j
                j+=1
        qc_object_properties = regionprops( qc_object_labels, self.input_data, )

        self.object_labels = qc_object_labels
        self.object_properties = qc_object_properties 

    def _remove_short_duration_objects(self): 
        ''' 
        When identify objects over a time duration, removes objects existing for a duration less than min_time
        Args,
            : input_data, 2D numpy array, original data from which the object labels were generated from.
                                          Used to recalculate the regionprops after qualtiy control is applied. 
            : object_labels_and_props, 2-tuple of 2D np array of object labels and the associated regionprops object 
            : min_time, int, minimum duration (in terms of number of time steps) 
            : time_argmax_idxs, When applying np.argmax over time dimensions of an array
        '''
        qc_object_labels = np.zeros( self.object_labels.shape, dtype=int)
        j=1
        time_argmax_idxs = self.qc_params['min_time'][1]
        for region in self.object_properties:    
            duration = len( np.unique( time_argmax_idxs[self.object_labels == region.label] ))  
            if duration >= self.qc_params['min_time'][0]:
                qc_object_labels[self.object_labels == region.label] = j
                j+=1
        qc_object_properties = regionprops( qc_object_labels, self.input_data, )        
        
        self.object_labels = qc_object_labels
        self.object_properties = qc_object_properties

    def _merge(self):
        ''' 
        Merges near-by objects to limit discontinuous objects from counting as multiple objects.
        Applicable when using the single threshold object identification method. 
        Args,
            : input_data, 2D numpy array, original data from which the object labels were generated from.
                                          Used to recalculate the regionprops after qualtiy control is applied. 
            : object_labels_and_props, 2-tuple of 2D np array of object labels and the associated regionprops object 
            : merge_thresh, float, merging distance (objects closer than the merging distance 
                                   are combined together; in grid point distance)
        '''
        qc_object_labels = np.copy( self.object_labels ) 
        original_labels = np.unique(qc_object_labels)[1:] #[label for label in np.unique(qc_object_labels)[1:]]
        remaining_labels = np.unique(qc_object_labels)[1:]  #[label for label in np.unique(qc_object_labels)[1:]]
        for label in original_labels:
            # Does 'label' still exist? 
            if int(label) in remaining_labels:
                y_idx, x_idx = np.where( qc_object_labels == label )
                label_xy_stack = np.dstack([x_idx.ravel(), y_idx.ravel()])[0]
                region_tree = scipy.spatial.cKDTree( label_xy_stack )
                for other_label in original_labels:
                    # Does 'other_label' still exist?
                    if other_label != label and int(other_label) in remaining_labels: 
                        y_idx, x_idx = np.where( qc_object_labels == other_label )
                        other_label_xy_stack = np.dstack([x_idx.ravel(), y_idx.ravel()])[0]
                        dist_btw_objects, _ = region_tree.query(other_label_xy_stack)
                        if round(np.amin(dist_btw_objects), 2) <= self.qc_params['merge_thresh']: 
                            qc_object_labels[qc_object_labels == other_label] = label
                            y_idx, x_idx = np.where( qc_object_labels == label )
                            xy_stack = np.dstack([x_idx.ravel(), y_idx.ravel()])[0]
                            region_tree = scipy.spatial.cKDTree( xy_stack )
                            remaining_labels = np.unique(qc_object_labels)[1:]
                            
        remaining_labels = np.unique(qc_object_labels)[1:]
        for i, label in enumerate(remaining_labels):
            qc_object_labels[qc_object_labels==label] = int(i+1) 

        qc_object_properties = regionprops( qc_object_labels, self.input_data, )
        self.object_labels = qc_object_labels
        self.object_properties = qc_object_properties

    def _remove_objects_unmatched_to_lsrs(self):
        '''
        Removes objects unmatched to Local Storm Reports (LSRs).
        '''
        qc_object_labels = np.zeros( self.object_labels.shape, dtype=int)
        j=1
        for region in self.object_properties:
            kdbelled_points, labelled_datatree = spatial.cKDTree( region.coords )
            dist_btw_region_and_lsr, _ = kdtree.query( self.qc_params['match_to_lsr']['lsr_points'] )
            if round( np.amin( dist_btw_region_and_lsr ), 10) < self.qc_params['match_to_lsr']['dist_to_lsr']:
                qc_object_labels[self.object_labels == region.label] = j
                j+=1
            
        qc_object_properties = regionprops( qc_object_labels, self.input_data, )
        self.object_labels = qc_object_labels
        self.object_properties = qc_object_properties