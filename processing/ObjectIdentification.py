import scipy
import numpy as np
import skimage.measure
from math import sqrt
from skimage.measure import regionprops 
from .EnhancedWatershedSegmenter import EnhancedWatershed, rescale_data
from .ObjectMatching import ObjectMatching
import xarray as xr 
import random
import copy 

def label_regions( input_data, params, method='watershed', return_object_properties=True):
    """ Identifies and labels objects in input_data using a single threshold or 
        the enhanced watershed algorithm (Lakshmanan et al. 2009, Gagne et al. 2016)
            
        Example usage provided in a jupyter notebook @ https://github.com/monte-flora/MontePython/
        
        Lakshmanan, V., K. Hondl, and R. Rabin, 2009: An Efficient, General-Purpose Technique for Identifying Storm Cells 
        in Geospatial Images. J. Atmos. Oceanic Technol., 26, 523–537, https://doi.org/10.1175/2008JTECHA1153.1
        
        Gagne II, D. J., A. McGovern, N. Snook, R. Sobash, J. Labriola, J. K. Williams, S. E. Haupt, and M. Xue, 2016: 
        Hagelslag: Scalable object-based severe weather analysis and forecasting. Proceedings of the Sixth Symposium on 
        Advances in Modeling and Analysis Using Python, New Orleans, LA, Amer. Meteor. Soc., 447.
            
        ARGS, 
            : input_data, 2D numpy array of data to be labelled 
            : method, string declaring the object labelling method ( 'single_threshold' or 'watershed' ) 
            : returnobject_properties, Boolean, if True return the object properties calculated by skimage.measure.regionprops
            : params, dictionary with the parameters for the respective labelling method selected.
                if watershed:
                    min_thresh (int): minimum pixel value for pixel to be part of a object
                    data_increment (int): quantization interval. Use 1 if you don't want to quantize (See Lakshaman et al. 2009) 
                    max_thresh (int): values greater than max_thresh are treated as the maximum threshold
                    size_threshold_pixels (int): clusters smaller than this threshold are ignored.
                    delta (int): maximum number of data increments the cluster is allowed to range over. Larger d results in clusters over larger scales.            
                if single_threshold:
                    bdry_thresh (float): intensity threshold used for the 'single_threshold' object ID method
        RETURNS, 
            object_labels, input_data labelled 
            object_props, properties of the labelled regions in object_labels (optional, if return_object_properties = True )    
    """
    if method == 'watershed': 
        if 'dist_btw_objects' not in list(params.keys()): 
            params['dist_btw_objects'] = 15
        # Initialize the EnhancedWatershed objects
        watershed_method = EnhancedWatershed( min_thresh = params['min_thresh'],
                                              max_thresh = params['max_thresh'], 
                                              data_increment = params['data_increment'],
                                              delta = params['delta'],
                                              size_threshold_pixels = params['size_threshold_pixels'],
                                              dist_btw_objects = params['dist_btw_objects']) 
        object_labels = watershed_method.label( input_data ) 
    elif method == 'single_threshold': 
        # Binarize the input array based on the boundary threshold 
        binary_array  = _binarize( input_data, params['bdry_thresh'] )
        # Label the binary_array using skimage 
        object_labels = skimage.measure.label(binary_array).astype(int)
        
    if return_object_properties: 
        # Calculate the object properties 
        object_props  = regionprops( object_labels, input_data )    
        return (object_labels, object_props)
    else: 
        return object_labels

def _binarize( input_data, bdry_thresh ):
    """ Binarizes input_data with the object boundary threshold"""
    binary  = np.where( np.round(input_data, 10) >= round(bdry_thresh, 10) , 1, 0) 
    binary  = binary.astype(int)
    return binary 

def quantize_wofs_probabilities(ensemble_probabilities):
    """
    Quantize the discrete WOFS probabilities
    """
    q_data = np.round(100.*ensemble_probabilities)
    for i, value in enumerate(np.unique(q_data)):
        q_data[q_data==value] = i
    return q_data

def calc_dist( set1, set2 ):
    return sqrt((set1[0] - set2[0])**2 + (set1[1] - set2[1])**2)

def _fix_regions(
                 current_region_props, 
                 previous_labelled_data,
                 current_labelled_data, 
                 ):
    '''
    Fix regions 
    1) Maintain all the original watershed objects 
    2) Keep all new verison of an object if it is 'coherent'
    '''
    fixed_labelled_data = copy.deepcopy(previous_labelled_data)
    num_of_changes = 0 
    for region in current_region_props:
        if (float(region.convex_area) / region.filled_area) < 1.5: 
            # Coherent enough to keep 
            fixed_labelled_data[current_labelled_data==region.label] = region.label 
        else:
            num_of_changes += 1   

    return fixed_labelled_data, num_of_changes 

def _fix_bad_pixels(labelled_data):
    '''
    Returns labelled regions where individual pixels 
    are fixed
    '''
    new_labels = copy.deepcopy( labelled_data )
    nonzero_points = np.where(new_labels>0)
    nonzero_points = np.c_[nonzero_points[0].ravel(), nonzero_points[1].ravel()]

    for point in nonzero_points: 
        y = point[0]
        x = point[1]
        min_y = max(y-2, 0)
        min_x = max(x-2, 0)
        max_x = min(x+2+1, new_labels.shape[1])
        max_y = min(y+2+1, new_labels.shape[0])
        data = new_labels[min_y:max_y, min_x:max_x]
        nonzero_data = data[np.nonzero(data)]
        unique_values, their_counts = np.unique(nonzero_data, return_counts=True)
        most_common_value = unique_values[np.argmax(their_counts)] 
        if new_labels[y,x] != most_common_value:
            #print (nonzero_data, np.argmax(counts), new_labels[y,x])   
            new_labels[y,x] = most_common_value

    return new_labels      


def _get_labelled_coords(region_props):
    '''
    Returns a dictionary with the various coordinates of
    alreay labelled regions as keys and their respective label as the value
    '''
    region_coords = { }
    for region in region_props:
        for coord in region.coords:
            region_coords[(coord[0], coord[1])] = region.label 

    return region_coords

def _get_unlabelled_coords(original_data, labelled_data):
    '''
    Returns the coordinates of the non-zero, unlabelled pixels
    '''
    original_array = copy.deepcopy(original_data)
    # Ignore points already assigned to an region
    original_array[labelled_data > 0] = 0

    # Get the coordinates of the points to be assigned
    y, x = np.where(original_array > 0)
    original_nonzero_coords = np.c_[y.ravel(), x.ravel()] 

    return original_nonzero_coords

def _find_the_closest_object(labelled_data, original_nonzero_coords, region_coords):
    '''
    For each non-zero pixel, find the closest (in eucledian distance)
    already identified object region 
    '''
    labelled_array = copy.deepcopy( labelled_data ) 
   
    # Cycle through the nonzero coordinates
    for unlabelled_point in original_nonzero_coords:
        all_dist = {}
        for labelled_point in list(region_coords.keys()):
            all_dist[((unlabelled_point[0], unlabelled_point[1]), \
                    (labelled_point[0], labelled_point[1]))] = calc_dist( unlabelled_point, labelled_point)

        closest_coord_pair = min(all_dist, key=all_dist.get)
        this_unlabelled_point = closest_coord_pair[0]
        this_labelled_point = closest_coord_pair[1]
        labelled_array[this_unlabelled_point] = region_coords[this_labelled_point]

    return labelled_array


def grow_objects(previous_region_props, previous_data_to_label, previous_labelled_data, num_of_points_to_label=[]): 
    '''
    Using labelled regions provided by the watershed algorithm,
    grow objects from the original field using distance
    to nearby objects 
    
    For a non-zero point in the domain not initially assigned 
    to a labelled region, check the distance of this point
    to every labelled point in the domain:
    The point it is closest to, give it that label
    but in future iterations, do not alter the original
    labelled region with new points. BUT remove the point
    so it is not further considered. 

    Args:
    ------------------
    regionprops, skimage object
    original_data, 2D numpy array of the data to be labelled
    labelled_data, 2D numpy array of labelled regions in original_data

    Returns:
    ---------------
    labelled_array, 2D numpy array of labelled regions 
    '''
    # Get the coordinates of the points to be assigned to a region
    original_nonzero_coords = _get_unlabelled_coords(previous_data_to_label, previous_labelled_data)
    # Get the coordinates of the points already assigned to a region
    region_coords = _get_labelled_coords(previous_region_props)
    # Assign unlabelled points to the label of the closest region
    current_labelled_data = _find_the_closest_object(previous_labelled_data, original_nonzero_coords, region_coords)
    # Fix any stray pixels 
    current_labelled_data = _fix_bad_pixels(current_labelled_data)

    return current_labelled_data

def grow_objects_recursive(previous_region_props, previous_data_to_label, previous_labelled_data, num_of_points_to_label=[]): 
    '''
    Using labelled regions provided by the watershed algorithm,
    grow objects from the original field using distance
    to nearby objects 
    
    For a non-zero point in the domain not initially assigned 
    to a labelled region, check the distance of this point
    to every labelled point in the domain:
    The point it is closest to, give it that label
    but in future iterations, do not alter the original
    labelled region with new points. BUT remove the point
    so it is not further considered. 

    Args:
    ------------------
    regionprops, skimage object
    original_data, 2D numpy array of the data to be labelled
    labelled_data, 2D numpy array of labelled regions in original_data

    Returns:
    ---------------
    labelled_array, 2D numpy array of labelled regions 
    '''

    # Get the coordinates of the points to be assigned to a region
    original_nonzero_coords = _get_unlabelled_coords(previous_data_to_label, previous_labelled_data)
    # Get the coordinates of the points already assigned to a region
    region_coords = _get_labelled_coords(previous_region_props)
    # Assign unlabelled points to the label of the closest region
    current_labelled_data = _find_the_closest_object(previous_labelled_data, original_nonzero_coords, region_coords)
    # Fix any bad pixels
    current_labelled_data  = _fix_bad_pixels(current_labelled_data)
    current_region_props = regionprops( current_labelled_data.astype(int), current_labelled_data, coordinates='rc' )
    # Fix any non-cohererent regions
    fixed_labelled_data, current_num_of_changes = _fix_regions(
                                                current_region_props = current_region_props,
                                                previous_labelled_data = previous_labelled_data,
                                                current_labelled_data = current_labelled_data
                                             )
    
    fixed_region_props = regionprops( fixed_labelled_data.astype(int), fixed_labelled_data, coordinates='rc' ) 
    num_of_points_to_label.append( np.shape(original_nonzero_coords)[0] ) 
    
    if (len(num_of_points_to_label) > 2) and (len(np.unique(num_of_points_to_label[-2:])) == 1):
        # Get the coordinates of the points to be assigned to a region
        original_nonzero_coords = _get_unlabelled_coords(previous_data_to_label, previous_labelled_data)
        # Get the coordinates of the points already assigned to a region
        region_coords = _get_labelled_coords(previous_region_props)
        # Assign unlabelled points to the label of the closest region
        current_labelled_data = _fix_bad_pixels(current_labelled_data)
        current_labelled_data = _find_the_closest_object(previous_labelled_data, original_nonzero_coords, region_coords) 
         
        return current_labelled_data
    
    if current_num_of_changes == 0:
        return fixed_labelled_data

    elif current_num_of_changes > 0:
        return grow_objects_recursive( 
                     previous_region_props = fixed_region_props,
                     previous_data_to_label = previous_data_to_label,
                     previous_labelled_data = fixed_labelled_data,
                     num_of_points_to_label = num_of_points_to_label
                     )

def label_ensemble_objects(ensemble_probabilities):
    '''
    Method for labeling probability objects. A first pass watershed method identifies
    the key region using no minimum threshold and a large minimum area threshold. 
    A second pass watershed method uses a higher minimum threshold to assess 
    if an object in the first pass should be sub-divided. Finally, the remaining
    unidentified non-zero points are clustered to existing objects based on 
    minimum distance. Clustering ensures that all points are labeled. 

    Args:
    --------------
    ensemble_probabilities, 2d array of ensemble probabilities (between 0-1)


    Returns:
    ---------------
    full_objects, 2d array of labeled regions in ensemble probabilities
    full_object_props, list of RegionProps from skimage.measure.regionprops

    '''
    input_data = quantize_wofs_probabilities(ensemble_probabilities)

    first_pass_params = {'min_thresh': 0,
          'max_thresh': 18,
          'data_increment': 1,
          'delta': 0,
          'size_threshold_pixels': 400,
          'dist_btw_objects': 15 }

    second_pass_params = {'min_thresh': 5,
          'max_thresh': 18,
          'data_increment': 1,
          'delta': 0,
          'size_threshold_pixels': 300,
          'dist_btw_objects': 25 }

    first_pass_labels = label_regions (
                              input_data = input_data,
                              params = first_pass_params,
                              return_object_properties=False
                              )

    second_pass_labels = label_regions(
                               input_data = input_data,
                               params = second_pass_params,
                               return_object_properties=False
                              )

    # Adjust label of the second pass objects 
    second_pass_labels = np.where(second_pass_labels>0, second_pass_labels+1000, 0)
    combined_labels = np.zeros(second_pass_labels.shape, dtype=int)
    for label in np.unique(first_pass_labels)[1:]:
        objects_within_this_label = np.unique(second_pass_labels[first_pass_labels==label])[1:]
        if len(objects_within_this_label) > 1:
            for label in objects_within_this_label:
                combined_labels[second_pass_labels==label] = label
        else:
            combined_labels[first_pass_labels==label] = label

    # Convert labels back
    for i, value in enumerate(np.unique(combined_labels)[1:]):
        combined_labels[combined_labels==value] = i+1

    combined_label_props = regionprops(combined_labels.astype(int), combined_labels)

    full_objects = grow_objects_recursive(
                            previous_region_props=combined_label_props,
                            previous_data_to_label=np.copy(input_data),
                            previous_labelled_data=np.copy(combined_labels)
                            )
    
    del first_pass_labels
    del second_pass_labels
    del combined_labels

    full_object_props = regionprops(full_objects.astype(int), ensemble_probabilities, coordinates='rc')

    return full_objects, full_object_props  


class QualityControl:
    '''
    QualityControl can multiple quality control measures to the identified objects.
    These include: 
        - Minimum Area
        - Merging
        - Maximum Length
        - Duration
        - Maximum Value Within
        - Matched to Local Storm Reports
    '''
    def quality_control( self, input_data, object_labels, object_properties, qc_params ): 
        """
        Applies quality control to identified objects. 
        ARGs, 
            : object_labels, 2D numpy array of labeled objects
            : object_properties, object properties calculated by skimage.measure.regionprops for object labels
            : input_data, original 2D numpy array of data from which object_labels was generated from
            : qc_params,  List of tuples or ordered dictionary of the quality control option to apply 
                          and the corresponding criteria. E.g., to apply a minimum area threshold 
                          qc_params = [('min_area', 10)]. 
                Potential quality control
                 - 'min_area', removing objects below the minimum area (in number of pixels)
                 - 'merge_thresh', merging objects together closer than a minimum distance (in gridpoints)
                 - 'max_length', removing objects with a major axis length greater the given criterion (in gridpoints)
                 - 'min_time', removing objects with duration less than the given criterion (in time steps) 
                 - 'max_thresh', removing objects if the 75th percentile value is less than the given criterion
                 - 'match_to_lsr', remove objects not matched to an local storm report
                For additional details, read the functions below. 
 
        RETURNS,
            2-tuple of (object_labels, object_props) 
            object_labels, quality-controlled version of the original object_labels
            object_props, object properties of the quality-controlled objects
        """
        self.qc_params = qc_params
        self.input_data = input_data
        self.object_labels = object_labels
        self.object_properties = object_properties
        for option, value in list(self.qc_params.items()):
            # Removes objects based on a minimum or maximum area threshold
            if option == 'min_area':
               self._remove_small_objects( )
            # Merge togerther near-by objects ( applicable for single threshold object identification methods )
            if option == 'merge_thresh':
                self._merge( )
            # Remove lengthy objects    
            if option == 'max_length':   
                self._remove_long_objects( )
            # If the input array is a time composite, then remove objects which only exist less then time step continuity threshold 
            if option == 'min_time':
                self._remove_short_duration_objects( )
            # Remove objects with a maximum intensity less than the threshold given. 
            if option == 'max_thresh':
                self._remove_low_intensity_objects( )
            # Remove objects not matched with an local storm report 
            if option == 'match_to_lsr': 
                self._remove_objects_unmatched_to_lsrs( )

        return (self.object_labels, self.object_properties) 

    def _remove_small_objects( self ):
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
        qc_object_properties = regionprops( qc_object_labels, self.input_data, coordinates='rc' ) 
        
        self.object_labels = qc_object_labels
        self.object_properties = qc_object_properties

    def _remove_long_objects( self ):
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
        qc_object_properties = regionprops( qc_object_labels, self.input_data, coordinates='rc' )

        self.object_labels = qc_object_labels
        self.object_properties = qc_object_properties

    def _remove_low_intensity_objects( self ):
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
        for region in self.object_properties: 
            if round( np.percentile(self.input_data[self.object_labels == region.label], 75), 10) >= round(self.qc_params['max_thresh'],10): 
                qc_object_labels[self.object_labels == region.label] = j
                j+=1
        qc_object_properties = regionprops( qc_object_labels, self.input_data, coordinates='rc' )

        self.object_labels = qc_object_labels
        self.object_properties = qc_object_properties 

    def _remove_short_duration_objects( self ): 
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
        qc_object_properties = regionprops( qc_object_labels, self.input_data, coordinates='rc' )        
        
        self.object_labels = qc_object_labels
        self.object_properties = qc_object_properties

    def _merge( self ):
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

        qc_object_properties = regionprops( qc_object_labels, self.input_data, coordinates='rc' )
        self.object_labels = qc_object_labels
        self.object_properties = qc_object_properties

    def _remove_objects_unmatched_to_lsrs( self ):
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
            
        qc_object_properties = regionprops( qc_object_labels, self.input_data, coordinates='rc' )
        self.object_labels = qc_object_labels
        self.object_properties = qc_object_properties

    @staticmethod 
    def convert_labels_to_intensity( input_data , object_labels, object_props ): 
        """Convert integer labelled objects back into the original intensity values from input_data"""
        filled_labels = np.zeros(( input_data.shape ))
        for region in object_props:    
            idx = np.where(object_labels == region.label)
            filled_labels[idx] = input_data[idx] 

        return filled_labels 
    
    @staticmethod
    def identify_storm_mode( object_labels_a, object_labels_b, object_labels_a_props, object_labels_b_props): 
        '''
        Labels objectively-identified storms as either MCS (-1) or non-MCS (1)
        Args:
            object_labels_a, 2D array, storm objects labelled at a high dbz threshold (e.g., 40 dbz)
            object_labels_b, 2D array, storm objects labelled at a lower dbz threshold (e.g., 35 dbz )
            
        Returns:
            2D array where regions in object_labels_a lablled at MCS (-1) or non-MCS (1) 
        '''
        obj_match = ObjectMatching( dist_max = 30 )

        objects_b_dict = { }
        for region in object_labels_b_props:
            if region.minor_axis_length < 10e-3:
                maj_min_ratio = -1 
            else:
                maj_min_ratio = region.major_axis_length/region.minor_axis_length

            objects_b_dict[region.label] = {'Major Axis Length':region.major_axis_length, 
                                            'Area':region.area, 
                                            'Maj/Min Ratio': maj_min_ratio}

        #print objects_b_dict
        matched_object_set_a_labels, matched_object_set_b_labels, cent_dist_of_matched_objects = obj_match.match_objects( object_labels_a, object_labels_b )
        matched_labels = list(zip( matched_object_set_a_labels, matched_object_set_b_labels ))

        storm_labels = np.zeros(( object_labels_a.shape ))
        for label_a, label_b in matched_labels:
            props = objects_b_dict[label_b]
            #print '\n', label_a, props
            if ( props['Major Axis Length'] > 70. or props['Area'] > 2000. or (props['Maj/Min Ratio'] > 3.0 and props['Area'] > 500.)):
                storm_labels[np.where(object_labels_a==label_a)] = -1
            else:
                storm_labels[np.where(object_labels_a==label_a)] = 1

        return storm_labels 

    @staticmethod
    def identify_storm_mode_of_meso_track( rot_labels, matched_rot_labels, matched_strm_labels, df ):
        '''
        Labels rotation tracks as either being associated with an MCS (-1) or non-MCS (1)
        Args:
            rot_labels, 2D array, labelled mesocyclone tracks
            matched_rot_labels, list, labels of the mesocyclone tracks matched to storm objects 
            matched_strm_labels, list, labels of the storm object matched to the mesocyclone tracks
            df, pandas dataframe, 
        Returns:
            storm_labels, 2D array, mesocyclone tracks labelled as MCS (1) or non-MCS (-1) 
        Returns: 
        '''
        storm_labels = np.zeros(( rot_labels.shape ))
        for rot_label, strm_label in zip( matched_rot_labels, matched_strm_labels ): 
            storm_labels[rot_labels == rot_label] = df.loc[ df['object label'] == strm_label, 'Storm Label'].values[0]
        
        return storm_labels

