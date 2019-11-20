from scipy import spatial
import numpy as np
import skimage.measure
from skimage.measure import regionprops 
from hagelslag.processing.EnhancedWatershedSegmenter import EnhancedWatershed, rescale_data
from ObjectMatching import ObjectMatching

def label( self, input_data, method='watershed', return_object_properties=True, **params):
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
        if 'local_max_area' not in list(params.keys()): 
            params['local_max_area'] = 16
        # Initialize the EnhancedWatershed objects
        watershed_method = EnhancedWatershed( min_thresh = params['min_thresh'],
                                              max_thresh = params['max_thresh'], 
                                              data_increment = params['data_increment'],
                                              delta = params['delta'],
                                              size_threshold_pixels = params['size_threshold_pixels'],
                                              local_max_area = params['local_max_area']) 
        object_labels = watershed_method.label( input_data ) 
    elif method == 'single_threshold': 
        # Binarize the input array based on the boundary threshold 
        binary_array  = self._binarize( input_data, params['bdry_thresh'] )
        # Label the binary_array using skimage 
        object_labels = skimage.measure.label(binary_array).astype(int)
        
    if return_object_properties: 
        # Calculate the object properties 
        object_props  = regionprops( object_labels, input_data )    
        return (object_labels, object_props)
    else: 
        return object_labels

def _binarize( self, input_data, bdry_thresh ):
    """ Binarizes input_data with the object boundary threshold"""
    binary  = np.where( np.round(input_data, 10) >= round(bdry_thresh, 10) , 1, 0) 
    binary  = binary.astype(int)
    return binary 

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
            kdtree = spatial.cKDTree( region.coords )
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

