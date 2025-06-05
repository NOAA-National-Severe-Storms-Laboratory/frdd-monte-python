import scipy
import numpy as np
import skimage.measure
from math import sqrt
from skimage.measure import regionprops 
from skimage.util import img_as_ubyte
from skimage.filters.rank import modal
from sklearn.neighbors import KDTree
from scipy.ndimage import generic_filter
from skimage.morphology import disk
import random
import copy 
import warnings
warnings.simplefilter("ignore", UserWarning)

#mcit
import cv2
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max


from .object_matching import ObjectMatcher
from .object_quality_control import QualityControler
from .EnhancedWatershedSegmenter import EnhancedWatershed, rescale_data


def label(input_data, params, method='watershed', return_object_properties=True):
    """ 
    Performs image segmentation using either (1) a single threshold method,
    (2) a modified version of the enhanced watershed algorithm, or (3) an iterative watershed 
    method [1]_ [2]_ [3]_ [4]_.

    .. note :: 
       The enhanced and iterative watershed algoritms are a powerful tools for image segementation 
       ,but they require significant parameter tuning. We recommend exploring the single
       threshold method first. 
    
    Parameters
    ----------
   
    input_data : {numpy.array or array-like} of shape (NY, NX) 
        2D data to be segmented and labelled. 
    
    method : ``"single_threshold"`` or ``"watershed"`` or ``"iterative_watershed"``
        The method used to segment and label the input data. 
        
        - ``"single_threshold"``, the single threshold method binarizes
                                  the input data where input data > threshold == 1 else 0
                                  and then connected regions are labelled. 
        - ``"watershed"``, uses multiple thresholds and grows objects 
                           until they reach an a given area threshold. 
        - ``"iterative_watershed"``, uses the watershed iteratively where 
                                     higher minimal thresholds and lower area thresholds
                                     are used for later iterations. This allows objects to
                                     be broken up. However, all non-zero regions must be assigned 
                                     to a region, thus it respects the original spatial extent
                                     of a region. 
        
    params : dict  
        The parameter(s) required for the respective labelling method used.
        if ``method == "watershed"``:
            - min_thresh (float), minimum pixel value for pixel to be part of a object. 
            - data_increment (int), quantization interval. default=1  
            - max_thresh (float),  values greater than max_thresh are treated as the maximum threshold
            - area_threshold (int),  clusters smaller than this threshold are ignored.
            - dist_btw_objects (int),  minimal clearance between objects. Objects will no be identified 
                                       closer together than this distance. Used for identifying the 
                                       local maxima for the start of the enhanced watershed. 
                                       default = 15 
        
        if ``method == "iterative_watershed"``:
            - params : list of dict 
                list of the parameters that would be passed for the watershed method (see above). 
                Number of dicts passed indicates the number of iterations. 
            - qc_params : list of tuples , default=None
                Input to the QualityControler to quality control the different 
                iterations of this method.
        
        if ``method == "single_threshold"``:
            bdry_thresh (float), intensity threshold to segment the input data
 
    return_object_properties : boolean, default=True
        If True, the skimage.measure.regionprops is also returned with the labeled data. 
        
 
    Returns
    -------
    labels : {numpy.array or array-like} of shape (NY, NX)
        Labeled array, where all the connected regions are assigned the same integer value.
        
    object_props : {skimage.measure.regionprops object}
        If return_object_properties, properties of each labeled regions 
        (e.g., area, major axis length) is returned 
    
 
    References
    ----------
    .. [1] Lakshmanan, V., K. Hondl, and R. Rabin, 2009: An Efficient, General-Purpose Technique for Identifying Storm Cells 
            in Geospatial Images. J. Atmos. Oceanic Technol., 26, 523–537, https://doi.org/10.1175/2008JTECHA1153.1
        
    .. [2] Gagne II, D. J., A. McGovern, N. Snook, R. Sobash, J. Labriola, J. K. Williams, S. E. Haupt, and M. Xue, 2016: 
            Hagelslag: Scalable object-based severe weather analysis and forecasting. Proceedings of the Sixth Symposium on 
            Advances in Modeling and Analysis Using Python, New Orleans, LA, Amer. Meteor. Soc., 447.
        
    .. [3] Flora, M. L., Skinner, P. S., Potvin, C. K., Reinhart, A. E., Jones, T. A., Yussouf, N., & Knopfmeier, K. H. (2019).
            Object-Based Verification of Short-Term, Storm-Scale Probabilistic Mesocyclone Guidance from an Experimental 
            Warn-on-Forecast System, Weather and Forecasting, 34(6), 1721-1739. Retrieved Sep 27, 2021, 
            from https://journals.ametsoc.org/view/journals/wefo/34/6/waf-d-19-0094_1.xml 
            
    .. [4] Flora, M. L., Potvin, C. K., Skinner, P. S., Handler, S., & McGovern, A. (2021). 
           Using Machine Learning to Generate Storm-Scale Probabilistic Guidance of Severe Weather Hazards in 
           the Warn-on-Forecast System, Monthly Weather Review, 149(5), 1535-1557. Retrieved Mar 4, 2022, 
           from https://journals.ametsoc.org/view/journals/mwre/149/5/MWR-D-20-0194.1.xml
    
    
    Examples
    ---------
    
    >>> import monte_python
    >>> input_data = 
    >>> params = {'min_thresh':25,
    ...           'max_thresh':80,
    ...           'data_increment':20,
    ...           'area_threshold': 150,
    ...           'dist_btw_objects': 50} 
    ...
    >>> centers = [(40, 45), (40, 58), (65, 90), (90, 55), (40,20)]
    >>> storms,_,_ = monte_python.create_fake_storms(centers) 
    >>> labels, label_props = monte_python.label(storms, 
    ...                           method = "watershed",
    ...                              params = params, 
    ...                              return_object_properties=True,
    ...                              )    
    """
    possible_methods = ['watershed', 'single_threshold', 'iterative_watershed', 'mcit']
    is_good_method = method in possible_methods
    if not is_good_method:
        raise ValueError(f"{method} is not a valid method. The valid methods include {possible_methods}.")
    
    # Initialize the ID'er 
    if method == "iterative_watershed":
        ider = IterativeWatershed(**params)
    elif method == "watershed":
        ider = EnhancedWatershed(**params)
    elif method == "mcit":
        ider = MCIT_Identifier(**params)
    else:
        if 'bdry_thresh' not in params.keys():
            raise KeyError("""
                           bdry_thresh is not in params. Must provide a boundary threshold for the single
                           threshold method
                           """
                          )
        
        ider = SingleThresholdIdentifier(params['bdry_thresh'])
    
    labels = ider.label(input_data)
        
    if return_object_properties: 
        # Calculate the object properties 
        object_props  = regionprops(labels, input_data)    
        return labels, object_props
    else: 
        return labels

def label_per_member(data_to_label, method, params, qc_params):
    """
    Identify storm tracks per ensemble member.

    Args:
    -------------------
        data_to_label : array, (NE,NY,NX)
            Ensemble data to label 
        method : str
            Object identification method
        params : dict
            parameters for the chosen object identification method
        qc_params : list of 2-tuples 
            list of tuples where the first element is the 
            str of quality control to apply and the second element is 
            the parameter. This list is turned into a ordered dictionary 
            so that order is maintained. 

    Returns: 
        Quality-controlled objects identified from data_to_label
            
    """
    object_labels_per_ens_mem = np.zeros((data_to_label.shape))
    for mem in range(np.shape(object_labels_per_ens_mem)[0]):
        object_labels_per_ens_mem[mem, :, :] = label_with_qc(data_to_label[mem,:,:], 
                                                             params, 
                                                             qc_params, 
                                                             method=method, 
                                                             return_object_properties=False)

    return object_labels_per_ens_mem    
    
    
def label_with_qc(input_data, params, qc_params, method='watershed', return_object_properties=True):
    """
    Couples the label function with QCing into a single step. 
    """
    qc = QualityControler()
    labels, props = label(
            input_data=input_data,
            method=method,
            params=params,
        )

    labels, props = qc.quality_control(
            input_data=input_data,
            object_labels=labels,
            object_properties=props,
            qc_params=qc_params,
        )
    
    if return_object_properties:
        return labels, props
    else:
        return labels
    

def quantize_probabilities(ensemble_probabilities, ensemble_size):
    """
    Quantize ensemble probabilities to a discrete field. 
    """
    possible_probabilities = {int(round(float(i/ensemble_size)*100)):i for i in range(ensemble_size+1)}
    q_data = np.round(100.*ensemble_probabilities)
    for value in np.unique(q_data)[1:]:
        q_data[q_data==value] = possible_probabilities[int(value)]
    return q_data

class SingleThresholdIdentifier:
    """
    SingleThresholdIdentifier segments and labels input data based on 
    a single threshold. 
    
    Attributes
    ------------
        thresh : float 
            Intensity threshold used to segment regions.
   
    """
    def __init__(self, thresh):
        self.thresh = thresh
    
    
    def label(self, input_data):
        # Binarize the input array based on the boundary threshold 
        binary_array  = self._binarize(input_data, self.thresh)
        # Label the binary_array using skimage 
        labels = skimage.measure.label(binary_array).astype(int)
    
        return labels
    
    def _binarize(self, input_data, thresh):
        """ Binarizes input_data with the object boundary threshold"""
        binary  = np.where(np.round(input_data, 10) >= round(thresh, 10), 1, 0) 
        binary  = binary.astype(int)
        return binary 

class IterativeWatershed:
    """
        The IterativeWatershed method is based on the algorithm developed for 
        identifying ensemble storm tracks [1]_ . The method iteratively uses the 
        a modified version of the hagelslag [2]_ EnhancedWatershed algorithm 
        with increasing minimum and area thresholds, respectively, to identify 
        objects across a multiple spatial scales (e.g., embedded objects to 
        large scale objects). The flow chart for 2 passes can be found in 
        Flora et al. 2021 [1]_ . This method, however, has been generalized 
        to any number of iterations. There is also the ability to quality control
        the objects from each iteration. 
        
        Attributes
        ----------
        
            params : list of dict
                The EnhancedWatershed parameters used for each iterations. 
                The number of iterations == len(params). 
            
            qc_params : nested list of tuples
            
    """
    def __init__(self, params, qc_params=None,):
        self.param_set = params 
        self.qc_params = qc_params
        
        self.solidity = 1.5 

    def label(self, input_data):
        """
        Label connected regions. 
        
        Parameters
        ----------
        
        input_data : {numpy.array or array-like} of shape (NY, NX) 
            2D data to be segmented and labelled.  

        """
        potential_objects = np.array([self._label(input_data = input_data,params = params,)  
                                      for params in self.param_set
                             ])
        
        # Quality control the second pass objects. 
        object_props = regionprops(potential_objects[-1], input_data)
        
        if self.qc_params is not None:
            potential_objects[-1] = QualityControler().quality_control(input_data, 
                                                             potential_objects[-1], 
                                                             object_props, 
                                                             self.qc_params)[0]
        
        labels = self.relabel_embedded_object(potential_objects)
        _max_label = np.max(labels)
        
        # Is there at least one object? 
        if _max_label > 0:
            label_props  = regionprops(labels.astype(int), input_data)
            labels = self.grow_objects_recursive(
                            previous_region_props=label_props,
                            previous_data_to_label=np.copy(input_data),
                            previous_labelled_data=np.copy(labels)
                            )
   
        return labels 

    def _label(self, input_data, params):
        """Label data with watershed method (FOR INTERNAL PURPOSES ONLY)"""
        # Initialize the EnhancedWatershed objects
        watershed_method = EnhancedWatershed( min_thresh = params['min_thresh'],
                                              max_thresh = params['max_thresh'], 
                                              data_increment = params['data_increment'],
                                              area_threshold= params['area_threshold'],
                                              dist_btw_objects = params['dist_btw_objects']) 
        labels = watershed_method.label(input_data) 

        return labels
    
    def calc_dist(self, set1, set2 ):
        return sqrt((set1[0] - set2[0])**2 + (set1[1] - set2[1])**2)

    def _fix_regions(self,
                 current_region_props, 
                 previous_labelled_data,
                 current_labelled_data, 
                 ):
        """
        Fix regions 
        1) Maintain all the original watershed objects 
        2) Keep all new verison of an object if it is 'coherent'
        """
        fixed_labelled_data = copy.deepcopy(previous_labelled_data)
        num_of_changes = 0 
        for region in current_region_props:
            if (float(region.convex_area) / region.filled_area) < self.solidity: 
                # Coherent enough to keep 
                fixed_labelled_data[current_labelled_data==region.label] = region.label 
            else:
                num_of_changes += 1   

        return fixed_labelled_data, num_of_changes 

    def _fix_bad_pixels(self, labelled_data):
        """
        Returns labelled regions where individual pixels 
        are fixed using a 2-grid point radius modal filter. 
        I.e., assigned the most common label within a 
        2-grid point radius
        """
       # Run modal filter
        labelled_data = img_as_ubyte(labelled_data)
        new_labels = modal(labelled_data, disk(2))
        new_labels[labelled_data==0] = 0

        return new_labels


    def _get_labelled_coords(self, region_props):
        """
        Returns a dictionary with the various coordinates of
        alreay labelled regions as keys and their respective label as the value
        """
        region_coords = { }
        for region in region_props:
            for coord in region.coords:
                region_coords[(coord[0], coord[1])] = region.label 

        return region_coords

    def _get_unlabelled_coords(self, original_data, labelled_data):
        """
        Returns the coordinates of the non-zero, unlabelled pixels
        """
        original_array = copy.deepcopy(original_data)
        # Ignore points already assigned to an region
        original_array[labelled_data > 0] = 0

        # Get the coordinates of the points to be assigned
        y, x = np.where(original_array > 0)
        original_nonzero_coords = np.c_[y.ravel(), x.ravel()] 

        return original_nonzero_coords

    def _find_the_closest_object(self, labelled_data, original_nonzero_coords, region_coords):
        """
        For each non-zero pixel, find the closest (in eucledian distance)
        already identified object region. Use a KDTree for efficiency.  
        """
        labelled_array = copy.deepcopy( labelled_data )

        if len(original_nonzero_coords) == 0:
            return labelled_array

        X = np.array(list(region_coords.keys()))
        tree = KDTree(X, leaf_size=40,)
        # Should I check the distance? 
        ind = tree.query(original_nonzero_coords, k=1, return_distance=False,)
        for k, idx in enumerate(ind):
            labelled_array[tuple(original_nonzero_coords[k,:])] = region_coords[tuple(X[idx[0],:])]

        return labelled_array


    def grow_objects(self, previous_region_props, previous_data_to_label, previous_labelled_data, num_of_points_to_label=[]): 
        """
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

        Parameters
        ------------------
        regionprops, skimage object
        original_data, 2D numpy array of the data to be labelled
        labelled_data, 2D numpy array of labelled regions in original_data

        Returns
        ---------------
        labelled_array, 2D numpy array of labelled regions 
        """
        # Get the coordinates of the points to be assigned to a region
        original_nonzero_coords = self._get_unlabelled_coords(previous_data_to_label, previous_labelled_data)
        # Get the coordinates of the points already assigned to a region
        region_coords = self._get_labelled_coords(previous_region_props)
        # Assign unlabelled points to the label of the closest region
        current_labelled_data = self._find_the_closest_object(previous_labelled_data, original_nonzero_coords, region_coords)
        # Fix any stray pixels 
        current_labelled_data = self._fix_bad_pixels(current_labelled_data)

        return current_labelled_data

    def grow_objects_recursive(self,
                           previous_region_props,
                           previous_data_to_label, 
                           previous_labelled_data, 
                           num_of_points_to_label=[]): 
        """
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

        Parameters
        ------------------
        regionprops, skimage object
        original_data, 2D numpy array of the data to be labelled
        labelled_data, 2D numpy array of labelled regions in original_data

        Returns
        ---------------
        labelled_array, 2D numpy array of labelled regions 
        """

        # Get the coordinates of the points to be assigned to a region
        original_nonzero_coords = self._get_unlabelled_coords(previous_data_to_label, previous_labelled_data)
        # Get the coordinates of the points already assigned to a region
        region_coords = self._get_labelled_coords(previous_region_props)
        # Assign unlabelled points to the label of the closest region
        current_labelled_data = self._find_the_closest_object(previous_labelled_data, original_nonzero_coords, region_coords)
        # Fix any bad pixels
        current_labelled_data  = self._fix_bad_pixels(current_labelled_data)
        current_region_props = regionprops( current_labelled_data.astype(int), current_labelled_data,)
        # Fix any non-cohererent regions
        fixed_labelled_data, current_num_of_changes = self._fix_regions(
                                                current_region_props = current_region_props,
                                                previous_labelled_data = previous_labelled_data,
                                                current_labelled_data = current_labelled_data
                                             )
    
        fixed_region_props = regionprops( fixed_labelled_data.astype(int), fixed_labelled_data, ) 
        num_of_points_to_label.append( np.shape(original_nonzero_coords)[0] ) 
    
        if (len(num_of_points_to_label) > 2) and (len(np.unique(num_of_points_to_label[-2:])) == 1):
            # Get the coordinates of the points to be assigned to a region
            original_nonzero_coords = self._get_unlabelled_coords(previous_data_to_label, previous_labelled_data)
            # Get the coordinates of the points already assigned to a region
            region_coords = self._get_labelled_coords(previous_region_props)
            # Assign unlabelled points to the label of the closest region
            current_labelled_data = self._fix_bad_pixels(current_labelled_data)
            current_labelled_data = self._find_the_closest_object(previous_labelled_data, original_nonzero_coords, region_coords) 
         
            return current_labelled_data
    
        if current_num_of_changes == 0:
            return fixed_labelled_data

        elif current_num_of_changes > 0:
            return self.grow_objects_recursive( 
                     previous_region_props = fixed_region_props,
                     previous_data_to_label = previous_data_to_label,
                     previous_labelled_data = fixed_labelled_data,
                     num_of_points_to_label = num_of_points_to_label
                     )
    
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

    def relabel_embedded_object(self, potential_objects):
        """For multiple passes, re-identify embedded objects"""
        potential_objects = self.get_unique_labels(potential_objects)
    
        # Re-iterate over 2-N objects 
        for i in range(potential_objects.shape[0]-1):
            these_objects = potential_objects[i+1,:,:]
            previous_objects = potential_objects[i,:,:]
        
            combined_labels = np.zeros(these_objects.shape, dtype=int)
            for label in np.unique(previous_objects)[1:]:
                objects_within_this_label = np.unique(these_objects[previous_objects==label])[1:]
                if len(objects_within_this_label) > 1:
                    for label in objects_within_this_label:
                        combined_labels[these_objects==label] = label
                else:
                    combined_labels[previous_objects==label] = label

            potential_objects[i+1,:,:] = combined_labels        
    
        final_labels = potential_objects[-1,:,:]
    
        # Convert labels back
        for i, value in enumerate(np.unique(final_labels)[1:]):
            final_labels[final_labels==value] = i+1
    
        return final_labels


class MCIT_Identifier:
    """
    MCIT_Identifier labels input data based on the MCIT method.

    This implementation is generic and can apply to any input data.
    Send the dict parameters:
        mcit_parameters["min_value"] = <some_value>
        mcit_parameters["valley_depth"] = <some_value>

    and the numpy array, input

    For weather related storm identification we recommend linear vil that 
    has had a 3x3 box median applied to it. Like this:

    Code from hotspots repository
    #Compute the linear version of VIL

    #This transform changes the data in an important way
    #lowering the peaks and stretching out the tails. The data
    #becomes less spiky. We think this allows the watershed to do
    #a better job
    linear_vil = 10.0 * np.log10(vil_xy.data)

    #We smooth the data to make the objects more contiguous using a 3x3 box
    # average filter
    kernel = np.ones((3, 3)) / 9
    smlin_vil = cv2.filter2D(linear_vil, ddepth=-1, kernel=kernel)

    with parameters of:
    min_value = 10.0 * np.log10(min_vil_value=1.5)
    valley_depth = 10.0 * np.log10(valley_depth=2.0)

    also the valley_depth and minimum_value must be in the same units.

    If you want to use something like dBZ Reflectivity try linearizing it.

      Reference:

        MCIT (multi-cell identification and tracking):
        Jiaxi Hu, Daniel Rosenfeld, Dusan Zrnic, Earle Williams, Pengfei Zhang,
            Jeffrey C. Snyder, Alexander Ryzhkov, Eyal Hashimshoni,
            Renyi Zhang, Richard Weitz,
        Tracking and characterization of convective cells through their
            maturation into stratiform storm elements using polarimetric
            radar and lightning detection,
        Atmospheric Research,
        Volume 226,
        2019,
        Pages 192-207,
        ISSN 0169-8095,
        https://doi.org/10.1016/j.atmosres.2019.04.015.

    
          Attributes
        ----------
        
            params : list of dict
                The parameters used for each iterations.
                min_value
                valley_depth
    """
    def __init__(self, params: dict = None):
        self.min_value = np.nan 
        self.valley_depth = np.nan
        self.params = params

        print("mcit params: " % (params))

        if self.params is not None:
            self.min_value = self.params['min_value'] 
            self.valley_depth = self.params['valley_depth']
        else:
            #guessint from the data before label, don't do this
            print("Warning: Guessing the min_value and valley depth for mcit is non-optimal")
    
    
    def label(self, input_data):

        if self.min_value == np.nan or self.valley_depth == np.nan:
            mean = np.mean(input_data)
            std = np.std(input_data)

            self.min_value = mean - 2.*std
            self.valley_depth = std/4.0
            print("Warning: Using generic min_value and valley depth for mcit")
            print("Warning: min_value %f Warning: valley_depth: %f " % (self.min_value, self.valley_depth))

        ######################################################################
        #from hotspots repository: 
        #    https://github.com/NOAA-National-Severe-Storms-Laboratory/hotspots
        ######################################################################
        #yeah I'm lazy...
        smlin_vil = input_data
    
        #We follow the techinque described in:
        #https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_watershed.html
        #https://docs.opencv.org/4.x/d3/db4/tutorial_py_watershed.html
        #  We use a similar technique to the above without all the prework.
        #  We can't know starting locations so we use all values not equal to the 
        # missing value as our unknown region
    
        #set the background data (white in image above) to 0 and the foreground data (non-white) to 1
        valid = np.where(smlin_vil>=self.min_value, 1, 0)
    
        #the c++ code uses the vincent-sollie method and c library which is slightly different from
        #the openCV interpretation
        #Open CV requires "starting points" to begin the watershed. We generate those by finding the local
        #maximums in the data.This will oversegment the image. We will join segments in a later step
        #a footprint of 3x3 is a 9km-sq local max on a 1km grid
        #the code cannot seperate peaks that are within the footprint
        #
        coords = peak_local_max(smlin_vil, footprint=np.ones((3, 3)),
                                threshold_abs=1.0, min_distance=7)
        mask = np.zeros(smlin_vil.shape, dtype=bool)
        mask[tuple(coords.T)] = True
        markers, _ = ndi.label(mask)
        labels_arr = watershed(-smlin_vil, markers, mask=valid)
    
        #note: that this happens after the watershed is complete
        #note: the min_value must be in linVIL units 
        #note: 'Not an object' locations are labeled -1
    
        labels_arr = np.where(smlin_vil >= self.min_value, labels_arr, -1).astype(int)
        #Using the trimmed labels we want to identify watersheds that are adjacent
        #to one another. We do this by walking the image pixel by pixel. When we
        #find adjacent watersheds we determine if those watersheds should be
        #combined into a single watershed based on the "valley depth". This
        #parameter compares the highest value of the data in each watershed to the
        #"valley" location at the pixel. If the difference between the max data
        # and the  "valley" depth isn't low enough then the watersheds are combined.
    
    
        #exit after no move ids are combined
        # dummy variable for the check below
        old_number_of_ids = 9999
        removed_ids = [] #once an id is removed it stays removed
    
        # iterate as long as we find new connections
        while len(np.unique(labels_arr)[1:]) < old_number_of_ids:
            ids = np.unique(labels_arr)[1:]
            old_number_of_ids = len(ids)
            max_vals = ndi.maximum(input_data, labels=labels_arr, index=ids)
    
            #Lets create a dict of ids/max_vil values for ease
            #of use and understanding
            idx = dict(zip(ids, max_vals))
            # Sort by values in descending order
            idx_by_max = dict(sorted(idx.items(), key=lambda item: item[1], reverse=True))
    
            #idx_sort = np.argsort(max_vals)
            #ids_by_maximum = ids[idx_sort[::-1]]
            #once an id is removed it stays removed
            #removed_ids = []
            for target_id in idx_by_max.keys():
                if target_id in removed_ids:
                    continue
    
                neighboring_labels, border_mask = self.get_neighboring_labels(
                    labels_arr, target_id)
    
                if neighboring_labels.size == 0:
                    continue
    
                for neighbor_id in neighboring_labels:
                    #we removed the -1 values in get_neighboring_labels
                    #if neighbor_id == -1:
                    #    continue
    
                    # Find the border between label and neighbor
                    border = (border_mask & (labels_arr == neighbor_id))
    
                    # our reference the weakest peak that is being checked
                    peak_val = np.min(
                        [idx_by_max[neighbor_id],
                        idx_by_max[target_id]]
                                     )
    
                    # Calculate the maximum value along the border in the data
                    max_val_along_watershed = np.max(input_data[border])
    
                    # if the maximum value is close enough to the smaller peak VIL,
                    # then we combine the objects
                    if (peak_val - max_val_along_watershed) < self.valley_depth:
                        #print("Combine: %d and %d"%(target_id, neighbor_id))
                        labels_arr[labels_arr == neighbor_id] = target_id
                        removed_ids.append(neighbor_id)


        #monte python likes "0" not -1 as no object flag
        labels_arr[labels_arr == -1] = 0
        return labels_arr

    def _binarize(self, input_data):
        """ Binarizes input_data with the object boundary threshold"""
        binary  = np.where(np.round(input_data, 10) >= round(thresh, 10), 1, 0) 
        binary  = binary.astype(int)
        return binary 

    def get_neighboring_labels(self, labels_arr: np.ndarray,
                           target_id: int):
        structure = ndi.generate_binary_structure(2, 2)
    
        # Create a mask for the current region
        region_mask = labels_arr == target_id
    
        # Dilate the region to find neighbors
        border_mask = ndi.binary_dilation(
            region_mask, structure=structure) & (labels_arr != target_id)
    
        # Find neighboring labels
        neighboring_labels = np.unique(labels_arr[border_mask])
    
        #remove the -1 values
        neighboring_labels = neighboring_labels[~(neighboring_labels == -1)]
    
        return neighboring_labels, border_mask

