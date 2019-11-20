import netCDF4
import numpy as np
import pandas as pd 
from skimage.measure import regionprops

def object_properties_list(  ):
    """ Initializes a dictionary with object properties as keys"""
    keys= [ 'obj_centroid_x',
            'obj_centroid_y',
            'area'          ,
            'eccentricity'  ,
            'extent'        ,
            'orientation'   ,
            'minor_axis_length',
            'major_axis_length',
            'matched_to_azshr',
            'matched_to_lsr', 
            'matched_to_torn_warn',
            'matched_to_svr_warn', 
            'object label' , 
            'Ensemble member'] 
    return keys     

def save_object_properties( filename, object_properties_dict=None ):

    if type( object_properties_dict) != type(None): 
        df = pd.DataFrame.from_dict( object_properties_dict )  
        df.to_hdf( filename, key = 'df', mode ='w' ) 
    else: 
        return pd.read_hdf( filename ) 

def append_object_properties( object_props, matched_to_torn_warn, matched_to_lsrs, matched_to_azshr, matched_to_svr_warn, **kwargs):
        """ Append the various object properties to a dictionary """
        n_props = len( object_properties_list( ))
        object_properties_per_object = [ ] 
        if len(object_props) > 0: 
            for region in object_props:
                object_properties_per_ens_mem = np.zeros(( n_props ))
                object_centroid = region.centroid
                object_properties_per_ens_mem[ 0] = object_centroid[1] #x 
                object_properties_per_ens_mem[ 1] = object_centroid[0] #y 
                object_properties_per_ens_mem[ 2] = region.area 
                object_properties_per_ens_mem[ 3] = region.eccentricity 
                object_properties_per_ens_mem[ 4] = region.extent
                object_properties_per_ens_mem[ 5] = region.orientation 
                object_properties_per_ens_mem[ 6] = region.minor_axis_length 
                object_properties_per_ens_mem[ 7] = region.major_axis_length
                object_properties_per_ens_mem[ 8] = matched_to_azshr[region.label]     
                object_properties_per_ens_mem[ 9] = matched_to_lsrs[region.label]  
                object_properties_per_ens_mem[10] = matched_to_torn_warn[region.label]
                object_properties_per_ens_mem[11] = matched_to_svr_warn[region.label]
                object_properties_per_ens_mem[12] = region.label
                object_properties_per_ens_mem[13] = kwargs['mem'] 
                
                object_properties_per_object.append( object_properties_per_ens_mem )  
                del object_properties_per_ens_mem

            return object_properties_per_object
        
        else:
            object_properties_per_object.append( np.zeros(( n_props )) )
            return object_properties_per_object            
