import itertools
import sys
sys.path.append('/home/monte.flora/machine_learning/extraction')
from StormBasedFeatureEngineering import StormBasedFeatureEngineering

env_vars_smryfiles = ['cape_0to3_ml', 'srh_0to1', 'srh_0to3', 'qv_2', 't_2', 'td_2']
env_vars_wofsdata  = ['buoyancy', 'rel_helicity_0to1', '10-500 m bulk shear', 'Mid-level Lapse Rate', 'Low-level Lapse Rate',
                            'Temp (1km)', 'Temp (3km)', 'Geopotential (1km)', 'Geopotential (5km)']
env_vars_mrms = ['MRMS Reflectivity']

storm_vars_smryfiles = ['uh_0to2', 'uh_2to5', 'wz_0to2', 'comp_dz', 'ws_80']
storm_vars_wofsdata = ['w_1km', 'w_down', 'div_10m']

object_properties_list = ['obj_centroid_x', 'obj_centroid_y', 'area', 'eccentricity', 'extent', 'orientation', 'minor_axis_length', 'major_axis_length', 'matched_to_azshr', 'matched_to_lsr', 'matched_to_torn_warn', 'matched_to_svr_warn',  
                          'object label', 'Ensemble member'] 

map_to_readable_names={ 
                    'MRMS Reflectivity':'MRMS Reflectivity', 
                    'obj_centroid_x': 'X-comp of Object Centroid',
                    'obj_centroid_y': 'Y-comp of Object Centroid',
                    'area': 'Area', 
                    'eccentricity': 'Eccentricity', 
                    'extent' : 'Extent',
                    'orientation': 'Orientation',
                    'minor_axis_length': 'Minor Axis Length',
                    'major_axis_length': 'Major Axis Length', 
                    'matched_to_azshr': 'Matched to Azimuthal Shear',
                    'matched_to_lsr': 'Matched to LSRs',
                    'matched_to_torn_warn': 'Matched to Tornado Warning Polygon',
                    'matched_to_svr_warn' : 'Matched to Severe Weather Warning Polygon',
                    'object label': 'Object Label',
                    'Ensemble member': 'Ensemble member',
                    'cape_0to3_ml':'0-3 km ML CAPE',
                    'srh_0to1': '0-1 km SRH',
                    'srh_0to3': '0-3 km SRH',
                    'qv_2': '2-m Water Vapor',
                    't_2': '2-m Temperature',
                    'td_2': '2-m Dewpoint Temperature',
                    'buoyancy': 'Buoyancy',
                    'rel_helicity_0to1': '0-1 km Relative Helicity',
                    '10-500 m bulk shear': '10-500 m Bulk Shear',
                    'Mid-level Lapse Rate': 'Mid-level Lapse Rate',
                    'Low-level Lapse Rate': 'Low-level Lapse Rate',
                    'Temp (1km)': 'Temperature (1km AGL)',
                    'Temp (3km)': 'Temperature (3km AGL)',
                    'Geopotential (1km)': 'Geopotential Height (1km AGL)',
                    'Geopotential (5km)': 'Geopotential Height (5km AGL)', 
                    'uh_0to2': '0-2 km Updraft Helicity', 
                    'uh_2to5': '2-5 km Updraft Helicity',
                    'wz_0to2': '0-2 km Vertical Vorticity', 
                    'comp_dz': 'Composite Reflectivity',
                    'ws_80'  : '80-m Wind Speed',
                    'w_1km'  : 'Low-level Updraft', 
                    'w_down' : 'Column-min Downdraft', 
                    'div_10m': '10-m Divergence' } 

obj_props_for_learning = ['area', 'eccentricity', 'extent', 'orientation', 'minor_axis_length', 'major_axis_length', 'matched_to_torn_warn', 'matched_to_svr_warn', 'obj_centroid_x', 'obj_centroid_y'] 
additional_vars = ['Initialization Time', 'YSU', 'MYJ', 'MYNN'] 

def combine( variable_list, only_mean):
    '''
    Mapping variable names to readable names.
    '''
    stat_funcs_readable = [ ' (Standard Deviation)', ' (Average)', ' (10th percentile) ', ' (Median)', ' (90th percentile)' ]
    if only_mean:
         stat_funcs_readable=[ ' (Average)']
    extract = StormBasedFeatureEngineering( )
    # Determine the feature names 
    stat_funcs  = extract._set_of_stat_functions( names=True, only_mean=only_mean )    

    feature_names = [ ''.join(tup) for tup in list(itertools.product( variable_list, stat_funcs )) ]
    feature_names_readable = [ ''.join((map_to_readable_names[var], stat_func)) for (var,stat_func) in list(itertools.product( variable_list, stat_funcs_readable )) ] 

    return feature_names, feature_names_readable


def _feature_names_for_traditional_ml( obj_props_train=False  ):
    '''
    Determines the feature names for the traditional machine learning     
    Args:
        storm_vars, 2-tuple of dictionaries where the 0 = smryfiles and 1 is wofsdata

    '''
    stat_funcs_readable = [ ' (Standard Deviation)', ' (Average)', ' (10th percentile) ', ' (Median)', ' (90th percentile)' ]
  
    storm_vars = storm_vars_smryfiles + storm_vars_wofsdata
    storm_feature_names, storm_feature_names_readable = combine( variable_list=storm_vars, only_mean=False )
    strm_feature_colors = ['lightblue'] * len(storm_feature_names)

    environment_vars = env_vars_smryfiles + env_vars_wofsdata + env_vars_mrms 
    env_feature_names, env_feature_names_readable = combine( variable_list=environment_vars, only_mean=True )
    env_feature_colors = ['navajowhite']* len(env_feature_names)

    # Feature names in the netcdf files
    if obj_props_train:
        object_properties=obj_props_for_learning
    else:
        object_properties = object_properties_list 
    feature_names = storm_feature_names + env_feature_names + object_properties + additional_vars

    obj_props_colors = ['lightgreen'] *len(object_properties)
    additional_var_colors = ['lightgreen'] *len(additional_vars) 

    object_properties_readable = [ map_to_readable_names[var] for var in object_properties]
    feature_names_readable = storm_feature_names_readable + env_feature_names_readable +  object_properties_readable + additional_vars
    feature_colors = strm_feature_colors + env_feature_colors + obj_props_colors + additional_var_colors

    VARIABLE_NAMES_DICT = {var: readable_name for readable_name, var in zip(feature_names_readable, feature_names) }
    VARIABLE_COLORS_DICT = {var: color for color, var in zip(feature_colors, feature_names) } 

    return feature_names, (storm_vars_smryfiles, storm_vars_wofsdata), (env_vars_smryfiles, env_vars_wofsdata), object_properties, (VARIABLE_NAMES_DICT, VARIABLE_COLORS_DICT)




