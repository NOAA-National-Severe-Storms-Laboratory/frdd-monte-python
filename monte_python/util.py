import re
from os.path import exists, basename
from datetime import datetime, timedelta

def isPath(s):
    """
    @param s string containing a path or url
    @return True if it's a path, False if it's an url'
    """
    if exists(s): 
        return True
    elif s.startswith("/"): 
        return True
    elif len(s.split("/")) > 1: 
        return True
    else:
        return False
    

def decompose_file_path(file_path, 
                        file_pattern = 'wofs', 
                        comp_names = None,
                        decompose_path = False, 
                       ):
    """
    Decompose a file into its components. Default behavior is to decompose 
    WoFS summary files into components, but could be used for other file paths.
    
    Parameters
    ----------------
    file_path : 'wofs', 'wrfin', str, path-like 
        Path to a file or the filename. If a path, then the code internally converts to 
        the file name. 
    
    file_pattern : re-based str
        A re-structured string 
        
    comp_names : list of strings
        Names of the components
        
    decompose_path : True/False (default=False)
        If True, then decompose_file_path assumes that file_pattern is a path-like string
        otherwise, the decompose_file_path will treat it as the file name. 
    
    Returns
    -------------
        components : dict 
            A dictionary of file component names and the components themselves. 
    
    Raises
    ------------
    ValueError
        Components names must be provide, if the user provides a file path (not using 
        one of the default options) 
    
    ValueError
        The given file must match the pattern given.
        
    AssertionError
        The Number of components has to equal the number of file path components. 
    
    """
    
    if file_pattern == 'wofs':
        file_pattern = 'wofs_(\S{3,14})_(\d{2,3}|RAD)_(\d{8})_(\d{4})_(\d{4}).(nc|json|feather)'
        comp_names = ['TYPE', 'TIME_INDEX', 'VALID_DATE', 'INIT_TIME', 'VALID_TIME', 'FILE_TYPE']

    if not decompose_path:
        if isPath(file_path):
            file_path = basename(file_path)
            if comp_names is None:
                raise ValueError('Must provide names for the file path components!') 
    
    
    dtre = re.compile(file_pattern)
    
    try:
        obj = dtre.match(file_path)
    except:
        raise ValueError('File given does not match the pattern!') 
    
    if obj is None:
        raise ValueError('File given does not match the pattern!') 
    
    comps = obj.groups()
    
    assert len(comps) == len(comp_names), f"""
                                          Number of component names does not equal the number of file components!
                                          components: {comps} 
                                          component names : {comp_names}
                                          """
    
    components = {n : c for n,c in zip(comp_names, comps)}
    
    return components 

def get_valid_time(wofs_file):
    # Get valid time from wofs file. 
    comps = decompose_file_path(wofs_file)
    init_date, init_time = comps['VALID_DATE'], comps['INIT_TIME']
    init_dt = datetime.strptime(init_date+init_time, '%Y%m%d%H%M')
    valid_duration = int(comps['TIME_INDEX'])*DT 
    valid_dt = init_dt + timedelta(minutes=valid_duration)
    valid_label = valid_dt.strftime('%Y%m%d_%H%M')
    
    return valid_label 

#from matplotlib.colors import (BoundaryNorm, ListedColormap,
#                               LinearSegmentedColormap, Normalize)
from matplotlib.colors import LinearSegmentedColormap
def get_obj_cmap():
    import matplotlib.pyplot as plt
    import numpy as np
    # Get the existing colormap
    cmap = plt.get_cmap('tab20b')
    # Extract the colors
    colors = cmap(np.linspace(0, 1, cmap.N))
    #
    #we need a colormap to 200 (or more?), but with alternating colors that are not close
    #
    color_index = [0,19,4,15,8,3,16,7,12,9,1,18,5,14,10,17,2,13,6,11]
    object_colors = []
    white = [1., 1., 1. , 1.]
    object_colors.append(white)

    for l in range(10): #need 200 so 20*10
        for n in range(cmap.N):
            object_colors.append(colors[color_index[n]])

    object_cmap = LinearSegmentedColormap.from_list(name='object_index', colors=object_colors)
    return object_cmap
