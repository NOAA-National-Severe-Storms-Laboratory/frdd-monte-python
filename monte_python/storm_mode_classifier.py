from ._mode_classifier import get_constituent_storms, get_storm_types
from .object_identification import label
from .object_quality_control import QualityControler

import numpy as np 

class StormModeClassifier:
    """
    StormModeClassifier is 7-scheme storm mode classification method developed by 
    Potvin et al. First, image segmentation creates composite reflectivity and 
    mid-level rotation regions using a single-threshold method. From the combination 
    of these two fields, the initial storm mode is determined as one of the following:
    the following:
     (1) NONROT
     (2) ROTATE
     (3) QLCS
     (4) AMORPHOUS
    Through an iterative process, a higher reflectivity threshold is used to 
    identify and classify embedded storms. The final modes include:
     (1) ORDINARY 
     (2) SUPERCELL 
     (3) QLCS
     (4) SUPERCELL CLUSTER (SUP_CLUST)
     (5) ORDINARY QLCS CELL (QLCS_ORD)
     (6) ROTATING QLCS CELL (QLCS_MESO)
     (7) OTHER
    
    Authors: Corey Potvin (NOAA NSSL), Montgomery Flora (OU/CIWRO)
    
    Parameters
    ----------------
    dbz_thresh : float (default = 43.0)
        The composite reflectivity threshold used for object identification. 
        
    rot_thresh : float (default = 55.0)
        The mid-level rotation threshold used for object identification. 
        Default is for WoFS mid-level updraft helicity. 
        
    dbz_qc_params : list of tuples (default = None) 
        QC parameters for the composite reflectivity object identification. 
        See monte_python.object_quality_control.QualityControler for more details
        The default is None and uses the parameters for WoFS data from Potvin et al. 2022, 
    
    rot_qc_params : list of tuples (default = None) 
        QC parameters for the mid-level rotation object identification. 
        See monte_python.object_quality_control.QualityControler for more details
        The default is None and uses the parameters for WoFS data from Potvin et al. 2022
    
    emb_qc_params : list of tuples (default = None) 
        QC parameters for the embedded composite reflectivity object identification. 
        See monte_python.object_quality_control.QualityControler for more details
        The default is None and uses the parameters for WoFS data from Potvin et al. 2022
    
    grid_spacing : integer/float
        The data grid spacing in meters. 
    
    Attributes
    --------------
    converter : dict 
        The conversion between the storm modes and their integer labels 
    
    """
    MODES = ['ORDINARY', 
             'SUPERCELL', 
             'QLCS', 
             'SUP_CLUST', 
             'QLCS_ORD', 
             'QLCS_MESO', 
             'OTHER']
    
    
    def __init__(self, dbz_thresh=43., rot_thresh=55., grid_spacing=3000, 
                 dbz_qc_params=None, rot_qc_params=None, emb_qc_params = None):
        
        self._dbz_thresh = dbz_thresh
        self._rot_thresh = rot_thresh
        self._ANALYSIS_DX = grid_spacing
        
        if dbz_qc_params is None:
            self._dbz_qc_params =  [
                    ("min_area", 15),
                    ("merge_thresh", 3),
                    ("max_thresh", (44.93, 100)),
                ]
        else:
            self._dbz_qc_params = dbz_qc_params
        
        if rot_qc_params is None:
            self._rot_qc_params = [
                                ("merge_thresh", 1),
                                ("min_area", 5),
                               ]
        else:
            self._rot_qc_params = rot_qc_params
        
        if emb_qc_params is None:
            self._emb_qc_params =  [
                ("min_area", 10),
                ("merge_thresh", 0),
                ("max_thresh", (44.93, 100)),
                ]
        else:
            self._emb_qc_params = emb_qc_params
    
    
        if not isinstance(self._dbz_thresh, float):
            raise ValueError('dbz_thresh must be a float!')
        
        if not isinstance(self._rot_thresh, float):
            raise ValueError('rot_thresh must be a float!')
        
        if not isinstance(self._dbz_qc_params, list):
            raise ValueError('dbz_qc_params must be a list!')
        
        if not isinstance(self._rot_qc_params, list):
            raise ValueError('rot_qc_params must be a list!')
    
    
    @property
    def converter(self):
        digitize_types = {n+1 : key for n, key in enumerate(self.MODES)}
        return digitize_types
    
    
    def _label_and_qc(self, input_data, thresh, qc_params):
        """ Label the data and apply QC """
        labels, label_props =  label(
            input_data=input_data,
            method="single_threshold",
            params={"bdry_thresh": thresh},
        )
  
        labels_qced, props_qced = QualityControler().quality_control(
                    object_labels=labels,
                    object_properties=label_props,
                    input_data=input_data,
                    qc_params=qc_params,
                )

        return labels_qced, props_qced
    
    def classify(self, dbz_vals, rot_vals):
        """
        Identify and classify composite reflectivity objects using the 
        7-mode scheme from Potvin et al. 
        
        Parameters
        ---------------------
            dbz_vals, array-like with shape (NY, NX)
                Composite reflectivity field at single time to identify and classify.
            
            rot_vals, array-like with shape (NY, NX) 
                Optional 30-minute time-maximum mid-level rotation field to aid in 
                the storm mode classification. For NWP, we recommend mid-level UH and 
                for radar data, azimuthal shear. Expected to be a 
                time-composite field (e.g., max over the last 30 or 60 minutes).
            
        Returns
        ---------------------
           storm_modes : array of shape (NY, NX) 
               Integer labelled array where values 1-7 indicate the storm mode. 
               
           merged_labels : array of shape (NY, NX)
               Integer labelled array of the composite reflectivity objects 
               (including the embedded regions).
               
           dbz_props : skimage.measure.RegionProps objects 
               Object properties of the the merged_labels array.

        """
        if np.ndim(dbz_vals) != 2:
            raise ValueError('dbz_vals must be a 2D array')
            
        if np.ndim(rot_vals) != 2:
            raise ValueError('rot_vals must be a 2D array')    
        
        dbz_labels, dbz_props = self._label_and_qc(dbz_vals, self._dbz_thresh, self._dbz_qc_params)
        rot_labels, rot_props = self._label_and_qc(rot_vals, self._rot_thresh, self._rot_qc_params)   
    
        storm_types, labels_with_matched_rotation = get_storm_types(
                        None,
                        dbz_labels,
                        dbz_props,
                        dbz_vals,
                        rot_labels,
                        rot_props,
                        rot_vals,
                        ANALYSIS_DX= self._ANALYSIS_DX
                    )
    
        min_thres_vals = np.arange(
                            self._dbz_thresh + 0, self._dbz_thresh + 23.1, 1
                        )
        
        for itr, min_thres in enumerate(min_thres_vals):
            (
             storm_types,
             storm_embs,
             dbz_props,
             storm_depths,
                ) = get_constituent_storms(
                                None,
                                min_thres,
                                self._emb_qc_params,
                                storm_types,
                                dbz_labels,
                                dbz_props,
                                dbz_vals,
                                rot_labels,
                                rot_props,
                                rot_vals,
                                itr + 1,
                                len(min_thres_vals),
                                self._ANALYSIS_DX,
                            )

        for n in range(len(dbz_props)):
            dbz_props[n].label = n + 1
    
        storm_modes, merged_labels, merged_storm_modes = self._embedding_to_2d_array( dbz_vals.shape,
                                                                 dbz_props, 
                                                                storm_types, 
                                                                storm_depths, 
                                                                storm_embs)
        return storm_modes, merged_labels, merged_storm_modes, dbz_props    
        
    def get_storm_labels(self, storm_emb, storm_type):
        convert_labels = {'ROTATE': 'SUPERCELL', 
                      'NONROT': 'ORDINARY', 
                      'SEGMENT': 'OTHER', 
                      'QLCS': 'QLCS', 
                      'CLUSTER': 'OTHER', 
                      'AMORPHOUS': 'OTHER', 
                      'QLCS_ROT': 'QLCS_MESO', 
                      'QLCS_NON': 'QLCS_ORD', 
                      'CLUS_ROT': 'SUPERCELL', 
                      'CLUS_NON': 'ORDINARY', 
                      'SUP_CLUST': 'SUP_CLUST',
                      'CELL_NON': 'ORDINARY', 
                      'CELL_ROT': 'SUPERCELL'}

        digitize_types = {y:x for x,y in self.converter.items()}
        
        if storm_emb == 'NONEMB':
            label = storm_type
        else:
            label = f'{storm_emb[4:]}_{storm_type[:3]}'

        type_str = convert_labels[label]
        type_int = int(digitize_types[type_str])

        return type_int, type_str
    
    def _embedding_to_2d_array(self, shape, dbz_props, storm_modes, storm_depths, storm_embs):
        """ Convert embedded modes and labels to 2D array """
        storm_mode_array = np.zeros(shape, dtype=int)
        merged_labels = np.zeros(shape, dtype=int)
        merged_storm_modes = []
        for storm_depth in range(3):
            inds = np.where(np.array(storm_depths)==storm_depth)[0]
            # get regionprops, storm types, and embedded status for each storm of current depth
            props = [dbz_props[index] for index in inds]
            types = [storm_modes[index] for index in inds]
            embs = [storm_embs[index] for index in inds]
            
            for region, mode, emb in zip(props, types, embs):
                # Insert each storm into labels array and merged labels array 
                # (labels in latter will later be overwritten by collocated deeper storms)
                coords = region.coords
                merged_labels[coords[:,0], coords[:,1]] = region.label
                # based on storm type and embedded status, get integer corresponding to storm mode, 
                # and insert into storm modes array
                storm_type_int, storm_type_str = self.get_storm_labels(emb, mode)
                merged_storm_modes.append(storm_type_int)
                storm_mode_array[coords[:,0], coords[:,1]] = storm_type_int
        
        return storm_mode_array, merged_labels, merged_storm_modes
