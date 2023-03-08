import warnings
import numpy as np
from math import atan2, ceil, pi, log, sqrt, pow, fabs, cos, sin
from skimage.measure import regionprops
import itertools
from collections import OrderedDict
from .object_identification import label
from .object_quality_control import QualityControler

import pandas as pd
import xarray as xr
import random
from copy import copy
from copy import copy

from .STORM_CLASSIFY_defs import get_constituent_storms, get_storm_types

class StormModeClassifier:
    """
    StormModeClassifier is 7-scheme storm mode classification method developed by 
    Potvin et al. First, image segmentation will label the composite reflectivity and 
    mid-level rotation field using a single-threshold method. From these initial labels, 
    an initial storm mode is determined. After an iterative process, it further refines 
    the image segmentation and storm mode classification. The 7 possible storm 
    modes include:
    (1) Supercell 
    (2) Supercell Cluster 
    (3) QLCS 
    (4) Ordinary 
    (5) 
    (6)
    (7) Other 
    
    Authors: Corey Potvin (NOAA NSSL), Montgomery Flora (OU/CIWRO) 

    Parameters
    ----------------
        dbz_thresh, float 
            Threshold used for the initial composite reflectivity object identification.
        
        rot_thresh, float, default=None
            Threshold used for mid-level rotation object identification. 
        
        grid_spacing, float (default=3000)
            Grid spacing (in meters). Used to determine lengths and areas for storm 
            mode classification. 
    
        dbz_qc_params, list of 2-tuples (default=None)
            QualityControler (see monte_python.object_quality_control) 
            qc_params for the reflectivity objects. 
       
        dbz_qc_params_emb, list of 2-tuples (default=None)
            QualityControler (see monte_python.object_quality_control) 
            qc_params for the constituent/embedded reflectivity objects. 
            
        rot_qc_params, list of 2-tuples (default=None)
            QualityControler (see monte_python.object_quality_control) 
            qc_params for the mid-level rotation objects.
        
        verbose, boolean (default=False)
            If True, various progress statements are printed. 
            Used for debugging. 
        
    """
    CONVERTER = {
        'ORDINARY' : 1,
        'NONROT' : 1,
        'CLUS_NON' : 1,
        
        'SUPERCELL' : 2,
        'ROTATE' : 2,
        
        'QLCS' : 3, 
        'SUP_CLUSTER' : 4,
        'SUP_CLUST' : 4,
        'QLCS_ORD' : 5, 
        'QLCS_MESO' : 6,
        'QLCS_ROT' : 6, 
        
        'OTHER'  : 7,
        'AMORPHOUS' : 7, 
        'CLUSTER' : 7, 
    }

    def __init__(self, 
                 dbz_thresh=43.0, 
                 rot_thresh=55., 
                 dbz_qc_params=None, 
                 dbz_qc_params_emb=None,
                 rot_qc_params=None, 
                 grid_spacing=3000, 
                verbose=False):
        
        near_storm_diameter = 120 * 1000 # in meters
        
        self.verbose = verbose
        self.dbz_thresh = dbz_thresh
        self.rot_thresh = rot_thresh
        self.grid_spacing = grid_spacing
        self.near_storm_diameter = near_storm_diameter / grid_spacing 

        # TODO: It might make sense to make to turn this into an argument 
        self.dbz_thresh_rng = np.arange(self.dbz_thresh+0, self.dbz_thresh+23.1, 1)

        if dbz_qc_params is None:
            self.dbz_qc_params = [
                    ("min_area", 15),
                    ("merge_thresh", 3),
                    ("max_thresh", (44.93, 100)),
                ]
        else:
            self.dbz_qc_params = dbz_qc_params
        
        if dbz_qc_params_emb is None:
            self.dbz_qc_params_emb = [
                ("min_area", 10),
                ("merge_thresh", 0),
                ("max_thresh", (44.93, 100)),
            ]
        else:
            self.dbz_qc_params_emb = dbz_qc_params_emb
        
        if rot_qc_params is None:
            self.rot_qc_params = [
                                ("merge_thresh", 1),
                                ("min_area", 5),
                               ]
        else:
            self.rot_qc_params = rot_qc_params
                        
    def classify(self, dbz_vals, rot_vals, label_embedded=True):
        """
        Identify and classify composite reflectivity objects using the 
        7-mode scheme from Potvin et al. Optionally, it can include mid-level rotation to aid 
        in the classification. 
        
        Parameters
        ---------------------
            dbz_vals, array-like with shape (NY, NX)
                Composite reflectivity field at single time to identify and classify.
            
            rot_vals, array-like with shape (NY, NX) 
                Optional 30-minute time-maximum mid-level rotation field to aid in 
                the storm mode classification. For NWP, we recommend mid-level UH and 
                for radar data, azimuthal shear. Expected to be a 
                time-composite field (e.g., max over the last 30 or 60 minutes).
            
            label_embedded, boolean (default=True)
                If True, then embedded storms are identified and classified. 
                Otherwise, storm modes are only based on a 3-mode scheme. 
            
        Returns
        ---------------------
            2-tuple of labelled composite reflectivity regions (of shape NY, NX) 
            and storm modes(of shape NY, NX) 
            
            Storm modes are interger-labelled, here is the conversion:
                1 : 'ORDINARY'
                2 : 'SUPERCELL'
                3 : 'QLCS'
                4 : 'SUP_CLUSTER'
                5 : 'QLCS_ORD'
                6 : 'QLCS_MESO'
                7 : 'OTHER'  
        """
        # Identify and quality control the composite reflectivity and mid-level rotation objects. 
        dbz_labels, dbz_props = self._label_and_qc(dbz_vals, 
                                                   thresh=self.dbz_thresh, 
                                                   qc_params=self.dbz_qc_params, 
                                     )
        
        labels = copy(dbz_labels)
        
        rot_labels, rot_props = self._label_and_qc(rot_vals, 
                                          thresh=self.rot_thresh, 
                                          qc_params=self.rot_qc_params,
                                         )
        
        # Get the storm mode classifications.
        #storm_modes, labels_with_matched_rotation = self.get_storm_modes(dbz_data, rot_data)
        
        temp_dbz_labels = dbz_labels.copy()
        
        storm_modes, labels_with_matched_rotation = get_storm_types(None, 
                                        temp_dbz_labels,
                                        dbz_props,
                                        dbz_vals,
                                        rot_labels,
                                        rot_props,
                                        rot_vals,
                                    )

        print(f'{labels_with_matched_rotation=}')
        print(f'Initial storm modes : {storm_modes}')
        
        storm_embs = ["NONEMB"] * len(storm_modes)
        storm_depths = [0]*len(storm_modes)
        # if True, find possibly embedded storms.
        if label_embedded:
            dbz_data = (dbz_vals, temp_dbz_labels, dbz_props)
            rot_data = (rot_vals, rot_labels, rot_props) 
                  
                    
            (storm_modes, 
             storm_embs, 
             dbz_props, 
             storm_depths) = self.get_embedded_storms(dbz_data, rot_data, storm_modes) 

        return labels, dbz_props, storm_modes, storm_embs, storm_depths, rot_labels

    def mode_to_2d(self, storm_modes, dbz_labels, dbz_props):
        """Convert the storm modes to a 2D array"""
        #storm_mode_array = np.zeros(dbz_labels.shape, dtype=int)
        #for region, mode in zip(dbz_props, storm_modes):
        #    coords = region.coords
        #    storm_mode_array[coords[:,0], coords[:,1]] = self.CONVERTER[mode]
        return storm_mode_array
    
    def get_regionprops(self, input_data, labels):
        return regionprops(labels, input_data)   
    
    def is_near_boundary(self, region, NY, NX):
        # TODO: Needs to be fixed! 
        # This is meant to exclude storm objects too close the WoFS domain boundaries.
        buffer = int(0.5*self.near_storm_diameter)
        print(f'{buffer=}', f'{region.centroid[0]=}')
        
        x,y = int(region.centroid[0]), int(region.centroid[1])
        
        print(x,y) 
        
        return (
                x <= buffer 
                or y <= buffer
                or (region.centroid[0] + buffer) > NY - 2 
                or (region.centroid[1] + buffer) > NX - 2
                     )
    
    def relabel(self, data):
        """Re-label data so that np.max(data) == number of objects."""
        relabelled_data = np.copy(data)
        #Ignore the zero label
        unique_labels = np.unique(data)[1:]
        for i, label in enumerate(unique_labels):
            relabelled_data[data==label] = i+1
    
        return relabelled_data
    
    
    def _label_and_qc(self, input_data, thresh, qc_params):
        """Identify and quality control objects (FOR INTERNAL PURPOSES ONLY)"""
        labels, label_props =  label(
            input_data=input_data,
            method="single_threshold",
            params={"bdry_thresh": thresh},
        )
        #if qc_params is None:
        #    return input_data, labels, label_props
        
        labels_qced, props_qced = QualityControler().quality_control(
                    object_labels=labels,
                    object_properties=label_props,
                    input_data=input_data,
                    qc_params=qc_params,
                )

        """
        NY, NX = input_data.shape[0], input_data.shape[1]
        
        if is_dbz and remove_objects_near_bdry:
            # Remove those reflectivity objects that are near the 
            # WoFS domain boundary. This is done since environmental 
            # statistics are computed over identical regions for each 
            # storm objects.             
            labels_to_remove = [region.label for region in props_qced if 
                                self.is_near_boundary(region, NY, NX)]
            
            # For those labels in the labels to remove, set them to 0. 
            labels_qced[np.where(np.isin(labels_qced, labels_to_remove))] = 0
            
            # Re-label. 
            labels_qced = self.relabel(labels_qced)
            
            # Re-compute the regionprops:
            props_qced = self.get_regionprops(input_data, labels_qced) 
        """
        
        return labels_qced, props_qced

    def get_storm_modes(self, dbz_data, rot_data=None,):
        """
        Initial classification of storm objects into one of the 
        four follwing categories: 
        
        (1) ROTATE : Isolated cell matched to a rotation track
        (2) NONROT : Isolated cell not matched to a rotation track
        (3) QLCS 
        (4) AMORPHOUS
        
        Authors: Corey Potvin, Montgomery Flora 
        
        Parmeters:
        -------------------
            dbz_data, 3-tuple of arrays of shape (NY, NX) including 
                      the input data, objects identified from input, 
                      and the sklearn.measure.regionprops of the objects.
                      
                      The labelled data should already be QC'd
                      
                Example : dbz_data = (dbz, dbz_labels, dbz_props)
        
            rot_data, 3-tuple of arrays of shape (NY, NX) including 
                      the input data, objects identified from input, 
                      and the sklearn.measure.regionprops of the objects.
                    
                      The labelled data should already be QC'd
                    
                Example : rot_data = (uh, rot_labels, rot_props)
                   
        Returns
        -------------------
            storm_modes : list of strings, of len(number of objects) 
                Storm types for each object in the dbz_props, (i.e., one for each reflectivity object)
            
            labels_with_matched_rotation : list of integers
                Reflectivity object labels matched to a rotation track. 
        """
        dbz, dbz_labels, dbz_props = dbz_data
        
        # Match the composite reflectivity objects to the rotation tracks. 
        labels_with_matched_rotation = []
        if rot_data is not None:
            rot, rot_labels, rot_props = rot_data
            labels_with_matched_rotation = self.match_to_rotation_tracks(dbz_props, rot_props)

        storm_modes = []
        for region in dbz_props:
            # see scikit-image regionprops documentation for definitions of object attributes
            region_length = region.major_axis_length * self.grid_spacing
            region_area = region.area * pow(self.grid_spacing, 2)

            if self.verbose:
                print(f'area: {region_area:.02e}, Leng: {region_length:.02f}, ecc:{region.eccentricity:.03f}') 

            # If very small or smaller and near-circular and relatively solid, then it is likely
            # an isolated cell. Isolated cells are then as identified as rotating or non-rotating
            # based on whether the track can be reasonable matched to a rotation track. 
            if round(region_area, 10) < 100e6 or (
                round(region_area,10) < 200e6 and round(region.eccentricity, 10) < 0.7 \
                and round(region.solidity,10) > 0.7
            ):
                mode = "ROTATE" if region.label in labels_with_matched_rotation else "NONROT"
            
            elif round(region.eccentricity, 5) > 0.97:
                if self.verbose:
                    print('Region has eccentricity > 0.97, entered QLCS/AMORPHOUS...') 
                mode = "QLCS" if round(region_length,10) > 75e3 else "AMORPHOUS"
            elif round(region.eccentricity,10) > 0.9 and round(region_length,10) > 150e3:
                mode = "QLCS"
            elif (
                round(region.eccentricity,10) > 0.93
                or round(region_area,10) > 500e6
                or round(region_length,10) > 75e3
                or round(region.solidity,10) < 0.7
            ):
                mode = "AMORPHOUS"
            elif region.label in labels_with_matched_rotation:
                mode = "ROTATE"
            else:
                mode = "NONROT"

            storm_modes.append(mode)

        # At this moment, there should a storm mode for each 
        # labelled region. 
        if len(storm_modes) != len(np.unique(dbz_labels)[1:]):
            print(f'{len(storm_modes)=}')
            print(f'{len(np.unique(dbz_labels)[1:])=}')
            raise ValueError('Number of storm modes and number of labelled regions do not match!')
        
        return storm_modes, labels_with_matched_rotation
    
    def match_to_rotation_tracks(self, dbz_regionprops, rot_regionprops):
        """Match reflectivity objects to rotation tracks; used for identifying supercellular modes"""
        labels_with_matched_rotation = []
        # Concatenate all coordinates of strong rotation, then identify regions with proximate rotation objects
        all_rot_coords = [[coords for coords in region.coords] for region in rot_regionprops]
        all_rot_coords = [item for sublist in all_rot_coords for item in sublist]
        if len(all_rot_coords) > 0:
            for region in dbz_regionprops:
                # Looking within a square region around the dBZ object for a 
                # rotation object. 
                ic, jc = int(region.centroid[0]), int(region.centroid[1])
                rad = max(2, ceil(region.equivalent_diameter / 3.0))
                imin, imax = ic - rad, ic + rad
                jmin, jmax = jc - rad, jc + rad
                dbz_coords = [[i,j] for i,j in 
                              itertools.product(range(imin, imax + 1), range(jmin, jmax + 1))]
                if self.is_overlapping(dbz_coords, all_rot_coords):
                    labels_with_matched_rotation.append(region.label)                                 
        
        return labels_with_matched_rotation
    
    def is_overlapping(self, region_a_coords, region_b_coords):
        """Do two objects overlap at even a single grid point?"""
        set_a = set(tuple(i) for i in region_a_coords)
        set_b = set(tuple(i) for i in region_b_coords)
        return len(list(set_a.intersection(set_b))) > 0

    def get_embedded_storms(self, dbz_data, rot_data, storm_modes):
        
        min_thres = 43.0
        min_thresh2 = 44.93 
        
        dbz, dbz_labels, dbz_props = dbz_data
        rot, rot_labels, rot_props = dbz_data
        
        temp_dbz_labels = dbz_labels.copy()
        
        max_itr = len(self.dbz_thresh_rng)
        
        for itr, min_thresh in enumerate(self.dbz_thresh_rng):
            (storm_modes, 
             storm_embs, 
             dbz_props, 
             storm_depths) = get_constituent_storms(None, 
                                                    min_thresh,
                                                    min_thresh2,
                                                    storm_modes, 
                                                    temp_dbz_labels, 
                                                    dbz_props, 
                                                    dbz, 
                                                    rot_labels, 
                                                    rot_props, 
                                                    rot, 
                                                    itr+1, 
                                                    max_itr, 
                                                    debug=False)
        
        # Re-label
        for n in range(len(dbz_props)):
            dbz_props[n].label = n+1
    
        return storm_modes, storm_embs, dbz_props, storm_depths
    
    
    def get_constituent_storms(self, data_dict):
        """
        Description: Identify reflectivity objects with a progressively
                     higher threshold. Helps to identify embedded storms. 
    
        Parameters
        -------------------
            data_dict : dict 
                keys include 
                   'storm_modes'  
                   'dbz_labels'    
                   'dbz_props' 
                   'rot_labels'    
                   'rot_props'          
        Returns
        -------------------
        """
        # Get the original reflectivity objects and storm mode estimates. 
        dbz_vals, init_dbz_labels, init_dbz_props = data_dict['dbz_vals'], data_dict['dbz_labels'], data_dict['dbz_props']
        rot_values, rot_labels, rot_props = data_dict['rot_vals'], data_dict['rot_labels'], data_dict['rot_props']
        init_storm_modes = data_dict['storm_modes']
        
        rot_data = (rot_values, rot_labels, rot_props) 
        
        # Initialize the updated properties. 
        updated_storm_modes = init_storm_modes.copy()
        updated_dbz_labels = init_dbz_labels.copy()
        updated_dbz_props = init_dbz_props.copy()
        
        # Cycle through higher dBZ thresholf for identifying embedded objects. 
        for itr, min_thresh in enumerate(self.dbz_thresh_rng):
            label_inc = 100 * (itr - 1)
            
            # Identify new composite reflectivity objects with a higher minimum threshold. 
            # By increasing the minimum threshold, it allows for the identification of 
            # embedded storms. 
            new_dbz_data = self._label_and_qc(dbz_vals, thresh=min_thresh, qc_params=self.dbz_qc_params_emb)
            new_storm_modes, labels_with_matched_rotation = self.get_storm_modes(new_dbz_data, rot_data)
            _, new_dbz_labels, new_dbz_props = new_dbz_data
            
            # Cycle through the original storms
            for parent_region in init_dbz_props:
                child_storm_modes, child_dbz_props, child_labels = [], [], []
                parent_coords = parent_region.coords
                
                # Cycle through the newly identified composite refl. objects
                for n, (mode , sub_region) in enumerate(zip(new_storm_modes, new_dbz_props)):
                    if mode in ["ROTATE", "NONROT"]:
                        if self.is_overlapping(sub_region.coords, parent_coords):
                            # If candidate storm object overlaps an existing storm object, 
                            # retain and characterize it for further testing. 
                            child_storm_modes.append(mode)
                            # ensure new object has unique label so as not to combine with preexisting object
                            new_dbz_props[n].label += label_inc  
                            child_dbz_props.append(new_dbz_props[n])
                            child_labels.append(new_dbz_props[n].label)

                # Cycle through the new, child storm objects
                for nn, mode in enumerate(child_storm_modes):
                    # if storm object is a CELL, determine if this object is likely just a smaller version 
                    # of an existing object (versus a constituent of a larger storm)
                    if mode in ["ROTATE", "NONROT"]:
                        similar_storm_found = False
                        child_region = child_dbz_props[nn]
                        child_i = child_region.centroid[0]
                        child_j = child_region.centroid[1]

                        for nnn, existing_region in enumerate(updated_dbz_props):
                            if updated_storm_modes[nnn] in ["ROTATE", "NONROT"]:  ### NEW
                                exist_i = existing_region.centroid[0]
                                exist_j = existing_region.centroid[1]
                                dist = sqrt(
                                    pow(child_i - exist_i, 2) + pow(child_j - exist_j, 2)
                                )
                                if (
                                    dist < max(2, 0.15 * existing_region.equivalent_diameter)
                                    and child_region.area <= 1.0 * existing_region.area
                                ):
                                    similar_storm_found = True
                                    break
                                
                        # if candidate object is not merely a smaller version of an existing object, retain it
                        if not similar_storm_found:
                            updated_storm_modes.append(child_storm_modes[nn])
                            updated_dbz_props.append(child_dbz_props[nn])
                        
                            # insert new object into label image
                            updated_dbz_labels[
                                new_dbz_labels == (child_labels[nn] - label_inc)
                            ] = child_labels[nn]
                 
        print('Before final QC\n')
        print('------------------------------')
        print(updated_storm_modes)
        print('------------------------------\n')
        
        storm_embs = ["NONEMB"] * len(updated_storm_modes)
        results = self.final_qc(updated_storm_modes, 
                               storm_embs, 
                               updated_dbz_labels, 
                               updated_dbz_props, 
                               )
        return results                
 
                            
    def final_qc(self, storm_modes, storm_embs, dbz_labels, dbz_props,):
        """
        After the final iteration, quality control the objects. 
        Create "family tree" of objects; new_storm_depths contains the "generation" of each object; 
        new_storm_parent_labels contains the parent label for each child object; new_storm_embs
        contains the parent type of each child object
        new_storm_depths is later used to process objects in order of their generation. 
        
        Parameters
        -----------------------
        
        Returns
        -----------------------
        
        """
        attrs = ['remove_stuff', 
                 'reclassify_qlcs', 
                 'check_if_embedded', 
                 'separate_supercell_clusters', 
                 'to_amorphous', 
                ]

        results = { 
         'storm_modes' : storm_modes, 
         'dbz_labels' : dbz_labels, 
         'dbz_props' : dbz_props, 
         'storm_embs' : storm_embs, 
               } 

        results = self.object_hierarchy(results)
        
        for attr in attrs:
            results = getattr(self, attr)(results,)
            if attr != 'to_amorphous':
                results = self.object_hierarchy(results)

        return results
 
    def remove_stuff(self, data,):
        """ Remove stuff """
        if self.verbose:
            print('Removing stuff...') 
            
        storm_modes = data['storm_modes']
        dbz_props = data['dbz_props']
        storm_embs = data['storm_embs']
        parent_labels = data['storm_parent_labels']
        
        remove_indices = []
        where_emb_qlcs = np.where(storm_embs == 'EMB-QLCS')[0]
        for n in where_emb_qlcs:
            parent_coords = dbz_props[n].coords
            for nn, storm2_region in enumerate(dbz_props):    
                if round(dbz_props[nn].area, 5) < round(dbz_props[n].area, 5):
                    if self.is_overlapping(storm2_region.coords, parent_coords):
                        if nn not in remove_indices:
                            remove_indices.append(nn)
                            if self.verbose:
                                print(
                                      "Removing %s within %s within %s"
                                       % (
                                        storm_modes[nn],
                                        mode,
                                         storm_embs[n][4:],
                                         )
                                     )
                                        
        for ind in sorted(remove_indices, reverse=True):
            storm_modes.pop(ind)
            storm_embs.pop(ind)
            dbz_props.pop(ind)
            parent_labels.pop(ind)
       
        # Update data with the new storm modes, storm_embs, dbz_props, and parent_labels.
        data['storm_modes'] = storm_modes
        data['storm_embs'] = storm_embs
        data['dbz_props'] = dbz_props
        data['storm_parent_labels'] = parent_labels
            
        return data
        
    def reclassify_qlcs(self, data):
        """ Reclassify QLCS storm type as cluster if object area dominated by CELLs"""
        if self.verbose:
            print('Reclassifying QLCSs as CELLs...') 
        
        storm_modes = data['storm_modes']
        dbz_props = data['dbz_props']
        storm_parent_labels = data['storm_parent_labels']
        
        updated_storm_modes = storm_modes.copy()
        
        for n, (mode, region) in enumerate(zip(storm_modes, dbz_props)):
            if (mode == "QLCS" and region.major_axis_length < 150 / 3.0):
                num_const, num_rot, const_area = 0,0,0 
                for nn in range(len(storm_modes)):
                    if (nn != n and storm_parent_labels[nn] == region.label):
                        num_const += 1
                        if storm_modes[nn] == "ROTATE":
                            num_rot += 1
                        const_area += region.area
                        
                const_area_frac = const_area / float(region.area)
                if const_area_frac > 0.75:
                    if num_rot > 1:
                        updated_storm_modes[n] = "SUP_CLUSTER"
                    elif num_const > 1:
                        updated_storm_modes[n] = "CLUSTER"
                    else:
                        updated_storm_modes[n] = "AMORPHOUS"

        # Update the new storm modes.                     
        data['storm_modes'] = updated_storm_modes              
                        
        return data

    def check_if_embedded(self, data):
        """
        Checking for potentially embedded storms:
            -If object embedded within QLCS_embedded object, discard the former
        
            -If object embedded within AMORPHOUS or CELL is the only object embedded therein, 
                 discard it. 
        
            -If parent object is AMORPHOUS, reclassify it as discarded child object type.
        
            -If object embedded within AMORPHOUS or CELL is one of multiple such objects, 
                and parent object is unembedded, reclassify parent as CLUSTER or SUP_CLUSTER.
        
            -If object embedded within AMORPHOUS or CELL is one of multiple such objects, 
                and parent object is embedded in another object, remove parent since 
                it we don't want embedded clusters
        """
        if self.verbose:
            print('Checking if there are embedded storms...') 
        
        storm_modes = data['storm_modes']
        dbz_props = data['dbz_props']
        storm_embs = data['storm_embs']
        storm_parent_labels = data['storm_parent_labels']
        storm_depths = data['storm_depths']
        
        updated_storm_modes = storm_modes.copy()
        updated_storm_embs = storm_embs.copy() 
        updated_dbz_props = dbz_props.copy()
        updated_storm_parent_labels = storm_parent_labels.copy()
        
        for depth in range(max(storm_depths), -1, -1):
            if self.verbose: 
                print("depth = %d" % depth)
            remove_indices = []
            for n, mode in enumerate(storm_modes):
                if storm_depths[n] == depth: 
                    if storm_embs[n] in ["EMB-QLCS999"]:
                        parent_coords = dbz_props[n].coords
                        for nn, storm2_region in enumerate(dbz_props):
                            if round(storm2_region.area,5) < round(storm2_region.area,5):
                                if self.is_overlapping(storm2_region.coords, parent_coords):
                                    if nn not in remove_indices:
                                        remove_indices.append(nn)
                                        if self.verbose:
                                            print(
                                                    "Removing %s within %s within %s"
                                                    % (
                                                        storm_modes[nn],
                                                        mode,
                                                        storm_embs[n][4:],
                                                    )
                                                )

                    elif storm_embs[n] in ["EMB-CELL", "EMB-CLUS"]:
                        parent_label = storm_parent_labels[n]
                        const_indices = [n]
                        for nn in range(len(storm_modes)):
                            if dbz_props[nn].label == parent_label:
                                parent_index = nn
                                break

                        num_sups = 1 if storm_modes[n] == "ROTATE" else 0
                        
                        for nn in range(len(storm_modes)):
                            if nn != n and storm_parent_labels[nn] == parent_label:
                                const_indices.append(nn)
                                if storm_modes[nn] == "ROTATE":
                                    num_sups += 1

                        if len(const_indices) == 1:
                            if n not in remove_indices:
                                remove_indices.append(n)
                                if storm_embs[n] == "EMB-CLUS":
                                    updated_storm_modes[parent_index] = mode
                                if self.verbose:
                                    print(
                                            "Removing singleton %s from %s (which is now a %s)"
                                            % (
                                                mode,
                                                storm_modes[parent_index],
                                                mode,
                                            )
                                        )

                        elif storm_embs[
                            parent_index
                            ] == "NONEMB" and storm_modes[parent_index] not in [
                                "CLUSTER",
                                "SUP_CLUSTER",
                            ]:

                            if num_sups > 1:
                                if self.verbose:
                                    print(
                                            "Converting parent %s to SUP_CLUSTER"
                                            % updated_storm_modes[parent_index]
                                        )
                                updated_storm_modes[parent_index] = "SUP_CLUSTER"
                            else:
                                if self.verbose:
                                    print(
                                            "Converting parent %s to CLUSTER"
                                            % updated_storm_modes[parent_index]
                                        )
                                updated_storm_modes[parent_index] = "CLUSTER"
                            for ind in const_indices:
                                updated_storm_embs[ind] = "EMB-CLUS"

                        elif storm_embs[parent_index] != "NONEMB":
                            if parent_index not in remove_indices:
                                if self.verbose:
                                    print(
                                            "Removing parent %s since it is actually an embedded cluster"
                                            % storm_modes[parent_index]
                                        )
                                remove_indices.append(parent_index)
                                for ind in const_indices:
                                    updated_storm_embs[ind] = "EMB-CLUS"

            if len(remove_indices) != len(set(remove_indices)):
                raise ValueError("DUPLICATES in remove_indices!!! Exiting.")
                
        for ind in sorted(remove_indices, reverse=True):
            updated_storm_modes.pop(ind)
            updated_storm_embs.pop(ind)
            updated_dbz_props.pop(ind)
            updated_storm_parent_labels.pop(ind)

        # Update the results. 
        data['storm_modes'] = updated_storm_modes
        data['storm_embs'] = updated_storm_embs
        data['dbz_props'] = updated_dbz_props
        data['storm_parent_labels'] = updated_storm_parent_labels        
                
        return data      

    def separate_supercell_clusters(self, data):
        """ Separate Supercell Clusters """
        storm_modes = data['storm_modes']
        dbz_props = data['dbz_props']
        storm_embs = data['storm_embs']
        
        updated_storm_modes = storm_modes.copy()

        where_cluster = np.where(storm_embs == 'CLUSTER')[0]
        for n in where_cluster:
            n_supercells = 0 
            parent_coords = dbz_props[n].coords
            for nn, storm2_region in enumerate(dbz_props):
                if (storm_modes[nn] == "ROTATE" and 
                    round(dbz_props[nn].area,5) < round(dbz_props[n].area,5)):
                    if self.is_overlapping(storm2_region.coords, parent_coords):
                        n_supercells += 1
                        if n_supercells == 2:
                            updated_storm_modes[n] = "SUP_CLUSTER"
                            print("CONVERTING CLUSTER TO SUP_CLUSTER!!! Exiting.")
                            break
        
        # Update the results. 
        data['storm_modes'] = updated_storm_modes
        
        return data 

    def to_amorphous(self, data):
        """ Recast irregular discrete cell objects as AMORPHOUS """
        storm_modes = data['storm_modes']
        updated_storm_modes = data['storm_modes'].copy()
        
        for n,mode in enumerate(storm_modes):
            if (
                mode in ["ROTATE", "NONROT"]
                and data['storm_embs'][n] == "NONEMB"
                and (
                     data['dbz_props'][n].major_axis_length > 75 / 3.0
                     or data['dbz_props'][n].solidity < 0.7
                     or data['dbz_props'][n].eccentricity > 0.97
                    )
                ):
                updated_storm_modes[n] = "AMORPHOUS"
                if self.verbose:
                    print(
                            "Converting DISCRETE CELL to AMORPHOUS",
                            data['dbz_props'][n].area,
                            data['dbz_props'][n].major_axis_length,
                            data['dbz_props'][n].eccentricity,
                            data['dbz_props'][n].solidity,
                        )

        # Update the results.                
        data['storm_modes'] = updated_storm_modes 
        
        return data
    

    def object_hierarchy(self, data):
        """
        Parameters
        ----------
        
        Returns
        --------
        """
        storm_modes = data['storm_modes']
        dbz_props = data['dbz_props']
        
        storm_parent_labels = np.zeros(len(storm_modes), dtype=object)
        storm_embs = np.zeros(len(storm_modes), dtype=object)

        for n in range(len(storm_modes)):
            child_coords = dbz_props[n].coords
            min_potential_parent_area = 9999999999
            parent_found = False
            for nn in range(len(dbz_props)):
                potential_parent_area = dbz_props[nn].area
                if potential_parent_area > dbz_props[n].area:
                    potential_parent_coords = dbz_props[nn].coords
                    if self.is_overlapping(child_coords, potential_parent_coords):
                        parent_found = True
                        if potential_parent_area < min_potential_parent_area:
                            min_potential_parent_area = potential_parent_area
                            potential_parent_index = nn

            if parent_found:
                storm_parent_labels[n] = dbz_props[potential_parent_index].label
                if storm_modes[potential_parent_index] == "QLCS":
                    storm_embs[n] = "EMB-QLCS"
                elif storm_modes[potential_parent_index] in [
                    "CLUSTER",
                    "AMORPHOUS",
                    "SUP_CLUSTER",
                ]:
                    storm_embs[n] = "EMB-CLUS"
                else:
                    storm_embs[n] = "EMB-CELL"
            else:
                storm_embs[n] = "NONEMB"

            storm_labels = [props.label for props in dbz_props]
        
        # Find the storm depths 
        storm_depths = []
        for n in range(len(storm_parent_labels)):
            depth = 0
            nn = n
            while storm_parent_labels[nn] > 0:
                depth += 1
                parent_label = storm_parent_labels[nn]
                nn = storm_labels.index(parent_label)
            storm_depths.append(depth)

        data['storm_parent_labels'] =  list(storm_parent_labels)  
        data['storm_embs'] =  list(storm_embs)
        data['storm_depths'] =  storm_depths
            
        return data

   
