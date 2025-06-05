# -*- coding: utf-8 -*-
"""
Created on August 21 2024

@author: John Krause (JK)

@license: BSD-3-Clause

@copyright Copyright 2024 John Krause

Update History:
    Errors, concerns, proposed modifications, can be sent to:
        John.Krause@noaa.gov, please help us improve this code

    Version 0:
        -JK Initial version, pulled mcit_trim from hotspots dir
        and made it generic to work with np.array
            
"""

import numpy as np
from scipy import ndimage as ndi

#copied from:
# mcit_trim:
#   https://github.com/NOAA-National-Severe-Storms-Laboratory/hotspots
#
# Thanks to Vincent Klaus and John Krause
# Report bugs to John.Krause@noaa.gov or also to hotspots repository
#
def object_trim(labels_arr: np.ndarray,
              input_data: np.ndarray,
              min_strength: (int, float) = None,
              min_size: (int) = None) -> np.ndarray:

    """
    Process the objects data into new objects by combing or removing
    small objects and/or weak objects.

    Only works with positive max values.
    Lowest min values, like sattelite temperatues flip the sign of the input_data

    Parameters
    ----------
    objects: np.ndarray 
        labeled objects
    input_data: np.ndarray 
        data field on the same grid as objects
    min_strength: (int, float)
        Peak data  required for an object not to be removed or combined with its
        neighbor. Set to -1 to skip this process.
    min_size: int
        Objects smaller than min_size will be removed or combined with their
        neighbor. Set to -1 to skip this process.
    """

    if min_strength == None:
        min_strength = -1
    if min_size == None:
        min_size = -1

    ids = np.unique(labels_arr)[1:]
    max_data = ndi.maximum(input_data, labels=labels_arr, index=ids)
    obj_sizes = ndi.sum(labels_arr > 0, labels=labels_arr, index=ids)

    # dict with maximum VIL of all labelled cells
    id_maxdata = {test_id: max_data[i] for i, test_id in enumerate(ids)}
    # dict with size of all labelled cells
    id_size = {test_id: int(obj_sizes[i]) for i, test_id in enumerate(ids)}

    ids_to_remove = []
    # now remove weak MCIT cells unless they have stronger neighbors
    for target_id in id_maxdata.keys():
        if id_maxdata[target_id] < min_strength:
            # mark the id for removal later
            ids_to_remove.append(target_id)

            neighboring_ids, bordermask = get_neighboring_labels(
                labels_arr, target_id)

            # all neighbor labels except for 0, which is background
            neighboring_ids = neighboring_ids[~(neighboring_ids == 0)]

            if len(neighboring_ids) == 0:
                labels_arr = np.where(labels_arr == target_id, 0, labels_arr)
            else:
                neighbor_dict = {i: id_maxdata[i] for i in neighboring_ids}
                sorted_neighbors = sorted(
                    neighbor_dict.items(), key=lambda x: x[1])
                assigned_id = sorted_neighbors[-1][0]
                labels_arr = np.where(labels_arr == target_id, assigned_id,
                                      labels_arr)

    # now delete the removed labels from the dictionary
    for target_id in ids_to_remove:
        del id_maxdata[target_id]
        del id_size[target_id]

    ids_to_remove = []
    for target_id in id_size.keys():
        if id_size[target_id] < min_size:
            # mark the id for removal later
            ids_to_remove.append(target_id)

            neighboring_ids, bordermask = get_neighboring_labels(
                labels_arr, target_id)
            # all neighbor labels except for 0, which is background
            neighboring_ids = neighboring_ids[~(neighboring_ids == 0)]

            if len(neighboring_ids) == 0:
                labels_arr = np.where(labels_arr == target_id, 0, labels_arr)
            else:
                neighbor_dict = {i: id_maxdata[i] for i in neighboring_ids}
                sorted_neighbors = sorted(
                    neighbor_dict.items(), key=lambda x: x[1])
                assigned_id = sorted_neighbors[-1][0]
                labels_arr = np.where(labels_arr == target_id, assigned_id,
                                      labels_arr)

    # now delete the removed labels from the dictionary
    for target_id in ids_to_remove:
        del id_maxdata[target_id]
        del id_size[target_id]

    return labels_arr 


def get_neighboring_labels(labels_arr: np.ndarray,
                           target_id: int):
    structure = ndi.generate_binary_structure(2, 2)

    # Create a mask for the current region
    region_mask = labels_arr == target_id

    # Dilate the region to find neighbors
    border_mask = ndi.binary_dilation(
        region_mask, structure=structure) & (labels_arr != target_id)

    # Find neighboring labels
    neighboring_labels = np.unique(labels_arr[border_mask])

    #remove the 0 values
    neighboring_labels = neighboring_labels[~(neighboring_labels == 0)]

    return neighboring_labels, border_mask
