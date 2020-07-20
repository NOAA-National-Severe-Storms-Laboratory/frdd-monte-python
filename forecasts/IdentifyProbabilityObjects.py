import numpy as np
import xarray as xr
from os.path import join, exists
import os
import pandas as pd
import cv2

from wofs.processing.ObjectIdentification import label_regions, QualityControl
from wofs.util.basic_functions import (
    get_key,
    personal_datetime,
    check_file_path,
    to_ordered_dict,
)
from wofs.data.loadEnsembleData import EnsembleData, calc_time_max
from wofs.processing.ObjectMatching import ObjectMatching, match_to_lsrs
from wofs.data.loadMRMSData import MRMSData
from wofs.data.loadWWAs import (
    load_reports,
    load_tornado_warning,
    load_svr_warning,
    is_severe,
)
from wofs.util.StoringObjectProperties import (
    get_object_properties,
    save_object_properties,
)
from wofs.data.loadMRMSData import MRMSData
from wofs.data.loadEnsembleData import EnsembleData, calc_time_max
from wofs.util import config
from scipy.ndimage import maximum_filter, gaussian_filter
from datetime import datetime


from wofs.processing.ObjectIdentification import label_ensemble_objects
from skimage.measure import regionprops

from wofs.processing.ObjectMatching import ObjectMatching, match_to_lsrs
from wofs.main.forecasts.IdentifyForecastTracks import _load_verification

get_time = personal_datetime()

qc = QualityControl()
###########################################################################
# Label forecast storm objects valid at a single time
###########################################################################
variable_key = "updraft"
###########################################
QC_PARAMS = config.VARIABLE_ATTRIBUTES[variable_key]["qc_params"]
QC_PARAMS = to_ordered_dict(QC_PARAMS)


def _load_forecast_data(date_and_time_dict, **kwargs):
    """
    """
    instance = EnsembleData(
        date_dir=date_and_time_dict["date"],
        time_dir=date_and_time_dict["time"],
        base_path="summary_files",
    )
    input_data = instance.load(
        variables=[config.VARIABLE_ATTRIBUTES[variable_key]["var_newse"]],
        time_indexs=kwargs["time_indexs"],
        tag='ENS')
        #get_key(
        #    config.newse_tag, config.VARIABLE_ATTRIBUTES[variable_key]["var_newse"]
        #),
    if input_data is None:
        return None, None

    ensemble_data_over_time = input_data[
        config.VARIABLE_ATTRIBUTES[variable_key]["var_newse"]
    ].values
    ens_data, ens_data_indices = calc_time_max(
        input_data=ensemble_data_over_time, argmax=True
    )

    return ens_data, ens_data_indices


def calc_ensemble_probs(date_and_time_dict, kwargs):
    """
    Identify updraft tracks 
    """
    ens_data, ens_data_indices = _load_forecast_data(date_and_time_dict, **kwargs)
    if ens_data is None:
        return None

    ens_object_labels = np.zeros((ens_data.shape))
    for mem in range(np.shape(ens_data)[0]):
        fcst_object_labels, fcst_object_props = label_regions(
            input_data=ens_data[mem, :, :],
            method="single_threshold",
            params={"bdry_thresh": 10.0},
        )
        QC_PARAMS["min_time"].append(ens_data_indices[mem, :, :])
        qc_fcst_object_labels, _ = qc.quality_control(
            input_data=ens_data[mem, :, :],
            object_labels=fcst_object_labels,
            object_properties=fcst_object_props,
            qc_params=QC_PARAMS,
        )
        ens_object_labels[mem, :, :] = qc_fcst_object_labels

    objects = np.where(ens_object_labels > 0, 1, 0)
    ensemble_probabilities = np.average(objects, axis=0)
    del ens_data, ens_data_indices, objects
    return ensemble_probabilities, ens_object_labels


def worker(date, time, kwargs):
    """
    thread worker function
    Functions bulit for multiprocessing
    """
    date_and_time_dict = {
        "date": date,
        "time": time,
        "fcst_time_idx": kwargs["fcst_time_idx"],
    }
    fcst_time_idx = kwargs["fcst_time_idx"]
    # print (f"Starting on {date}-{time}...")
    fname = join(
        config.OBJECT_SAVE_PATH,
        str(date),
        f"updraft_ensemble_objects_{date}-{time}_t:{fcst_time_idx}.nc",
    )

    # Load the individual objects from the ensemble members
    ensemble_probabilities, ens_object_labels = calc_ensemble_probs(date_and_time_dict, kwargs)
    if ensemble_probabilities is None:
        print(f"{date}-{time} is empty, returning None")
        return None
    full_objects, full_object_props = label_ensemble_objects(
        ensemble_probabilities=ensemble_probabilities
    )

    data = {}
    data["2D Probabilities"] = (["Y", "X"], ensemble_probabilities)
    data["Probability Objects"] = (["Y", "X"], full_objects)
    data["Storm Tracks"] = (['NE', 'Y', 'X'], ens_object_labels)

    (
        valid_date_and_time,
        initial_date_and_time,
    ) = get_time.determine_forecast_valid_datetime(
        date_dir=str(date),
        time_dir=time,
        fcst_time_idx=date_and_time_dict["fcst_time_idx"],
    )

    verification_dict = _load_verification(
        date, valid_date_and_time, initial_date_and_time, **kwargs
    )
    matched_at_15km = {
        "matched_to_{}_15km".format(atype): match_to_lsrs(
            object_properties=full_object_props,
            lsr_points=verification_dict[atype],
            dist_to_lsr=5,
        )
        for atype in verification_dict.keys()
    }
    matched_at_0km = {
        "matched_to_{}_0km".format(atype): match_to_lsrs(
            object_properties=full_object_props,
            lsr_points=verification_dict[atype],
            dist_to_lsr=1,
        )
        for atype in verification_dict.keys()
    }

    matched_at_15km = is_severe(matched_at_15km, "15km")
    matched_at_0km = is_severe(matched_at_0km, "0km")
    all_matched = {**matched_at_15km, **matched_at_0km}

    df = get_object_properties(object_props=full_object_props, matched=all_matched)
    for object_property in df.columns:
        data[object_property] = (["Object"], df[object_property].values)

    ds = xr.Dataset(data)
    comp = dict(zlib=True, complevel=5)
    encoding = {var: comp for var in ds.data_vars}

    os.makedirs(os.path.dirname(fname), exist_ok=True)
    # print(f"Saving {fname}")
    ds.to_netcdf(path=fname, encoding=encoding)
    ds.close()
    del ds, data, full_objects
