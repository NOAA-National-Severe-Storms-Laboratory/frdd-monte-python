from wofs.util import config
from wofs.util.MultiProcessing import multiprocessing_in_chunks
import numpy as np
from datetime import datetime
from IdentifyProbabilityObjects import worker

""" usage: stdbuf -oL python -u run_IdentifyProbabilityObjects.py 2 > & log & """
###########################################################################
# Label forecast storm objects valid at a single time
###########################################################################
debug = True
duration = 6

# /work/mflora/ML_DATA/INPUT_DATA/20190509/PROBABILITY_OBJECTS_20190509-0030_24.nc
# 20190626/1700
if debug:
    print("\n Working in DEBUG MODE...\n")
    date = "20180501"
    time = "2330"
    fcst_time_idx = 0
    kwargs = {
        "time_indexs": np.arange(duration + 1) + fcst_time_idx,
        "fcst_time_idx": fcst_time_idx,
        "debug": debug,
    }
    worker(date, time, kwargs)

else:
    datetimes = config.datetimes_ml
    print("Total number of dates: {}".format(len(datetimes)))
    for fcst_time_idx in range(0, 24 + 1):
        print("\n Start Time:", datetime.now().time())
        print("Forecast Time Index: ", fcst_time_idx)
        kwargs = {
            "time_indexs": np.arange(duration + 1) + fcst_time_idx,
            "fcst_time_idx": fcst_time_idx,
        }
        multiprocessing_in_chunks(
            datetimes=datetimes, n_date_per_chunk=6, worker=worker, kwargs=kwargs
        )
        print("End Time: ", datetime.now().time())
