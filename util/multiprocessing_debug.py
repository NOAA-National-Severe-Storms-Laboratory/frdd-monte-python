from datetime import datetime as timing
import time as time_module
import multiprocessing as mp
from basic_functions import chunks
import itertools

def multiprocessing_per_date( datetimes, n_date_per_chunk, func, kwargs, verbose=True):
    chunks_of_dates = list(chunks( list(datetimes.keys()), n_date_per_chunk))
    if verbose:
        print("Total number of dates: ", len(list(datetimes.keys( ))))
        print(chunks_of_dates[0])
    for i, dates in enumerate(chunks_of_dates):
        if verbose:
            print(("Starting on set %s out of %s at %s" % ( i+1, len(chunks_of_dates), timing.now() ) ))
        if i+1 == 9:
            print(dates) 
            

