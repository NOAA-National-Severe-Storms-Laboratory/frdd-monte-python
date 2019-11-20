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
            date_processes = [[ mp.Process(target=func, args=(str(date), time, kwargs)) for time in datetimes[date]] for date in dates]
            date_processes = list( itertools.chain.from_iterable(date_processes))
            for p in date_processes:
                p.start()        
            activity_list = [True]
            # Once all processes are dead, move on to the next chunk of dates 
            while all( i == False for i in activity_list ) == False:
                activity_list = [ p.is_alive() for p in date_processes ]
                time_module.sleep( 1 )



