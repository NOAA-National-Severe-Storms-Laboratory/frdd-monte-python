from datetime import datetime as timing
import time as time_module
import multiprocessing as mp
from basic_functions import chunks 
import itertools

def multiprocessing_per_date( datetimes, n_date_per_chunk, func, kwargs, verbose=True):
    #
    #Performs parallelization for a function over half of the processors.
    #
    if verbose:
        print("Total number of dates: ", len(list(datetimes.keys( ))))
    pool = mp.Pool(int(mp.cpu_count()/2))
    for date in datetimes.keys():
        for time in datetimes[date]:
            pool.apply_async(func, args=(str(date), time, kwargs) )
    pool.close()
    pool.join( )

'''
def multiprocessing_per_date( datetimes, n_date_per_chunk, func, kwargs, verbose=True):
    chunks_of_dates = list(chunks( list(datetimes.keys()), n_date_per_chunk))
    if verbose:
        print("Total number of dates: ", len(list(datetimes.keys( ))))
        print(chunks_of_dates[0])
    for i, dates in enumerate(chunks_of_dates):
        if verbose:
            print(("Starting on set %s out of %s at %s" % ( i+1, len(chunks_of_dates), timing.now() ) ))
        date_processes = [[ mp.Process(target=func, args=(str(date), time, kwargs)) for time in datetimes[date]] for date in dates]
        date_processes = list( itertools.chain.from_iterable(date_processes))
        for p in date_processes:
            p.start()        
        activity_list = [True]
        # Once all processes are dead, move on to the next chunk of dates 
        while all( i == False for i in activity_list ) == False:
            activity_list = [ p.is_alive() for p in date_processes ]
            time_module.sleep( 1 )
'''

def multiprocessing_func( processes_list, n_processes_per_chunk, func, verbose = True ):
    chunks_of_processes = list( chunks( processes_list, n_processes_per_chunk ) )
    if verbose:
        print("Total number of processes: ", len(processes_list))
    for i, a_chunk in enumerate( chunks_of_processes ):
        if verbose:
            print(("Starting on set %s out of %s at %s" % ( i+1, len(chunks_of_processes), timing.now() ) ))
        processes = [ mp.Process(target=func, args=( storm_file, )) for storm_file in a_chunk ]
        for p in processes:
            p.start( )
        activity_list = [True]
        # Once all processes are dead, move on to the next chunk of dates 
        while all( i == False for i in activity_list ) == False:
            activity_list = [ p.is_alive() for p in processes ]
            time_module.sleep( 1 )
    

