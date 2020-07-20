from wofs.util import config
from os.path import exists, join 

def _determine_base_path( date):
    '''
    Find the correct base directory path; based on the date.
    '''
    if exists( join( '/work3/JTTI/HMT_FFaIR/FCST', date)):
        return join( '/work3/JTTI/HMT_FFaIR/FCST', date)
    else:
        return join( join( '/work3/wof/realtime/FCST', date[:4]), date)    


datetimes = config.datetimes_ml
for date in datetimes.keys():
    for time in datetimes[date]:
        #print( date, _determine_base_path( str(date) ), exists(_determine_base_path(str(date))) )
        base_path = join(_determine_base_path( str(date) ), time)
        if not exists(base_path):
            base_path = join( join( _determine_base_path(str(date)), 'RLT'), time)
        if not exists(base_path):
            print ( date, time, exists(base_path)) 
    



