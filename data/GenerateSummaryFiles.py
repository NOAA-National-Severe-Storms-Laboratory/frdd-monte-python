import numpy as np
import xarray as xr
import os 
from os.path import join, exists 
from loadWRFGrid import WRFData 
from loadWRFGrid import calc_cold_pool, calc_divergence, calc_relative_helicity, integrate_over_a_layer, calc_vorticity
from loadWRFGrid import calc_lowlevel_bulk_wind_shear, find_z_index, calc_lapse_rate
from loadEnsembleData import EnsembleData
import sys
sys.path.append('/home/monte.flora/wofs/util')
import config
from news_e_post_cbook import calc_height, calc_t
from MultiProcessing import multiprocessing_per_date
from datetime import datetime

usage = ' stdbuf -oL python -u loadAdditionalVars.py 2 > & log & '
debug = True

var_env_list = [ 'u_10', 'v_10', 'th_e_ml', 'bunk_r_u', 'bunk_r_v', 'u_500', 'v_500']
variables = ['U', 'V', 'W', 'PH', 'PHB'] 

def function_for_multiprocessing( date, time, kwargs ):   
    '''
    Functions used for multiprocessing.
    '''
    out_path = join(join( config.WOFS_DATA_PATH, str(date) ), time)
    fname = join( out_path, 'wofs_data_%s_%s_%s.nc' % ( str(date), time, kwargs['time_indexs'][0] ))
    #if exists(fname):
    #    print ('{} already exists!'.format(fname))
    #    return None 
    #print date, time
    wrf_data = WRFData( date=str(date), time=time, time_indexs=kwargs['time_indexs'], variables = ['PH', 'PHB', 'HGT'] )
    data = wrf_data._load_single_ens_mem( mem_idx=1) #dict:( NT, NZ, NY, NX )
    if data is None:
        return None
    ph = data['PH'][0,:] 
    phb = data['PHB'][0,:] 
    hgt = data['HGT'][0,:]
    z, dz = calc_height( ph, phb ) 
    avg_dz = np.mean( np.mean( dz, axis = 2), axis = 1 )
    z_1km = find_z_index( ph, phb, hgt, height_in_m=1000.)
    z_3km = find_z_index( ph, phb, hgt, height_in_m=3000.)
    z_5km = find_z_index( ph, phb, hgt, height_in_m=5000.)
    del data
   
    ens_data = EnsembleData( date_dir=str(date), time_dir=time, base_path ='summary_files' )
    data_env = ens_data.load( variables=var_env_list, time_indexs = kwargs['time_indexs'], tag = 'ENV' ) # shape = (nv, ne, ny, nx)
   
    # Initialize all the variables to be calculated 
    div = np.zeros(( data_env.shape[2:] ))
    perturb_theta_e = np.zeros(( data_env.shape[2:] ))
    relative_helicity = np.zeros(( data_env.shape[2:] ))
    bulk_shear = np.zeros(( data_env.shape[2:] ))
    w_down = np.zeros(( data_env.shape[2:] ))
    w_1km = np.zeros(( data_env.shape[2:] ))
    temp_1km = np.zeros(( data_env.shape[2:] ))
    temp_3km = np.zeros(( data_env.shape[2:] ))
    geopotential_5km = np.zeros(( data_env.shape[2:] ))
    geopotential_1km =  np.zeros(( data_env.shape[2:] ))
    mid_level_lapse_rate = np.zeros(( data_env.shape[2:] ))
    low_level_lapse_rate = np.zeros(( data_env.shape[2:] ))

    wrf_data = WRFData( date = str(date), time =time, time_indexs = kwargs['time_indexs'], variables = ['U', 'V', 'W', 'T', 'P', 'PB', 'PH', 'PHB'] )
    for mem in range(config.N_ENS_MEM):
        data = wrf_data._load_single_ens_mem( mem_idx = mem+1)
        u = data['U'][0,:]
        v = data['V'][0,:]
        w = data['W'][0,:]
        geopotential_height = data['PH'][0,:] #+ data['PHB'][0,:]
        geopotential_5km[mem,:,:] = geopotential_height[z_5km,:,:]
        geopotential_1km[mem,:,:] = geopotential_height[z_1km,:,:]
        temp = calc_t( th = data['T'][0,:]+300., p=(data['P'][0,:]+data['PB'][0,:])/ 100.)
        temp_1km[mem,:,:] = temp[z_1km,:,:]
        temp_3km[mem,:,:] = temp[z_3km,:,:]
        mid_level_lapse_rate[mem,:,:] = calc_lapse_rate( temp, z_top=z_5km, z_bottom=z_3km, dz=2 )
        low_level_lapse_rate[mem,:,:] = calc_lapse_rate( temp, z_top=z_3km, z_bottom=0, dz=2 )
        # Calculate the 1-km AGL updraft 
        w_1km[mem,:,:]  = w[8,:,:]     
        # Calculate the column-min vertical velocity ( max downdraft )
        w_down[mem,:,:] = np.amin( w, axis = 0 )
        div[mem,:,:] = calc_divergence( u = data_env[0,0,mem], v = data_env[0,1,mem] )
        perturb_theta_e[mem,:,:] = calc_cold_pool( theta_e = data_env[0,2,mem] )
        cx = data_env[0,3,mem]
        cy = data_env[0,4,mem]
        cx = np.repeat( cx[np.newaxis,:,:], u.shape[0], axis = 0 )
        cy = np.repeat( cy[np.newaxis,:,:], v.shape[0], axis = 0 )
        u_storm_relative = u - cx
        v_storm_relative = v - cy 
        vort_x, vort_y = calc_vorticity( u=u, v=v, dz=avg_dz )
        rel_hel = calc_relative_helicity(  u=u_storm_relative, v=v_storm_relative, vort_x=vort_x, vort_y=vort_y)
        relative_helicity[mem,:,:] = integrate_over_a_layer( f=rel_hel, z=z, dz=dz, lower=0., upper=1000. )
        bulk_shear[mem,:,:] = calc_lowlevel_bulk_wind_shear( u_10=data_env[0,0,mem], v_10=data_env[0,1,mem], u_500=data_env[0,5,mem], v_500=data_env[0,6,mem] )
        del data 

    data = { }
    data['w_down'] = (['Ens. Mem.', 'Y', 'X'], w_down)
    data['w_1km'] = (['Ens. Mem.', 'Y', 'X'], w_1km)
    data['div_10m'] = (['Ens. Mem.', 'Y', 'X'], div)
    data['buoyancy'] = (['Ens. Mem.', 'Y', 'X'], perturb_theta_e)
    data['rel_helicity_0to1'] = (['Ens. Mem.', 'Y', 'X'], relative_helicity)
    data['10-500 m bulk shear'] = (['Ens. Mem.', 'Y', 'X'], bulk_shear)
    data['Mid-level Lapse Rate'] = (['Ens. Mem.', 'Y', 'X'], mid_level_lapse_rate)
    data['Low-level Lapse Rate'] = (['Ens. Mem.', 'Y', 'X'], low_level_lapse_rate)
    data['Temp (1km)'] = (['Ens. Mem.', 'Y', 'X'], temp_1km)
    data['Temp (3km)'] = (['Ens. Mem.', 'Y', 'X'], temp_3km)
    data['Geopotential (1km)'] = (['Ens. Mem.', 'Y', 'X'], geopotential_1km)
    data['Geopotential (5km)'] = (['Ens. Mem.', 'Y', 'X'], geopotential_5km)

    del w_down, w_1km, div, perturb_theta_e, relative_helicity
    del bulk_shear, mid_level_lapse_rate, low_level_lapse_rate
    del temp_1km, temp_3km, geopotential_1km, geopotential_5km

    ds = xr.Dataset( data )
    comp = dict(zlib=True, complevel=5)
    encoding = {var: comp for var in ds.data_vars}
    if debug:
        ds.to_netcdf( 'wofs_data_%s_%s_%s.nc' % ( str(date), time, kwargs['time_indexs'][0]))
    else:
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        ds.to_netcdf( fname, encoding=encoding )
        ds.close( ) 
        del data

if debug:
    kwargs = {'time_indexs':[10]}
    function_for_multiprocessing( date='20180501', time='2300', kwargs=kwargs )
else:
    # Cycle through the time index
    datetimes = config.datetimes_ml
    fcst_time_idx_set = list(range(37))
    for t in fcst_time_idx_set:
        print('Start Time:', datetime.now().time())
        print("Forecast Time Index: ", t) 
        kwargs = {'time_indexs':[t]}
        multiprocessing_per_date( datetimes=datetimes, n_date_per_chunk=8, func=function_for_multiprocessing, kwargs=kwargs)
        print('End Time: ', datetime.now().time())


