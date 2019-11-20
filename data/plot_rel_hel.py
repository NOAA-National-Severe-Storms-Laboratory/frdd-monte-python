import sys
sys.path.append('/home/monte.flora/wofs/plotting')
sys.path.append('/home/monte.flora/wofs/data')
sys.path.append('/home/monte.flora/wofs/util')
from Plot import Plotting 
import numpy as np
from loadWRFGrid import WRFData
from loadWRFGrid import calc_cold_pool, calc_divergence, calc_relative_helicity, calc_vorticity
from loadWRFGrid import calc_relative_streamwise_vorticity
from loadEnsembleData import EnsembleData
import config
from news_e_post_cbook import calc_height
from scipy.ndimage import gaussian_filter

date = '20180501'
time = '2300'
time_indexs = [12] 
hgt_1km = 8

wrf_data = WRFData( date = date, time =time, time_indexs = time_indexs, var_name_list = ['U', 'V', 'PH', 'PHB'] )
data = wrf_data._load_single_ens_mem( mem_idx=1 ) 

# Read in velocity components 
u = data['U'][0,:]
v = data['V'][0,:]
ph = data['PH'][0,:]
phb = data['PHB'][0,:]
z, dz = calc_height( ph, phb )
del data

dz = np.mean( np.mean( dz, axis = 2), axis = 1 ) 

var_env_list = ['bunk_r_u', 'bunk_r_v']
ens_data = EnsembleData( date_dir=date, time_dir=time, var_name_list=var_env_list, time_indexs = time_indexs, tag = 'ENV' )
ens_strm = EnsembleData( date_dir=date, time_dir=time, var_name_list=['comp_dz'], time_indexs = time_indexs, tag = 'ENS' )
data_strm = ens_strm.load_ensemble_data_single_time( )
data_env = ens_data.load_ensemble_data_single_time( )

cx = data_env[0,0]
cy = data_env[1,0]
uh_2to5 = gaussian_filter( data_strm[0,:,:], sigma = 1.0) 

cx = np.repeat( cx[np.newaxis,:,:], u.shape[0], axis = 0 )
cy = np.repeat( cy[np.newaxis,:,:], v.shape[0], axis = 0 )

u_storm_relative = u - cx
v_storm_relative = v - cy

vort_x, vort_y    = calc_vorticity( u=u, v=v, dz=dz ) 
relative_helicity = calc_relative_helicity(  u=u_storm_relative, v=v_storm_relative, vort_x=vort_x, vort_y=vort_y)

print(np.amin( relative_helicity), np.amax( relative_helicity )) 
rel_streamwise_vorticity = calc_relative_streamwise_vorticity( u=u_storm_relative, v=v_storm_relative, vort_x=vort_x, vort_y=vort_y)

#print np.amax( rel_streamwise_vorticity - relative_helicity )

kwargs = {'cblabel': 'Storm-Relative Relative Helicity ( 1 km AGL )', 'alpha':0.5, 'extend': 'neither', 'cmap': 'diverge' }
plt = Plotting( date='20180501', z1_levels=np.arange(-1, 1.25, 0.25), z2_levels = [35.], **kwargs )
fig, axes, map_axes, x, y = plt._create_fig( fig_num = 0, plot_map = True, figsize = (10, 9))

#z1 = rel_streamwise_vorticity[hgt_1km,:,:] 
z1 = relative_helicity[hgt_1km,:,:]

plt.spatial_plotting(fig, axes, map_axes[0], x, y, z1=z1, z2 = uh_2to5, plot_colorbar = True,
        quiver_u_1=u_storm_relative[hgt_1km,:,:],
        quiver_v_1=v_storm_relative[hgt_1km,:,:],
        quiver_u_2=vort_x[hgt_1km,:,:],
        quiver_v_2=vort_y[hgt_1km,:,:])

#plt._save_fig( fig, fname = 'relative_streamwise_vorticity_%s_%s.png' % (date, time) ) 
plt._save_fig( fig, fname = 'relative_helicity_%s_%s.png' % (date, time) )




