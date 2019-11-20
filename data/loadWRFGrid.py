import numpy as np 
from os.path import exists, join, isdir
from glob import glob
import netCDF4 as nc
from scipy.ndimage import gaussian_filter, convolve

import sys 
sys.path.append('/home/monte.flora/wofs/util')
import config
from news_e_post_cbook import calc_height, calc_t

class WRFData:
    '''
    '''
    def __init__(self, date, time, time_indexs=None, variables=None):
        self.date = date
        self.time = time
        self.time_indexs = time_indexs
        self.variables = variables
        self.base_path = join(self._determine_base_path(date), time) 
        print ( self.base_path, isdir(self.base_path) )
        if not exists(self.base_path):
            self.base_path = join( join( self._determine_base_path(date), 'RLT'), time)

    def _determine_base_path(self, date):
        '''
        Find the correct base directory path; based on the date.
        '''
        if isdir( join( '/work3/JTTI/HMT_FFaIR/FCST', date)):
            return join( '/work3/JTTI/HMT_FFaIR/FCST', date)
        else:
            return join( join( '/work3/wof/realtime/FCST', date[:4]), date)

    def _generate_filename_list( self, mem_idx ):
        '''
        Gets a list of wrf filenames for a particular ensemble member.
        '''
        wrf_files = [ ]
        in_path = join( self.base_path, 'ENS_MEM_%s' % (mem_idx))
        # wrfout_d01_2018-05-02_00:00:00
        all_wrf_files = list(sorted( glob(join(in_path, 'wrfout_d01*')))) 
        if len(all_wrf_files) == 0: 
            print(in_path) 
            print((self.date, self.time, "No wrfout files for this ensemble member!!"))
            print("Likely an issue with the base path.") 
        else: 
            for t in self.time_indexs:
                wrf_files.append( nc.Dataset( all_wrf_files[t], 'r') )  

        return wrf_files
    
    def _load_single_ens_mem( self, mem_idx ):
        '''
        Load raw WRFOUT files for one ensemble member.
        Returns: data dictionary, 
                    variables are the keys
                    shape = ( NT, NZ, NY, NX ) 
        '''
        data = {key: [ ] for key in self.variables} 

        wrf_files_at_mem = self._generate_filename_list( mem_idx )
        if wrf_files_at_mem == [ ]:
            return None 
        
        for wrf_file in wrf_files_at_mem:
            for var in self.variables:
                if var == 'U':
                    # Average of x-dim (nt, nz, ny, nx)
                    data[var].append( 0.5* (wrf_file.variables[var][0,:,:,1:]+wrf_file.variables[var][0,:,:,:-1]) )
                    #temp.append( 0.5* (wrf_file.variables[var][0,:,:,1:]+wrf_file.variables[var][0,:,:,:-1]) )
                elif var == 'V':
                    # Average over y-dim
                    data[var].append( 0.5* (wrf_file.variables[var][0,:,1:,:]+wrf_file.variables[var][0,:,:-1,:]))
                elif var in 'W':
                    # Average over z-dim
                    data[var].append( 0.5* (wrf_file.variables[var][0,1:,:,:]+wrf_file.variables[var][0,:-1,:,:]) )
                else:
                    data[var].append( wrf_file.variables[var][0,:] ) 
            wrf_file.close()
            del wrf_file    

        # Convert lists to numpy arrays
        for key in list(data.keys( )):
            data[key] = np.array( data[key] )

        return data 

    def _load_data( self, verbose=True):
        '''
        Load the data.

        Returns: nested dictionary 
                 ensemble members are the first layer keys, 
                 variables are the next layer keys, (NT, NZ, NY, NX ) 
        '''
        data = {key: [ ] for key in np.arange(config.N_ENS_MEM)+1}
        for mem in range(self.N_ENS_MEM):
            print(mem) 
            data[mem+1] = self._load_single_ens_mem( mem_idx = mem+1 )
        return data 

def integrate_over_a_layer( f, z, dz, lower, upper ): 
    '''
    Integrate some function f in between layer lower - upper km AGL.
    '''
    f[np.where(z < lower)] = 0.
    f[np.where(z > upper)] = 0. 
    
    integrated_f = np.trapz(f, dx = dz[:-1], axis=0)

    return integrated_f

def calc_cold_pool( theta_e, g = 9.18 ): 
    '''
    Calculate buoyancy from the perturbation equivalent potential temperature.
    '''
    avg_theta_e = np.mean( theta_e )
    perturb_theta_e = theta_e - avg_theta_e

    B = (g/avg_theta_e)*perturb_theta_e

    return B 

def calc_divergence( u, v, dx = 3000., dy = 3000. ):
    ''' 
    Calculate horizontal divergence. 
    Formula: du/dx + dv/dy
    param: u, x-comp of velocity, shape = (ny,nx)
    param: v, y-comp of velocity, shape = (ny,nx)

    Returns: Divergence, shape (ny,nx)
    '''
    du_dx = np.gradient( u, dx, edge_order = 2, axis = 1 )
    dv_dy = np.gradient( v, dy, edge_order = 2, axis = 0 ) 

    return du_dx + dv_dy 

def calc_lowlevel_bulk_wind_shear( u_10, v_10, u_500, v_500):
    '''
    Calculate 10-500 m bulk wind shear
    '''
    diff_u = u_500 - u_10 
    diff_v = v_500 - v_10 

    bulk_shear = np.sqrt( diff_u**2 + diff_v**2 ) 

    return bulk_shear 

def find_z_index( ph, phb, hgt, height_in_m):
    '''
    Find z index for a given height in meters.
    '''
    z, dz = calc_height(ph, phb)
    terrian_relative_hght = z - hgt 

    avg_hght = np.mean( np.mean( terrian_relative_hght, axis = 2), axis = 1 ) 

    z_index = np.abs( avg_hght - height_in_m ).argmin( ) 

    #print avg_hght[z_index]

    return z_index 

def central_difference( f, dx,dz ):

    dz = np.mean( np.mean( dz, axis = 0), axis = 0 )
    dfdx = np.zeros(( f.shape ))
    for i in range( 1, f.shape[0]-1):
        #dfdx[:,i,:] = (f[:,i+1,:] - f[:,i-1,:]) / ( 2*3000.)
        dfdx[i,:,:] = (f[i+1,:,:] - f[i-1,:,:]) / (dz[i]+dz[i+1])

    #dfdz =[1:-1, :, :] = (f[2:,:,:]-f[:-2,:,:]) /(2*dz)

    dfdx[0,:,:] = (f[1,:,:] - f[0,:,:]) / dz[0]
    dfdx[-1,:,:] = (f[-1,:,:] - f[-2,:,:]) / dz[-1]

    return dfdx 

def angle_btw_vectors( v1, v2 ):
    '''
    Calculate angle between two vectors.
    '''
    #angle = np.arccos( np.dot(v1, v2) 
    a = 0 

def calc_vector_magnitude( x, y, z=None):
    '''
    Calculate vector magnitude.
    '''
    if z is None:
        return np.sqrt( x**2 + y**2 )
    else:
        return np.sqrt( x**2 + y**2 + z**2) 

def calc_lapse_rate( temp, z_top, z_bottom, dz ):
    '''
    Calculate temperature lapse rate.
    '''
    dT = temp[z_top,:,:] - temp[z_bottom,:,:]
    return -1.0* ( dT / dz )

def calc_helicity ( u, v, vort_x, vort_y, w=None, vort_z=None ): 
    '''
    Calculate helicity 
    '''
    if (w is None and vort_z is None):
        return u*vort_x + v*vort_y
    else:
        return u*vort_x + v*vort_y + w*vort_z

def calc_streamwise_vort( u,v,vort_x,vort_y, w=None, vort_z=None ):
    '''
    Calculate stream-wise vorticity 
    '''
    helicity = calc_helicity ( u=u, v=v, vort_x=vort_x, vort_y=vort_y, w=w, vort_z=vort_z )
    velocity_vector_norm = calc_vector_magnitude( x=u, y=v, z=w)

    return helicity / velocity_vector_norm 

def calc_vorticity( u, v, dz, w=None, dy=3000., dx=3000. ):
    '''
    Calculate the three components of vorticity.
    (Works fairly well!, values slightly lower, probably from calculating from averaged velocity field)
    '''
    dvdz, dvdy, dvdx = np.gradient( v, dz, dy, dx )
    dudz, dudy, dudx = np.gradient( u, dz, dy, dx )

    if w is None:
        vort_x = -1.*dvdz
        vort_y = dudz
        return vort_x, vort_y 
    else:
        dwdz, dwdy, dwdx = np.gradient( w, dz, dy, dx )
        vort_x = dwdy - dvdz
        vort_y = dudz - dwdx
        vort_z = dvdx - dudy
        return vort_x, vort_y, vort_z

def calc_relative_streamwise_vorticity( u, v, vort_x, vort_y, w=None, vort_z=None):
    '''
    Calculate the (storm-relative) relative streamwise vorticity.
    '''
    streamwise_vort = calc_streamwise_vort( u=u, v=v, vort_x=vort_x, vort_y=vort_y, w=w, vort_z=vort_z )
    vorticity_vector_norm = calc_vector_magnitude( x=vort_x, y=vort_y, z=vort_z )

    return streamwise_vort / vorticity_vector_norm

def calc_relative_helicity( u, v, vort_x, vort_y, w =None, vort_z=None, dx = 3000., dy = 3000., integrate=False ):
    '''
    Calculate the relative helicity according to Lilly (1986b).
    Can be either storm-relative or not. 
    '''
    vorticity_magnitude = calc_vector_magnitude( x=vort_x, y=vort_y, z=vort_z )
    velocity_magnitude = calc_vector_magnitude( x=u, y=v, z=w )
    helicity = calc_helicity( u=u, v=v, w=w, vort_x=vort_x, vort_y=vort_y, vort_z=vort_z )

    # Correct for no vorticity 
    vorticity_magnitude[np.where(vorticity_magnitude == 0.)] = 1e-8
    
    relative_helicity = helicity / (vorticity_magnitude * velocity_magnitude)

    return relative_helicity 

