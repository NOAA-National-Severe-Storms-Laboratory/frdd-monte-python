import datetime 
import numpy as np 
import os, subprocess 
import netCDF4 as nc 
import pickle
from scipy import ndimage 
from scipy.ndimage import generic_filter

import sys 
sys.path.append('/home/monte.flora/NEWSeProbs/observed_rotation_tracks' )
import config

def remove_items_from_list(  llist , criteria ): 
   """ Removes strings from a list if they do not contain the criteria"""
   reduced_list = [ x for x in alist if criteria not in x ] 
   return reduced_list 

def find_in_list_of_list(mylist, char):
    for sub_list in mylist:
        if char in sub_list:
            return (mylist.index(sub_list), sub_list.index(char))
    raise ValueError("'{char}' is not in list".format(char = char))

def load_or_save_verify( load_or_save, var_newse, fcst_time_idx, matching_dist, prob_bdry, percent_value, 
         min_prob_area, hits_option, obs_label_option, path = config.verification_dir, plot_type=None, verify_data=None ):
   filename = "AVerificationData_%s_bdry=%s_percent=%s_min_prob_area=%s_matching_dist=%skm_time_idx=%s_hits=%s_obs_label_option=%s" % ( var_newse, round(prob_bdry, 3), percent_value, min_prob_area, str(3*matching_dist), fcst_time_idx, hits_option, obs_label_option )
   if load_or_save == 'save':
          with open( path + filename + '.pkl', 'wb') as f:
              pickle.dump(verify_data, f, pickle.HIGHEST_PROTOCOL)
   elif load_or_save == 'load' : 
      with open( path + filename + '.pkl', 'rb') as f:
              verify_data = pickle.load(f)
      
      return verify_data[plot_type] 

def load_or_save_ml_verify( load_or_save_option, option, fold, fcst_time_idx, matching_dist, num_conv_blocks=None, 
        num_dense_layers=None, num_conv_layers_in_a_block=None, first_num_filters=None,
        use_batch_normalization=None, kernel_size=None,  dropout_fraction=None, l1_weight=None, l2_weight=None,
        activation_function_name=None , pooling_type=None , dim_option=None , conv_type=None, 
        patience=None , min_delta=None, plot_type = None, verify_data = None  ):

    if option == 'machine_learning': 
        fcst_filename = 'VERIFYDATA_%s_DIM=%s_fcst_time_idx=%02d_MATCHDIST=%s_NCONVBLOCKS=%s_NCONV_PER_BLOCK=%s_NDENSE=%s_1STFILTER=%s_BATCHNORM=%s_KERNEL=%s_DO=%s_L1=%s_L2=%s_ACT=%s_POOL=%s_CONVTYPE=%s_EARLYSTOP=%s_%s_fold=%s.pkl' % (
                                 option, dim_option,  fcst_time_idx,  matching_dist, num_conv_blocks,  num_conv_layers_in_a_block,  num_dense_layers,  first_num_filters,  use_batch_normalization,  kernel_size,
                                 dropout_fraction,  l1_weight,  l2_weight,  activation_function_name,  pooling_type,  conv_type,  patience,  min_delta,  fold )
    elif option == 'baseline': 
        fcst_filename = 'VERIFYDATA_%s_fcst_time_idx_%02d_MATCHDIST_%s_fold=%s.pkl' % (option, fcst_time_idx, matching_dist, fold) 
        
    if load_or_save_option == 'save': 
        with open( os.path.join( config.results_dir_dl, fcst_filename) , 'wb') as f: 
            pickle.dump(verify_data, f, pickle.HIGHEST_PROTOCOL)
    elif load_or_save_option == 'load' :
        with open( os.path.join( config.results_dir_dl, fcst_filename) , 'rb') as f:
            verify_data = pickle.load(f)
            return verify_data[plot_type]



'''
###################################################################################################
# run_script is a function that runs a system command
def run_script(cmd, cmd2=0):
    print "Executing command:  " + cmd
    if cmd2 == 0: 
        os.system(cmd)
    else: 
        os.system( cmd & cmd2 ) 
    #print cmd + "  is finished...."
    return

def calc_grad_mag( inputArray, dx=3000.):
    varargs = [dx]
    x_grad  = np.gradient( inputArray, *varargs, edge_order = 2, axis = 3 )
    y_grad  = np.gradient( inputArray, *varargs, edge_order = 2, axis = 2 )
    return np.sqrt( x_grad**2 + y_grad**2 )

def spatial_filter( ensemble, func , dx ): 
   for i in range( np.shape(ensemble)[0] ): 
      ensemble[i,:,:] = generic_filter( ensemble[i,:,:], func, footprint =np.ones((dx,dx)) ) 

   return ensemble
 
def map2D( clf_probs, ny, nx ):
                """
                DESCRIPTION: Map 1D Probability array into 2D space 
                """
                probs_2D = np.zeros(( ny, nx ))
                for i in range(nx):
                        for j in range(ny):
                                probs_2D[j,i] = clf_probs[j+i*nx]
                return probs_2D

def fill_object_with_single_value( obj_labels, obj_props, init_array ):
        obj_prob_swath = 0.0*obj_labels
        for i in range( len(obj_props) ):
                indices        = np.where(obj_labels == obj_props[i].label)
                per_90         = np.percentile( init_array[indices] , 90 )
                obj_prob_swath[indices] = per_90
        return obj_prob_swath
'''
