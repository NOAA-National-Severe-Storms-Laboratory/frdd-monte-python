###################################################################################################

from mpl_toolkits.basemap import Basemap
import matplotlib
import math
from scipy import *
import pylab as P
import numpy as np
import sys, glob
import os
import time
from optparse import OptionParser
import netCDF4
from news_e_post_cbook import *

from multiprocessing import Pool

sys.path.append("/scratch/software/Anaconda2/bin")

###################################################################################################
# run_script is a function that runs a system command

def run_script(cmd):
    print("Executing command:  " + cmd)
    os.system(cmd)
    print(cmd + "  is finished....")
    return

###################################################################################################

parser = OptionParser()
parser.add_option("-d", dest="exp_dir", type="string", default=None, help="Input Directory (summary files)")
parser.add_option("-f", dest="fcst_dir", type="string", default=None, help="Forecast Directory")
parser.add_option("-n", dest="ne", type="int", help = "Total number of ensemble members")

(options, args) = parser.parse_args()

if ((options.exp_dir == None) or (options.fcst_dir == None) or (options.ne == None)):
    print()
    parser.print_help()
    print()
    sys.exit(1)
else:
    exp_dir = options.exp_dir
    fcst_dir = options.fcst_dir
    ne = options.ne

######### Get number of forecast wrfouts (fcst_nt): #########

member_dirs = []
member_dirs_temp = os.listdir(fcst_dir)

for d, dir in enumerate(member_dirs_temp):
   if (dir[0:3] == 'ENS'):
      member_dirs.append(dir)

member_files = []
member_dir_temp = os.path.join(fcst_dir, member_dirs[0])

member_files_temp = os.listdir(member_dir_temp)

for f, file in enumerate(member_files_temp):
   if (file[0:9] == 'wrfout_d0'):                             #assumes filename format of: "wrfout_d02_yyyy-mm-dd_hh:mm:ss
      member_files.append(file)

fcst_nt = len(member_files)

############### Probability matched mean composite reflectivity path is hard coded: #################

pmm_file = os.path.join(exp_dir, 'pmm_dz.nc')

pool = Pool(processes=32)              # set up a queue to run

################# Create Summary Files: ####################################

for t in range(0, fcst_nt): 
   for n in np.arange(0, ne):
      dir = os.path.join(fcst_dir, member_dirs[n])

      time.sleep(2)
      cmd = "python news_e_post_retro.py -d %s -o %s -e %d -t %d" % (dir, exp_dir, fcst_nt, t)
      pool.apply_async(run_script, (cmd,))
   if (t == 0): 
      time.sleep(90) #give time to write file for first timestep
   else: 
      time.sleep(2)

time.sleep(2)

################# Create PMM Summary File: ####################################

#for t in range(0, fcst_nt):
#   time.sleep(2)
#   cmd = "python news_e_pmm_retro.py -d %s -e %d -t %d" % (exp_dir, fcst_nt, t)
#   pool.apply_async(run_script, (cmd,))

pool.close()
pool.join()

