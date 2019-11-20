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
parser.add_option("-i", dest="image_dir", type="string", default=None, help="Image Directory")
parser.add_option("-n", dest="ne", type="int", help = "Total number of ensemble members")

(options, args) = parser.parse_args()

if ((options.exp_dir == None) or (options.image_dir == None) or (options.ne == None)):
    print()
    parser.print_help()
    print()
    sys.exit(1)
else:
    exp_dir = options.exp_dir
    image_dir = options.image_dir
    ne = options.ne

######### Swath variable plots to create: #########

var_names = ['wz0to2', 'wz2to5', 'uh0to2', 'uh2to5', 'rain', 'graupmax', 'ws', 'dz', 'wup']
nv = len(var_names)

################## Get number of forecast times (fcst_nt) from first summary file: ########################

files = []
files_temp = os.listdir(exp_dir)
for f, file in enumerate(files_temp):
   if (file[0] == '2'):
      files.append(file)

exp_file = os.path.join(exp_dir, files[0])

try:
   fin = netCDF4.Dataset(exp_file, "r")
   print("Opening %s \n" % exp_file)
except:
   print("%s does not exist! \n" % exp_file)
   sys.exit(1)

temp_time = fin.variables['TIME'][:]
fcst_nt = len(temp_time)

print('nt is: ', fcst_nt)

fin.close()
del fin

############### Probability matched mean composite reflectivity path is hard coded: #################

pmm_file = os.path.join(exp_dir, 'pmm_dz.nc')

pool = Pool(processes=(32))              # set up a queue to run

#########################plot stuff

for t in range(0, fcst_nt):
   if (t == 0): 
      for n in np.arange(0, nv):
         time.sleep(1)
         cmd = "python news_e_swath_retro.py -d %s -o %s -n %s -t %d -s %d" % (exp_dir, image_dir, var_names[n], fcst_nt, t)
         pool.apply_async(run_script, (cmd,))
   time.sleep(1)
   cmd = "python news_e_timestep_retro.py -d %s -o %s -t %d" % (exp_dir, image_dir, t)
   pool.apply_async(run_script, (cmd,))

   time.sleep(5)

pool.close()
pool.join()


