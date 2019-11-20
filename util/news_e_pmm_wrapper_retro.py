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
parser.add_option("-n", dest="ne", type="int", help = "Total number of ensemble members")

(options, args) = parser.parse_args()

if ((options.exp_dir == None) or (options.ne == None)):
    print()
    parser.print_help()
    print()
    sys.exit(1)
else:
    exp_dir = options.exp_dir
    ne = options.ne

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

################################

pool = Pool(processes=(32))              # set up a queue to run

#########################plot stuff

for t in range(0, fcst_nt):
   time.sleep(1)
   cmd = "python news_e_pmm_retro.py -d %s -e %d -t %d" % (exp_dir, fcst_nt, t)
   pool.apply_async(run_script, (cmd,))

   time.sleep(10)

pool.close()
pool.join()


