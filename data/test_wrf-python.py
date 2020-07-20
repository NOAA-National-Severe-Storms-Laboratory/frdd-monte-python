from wrf import getvar, interplevel
from netCDF4 import Dataset

f = '/scratch/wof/realtime/FCST/20190524/2300/ENS_MEM_1/wrfwof_d01_2019-05-24_23:05:00'
ncfile = Dataset(f)

temperature_C = getvar(wrfin=ncfile, varname= 'avo')
p = getvar(wrfin=ncfile, varname='pressure')
print (temperature_C.shape)






