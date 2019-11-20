import fnmatch
import os


#news-e_ENS_09_20180510_0000_0045.nc


basePath = '/scratch/skinnerp/2018_newse_post/summary_files/20180509/0000'
nc_file_tag = 'ENS' 
time_idx = [ 4, 5, 6, 7, 8 ] 

all_files  = os.listdir( basePath )
files      = [ ]
for t in time_idx:	
	files.append( os.path.join( basePath, fnmatch.filter( all_files, 'news-e_%s_%02d*' % (nc_file_tag, t ) )[0] ) )

print files 
