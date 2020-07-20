from wofs.data.loadEnsembleData import EnsembleData

date = '20180501'
time = '2300'

data_smryfiles = EnsembleData( date_dir=date, time_dir = time, base_path ='summary_files')

strm_data_smryfiles = data_smryfiles.load( variables=['uh_0to2', 'uh_2to5'], time_indexs=[0,2,4,6,8], tag='ENS' )



