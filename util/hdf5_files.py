import h5py

def save_hdf5_files( hdf5_filename, variable_name_list, save_data ):
    """ Saves an hdf5 file """
    with h5py.File( hdf5_filename , 'w' ) as f:
        for i, variable_name in enumerate(variable_name_list):
            f.create_dataset( variable_name , data=save_data[i] )

def load_hdf5_files( hdf5_filename, variable_name_list ):
    """ Loads an hdf5 file of multiple variables into a list """
    x = [ ]
    f = h5py.File(hdf5_filename, 'r')
    for i, variable_name in enumerate(variable_name_list):
        x.append( f[variable_name][:] )
    f.close( )

    return x

