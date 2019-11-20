import numpy as np 
from scipy.ndimage import maximum_filter

class EnsembleProbability: 
	"""
	EnsembleProbability calculates the 2D ensemble probability with the capability of calculating the 
	neighborhood maximum ensemble probability (NMEP; Schwartz and Sobash 2017) 

	"""
	def get_local_maxima2d( self, input_array, nghbrd ):
		"""
		DESCRIPTION: Runs a maximum value filter with a radius 'radius' over a 2D numpy array 
		INPUT: 
			input  , numpy array, 2D numpy array of data to filter 
			nghbrd , integer	, filter radius (in gridpoints ) 
		RETURNS:
			2D numpy array with the filter applied 
		"""
		return maximum_filter(input_array, size=( nghbrd, nghbrd))	

	def calc_ensemble_probability( self, ensemble_data, intensity_threshold, max_nghbrd=0 ): 
		"""
		DESCRIPTION: Calculates the ensemble probability of exceedance ( e.g., source?)
					by masking values of 'ensemble_array' < 'thresh', counting occurences of exceedance
					at each gridpoint and then calculating probability as fraction
		INPUT:
			ensemble_array, numpy array, 4D numpy array ( nt, ne, ny, nx )
			thresh		, float	  , exceedance value
			radmask		  , numpy array,
			nghbrd		  , integer	, (default = 3; 9 km )
			smooth		, string	 ,
		RETURN:
			prob, 2D array of probability of exceedance values between (0, 1)
		"""
		if max_nghbrd > 0:
			for mem in range( ensemble_data.shape[0] ):
				ensemble_data[mem, :, :] = self.get_local_maxima2d( ensemble_data[mem, :, :], max_nghbrd )
		masked_ensemble = np.ma.masked_array(ensemble_data, np.round( ensemble_data, 10 ) < round(intensity_threshold, 10))
		count_ensemble  = np.float64( np.ma.count(masked_ensemble, axis=0))
		ensemble_probability = count_ensemble / float( ensemble_data.shape[0] )
		return ensemble_probability
