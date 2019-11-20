from news_e_post_cbook import * 
import datetime 

# 1. Swath low-level Azimuthal Shear in a 1-h window to produce low-level rotation tracks 
# 2. Maximize and Convolve the Observed low-level Azimuthal Shear in subjectively-defined spatial neighborhood 
# 3. Input this 2D array into the following code to produce objects  
        # obj = find_objects_timestep(var, radmask, thresh, area_thresh)
        # Code automatically eliminates azimuthal shear beyond the radar mask  
# 4. For each low-level rotation object, Evaluate  
#       a.  Distance of the object centroid from NWS-issued tornado warning box
#       b.  "                                   " SPC tornado reports
#       c.  "                                   " documented tornado damage paths 
#       d.  Whether in a buffer region from the object centroid, a maximum in mid-level rotation exists 
#       e.  "                                                 ", a maximum in reflectivity exists 


class TornadoPathObjects:
	"""
		Evaluate azimuthal shear objects and discriminate out objects that are 
		not within a distance and timing of tornado storm reports, tornado warning boxes, 
		and observed mid-level rotation and reflectivity 
	""" 
	def __init__( self ): 
		self.shapefile_lsr = '/home/monte.flora/PHD_RESEARCH/LSR/lsr_201704010000_201706150000'
		self.shapefile_wwa = '/home/monte.flora/PHD_RESEARCH/WWA/wwa_201604010000_201806150000'

	def currentTime( self ): 
		dt1       = datetime.timedelta( hours = 1, minutes = window )
                dt2       = datetime.timedelta( minutes = window )
                initTime  = datetime.datetime( year , date[0:3]      , date[2:]      , time[0:3] , time[2:])
                startTime = initTime - dt2
                endTime   = startTime + dt1

		return initTIme, startTime, endTime

	def FuzzyLogicFunction( self, x, dist1, dist2): 
		"""
			Piece-wise function used for Fuzzy logic 		 
		"""
		if x <= dist1: 
			y = 1 
		elif (x > dist1) and (x <= dist2): 
			m = -1.0 / ( dist2 - dist1 )  
			y = m*x + (dist2/(dist2-dist1))  
		else:
			y = 0 

		return y 
	
	def tornadoWarningBoxLocations( self, initTime, date, time ):	
		"""
			Reads in the NWS-issued tornado warning boxes 
		"""	
		year = date[0:4]

		map.readshapefile(self.shapefile_wwa, 'warnings', drawbounds = False)
			
		for info, shape in zip(map.warnings_info, map.warnings):
		 	if (info['PHENOM'] == 'TO'):
				temp_initdate    = info['ISSUED'][4  :8 ]
            			temp_inithr      = info['ISSUED'][8  :10]
            			temp_initmin     = info['ISSUED'][10 :12]
				
				temp_expiredate  = info['EXPIRED'][4 :8 ]
				temp_expirehr    = info['EXPIRED'][8 :10]
				temp_expiremin   = info['EXPIRED'][10:12]

				issuedTime  = datetime.datetime( year , temp_date[0:3] , temp_date[2:] , temp_hr   , temp_min)
				expiredTime = datetime.datetime( year , temp_date[0:3] , temp_date[2:] , temp_hr   , temp_min)

				if (initTime <= expiredTime) and (initTime >= issuedTime): 
					wx, wy = zip(*shape)
		return wx, wy 

	def tornadoReportLocations( self, map, date, time, window = 30. ): 
	
		"""
			Location of SPC storm reports
			
			Inputs: 
				map ,   basemap object 
				date,   string of 8-digit date (e.g., '20170508' ) 
				time,   string of 4-digit time (e.g., '2300'     ) 
				window, buffer zone (in minutes)  
		"""
		
		year = date[0:4] 

		map.readshapefile(self.shapefile_lsr, 'lsr', drawbounds = False) #read shapefile
	
		dt1       = datetime.timedelta( hours = 1, minutes = window )
		dt2	  = datetime.timedelta( minutes = window )
		initTime  = datetime.datetime( year , date[0:3]      , date[2:]      , time[0:3] , time[2:]) 
		startTime = initTime - dt2
		endTime	  = startTime + dt1

		for info, shape in zip(map.lsr_info, map.lsr):
	    		if (info['TYPECODE'] == 'T'):
				temp_date= info['VALID'][4 :8 ] 
         			temp_hr  = info['VALID'][8 :10]
         			temp_min = info['VALID'][10:12]
				fullDate  = datetime.datetime( year , temp_date[0:3] , temp_date[2:] , temp_hr   , temp_min) 
					
				if (fullDate >= startTime) and ( fullDate < endTime ): 
            				temp_lat = double(info['LAT'])
            				temp_lon = double(info['LON'])
            				wx, wy   = map(temp_lon, temp_lat)
		return wx, wy 	

	def additionalStormFeatureLocations( self, mid_level, reflect, threshold1, threshold2 ):
		"""
			Determine locations of observed mid-level rotation and high reflectivity 
			exceeding predetermined thresholds
		"""
		# Determine in advance where mid_level > threshold 
                # ny, nx = np.where(( mid_level > threshold1 & reflect > threshold2), 1, 0) 

		# Just want the locations where it exceeds the threshold

		return ny, nx 

	def lowlevelrotation_objects (  self, obj, map, mid_level, dbz, date, time ):
		"""
		"""
		threshold = 0.6 	
		# pass and object through and then determine the centroid distance 
		#ny1, nx1 = self.additionalStormFeatureLocations( mid_level, reflect, threshold1 = 100. , threshold2 = 30. ) 
		#ny2, nx2 = self.tornadoReportLocations( map, date, time ) 
		#ny3, nx3 = self.tornadoWarningBoxLocations( initTime, date, time )	

		#for object in obj:
			# val = np.zeros(( 3 )) 
			# determine object centroid using regionprops 
		
			# y,x = obj.centroid_distance

			# Need the distance from object centroid and a spc reports 
			# dist_from_report = object centroid - storm report location 
		#val[0] = self.FuzzyLogicFunction( dist_from_report, dist1=20, dist2=30)
		
			# Need the distance from object centroid and tornado warning box  
                        # dist_from_warn = object centroid - tornado warning box 
                 #       val[1] = self.FuzzyLogicFunction( dist_from_warn, dist1=20, dist2=30)

		#	criteria np.prod( val ) 

		#	if criteria > threshold : 
				# Keep it 
		#	else : 
				# Get rid of it 
		#return obj 





