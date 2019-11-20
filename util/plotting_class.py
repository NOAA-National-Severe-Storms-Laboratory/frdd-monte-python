import sys, os  
sys.path.append( '/home/monte.flora/NEWSeProbs/misc_python_scripts/') 
import numpy as np 

from news_e_plotting_cbook_v2 import plot_lsr, plot_warn
import matplotlib 
import matplotlib.pyplot as p
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="matplotlib")
warnings.filterwarnings("ignore", module="matplotlib")
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)
from mpl_toolkits.basemap import Basemap
import netCDF4
from news_e_plotting_cbook_v2 import cb_colors

class SpatialPlotting:
        def __init__(self, filename, date, time, var1_levels=[1.,2.,3], var2_levels=[0.] , cmap=matplotlib.cm.rainbow, extend='neither', alpha=0.8, cblabel=''):
                self.date                 = date
                self.time                 = time
                self.filePath             = self.WRFPath( )
                self.filename             = filename              #string name for plot (e.g. wz0to2rot90)
                self.var1_levels          = var1_levels           #contour levels for var1 plot
                self.var2_levels          = var2_levels           #contour levels for var2 plot
		 #colormap for var1 plot
                self.cmap                 =  matplotlib.colors.ListedColormap([cb_colors.blue2, cb_colors.blue3, cb_colors.blue4, cb_colors.red2, 
							cb_colors.red3, cb_colors.red4, cb_colors.red5, cb_colors.red6, cb_colors.red7])                 
                self.extend               = extend                #string for extend option in var1 plot (e.g. 'neither', 'max', ...)
                self.alpha                = alpha                 #float value for transparency of var1 plot
                self.cblabel              = cblabel

        def WRFPath(self ):
                wrfPath    = '/work1/wof/realtime/%s/EXPS/RLT/WRFOUT/' % (self.date)
                if os.path.exists( wrfPath ):
                        wrfSet       = os.listdir( wrfPath )
                        wrfPath      = wrfPath+wrfSet[0]
                        return wrfPath
                else:
                        wrfPath    = '/work1/wof/realtime/%s/WRFOUT/' % (self.date)
                        wrfSet       = os.listdir( wrfPath )
                        wrfPath      = wrfPath+wrfSet[0]
                        return wrfPath

	def baseMap( self, counties='True'):
                """
                DESCRIPTION: Calculates the base map for spatial plotting 
        
                INPUT: filePath (string) , path to raw WRF output for a given date and time 

                OUTPUT: map object along with various parameters for plotting 

                """
                f        = netCDF4.Dataset( self.filePath, 'r')
                sw_lat   = f.variables['XLAT'][0,0,0]
                sw_lon   = f.variables['XLONG'][0,0,0]
                ne_lat   = f.variables['XLAT'][0,-1,-1]
                ne_lon   = f.variables['XLONG'][0,-1,-1]
                tlat1    = f.TRUELAT1; tlat2    = f.TRUELAT2
                stlon    = f.STAND_LON
                DX       = f.DX; DY       = f.DY
                xx       = np.arange(0.5,0.5+len(f.dimensions['west_east']))*DX
                yy       = np.arange(0.5,0.5+len(f.dimensions['south_north']))*DY
                xxu      = np.arange(0,len(f.dimensions['west_east_stag']))*DX
                yyv      = np.arange(0,len(f.dimensions['south_north_stag']))*DY
                we       = len(f.dimensions['west_east_stag'])
                sn       = len(f.dimensions['south_north_stag'])

                xs = (np.arange(0, we-1)+0.5)*DX
                ys = (np.arange(0, sn-1)+0.5)*DY

                xv = (xs[0:-1]+xs[1:])/2
                yv = (ys[0:-1]+ys[1:])/2

                xxv, yyv = np.meshgrid(xv, yv)
                xx1, yy1 = np.meshgrid(xx, yy)

                map      = Basemap(projection='lcc', llcrnrlon=sw_lon,llcrnrlat=sw_lat,urcrnrlon=ne_lon,urcrnrlat=ne_lat, \
                  lat_1 = tlat1, lat_2 = tlat2, lon_0 = stlon, resolution='l',area_thresh=1.)

                if (counties == 'True'):
                        map.drawcounties(linewidth=0.5, color=cb_colors.gray3)
                map.drawstates(linewidth=1., color=cb_colors.gray5)
                map.drawcoastlines(linewidth=1., color=cb_colors.gray5)
                map.drawcountries(linewidth=1., color=cb_colors.gray5)

                lat_loc   = f.variables['XLAT'][0,  15, 30]
                lon_loc   = f.variables['XLONG'][0, 15, 30]

                #map.drawmapscale( lon_loc, lat_loc, sw_lon, sw_lat, 60., barstyle='fancy',
                #                labelstyle='simple', fillcolor1='w', fillcolor2='#555555', fontcolor='k', zorder=5, fontsize = 8.)

                f.close()
                return map , xx1, yy1, xxv, yyv, DX, DY

        def create_fig( self, num ):
                fig                             = p.figure( num, figsize=(8., 9.) )
                map, xx1, yy1, xxv, yyv, DX, DY = self.baseMap( )
                return fig, map, xx1, yy1, xxv, yyv


	def mapContourPlot(self, filename, fig, map, x, y, var2D_1, var2D_2=[ ], spec='False', scat ='False', lsr_warn = 'False', date = '', init_time ='', Gauss=False, \
                           mapScale = False, scaY=0, scaX=0, text=False, text_string=None, integer_plot=False ):
                """
                Handles plotting 2D contour and filled-contoured maps 

                """
                if Gauss:
                        var2D_1 = gaussian_filter( var2D_1, sigma = 2.0)

                ############# Create filled contour plot, with intervals highlighted for var1 data:    #################
                if integer_plot:
                        p1 = map.imshow( var2D_1, cmap=self.cmap, alpha=self.alpha )
                        p2 = map.contour( x, y, var2D_1, colors='k',     levels = self.var1_levels, linewidths=0.4, alpha=0.25, linestyles='solid')
                else:
                        p1 = map.contourf(x, y, var2D_1, cmap=self.cmap, levels = self.var1_levels, alpha=self.alpha, extend=self.extend)
                        p2 = map.contour( x, y, var2D_1, colors='k',     levels = self.var1_levels, linewidths=0.4, alpha=0.25, linestyles='solid')

                ############# If special contour is to be plotted, plot it: ###############
                if (spec == 'True'):
                        p3 = map.contour(x, y, var2D_2, colors='k', levels=self.var2_levels, linewidths=1.0)

                if (scat == 'True'):
                        p4 = map.scatter(scaX*3000., scaY*3000., s = 2.5)

                #cb  = fig.colorbar(p1, orientation = 'horizontal', label = self.cblabel)

                if ( lsr_warn == 'True'):
                        if '2017' in date:
                                shapefile_lsr = '/home/monte.flora/PHD_RESEARCH/LSR/lsr_201704010000_201706150000'
                        elif '2016' in date:
                                shapefile_lsr = '/home/monte.flora/PHD_RESEARCH/LSR/lsr_201604010000_201606100000'
                        shapefile_wwa = '/home/monte.flora/PHD_RESEARCH/WWA/wwa_201604010000_201806150000'

                        ti     = (int( init_time[:3])) * 3600. + int(init_time[2:]) * 60.
                        time   = ti + 7200.

                        print(init_time[:3] , float( init_time[:3] ))
                        print(int( init_time[:3])-1, ti)

                        plot_lsr(map, fig, shapefile_lsr, date[4:], ti, time, plot_h='False', plot_w='False', plot_t='True')
                        plot_warn(map, fig, shapefile_wwa, date[4:],ti, svr_color='k', tor_color='r', ff='False')

                if text:
                        p.annotate(text_string, xy=(100,100), xycoords='axes pixels')

                p.savefig(self.filename, bbox_inches = 'tight', format = 'png' , dpi = 300)


