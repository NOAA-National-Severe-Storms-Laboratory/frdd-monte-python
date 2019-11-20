import matplotlib
matplotlib.use('Agg')

import sys, os  
sys.path.append('/home/monte.flora/wofs/util') 
sys.path.append('/home/monte.flora/wofs/evaluation') 
from news_e_plotting_cbook_v2 import cb_colors as wofs
from os.path import join
from glob import glob
import netCDF4

import ctables
import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.pyplot import cm
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.basemap import Basemap
import matplotlib.ticker as ticker
from matplotlib.colors import ListedColormap
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.collections import PolyCollection
from news_e_plotting_cbook_v2 import plot_lsr, plot_warn
import matplotlib
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="matplotlib")
warnings.filterwarnings("ignore", module="matplotlib")
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.axes_grid.inset_locator import (inset_axes, InsetPosition,
                                                  mark_inset)
from sklearn.metrics import brier_score_loss
from scipy.stats import gaussian_kde
from verification_metrics import ContingencyTable
from matplotlib import rcParams

class Plotting:
    ''' Plotting handles spatial plotting and 2D plots '''
    # {'cblabel': None, alpha:0.7, extend: 'neither', cmap: 'wofs', tick_labels: } 
    def __init__(self, date=None, z1_levels = None, z2_levels=None, z3_levels = None, z4_levels=None, cmap ='wofs', 
                 z2_color='k', z3_color='b', z4_color='r', 
                 z2_linestyles = '-', z3_linestyles = '-', z4_linestyles ='-', **kwargs):
        self.date = date
        self.z1_levels = z1_levels   
        self.z2_levels = z2_levels  
        self.z3_levels = z3_levels
        self.z4_levels = z4_levels
        self.z2_color = z2_color
        self.z3_color = z3_color
        self.z4_color = z4_color
        self.z2_linestyles = z2_linestyles
        self.z3_linestyles = z3_linestyles
        self.z4_linestyles = z4_linestyles
        if not isinstance(cmap, str):
            self.cmap = cmap
        elif cmap == 'wofs':
            self.cmap  =  ListedColormap([wofs.blue2, wofs.blue3, wofs.blue4, wofs.red2,               
                            wofs.red3, wofs.red4, wofs.red5, wofs.red6, wofs.red7])       
        elif cmap == 'basic':
            self.cmap = cm.rainbow

        elif cmap == 'diverge':
            self.cmap = cm.seismic
    
        elif cmap == 'dbz':
            self.cmap = ctables.NWSRef 
   
        elif cmap == 'red':
            self.cmap = cm.Reds

        elif cmap == 'precip':
            self.cmap = ctables.NWSRefPrecip

        elif cmap == 'qualitative':
            self.cmap = cm.Accent

        if 'extend' in kwargs.keys( ):
            self.extend = kwargs['extend']
        else:
            self.extend = 'neither' 
    
        if 'alpha' in kwargs.keys( ):
            self.alpha = kwargs['alpha']
        else:
            self.alpha = 0.7        

        if 'cblabel' in kwargs.keys( ):
            self.cblabel = kwargs['cblabel']
        else:
            self.cblabel = ''

        if 'tick_labels' in kwargs.keys():
            self.tick_labels = kwargs['tick_labels']
            self.tick_labels_str =None
        else:
            self.tick_labels = self.z1_levels
            self.tick_labels_str = 'self.z1_levels'

    def _generate_base_map( self,fig, axes, draw_counties = True, draw_map_scale = False, lat_lon_tuple=None):
        """ Creates the BaseMap object for spatial plotting """
        if lat_lon_tuple is None:
            base_path = '/oldscratch/skinnerp/2018_newse_post/summary_files/'
            in_path = join(join(base_path, self.date), '0000') 
            file_with_plot_data = glob( join(in_path, 'news-e_ENS*'))[0]
            file_in = netCDF4.Dataset( file_with_plot_data, 'r')
            lat = file_in.variables['xlat'][:]
            lon = file_in.variables['xlon'][:]      
    
            cen_lat = file_in.CEN_LAT
            cen_lon = file_in.CEN_LON
            stand_lon = file_in.STAND_LON
            true_lat1 = file_in.TRUE_LAT1
            true_lat2 = file_in.TRUE_LAT2
            file_in.close( ) 
            del file_in
        else:
            lat = lat_lon_tuple[0]
            lon = lat_lon_tuple[1]
            shape = lat.shape
            ny = int(shape[0] / 2)
            nx = int(shape[1]/ 2)
            
            lat_0 = lat[ny, nx]
            lon_0 = lon[ny,nx]

        sw_lat = lat[0,0]
        sw_lon = lon[0,0]
        ne_lat = lat[-1,-1]
        ne_lon = lon[-1,-1] 

        map_axes = [ ] 
        for i, ax in enumerate(fig.axes):     
            #print ('Plotting BaseMap...')
            map_ax = Basemap(projection='lcc', llcrnrlon=sw_lon,llcrnrlat=sw_lat, urcrnrlon=ne_lon, urcrnrlat=ne_lat, \
                  lat_1 = true_lat1, lat_2 = true_lat2, lon_0 = stand_lon, resolution='l',area_thresh=1., ax = ax)

            #map_ax = Basemap(projection='lcc', llcrnrlon=sw_lon,llcrnrlat=sw_lat, urcrnrlon=ne_lon, urcrnrlat=ne_lat, \
            #                 resolution='l',area_thresh=1., lon_0=lon_0, lat_0=lat_0, ax = ax)

            if draw_counties:
                map_ax.drawcounties(linewidth=0.5, color=wofs.gray3)
            map_ax.drawstates(linewidth=1., color=wofs.gray5)
            map_ax.drawcoastlines(linewidth=1., color=wofs.gray5)
            map_ax.drawcountries(linewidth=1., color=wofs.gray5)

            if draw_map_scale:
                lat_loc = lat[15,30]
                lon_loc = lon[15,30] 

                map_ax.drawmapscale( lon_loc, lat_loc, sw_lon, sw_lat, 60., barstyle='fancy',
                    labelstyle='simple', fillcolor1='w', fillcolor2='#555555', fontcolor='k', zorder=5, fontsize = 8.)
    
            map_axes.append( map_ax ) 
        
        x,y = map_axes[0](lon[:], lat[:]) 
        return map_axes, x, y 

    def _create_fig( self, fig_num, plot_map = False, sub_plots = None, figsize=(8,9), sharex='none', sharey='none', draw_counties = True, draw_map_scale = False, lat_lon_tuple=None ):
        ''' Creates a figure with a single panels or the prescribed subplot panels
            param: fig_num, figure number 
            param: sub_plots, default=None, otherwise tuple of (nrows, ncols)
            param: figsize, Figure size (as tuple [width, height] in inches)
            param: sharex, if using sub_plots, set to {'row' or 'col'} to share x-axis
            param: sharex, if using sub_plots, set to {'row' or 'col'} to share y-axis
            param: draw_counties, bool, whether to draw counties boundaries on the base map object
            param: draw_map_scale, bool, whether to draw a map scale on the base map object 
        '''
        ## Plot Parameters and Stuff 
        #plt.rc('xtick', labelsize=22)
        #plt.rc('ytick', labelsize=22)
        
        if sub_plots is None:
            sub_plots = (1,1)
        fig, axes  = plt.subplots( nrows=sub_plots[0], ncols=sub_plots[1], sharex = sharex, sharey=sharey, figsize=figsize )
        # Removing extra white space.
        plt.subplots_adjust(wspace=0.05, hspace=0.05)
        if plot_map: 
            map_axes, x, y = self._generate_base_map( fig=fig, axes = axes, draw_counties = draw_counties, draw_map_scale = draw_map_scale, lat_lon_tuple=lat_lon_tuple)
            return fig, axes, map_axes, x, y
        else:
            return fig, axes 

    def _save_fig(self, fig, fname, bbox_inches = 'tight', dpi = 300, aformat = 'png' ): 
        ''' Saves the current figure '''
        return plt.savefig( fname, bbox_inches = bbox_inches, dpi = dpi, format = aformat ) 
      
    def _add_major_frame( self, fig, xlabel_str, ylabel_str, fontsize=35, title=None, title_height=0.92 ):
        ''' Create a large frame around the subplots. Used to create large X- and Y- axis labels. '''
        fig.add_subplot(111, frameon=False)
        # hide tick and tick label of the big axes
        plt.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
        plt.grid(False)
        plt.xlabel( xlabel_str, fontsize = fontsize, alpha = 0.7, labelpad = 20.5)
        plt.ylabel( ylabel_str, fontsize = fontsize, alpha = 0.7, labelpad = 20.5)
        if title is not None:
            plt.suptitle( title, fontsize = fontsize, alpha = 0.6, y = title_height)

    def _add_major_colobar( self, fig, contours, label, labelpad=50.5, tick_fontsize=9, fontsize=35, coords = [0.92, 0.11, 0.03, 0.37], rotation=270, orientation='vertical' ):
        ''' Adds a single colorbar to the larger frame around the subplots 
            Args:
                coords = [X,Y,W,L] X,Y coordinates in (0,1) and L,W Length and Width of the colorbar
        '''
        cax = fig.add_axes(coords)
        cbar = plt.colorbar( contours , cax=cax, orientation = orientation)
        cbar.set_label( label, rotation = rotation, fontsize = fontsize, alpha = 0.7, labelpad = labelpad) 
        cbar.ax.tick_params(labelsize=tick_fontsize)
        if self.tick_labels_str != 'self.z1_levels':
            cbar.set_ticklabels(self.tick_labels)
        if orientation == 'horizontal':
            cbar.ax.xaxis.set_label_position('top')

    def plot_histogram( self, ax, x, bins=range(0,250,20), histtype ='step', color='k', label=None, ylog=False, density=True, cumulative=True ):
        '''
        Plots a histogram
        '''
        ax.hist( x, bins=bins, histtype=histtype, color=color, label=label, alpha = 0.7, lw=2.5, cumulative=cumulative, density=density)
        plt.legend( )
        if ylog:
            ax.set_yscale('log', nonposy='clip')
            ax.set_yticks([1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7])
        else:
             ax.set_yticks(np.arange(0, 1.1, 0.1))
        
        if density:
            ylabel = 'Relative Frequency'
        else:
            ylabel = 'Frequency'
        
        
        plt.grid(alpha = 0.5)
        ax.set_xticks(bins)  
        ax.set_ylabel( ylabel, fontsize = 25 )
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.set_xlabel( '6-hour Accumulated Precipitation (mm)', fontsize =25)

    def spatial_plotting( self, fig, ax, x, y, z1, map_ax=None, z2=None, z3=None, z4=None, scaY=None, scaX=None, quiver_u_1=None, quiver_u_2=None, quiver_v_1=None, quiver_v_2=None, 
            z1_is_integers=False, image=False, lsr_points=None, wwa_points=None, plot_colorbar=False, text=False, textx = None, texty = None, title=None): 
        """ Plots various spatial plots (e.g., contour & contourf) """
        if map_ax is None:
            map_ax = ax 
        
        if z1_is_integers:
            plt1 = map_ax.pcolormesh(x, y, z1, cmap=self.cmap, alpha=self.alpha, norm=BoundaryNorm(self.z1_levels, ncolors=self.cmap.N, clip=True)  )
        elif image:
            plt1 = map_ax.imshow( z1, cmap=self.cmap )         
        else:
            plt1 = map_ax.contourf(x, y, z1, cmap=self.cmap, levels = self.z1_levels, alpha=self.alpha, extend=self.extend)
            map_ax.contour( x, y, z1, colors='k', levels = self.z1_levels, linewidths=0.4, alpha=0.25, linestyles='solid')

        # CONTOUR PLOTS 
        if z2 is not None:
            plt2 = map_ax.contour(x, y, z2, colors=self.z2_color, levels=self.z2_levels, linestyles =self.z2_linestyles, linewidths=1.0, alpha = 0.8 )
        if z3 is not None: 
            plt3 = map_ax.contour(x, y, z3, colors=self.z3_color, levels=self.z3_levels, linestyles =self.z3_linestyles, linewidths=1.0, alpha = 0.8 )
        if z4 is not None: 
            plt4 = map_ax.contour(x, y, z4, colors=self.z4_color, levels=self.z4_levels, linestyles =self.z4_linestyles, linewidths=1.0, alpha = 0.8) 

        # SCATTER PLOTS
        if scaX is not None:
            plt4 = map_ax.scatter(scaX, scaY, s = 2.5, color='k', alpha =0.5)
       
        # QUIVER PLOTS (Vectors/arrows) 
        if quiver_u_1 is not None:
            plt5 = map_ax.quiver( x[::6, ::6], y[::6, ::6], quiver_u_1[::6, ::6], quiver_v_1[::6, ::6] )
            #map_ax.quiverkey(q, X=0.3, Y=1.1, U=10,
            #                     label='Quiver key, length = 10', labelpos='E')
        if quiver_u_2 is not None:
            plt6 = map_ax.quiver( x[::6, ::6], y[::6,::6], quiver_u_2[::6, ::6], quiver_v_2[::6, ::6], color = 'green' )
            #map_ax.quiverkey(q, X=0.3, Y=1.1, U=10,
            #       label='Quiver key, length = 10', labelpos='E')

        if lsr_points is not None:
            if 'wind' in lsr_points.keys():
                x,y = map_ax( lsr_points['wind'][1], lsr_points['wind'][0] )
                plt7 = map_ax.scatter(x, y, s=25, marker='o', facecolor=wofs.blue6, edgecolor='k', linewidth=0.4, zorder=10)
            if 'hail' in lsr_points.keys():
                x,y = map_ax( lsr_points['hail'][1], lsr_points['hail'][0] )
                plt8 = map_ax.scatter(x, y, s=25, marker='o', facecolor=wofs.green6, edgecolor='k', linewidth=0.4, zorder=10)
            if 'tornado' in lsr_points.keys():
                x,y = map_ax( lsr_points['tornado'][1], lsr_points['tornado'][0] )
                plt9 = map_ax.scatter(x, y, s=25, marker='o', facecolor=wofs.red6, edgecolor='k', linewidth=0.4, zorder=10)   

        if wwa_points is not None:
            if 'tornado' in wwa_points.keys():
                patches = [ ]
                for coords in wwa_points['tornado']:
                    coords = [ map_ax(pairs[0], pairs[1]) for pairs in coords ] 
                    poly = Polygon(coords,facecolor='none',edgecolor='r')
                    ax.add_patch( poly )
        if plot_colorbar:
            cb  = fig.colorbar(plt1, orientation = 'horizontal', ticks = self.z1_levels.tolist() )
            cb.set_label( self.cblabel, fontsize = 18 , labelpad = 10 ) 
            cb.ax.xaxis.set_label_position('top')
            cb.ax.tick_params(labelsize=12)
            cb.ax.set_xticklabels(self.tick_labels) 
    
        if text:
            plt.annotate(text_string, xy=(textx,texty), xycoords='axes pixels')
    
        if title is not None:
            ax.set_title( title ) 

        # Keep an equal aspect ratio 
        ax.set_aspect('equal') 
        
        return plt1

    def paint_ball_plot( self, fig, ax, map_ax, x, y, multi_var, title=None ): 

        var_colors = [wofs.q1, wofs.q2, wofs.q3, wofs.q4, wofs.q5, wofs.q6, wofs.q7, wofs.q8, wofs.q9, wofs.q10, 
                wofs.q11, wofs.q12, wofs.b1, wofs.b2, wofs.b3, wofs.b4, wofs.b5, wofs.b6, wofs.q1, wofs.q2, 
                wofs.q3, wofs.q4, wofs.q5, wofs.q6, wofs.q7, wofs.q8, wofs.q9, wofs.q10,
                                wofs.q11, wofs.q12, wofs.b1, wofs.b2, wofs.b3, wofs.b4, wofs.b5, wofs.b6    ]  

        for i in range( len(multi_var)): 
            p1 = map_ax.contourf(x, y, multi_var[i,:,:], colors=[var_colors[i], var_colors[i]], levels=[0.9, 200.], alpha=0.5)            

        if title is not None:
            ax.set_title( title )

    def prob_plot(self, fig, ax, map_ax, x, y, z):
        '''
        Plots multiple...
        '''
        max_z = np.amax( z, axis = 0 )
        contours = map_ax.pcolormesh(x, y, max_z, cmap=self.cmap, alpha=self.alpha, norm=BoundaryNorm(self.z1_levels, ncolors=self.cmap.N, clip=True))

        return contours


    def line_plot( self, fig, ax, x, variable, xlabel, ylabel):
        '''
        Plots traditional line plots.
        '''
        ax.plot( x, variable )

    def decorate_line_plot( ax, xlabel, ylabel ): 
        '''
        Adds decorations to traditional line plots like grids, axis labels, etc. 
        '''
        ax.grid(alpha=0.5)
        ax.set_ylabel(xlabel, fontsize=fontsize)
        ax.set_xlabel(ylabel, fontsize=fontsize)
        ax.set_title(title)

def kernal_density_estimate( dy, dx ):
    dy_min = np.amin( dy ) ; dx_min = np.amin( dx )
    dy_max = np.amax( dy ) ; dx_max = np.amax( dx )

    x, y = np.mgrid[ dx_min:dx_max:100j, dy_min:dy_max:100j ]
    positions = np.vstack( [x.ravel(), y.ravel( ) ] )
    values = np.vstack( [ dx, dy ] )
    kernel = gaussian_kde( values )
    f = np.reshape(kernel(positions).T, x.shape )

    return x, y, f

class verification_plots: 
    rcParams['axes.titlepad'] = 20
    rcParams['xtick.labelsize'] = 20 
    rcParams['ytick.labelsize'] = 20
    @staticmethod
    def plot_kde_scatter(ax, dy, dx, colors, alphas, fig_labels, grid_spacing=3., ylim=[-60, 60], xlim=[-60, 60]  ):
        xx = np.arange(0, max(ylim)+1)
        ax.plot( xx, xx, linestyle='dashed', color='k', alpha=0.7)
        ax.scatter( grid_spacing*dx, grid_spacing*dy, color = colors[-1], alpha = alphas[-1]  )

        x, y, f = kernal_density_estimate( grid_spacing*dy, grid_spacing*dx)
        temp_colors = [wofs.red4, wofs.red5, wofs.red6, wofs.red7]
        temp_linewidths = [1.5, 1.75, 2., 2.25]
        temp_thresh = [95., 97.5, 99., 99.9]
        temp_levels = [0., 0., 0., 0.]
        for i in range(0, len(temp_thresh)):
            temp_levels[i] = np.percentile(f.ravel(), temp_thresh[i])

        masked_f  = np.ma.masked_where(f < 1.6e-5, f)
        ax.contour( x, y, masked_f ,  levels = temp_levels, colors=temp_colors, linewidths=temp_linewidths, alpha=0.7)
        ax.axhline(y=0, color='k', alpha = 0.5)
        ax.axvline(x=0, color='k', alpha = 0.5)
        #ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        #ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.5))
        #ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
        #ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.5))
        ax.set_ylim( ylim )
        ax.set_xlim( xlim )
        ax.grid( )
        ax.text( 3.25, 0.1 , fig_labels[1], fontsize = 35, alpha = 0.9 )
        ax.set_aspect('equal')
        ax.set_title( fig_labels[0], fontsize = 25, alpha = 0.85 )

    @staticmethod
    def plot_roc_curve( ax, POD, POFD, colors, labels, fig_labels, counter ):
        x = np.arange(0, 1.1, 0.1)
        # Plot random predictor line
        ax.plot(x, x, linestyle = 'dashed', color = 'gray' )
        ax.xaxis.set_major_locator(ticker.MaxNLocator(5, prune='lower'))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(5, prune='lower'))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
        for i in range( len(POD) ):
            ax.plot( POFD[i], POD[i], color = colors[i] , alpha = 0.7, linewidth = 1.5, label = labels[i] )
            ax.scatter( POFD[i][::2], POD[i][::2], s = 50, color = colors[i], marker = '.', alpha = 0.8)
        if counter == 2:
            ax.legend( loc='lower left' , fontsize = 15, fancybox=True, shadow=True)
        ax.text( 0.02, 0.925 , fig_labels[0], fontsize = 25, alpha = 0.65 )
        ax.text( 0.87, 0.925 , fig_labels[1], fontsize = 35, alpha = 0.9 )

    @staticmethod
    def plot_performance_diagram( ax, pod, sr, line_colors, line_labels, subpanel_labels, linestyles, counter, error=False, 
                                    mode='line_plot', colors=None, markers=None, legend_elements=None ):
        '''
        Creates a performance diagram plot with potential multiple lines per axes.
        '''
        bias_slopes  = [0.5, 1.0, 1.5, 2.0, 4.0]
        x1 = y1 = np.arange(0 , 1.01, 0.01)
        xx,yy   = np.meshgrid( x1, y1 )
        bias    = ContingencyTable.calc_bias( yy, xx )
        csi     = ContingencyTable.calc_csi(yy, xx)
        levels  = [ 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        csiContours = ax.contourf( xx, yy, csi , levels, cmap=cm.Blues, extend = 'max', alpha = 0.6 )
        biasLines   = ax.contour( xx, yy, bias, colors='k', levels = bias_slopes, inline=True, fmt='%1.1f', linestyles='dashed', linewidths = 0.8 )
        ax.set_ylim( [0,1] )
        ax.set_xlim( [0,1] )
        ax.set_xticks( np.arange( 0.1, 1.1, .1) )
        ax.set_yticks( np.arange( 0.1, 1.1, .1) )
        ax.xaxis.set_major_locator(ticker.MaxNLocator(5, prune='lower'))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(5, prune='lower'))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
        manual_locations = [(0.1, 0.7) , (0.3, 0.75), (0.5, 0.75), (0.6, 0.6) , (0.7, 0.4)]
        ax.clabel(biasLines, fmt = '%1.1f', inline = True, fontsize = 15, manual=manual_locations)
    
        if mode == 'line_plot':
            for i in range( len(pod) ):
                if error:
                    mean_sr  = np.mean( sr[i], axis = 0 )
                    mean_pod = np.mean( pod[i], axis = 0 )
                    # plot spread envelope 
                    pod_5  = np.percentile( pod[i], 5 , axis = 0 )
                    pod_95 = np.percentile( pod[i], 95, axis = 0 )

                    ax.fill_between( mean_sr, pod_5, pod_95, facecolor = line_colors[i], alpha = 0.4 )
                    ax.plot( mean_sr, mean_pod, color = line_colors[i] , alpha = 0.9, linewidth = 2.5, label = line_labels[i], linestyle = linestyles[i] )
                    ax.scatter( mean_sr[::2], mean_pod[::2], s = 100, color = line_colors[i], marker = '.', alpha = 0.8) 

                else:
                    ax.plot( sr[i], pod[i], color = line_colors[i] , alpha = 0.7, linewidth = 1.8, label = line_labels[i], linestyle=linestyles[i] )
                    ax.scatter( sr[i][::2], pod[i][::2], s = 100, color = line_colors[i], marker = '.', alpha = 0.8)
        elif mode == 'scatter_plot':
            for pod_set, sr_set, marker in zip(pod,sr,markers):
                for pod, sr, color in zip(pod_set, sr_set, colors):
                    scatter = ax.scatter(sr, pod, s=100, color = color, marker=marker, alpha = 0.75) 

        if counter == 2:
            ax.legend(handles=legend_elements, loc='lower left', fontsize = 18, fancybox=True, shadow=True)

        # Forecast Time ( 0- 60 min )
        ax.text( 0.02, 0.925 , subpanel_labels[0], fontsize = 25, alpha = 0.65 )
        # Panel label ( e.g., (a) )
        ax.text( 0.87, 0.925 , subpanel_labels[1], fontsize = 35, alpha = 0.9 )
        
        return csiContours

    @staticmethod 
    def plot_attribute_diagram( ax, mean_prob, event_frequency, fcst_probs, line_colors, line_labels, linestyles, subpanel_labels, counter, inset_loc='lower right', 
        xticks = np.arange( 0.1, 1.1, .1), yticks=np.arange( 0.1, 1.1, .1), bin_rng = np.round( np.arange(0, 1+1./18., 1./18.), 5 ), inset_yticks=[1e0, 1e1, 1e2, 1e3], 
        truths=None, event_freq_err = None, error=False  ):
        '''
            Creates an attribute diagram plot with potential multiple lines per axes.
        '''
        x = np.arange(0, 1, 1./18.)
        # Plot perfect reliability line (and possible error)
        ax.plot( x, x, linestyle='dashed', color='gray')
         
        if event_freq_err is not None:
            lower_end = np.nanpercentile( event_freq_err, 5, axis = 0 )
            upper_end = np.nanpercentile( event_freq_err, 95, axis = 0 ) 
            for i in range( event_freq_err.shape[1] ):
                ax.axvline( mean_prob[-1][i], ymin = lower_end[i], ymax = upper_end[i], color = 'k', alpha = 0.5 )

        if truths is not None:
            climo = np.mean( truths ) 
            # Resolution line
            ax.axhline( climo, linestyle='dashed', color='gray' )
            # Uncertainty Line
            ax.axvline( climo, linestyle='dashed', color='gray' )
            # No Skill line
            y = 0.5 * x + climo
            ax.plot([0,1], [0.5*climo, 0.5*( 1.0 + climo )], color = 'k', alpha = 0.75, linewidth = 0.9 )
        
        ax.set_xticks( xticks )
        ax.set_yticks( xticks )        

        ax.xaxis.set_major_locator(ticker.MaxNLocator(5, prune='lower'))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
        
        ax.yaxis.set_major_locator(ticker.MaxNLocator(5, prune='lower'))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
        
        for i in range( len(mean_prob) ):
            if error:
                mean_mean_prob = np.mean( mean_prob[i], axis = 0 )
                mean_event_freq = np.mean( event_frequency[i], axis = 0 )

                event_freq_5  = np.percentile( event_frequency[i], 5 , axis = 0 )
                event_freq_95 = np.percentile( event_frequency[i], 95, axis = 0 )

                ax.fill_between( mean_mean_prob, event_freq_5, event_freq_95, facecolor = line_colors[i], alpha = 0.4 )
                ax.plot( mean_mean_prob, mean_event_freq, color = line_colors[i] , alpha = 0.9, linewidth = 2.5, label = line_labels[i], linestyle = linestyles[i] )

            else:
                ax.plot( mean_prob[i], event_frequency[i], color = line_colors[i], linewidth = 1.8, label = line_labels[i], linestyle = linestyles[i], alpha = 0.7 )


        if inset_loc == 'lower right': 
                small_ax = inset_axes(ax, width="60%", height="75%",
                           bbox_to_anchor=(.60, .05, .6, .5),
                           bbox_transform=ax.transAxes, loc=3)
                ax.text( 0.02, 0.925 , subpanel_labels[0], fontsize = 25, alpha = 0.65 )

        elif inset_loc == 'upper left': 
                small_ax = inset_axes(ax, width="60%", height="75%",
                                bbox_to_anchor=(.125, .525, .5, .4), # (x, y , x size, y size ) 
                                bbox_transform=ax.transAxes, loc=2)
                ax.text( 0.65, 0.05, subpanel_labels[0], fontsize = 26, alpha = 0.70 )

        # Plot the inset histogram figure
        fcst_probs = np.round(fcst_probs, 5)
        small_ax.hist( fcst_probs, bins = bin_rng, histtype ='bar',
                        rwidth=0.5, color = line_colors[0], alpha = 0.9 )
        small_ax.set_yscale('log', nonposy='clip')

        small_ax.set_xticks( [0  , 0.5, 1] )
        small_ax.set_yticks( inset_yticks )
        ax.set_ylim( [0,1] )
        ax.set_xlim( [0,1] )
        ax.grid( alpha = 0.5 ) 

        if counter == 2:
            ax.legend( ncol=2, loc='lower left' , fontsize = 15, fancybox=True, shadow=True, framealpha=0.6)
            ax.text( 0.87, 0.925 , subpanel_labels[1], fontsize = 35, alpha = 0.9 )


