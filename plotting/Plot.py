import matplotlib

matplotlib.use("Agg")
from wofs.util.news_e_plotting_cbook_v2 import cb_colors as wofs
from os.path import join
from glob import glob
import netCDF4

from scipy.ndimage import gaussian_filter
import shapely.geometry
from descartes import PolygonPatch
from wofs.util import ctables
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
import matplotlib
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning, module="matplotlib")
warnings.filterwarnings("ignore", module="matplotlib")
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from sklearn.metrics import brier_score_loss
from scipy.stats import gaussian_kde
from wofs.evaluation.verification_metrics import ContingencyTable
from matplotlib import rcParams
from wofs.evaluation.verification_metrics import reliability_uncertainty

POLYGON_OPACITY = 0.5


class Plotting:
    """ Plotting handles spatial plotting and 2D plots """

    # {'cblabel': None, alpha:0.7, extend: 'neither', cmap: 'wofs', tick_labels: }
    def __init__(
        self,
        date=None,
        z1_levels=None,
        z2_levels=None,
        z3_levels=None,
        z4_levels=None,
        cmap="wofs",
        z2_color="k",
        z3_color="b",
        z4_color="r",
        z2_linestyles="-",
        z3_linestyles="-",
        z4_linestyles="-",
        **kwargs,
    ):
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
        elif cmap == "wofs":
            self.cmap = ListedColormap(
                [
                    wofs.blue3,
                    wofs.blue4,
                    wofs.blue5,
                    wofs.blue6,
                    wofs.red3,
                    wofs.red4,
                    wofs.red5,
                    wofs.red6,
                    wofs.red7,
                    wofs.purple6,
                    wofs.purple7,
                    wofs.purple8
                ]
            )
        elif cmap == "basic":
            self.cmap = cm.rainbow

        elif cmap == "diverge":
            self.cmap = cm.seismic

        elif cmap == "dbz":
            self.cmap = ctables.NWSRef

        elif cmap == "red":
            self.cmap = cm.Reds

        elif cmap == "precip":
            self.cmap = ctables.NWSRefPrecip

        elif cmap == "qualitative":
            self.cmap = ListedColormap([wofs.q1, 
                                        wofs.b8, 
                                        wofs.q3,
                                        wofs.b1,
                                        wofs.b3,
                                        wofs.q4,
                                        wofs.b4,
                                        wofs.q5, 
                                        wofs.q6, 
                                        wofs.q7, 
                                        wofs.q8, 
                                        wofs.b6, 
                                        wofs.q10, 
                                        wofs.q11, 
                                        wofs.b3, 
                                        wofs.b2, 
                                        wofs.purple5, 
                                        wofs.red5, 
                                        wofs.green5, 
                                        wofs.blue5, 
                                        wofs.orange5,
                                        wofs.purple7,
                                        wofs.red3, 
                                        wofs.q2,
                                        wofs.b1,
                                        wofs.b3, 
                                        wofs.q7
                                        ])

        if "extend" in kwargs.keys():
            self.extend = kwargs["extend"]
        else:
            self.extend = "neither"

        if "alpha" in kwargs.keys():
            self.alpha = kwargs["alpha"]
        else:
            self.alpha = 0.7

        if "cblabel" in kwargs.keys():
            self.cblabel = kwargs["cblabel"]
        else:
            self.cblabel = ""

        if "tick_labels" in kwargs.keys():
            self.tick_labels = kwargs["tick_labels"]
            self.tick_labels_str = None
        else:
            self.tick_labels = self.z1_levels
            self.tick_labels_str = "self.z1_levels"

    def _generate_base_map(
        self, fig, axes, draw_counties=True, draw_map_scale=False, basemap_file=None
    ):
        """ Creates the BaseMap object for spatial plotting """
        if basemap_file is None:
            try:
                base_path = "/oldscratch/skinnerp/2018_newse_post/summary_files"
                in_path = join(base_path, self.date, "0000")
                file_with_plot_data = glob(join(in_path, "news-e_ENS*"))[0]
            except IndexError:
                base_path = "/work/skinnerp/WoFS_summary_files/2019/"
                in_path = join(join(base_path, self.date), "0000")
                file_with_plot_data = glob(join(in_path, "news-e_ENS*"))[0]
        else:
            file_with_plot_data = basemap_file

        file_in = netCDF4.Dataset(file_with_plot_data, "r")
        lat = file_in.variables["xlat"][:]
        lon = file_in.variables["xlon"][:]

        cen_lat = file_in.CEN_LAT
        cen_lon = file_in.CEN_LON
        stand_lon = file_in.STAND_LON
        try:
            true_lat1 = file_in.scale_lat_1
            true_lat2 = file_in.scale_lat_2
        except:
            true_lat1 = file_in.TRUE_LAT1
            true_lat2 = file_in.TRUE_LAT2
        
        file_in.close()
        del file_in
        
        sw_lat = lat[0, 0]
        sw_lon = lon[0, 0]
        ne_lat = lat[-1, -1]
        ne_lon = lon[-1, -1]

        map_axes = []
        for i, ax in enumerate(fig.axes):
            # print ('Plotting BaseMap...')
            map_ax = Basemap(
                projection="lcc",
                llcrnrlon=sw_lon,
                llcrnrlat=sw_lat,
                urcrnrlon=ne_lon,
                urcrnrlat=ne_lat,
                lat_1=true_lat1,
                lat_2=true_lat2,
                lon_0=stand_lon,
                resolution="l",
                area_thresh=1.0,
                ax=ax,
            )

            #map_ax = Basemap(projection='lcc', llcrnrlon=sw_lon,llcrnrlat=sw_lat, urcrnrlon=ne_lon, urcrnrlat=ne_lat, \
            #                 resolution='l',area_thresh=1., lon_0=stand_lon, ax = ax)

            if draw_counties:
                map_ax.drawcounties(linewidth=0.5, color=wofs.gray3)
            map_ax.drawstates(linewidth=1.0, color=wofs.gray5)
            map_ax.drawcoastlines(linewidth=1.0, color=wofs.gray5)
            map_ax.drawcountries(linewidth=1.0, color=wofs.gray5)

            if draw_map_scale:
                lat_loc = lat[15, 30]
                lon_loc = lon[15, 30]

                map_ax.drawmapscale(
                    lon_loc,
                    lat_loc,
                    sw_lon,
                    sw_lat,
                    60.0,
                    barstyle="fancy",
                    labelstyle="simple",
                    fillcolor1="w",
                    fillcolor2="#555555",
                    fontcolor="k",
                    zorder=5,
                    fontsize=8.0,
                )

            map_axes.append(map_ax)

        x, y = map_axes[0](lon[:], lat[:])
        return map_axes, x, y

    def _create_fig(
        self,
        fig_num,
        plot_map=False,
        sub_plots=None,
        figsize=(8, 9),
        sharex="none",
        sharey="none",
        draw_counties=True,
        draw_map_scale=False,
        basemap_file=None,
        wspace=0.05,
        hspace=0.05
    ):
        """ Creates a figure with a single panels or the prescribed subplot panels
            param: fig_num, figure number 
            param: sub_plots, default=None, otherwise tuple of (nrows, ncols)
            param: figsize, Figure size (as tuple [width, height] in inches)
            param: sharex, if using sub_plots, set to {'row' or 'col'} to share x-axis
            param: sharex, if using sub_plots, set to {'row' or 'col'} to share y-axis
            param: draw_counties, bool, whether to draw counties boundaries on the base map object
            param: draw_map_scale, bool, whether to draw a map scale on the base map object 
        """
        ## Plot Parameters and Stuff
        # plt.rc('xtick', labelsize=22)
        # plt.rc('ytick', labelsize=22)

        if sub_plots is None:
            sub_plots = (1, 1)
        fig, axes = plt.subplots(
            nrows=sub_plots[0],
            ncols=sub_plots[1],
            sharex=sharex,
            sharey=sharey,
            figsize=figsize,
        )
        # Removing extra white space.
        #plt.subplots_adjust(wspace=wspace, hspace=hspace)
        #plt.subplots_adjust(bottom=0.1, right=0.4, top=0.4)
        
        #for a in axes.flat:
        #    a.set_xticklabels([])
        #    a.set_yticklabels([])
        #    a.set_aspect('equal')
         
        if plot_map:
            map_axes, x, y = self._generate_base_map(
                fig=fig,
                axes=axes,
                draw_counties=draw_counties,
                draw_map_scale=draw_map_scale,
                basemap_file=basemap_file
            )
            return fig, axes, map_axes, x, y
        else:
            return fig, axes

    def _save_fig(self, fig, fname, bbox_inches="tight", dpi=300, aformat="png"):
        """ Saves the current figure """
        return plt.savefig(fname, bbox_inches=bbox_inches, dpi=dpi, format=aformat)

    def _add_major_frame(
        self, fig, xlabel_str='', ylabel_str='', fontsize=35, title=None, title_height=0.92
    ):
        """ Create a large frame around the subplots. Used to create large X- and Y- axis labels. """
        fig.add_subplot(111, frameon=False)
        # hide tick and tick label of the big axes
        plt.tick_params(
            labelcolor="none", top="off", bottom="off", left="off", right="off"
        )
        plt.grid(False)
        plt.xlabel(xlabel_str, fontsize=fontsize, alpha=0.7, labelpad=20.5)
        plt.ylabel(ylabel_str, fontsize=fontsize, alpha=0.7, labelpad=20.5)
        if title is not None:
            plt.suptitle(title, fontsize=fontsize, alpha=0.6, y=title_height)

    def _add_major_colobar(
        self,
        fig,
        contours,
        label,
        labelpad=50.5,
        tick_fontsize=9,
        fontsize=35,
        coords=[0.92, 0.11, 0.03, 0.37],
        rotation=270,
        orientation="vertical",
    ):
        """ Adds a single colorbar to the larger frame around the subplots 
            Args:
                coords = [X,Y,W,L] X,Y coordinates in (0,1) and L,W Length and Width of the colorbar
        """
        cax = fig.add_axes(coords)
        cbar = plt.colorbar(contours, cax=cax, orientation=orientation)
        cbar.set_label(
            label, rotation=rotation, fontsize=fontsize, alpha=0.7, labelpad=labelpad
        )
        cbar.ax.tick_params(labelsize=tick_fontsize)
        if self.tick_labels_str != "self.z1_levels":
            cbar.set_ticklabels(self.tick_labels)
        if orientation == "horizontal":
            cbar.ax.xaxis.set_label_position("top")

    def plot_histogram(
        self,
        ax,
        x,
        bins=range(0, 250, 20),
        histtype="step",
        color="k",
        label=None,
        ylog=False,
        density=True,
        cumulative=True,
    ):
        """
        Plots a histogram
        """
        ax.hist(
            x,
            bins=bins,
            histtype=histtype,
            color=color,
            label=label,
            alpha=0.7,
            lw=2.5,
            cumulative=cumulative,
            density=density,
        )
        plt.legend()
        if ylog:
            ax.set_yscale("log", nonposy="clip")
            ax.set_yticks([1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7])
        else:
            ax.set_yticks(np.arange(0, 1.1, 0.1))

        if density:
            ylabel = "Relative Frequency"
        else:
            ylabel = "Frequency"

        plt.grid(alpha=0.5)
        ax.set_xticks(bins)
        ax.set_ylabel(ylabel, fontsize=25)
        ax.tick_params(axis="both", which="major", labelsize=10)
        ax.set_xlabel("6-hour Accumulated Precipitation (mm)", fontsize=25)

    def spatial_plotting(
        self,
        fig,
        ax,
        x,
        y,
        z1,
        map_ax=None,
        z2=None,
        z3=None,
        z4=None,
        scaY=None,
        scaX=None,
        quiver_u_1=None,
        quiver_u_2=None,
        quiver_v_1=None,
        quiver_v_2=None,
        z1_is_integers=False,
        image=False,
        lsr_points=None,
        wwa_points=None,
        plot_colorbar=False,
        text=False,
        title=None,
    ):
        """ Plots various spatial plots (e.g., contour & contourf) """
        if map_ax is None:
            map_ax = ax

        if z1_is_integers:
            plt1 = map_ax.pcolormesh(
                x,
                y,
                z1,
                cmap=self.cmap,
                alpha=self.alpha,
                norm=BoundaryNorm(self.z1_levels, ncolors=self.cmap.N, clip=True),
            )
        elif image:
            plt1 = map_ax.imshow(z1, interpolation='none', vmin=0) #cmap=self.cmap)
        else:
            plt1 = map_ax.contourf(
                x,
                y,
                z1,
                cmap=self.cmap,
                levels=self.z1_levels,
                alpha=self.alpha,
                extend=self.extend,
            )
            map_ax.contour(
                x,
                y,
                z1,
                colors="k",
                levels=self.z1_levels,
                linewidths=0.4,
                alpha=0.25,
                linestyles="solid",
            )

        # CONTOUR PLOTS
        if z2 is not None:
            plt2 = map_ax.contour(
                x,
                y,
                z2,
                colors=self.z2_color,
                levels=self.z2_levels,
                linestyles=self.z2_linestyles,
                linewidths=1.0,
                alpha=0.8,
            )
        if z3 is not None:
            plt3 = map_ax.contour(
                x,
                y,
                z3,
                colors=self.z3_color,
                levels=self.z3_levels,
                linestyles=self.z3_linestyles,
                linewidths=1.0,
                alpha=0.8,
            )
        if z4 is not None:
            plt4 = map_ax.contour(
                x,
                y,
                z4,
                colors=self.z4_color,
                levels=self.z4_levels,
                linestyles=self.z4_linestyles,
                linewidths=1.0,
                alpha=0.8,
            )

        # SCATTER PLOTS
        if scaX is not None:
            plt4 = map_ax.scatter(scaX, scaY, s=2.5, color="k", alpha=0.5)

        # QUIVER PLOTS (Vectors/arrows)
        if quiver_u_1 is not None:
            plt5 = map_ax.quiver(
                x[::6, ::6], y[::6, ::6], quiver_u_1[::6, ::6], quiver_v_1[::6, ::6]
            )
            # map_ax.quiverkey(q, X=0.3, Y=1.1, U=10,
            #                     label='Quiver key, length = 10', labelpos='E')
        if quiver_u_2 is not None:
            plt6 = map_ax.quiver(
                x[::6, ::6],
                y[::6, ::6],
                quiver_u_2[::6, ::6],
                quiver_v_2[::6, ::6],
                color="green",
            )
            # map_ax.quiverkey(q, X=0.3, Y=1.1, U=10,
            #       label='Quiver key, length = 10', labelpos='E')

        if lsr_points is not None:
            if "wind" in lsr_points.keys():
                x, y = map_ax(lsr_points["wind"][1], lsr_points["wind"][0])
                plt7 = map_ax.scatter(
                    x,
                    y,
                    s=8,
                    marker="o",
                    facecolor=wofs.blue6,
                    edgecolor="k",
                    linewidth=0.2,
                    zorder=10,
                )
            if "hail" in lsr_points.keys():
                x, y = map_ax(lsr_points["hail"][1], lsr_points["hail"][0])
                plt8 = map_ax.scatter(
                    x,
                    y,
                    s=8,
                    marker="o",
                    facecolor=wofs.green6,
                    edgecolor="k",
                    linewidth=0.2,
                    zorder=10,
                )
            if "tornado" in lsr_points.keys():
                x, y = map_ax(lsr_points["tornado"][1], lsr_points["tornado"][0])
                plt9 = map_ax.scatter(
                    x,
                    y,
                    s=8,
                    marker="o",
                    facecolor=wofs.red6,
                    edgecolor="k",
                    linewidth=0.2,
                    zorder=10,
                )

            #plt.legend([plt7, plt8, plt9], ['Wind', 'Hail', 'Tornado'], loc='lower right', fontsize=10)

        if wwa_points is not None:
            if "tornado" in wwa_points.keys():
                patches = []
                for coords in wwa_points["tornado"]:
                    coords = [map_ax(pairs[0], pairs[1]) for pairs in coords]
                    poly = Polygon(coords, facecolor="none", edgecolor="r")
                    ax.add_patch(poly)
        if plot_colorbar:
            cb = fig.colorbar(
                plt1, orientation="horizontal", ticks=self.z1_levels.tolist()
            )
            cb.set_label(self.cblabel, fontsize=18, labelpad=10)
            cb.ax.xaxis.set_label_position("top")
            cb.ax.tick_params(labelsize=12)
            cb.ax.set_xticklabels(self.tick_labels)

        if text:
            ax.annotate(text[0], xy=text[1], xycoords="axes pixels")

        if title is not None:
            ax.set_title(title)

        # Keep an equal aspect ratio
        ax.set_aspect("equal")

        return plt1

    def paint_ball_plot(self, fig, ax, map_ax, x, y, multi_var, title=None):

        var_colors = [
            wofs.q1,
            wofs.q2,
            wofs.q3,
            wofs.q4,
            wofs.q5,
            wofs.q6,
            wofs.q7,
            wofs.q8,
            wofs.q9,
            wofs.q10,
            wofs.q11,
            wofs.q12,
            wofs.b1,
            wofs.b2,
            wofs.b3,
            wofs.b4,
            wofs.b5,
            wofs.b6,
            wofs.q1,
            wofs.q2,
            wofs.q3,
            wofs.q4,
            wofs.q5,
            wofs.q6,
            wofs.q7,
            wofs.q8,
            wofs.q9,
            wofs.q10,
            wofs.q11,
            wofs.q12,
            wofs.b1,
            wofs.b2,
            wofs.b3,
            wofs.b4,
            wofs.b5,
            wofs.b6,
        ]

        for i in range(len(multi_var)):
            p1 = map_ax.contourf(
                x,
                y,
                multi_var[i, :, :],
                colors=[var_colors[i], var_colors[i]],
                levels=[0.9, 200.0],
                alpha=0.5,
            )

        if title is not None:
            ax.set_title(title)

    def prob_plot(self, fig, ax, map_ax, x, y, z):
        """
        Plots multiple...
        """
        max_z = np.amax(z, axis=0)
        contours = map_ax.pcolormesh(
            x,
            y,
            max_z,
            cmap=self.cmap,
            alpha=self.alpha,
            norm=BoundaryNorm(self.z1_levels, ncolors=self.cmap.N, clip=True),
        )

        return contours

    def line_plot(self, fig, ax, x, variable, xlabel, ylabel):
        """
        Plots traditional line plots.
        """
        ax.plot(x, variable)

    def decorate_line_plot(ax, xlabel, ylabel):
        """
        Adds decorations to traditional line plots like grids, axis labels, etc. 
        """
        ax.grid(alpha=0.5)
        ax.set_ylabel(xlabel, fontsize=fontsize)
        ax.set_xlabel(ylabel, fontsize=fontsize)
        ax.set_title(title)


def kernal_density_estimate(dy, dx):
    dy_min = np.amin(dy)
    dx_min = np.amin(dx)
    dy_max = np.amax(dy)
    dx_max = np.amax(dx)

    x, y = np.mgrid[dx_min:dx_max:100j, dy_min:dy_max:100j]
    positions = np.vstack([x.ravel(), y.ravel()])
    values = np.vstack([dx, dy])
    kernel = gaussian_kde(values)
    f = np.reshape(kernel(positions).T, x.shape)

    return x, y, f


def plot_event_frequency_bars(ax, predictions, targets, mean_prob, color):
    """
        Plots the event frequency bars on the reliability diagrams
        """
    event_freq_err = reliability_uncertainty(
        X=predictions, Y=targets, bins=np.arange(0, 1.0, 0.1)
    )
    lower_end = np.nanpercentile(event_freq_err, 5, axis=0)
    upper_end = np.nanpercentile(event_freq_err, 95, axis=0)
    for i in range(event_freq_err.shape[1]):
        ax.axvline(
            mean_prob[i], ymin=lower_end[i], ymax=upper_end[i], color=color, alpha=0.5
        )


translate = {'RandomForest': 'RF', 'XGBoost':'XGB', 'LogisticRegression':'LR'}
class verification_plots:
    rcParams["axes.titlepad"] = 15
    rcParams["xtick.labelsize"] = 10
    rcParams["ytick.labelsize"] = 10

    def plot_kde_scatter(
        self,
        ax,
        dy,
        dx,
        colors,
        alphas,
        fig_labels,
        grid_spacing=3.0,
        ylim=[-60, 60],
        xlim=[-60, 60],
    ):
        xx = np.arange(0, max(ylim) + 1)
        ax.plot(xx, xx, linestyle="dashed", color="k", alpha=0.7)
        ax.scatter(
            grid_spacing * dx, grid_spacing * dy, color=colors[-1], alpha=alphas[-1]
        )

        x, y, f = kernal_density_estimate(grid_spacing * dy, grid_spacing * dx)
        temp_colors = [wofs.red4, wofs.red5, wofs.red6, wofs.red7]
        temp_linewidths = [1.5, 1.75, 2.0, 2.25]
        temp_thresh = [95.0, 97.5, 99.0, 99.9]
        temp_levels = [0.0, 0.0, 0.0, 0.0]
        for i in range(0, len(temp_thresh)):
            temp_levels[i] = np.percentile(f.ravel(), temp_thresh[i])

        masked_f = np.ma.masked_where(f < 1.6e-5, f)
        ax.contour(
            x,
            y,
            masked_f,
            levels=temp_levels,
            colors=temp_colors,
            linewidths=temp_linewidths,
            alpha=0.7,
        )
        ax.axhline(y=0, color="k", alpha=0.5)
        ax.axvline(x=0, color="k", alpha=0.5)
        # ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        # ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.5))
        # ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
        # ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.5))
        ax.set_ylim(ylim)
        ax.set_xlim(xlim)
        ax.grid()
        ax.text(3.25, 0.1, fig_labels[1], fontsize=35, alpha=0.9)
        ax.set_aspect("equal")
        ax.set_title(fig_labels[0], fontsize=25, alpha=0.85)

    def plot_roc_curve(
        self, ax, results, line_colors, line_labels, counter, subpanel_labels, title=''
    ):
        x = np.arange(0, 1.1, 0.1)
        # Plot random predictor line
        ax.plot(x, x, linestyle="dashed", color="gray", linewidth=0.8)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(5, prune="lower"))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(5, prune="lower"))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
        for name in line_labels:
            ci_dict = calc_confidence_intervals(
                results[name]["pofd"], results[name]["pod"]
            )
            mean_auc = np.mean(results[name]["auc"], axis=0)
            ax.plot(
                ci_dict["x_mean"],
                ci_dict["y_mean"],
                color=line_colors[name],
                alpha=0.7,
                linewidth=1.5,
                label="{0} {1:.3f}".format(translate[name], mean_auc),
            )
            ax.scatter(
                ci_dict["x_mean"][::2],
                ci_dict["y_mean"][::2],
                s=15,
                color=line_colors[name],
                marker=".",
                alpha=0.8,
            )

            polygon_object = _confidence_interval_to_polygon(
                x_coords_bottom=ci_dict["x_bottom"],
                y_coords_bottom=ci_dict["y_bottom"],
                x_coords_top=ci_dict["x_top"],
                y_coords_top=ci_dict["y_top"],
                for_performance_diagram=False,
            )
            polygon_colour = matplotlib.colors.to_rgba(line_colors[name], 0.4)

            polygon_patch = PolygonPatch(
                polygon_object, lw=0, ec=polygon_colour, fc=polygon_colour
            )

            ax.add_patch(polygon_patch)

        if counter == 2:
            ax.legend(loc="lower right", fontsize=8, fancybox=True, shadow=True)
        ax.set_ylim([0, 1])
        ax.set_xlim([0, 1])
        ax.grid()
        #ax.text(0.43,0.55,
        #    'No Skill',
        #    rotation=45,
        #    fontsize=8,
        #    color='gray'
        #    )
        ax.text(0.02, 0.925, subpanel_labels[0], fontsize=25, alpha=0.65)
        ax.text(0.87, 0.925, subpanel_labels[1], fontsize=35, alpha=0.9)
        #ax.set_title(title, fontsize=15, alpha=0.75)

    def plot_performance_diagram(
        self,
        ax,
        results,
        line_colors,
        line_labels,
        subpanel_labels,
        counter,
        mode="line_plot",
        colors=None,
        markers=None,
        legend_elements=None,
        title = ''
    ):
        """
        Creates a performance diagram plot with potential multiple lines per axes.
        """
        bias_slopes = [0.5, 1.0, 1.5, 2.0, 4.0]
        x1 = y1 = np.arange(0, 1.01, 0.01)
        xx, yy = np.meshgrid(x1, y1)
        bias = ContingencyTable.calc_bias(yy, xx)
        csi = ContingencyTable.calc_csi(yy, xx)
        levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        csiContours = ax.contourf(
            xx, yy, csi, levels, cmap=cm.Blues, extend="max", alpha=0.6
        )
        biasLines = ax.contour(
            xx,
            yy,
            bias,
            colors="k",
            levels=bias_slopes,
            inline=True,
            fmt="%1.1f",
            linestyles="dashed",
            linewidths=0.8,
            alpha=0.9
        )
        ax.set_ylim([0, 1])
        ax.set_xlim([0, 1])
        ax.set_xticks(np.arange(0.1, 1.1, 0.1))
        ax.set_yticks(np.arange(0.1, 1.1, 0.1))
        ax.xaxis.set_major_locator(ticker.MaxNLocator(5, prune="lower"))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(5, prune="lower"))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
        manual_locations = [
            (0.1, 0.7),
            (0.3, 0.75),
            (0.5, 0.75),
            (0.6, 0.6),
            (0.7, 0.4),
        ]
        ax.clabel(
            biasLines, fmt="%1.1f", inline=True, fontsize=8, manual=manual_locations
        )

        # No-Skill Line
        # Uncertainty Line
        truths = results[list(results.keys())[0]]["targets"][0]
        climo = np.mean(truths)
        ax.axvline(climo, linestyle="dashed", color="gray", linewidth=0.9)
        ax.text(climo + 0.01, 0.5, "No-Skill", rotation=90, fontsize=8, color="gray")

        if mode == "line_plot":
            for name in line_labels:
                ci_dict = calc_confidence_intervals(
                    results[name]["sr"], results[name]["pod"]
                )
                mean_auprc = np.mean(results[name]["auprc"], axis=0)
                ax.plot(
                    ci_dict["x_mean"],
                    ci_dict["y_mean"],
                    color=line_colors[name],
                    alpha=0.7,
                    linewidth=.9,
                    label="{0} {1:.3f}".format(translate[name], mean_auprc),
                )

                ax.scatter(
                    ci_dict["x_mean"][::2],
                    ci_dict["y_mean"][::2],
                    s=25,
                    color=line_colors[name],
                    marker=".",
                    alpha=0.8,
                )

                polygon_object = _confidence_interval_to_polygon(
                    x_coords_bottom=ci_dict["x_bottom"],
                    y_coords_bottom=ci_dict["y_bottom"],
                    x_coords_top=ci_dict["x_top"],
                    y_coords_top=ci_dict["y_top"],
                    for_performance_diagram=True,
                )

                polygon_colour = matplotlib.colors.to_rgba(line_colors[name], 0.4)

                polygon_patch = PolygonPatch(
                    polygon_object, lw=0, ec=polygon_colour, fc=polygon_colour
                )

                ax.add_patch(polygon_patch)

        elif mode == "scatter_plot":
            pass
            # for pod_set, sr_set, marker in zip(pod,sr,markers):
            # for pod, sr, color in zip(pod_set, sr_set, colors):
            #       scatter = ax.scatter(sr, pod, s=100, color = color, marker=marker, alpha = 0.75)

        if counter == 2:
            ax.legend(
                handles=legend_elements,
                loc="upper right",
                fontsize=8,
                fancybox=True,
                shadow=True,
            )

        # Forecast Time ( 0- 60 min )
        ax.text(0.02, 0.925, subpanel_labels[0], fontsize=25, alpha=0.65)
        # Panel label ( e.g., (a) )
        ax.text(0.87, 0.925, subpanel_labels[1], fontsize=35, alpha=0.9)
        ax.set_title(title, fontsize=15, alpha=0.75)

        return csiContours

    def generate_attribute_diagram(
        self,
        ax, 
        bin_rng=np.arange(0, 1.1, 0.1),
        inset_yticks=[1e1, 1e3, 1e5],
        climo=0.0,
        title=''
    ):
        """
        Generate a template attribute diagram 
        """
        #fig, ax = plt.subplots(figsize=(8, 8))
        rcParams["xtick.labelsize"] = 8
        rcParams["ytick.labelsize"] = 8

        xticks = np.arange(0.0, 1.1, 0.1)

        ax.set_xticks(xticks)
        ax.set_yticks(xticks)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(5, prune="lower"))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(5, prune="lower"))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))

        x = np.linspace(0, 1, 100)
        ax.plot(x, x, linestyle="dashed", color="gray", alpha=0.8, linewidth=0.7)

        # Resolution Line
        #ax.axhline(climo, linestyle="dashed", color="gray")
        #ax.text(0.15, climo + 0.015, "No Resolution", fontsize=8, color="gray")

        # Uncertainty Line
        #ax.axvline(climo, linestyle="dashed", color="gray")
        #ax.text(
            #climo + 0.01, 0.35, "Uncertainty", rotation=90, fontsize=8, color="gray"
        #)

        # No Skill line
        y = 0.5 * x + climo
        ax.plot(
            [0, 1],
            [0.5 * climo, 0.5 * (1.0 + climo)],
            color="k",
            alpha=0.65,
            linewidth=0.5,
        )
        #ax.text(
        #    0.8,
        #    (0.5 * 0.915) + (climo * 0.5),
        #    "No Skill",
        #    rotation=27,
        #    fontsize=12,
        #    color="gray",
        #)

        # Histogram inset
        small_ax = inset_axes(
            ax,
            width="50%",
            height="50%",
            bbox_to_anchor=(0.115, 0.58, 0.5, 0.4),
            bbox_transform=ax.transAxes,
            loc=2,
        )

        small_ax.set_yscale("log", nonposy="clip")
        small_ax.set_xticks([0, 0.5, 1])
        small_ax.set_yticks(inset_yticks)
        small_ax.set_ylim([1e0, 1e5])
        small_ax.set_xlim([0, 1])

        ax.set_ylim([0, 1])
        ax.set_xlim([0, 1])
        ax.grid(alpha=0.5)

        #ax.set_ylabel("Observed Frequency", fontsize=22, alpha=0.75)
        #ax.set_xlabel("Mean Forecast Probability", fontsize=22, alpha=0.75)
        #ax.set_title(title, fontsize=15, alpha=0.75)

        return ax, small_ax

    def plot_attribute_diagram(
        self, ax, results, line_colors, line_labels, climo, event_freq_err=None, title=''
    ):
        """
            Creates an attribute diagram plot with potential multiple lines per axes.
        """
        translate = {'RandomForest': 'RF', 'XGBoost':'XGB', 'LogisticRegression':'LR'}
        print(f"Climo: {climo:.3f}")
        bin_rng = np.arange(0, 1.1, 0.1)
        inset_yticks = [1e1, 1e3, 1e5]
        ax, small_ax = self.generate_attribute_diagram(
            ax, bin_rng=bin_rng, inset_yticks=inset_yticks, climo=0.0, title=title
        )

        for name in line_labels:
            ci_dict = calc_confidence_intervals(
                results[name]["mean fcst prob"], results[name]["event frequency"]
            )
            plot_event_frequency_bars(
                ax,
                results[name]["predictions"][0],
                results[name]["targets"][0],
                ci_dict["x_mean"],
                line_colors[name],
            )
            mean_bss = np.mean(results[name]["bss"], axis=0)
            ax.plot(
                ci_dict["x_mean"],
                ci_dict["y_mean"],
                color=line_colors[name],
                alpha=0.7,
                linewidth=1.0,
                label="{0} {1:.3f}".format(translate[name], mean_bss),
            )

            polygon_object = _confidence_interval_to_polygon(
                x_coords_bottom=ci_dict["x_bottom"],
                y_coords_bottom=ci_dict["y_bottom"],
                x_coords_top=ci_dict["x_top"],
                y_coords_top=ci_dict["y_top"],
                for_performance_diagram=False,
            )
            polygon_colour = matplotlib.colors.to_rgba(line_colors[name], 0.4)

            polygon_patch = PolygonPatch(
                polygon_object, lw=0, ec=polygon_colour, fc=polygon_colour
            )

            ax.add_patch(polygon_patch)

        colors = [line_colors[name] for name in line_labels]
        for name in line_labels:
            fcst_probs = np.round(results[name]["predictions"][0], 5)
            n, x = np.histogram(a=fcst_probs, bins=bin_rng)
            bin_centers = 0.5 * (bin_rng[1:] + bin_rng[:-1])
            n = np.ma.masked_where(n==0, n)
            small_ax.plot(bin_centers, n, color=line_colors[name], linewidth=0.6)

        ax.legend(
            ncol=1, loc="lower right", fontsize=6, fancybox=True, shadow=True
        )

def plot_timeseries(fcst_time_idx_set, results, line_labels, line_colors, title='', duration=30, time_step=5):
    """
    Plot Maximum Critical Success Index, AUPRC, AUC, and BSS as a time series. 
    
    Args:
        fcst_time_idx_set: 
        
    """
    rcParams["axes.titlepad"] = 15
    rcParams["xtick.labelsize"] = 15
    rcParams["ytick.labelsize"] = 15

    fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, figsize=(10,5))
    # Gives some space for the Y-axis labels
    fig.subplots_adjust(wspace=.4)
    x_ticks = [f'{int(t)*time_step}-{int(t)*time_step+duration}' for t in fcst_time_idx_set]
    fig.suptitle(title, alpha=0.75)
    
    metrics = ['Maximum CSI', 'AUC', 'AUPRC', 'BSS']
    
    line_set = []
    for mertic, ax in zip(metrics, fig.axes):  
        ax.set_xticks(fcst_time_idx_set)
        ax.set_xticklabels(x_ticks)
        ax.set_ylabel(mertic, fontsize=15, alpha=0.75)
        ax.set_xlim([0,24])
        ax.grid(alpha=0.5)

        for name in line_labels:
            line = ax.plot(fcst_time_idx_set, 
                           results[mertic][name], 
                           color=line_colors[name], 
                           linewidth=1.0, 
                           label=name)
            line_set.append(line)
        
    fig.add_subplot(111, frameon=False)
    plt.tick_params(
            labelcolor="none", top=False, bottom=False, left=False, right=False
        )
    plt.xlabel('Forecast Lead Time', fontsize=15, alpha=0.75, labelpad=20.5)
    # use parameter bbox_to_anchor to reposition
    # the legend box outside the plot area
    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    return fig


def calc_confidence_intervals(x, y):
    """
    Calculate confidence intervals
    """
    mean_x = np.mean(x, axis=0)
    mean_y = np.mean(y, axis=0)

    x_95 = np.percentile(x, 97.5, axis=0)
    x_5 = np.percentile(x, 2.5, axis=0)

    y_95 = np.percentile(y, 97.5, axis=0)
    y_5 = np.percentile(y, 2.5, axis=0)

    return {
        "x_mean": mean_x,
        "x_bottom": x_5,
        "x_top": x_95,
        "y_mean": mean_y,
        "y_bottom": y_5,
        "y_top": y_95,
    }


def _confidence_interval_to_polygon(
    x_coords_bottom,
    y_coords_bottom,
    x_coords_top,
    y_coords_top,
    for_performance_diagram=False,
):
    """Generates polygon for confidence interval.
    P = number of points in bottom curve = number of points in top curve
    :param x_coords_bottom: length-P np with x-coordinates of bottom curve
        (lower end of confidence interval).
    :param y_coords_bottom: Same but for y-coordinates.
    :param x_coords_top: length-P np with x-coordinates of top curve (upper
        end of confidence interval).
    :param y_coords_top: Same but for y-coordinates.
    :param for_performance_diagram: Boolean flag.  If True, confidence interval
        is for a performance diagram, which means that coordinates will be
        sorted in a slightly different way.
    :return: polygon_object: Instance of `shapely.geometry.Polygon`.
    """

    nan_flags_top = np.logical_or(np.isnan(x_coords_top), np.isnan(y_coords_top))
    if np.all(nan_flags_top):
        return None

    nan_flags_bottom = np.logical_or(
        np.isnan(x_coords_bottom), np.isnan(y_coords_bottom)
    )
    if np.all(nan_flags_bottom):
        return None

    real_indices_top = np.where(np.invert(nan_flags_top))[0]
    real_indices_bottom = np.where(np.invert(nan_flags_bottom))[0]

    if for_performance_diagram:
        y_coords_top = y_coords_top[real_indices_top]
        sort_indices_top = np.argsort(y_coords_top)
        y_coords_top = y_coords_top[sort_indices_top]
        x_coords_top = x_coords_top[real_indices_top][sort_indices_top]

        y_coords_bottom = y_coords_bottom[real_indices_bottom]
        sort_indices_bottom = np.argsort(-y_coords_bottom)
        y_coords_bottom = y_coords_bottom[sort_indices_bottom]
        x_coords_bottom = x_coords_bottom[real_indices_bottom][sort_indices_bottom]
    else:
        x_coords_top = x_coords_top[real_indices_top]
        sort_indices_top = np.argsort(-x_coords_top)
        x_coords_top = x_coords_top[sort_indices_top]
        y_coords_top = y_coords_top[real_indices_top][sort_indices_top]

        x_coords_bottom = x_coords_bottom[real_indices_bottom]
        sort_indices_bottom = np.argsort(x_coords_bottom)
        x_coords_bottom = x_coords_bottom[sort_indices_bottom]
        y_coords_bottom = y_coords_bottom[real_indices_bottom][sort_indices_bottom]

    polygon_x_coords = np.concatenate(
        (x_coords_top, x_coords_bottom, np.array([x_coords_top[0]]))
    )
    polygon_y_coords = np.concatenate(
        (y_coords_top, y_coords_bottom, np.array([y_coords_top[0]]))
    )

    return vertex_arrays_to_polygon_object(polygon_x_coords, polygon_y_coords)


def vertex_arrays_to_polygon_object(
    exterior_x_coords,
    exterior_y_coords,
    hole_x_coords_list=None,
    hole_y_coords_list=None,
):
    """Converts polygon from vertex arrays to `shapely.geometry.Polygon` object.
    V_e = number of exterior vertices
    H = number of holes
    V_hi = number of vertices in [i]th hole
    :param exterior_x_coords: np array (length V_e) with x-coordinates of
        exterior vertices.
    :param exterior_y_coords: np array (length V_e) with y-coordinates of
        exterior vertices.
    :param hole_x_coords_list: length-H list, where the [i]th item is a np
        array (length V_hi) with x-coordinates of interior vertices.
    :param hole_y_coords_list: Same as above, except for y-coordinates.
    :return: polygon_object: `shapely.geometry.Polygon` object.
    :raises: ValueError: if the polygon is invalid.
    """

    exterior_coords_as_list = _vertex_arrays_to_list(
        exterior_x_coords, exterior_y_coords
    )
    if hole_x_coords_list is None:
        return shapely.geometry.Polygon(shell=exterior_coords_as_list)

    num_holes = len(hole_x_coords_list)
    outer_list_of_hole_coords = []
    for i in range(num_holes):
        outer_list_of_hole_coords.append(
            _vertex_arrays_to_list(hole_x_coords_list[i], hole_y_coords_list[i])
        )

    polygon_object = shapely.geometry.Polygon(
        shell=exterior_coords_as_list, holes=tuple(outer_list_of_hole_coords)
    )

    if not polygon_object.is_valid:
        raise ValueError("Resulting polygon is invalid.")

    return polygon_object


def _vertex_arrays_to_list(vertex_x_coords, vertex_y_coords):
    """Converts vertices of simple polygon from two arrays to one list.
    x- and y-coordinates may be in one of three formats (see docstring at top of
    file).
    V = number of vertices
    :param vertex_x_coords: See documentation for _check_vertex_arrays.
    :param vertex_y_coords: See documentation for _check_vertex_arrays.
    :return: vertex_coords_as_list: length-V list, where each element is an
        (x, y) tuple.
    """
    num_vertices = len(vertex_x_coords)
    vertex_coords_as_list = []
    for i in range(num_vertices):
        vertex_coords_as_list.append((vertex_x_coords[i], vertex_y_coords[i]))

    return vertex_coords_as_list
