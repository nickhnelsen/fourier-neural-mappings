# -*- coding: utf-8 -*-
"""
Created on Tue May  5 10:53:28 2020

@author: nickh

Written by:
Nicholas H. Nelsen
California Institute of Technology
Email: nnelsen@caltech.edu

A suite of helper functions for plotting 1D and 2D data, in a class format. Full init options. Primarily for SIAM Journals.
    
Last updated: Jun. 10, 2022
"""

# Import modules/packages
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.colors import TwoSlopeNorm
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import Axes3D  
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Define and set custom LaTeX style
styleNHN = {
        "pgf.rcfonts": False,
        "pgf.texsystem": "pdflatex",   
        "text.usetex": True,                
        "font.family": "serif"
        }
mpl.rcParams.update(styleNHN)
FIG_MAX_WIDTH = 5.1667                  # Max figure width for SIAM
FIG_MAX_HEIGHT = 8.5166                 # Max figure height for SIAM
SMALL_SIZE = 8                          # Tiny font size
MEDIUM_SIZE = 10                        # Small font size
BIGGER_SIZE = 16                        # Large font size
SHRINK_SIZE = 0.75                      # Colorbar scaling
ASPECT_SIZE = 15                        # Colorbar width
DPI_SET = 300                           # Default DPI for non-vector graphics figures
FORMAT_VEC_SET = 'pdf'                  # Default format for saving vector graphics figures
FORMAT_IM_SET = 'png'                   # Default format for saving raster graphics figures
BBOX_SET = 'tight'                      # Bounding box setting for plots
NFRAME_SET = 50                         # (50) total number of frames for movie writing
NFPS_SET = 20                           # (20) frames per second for movie writing
CMAPCHOICE2D_SET = cm.inferno           # Colorbar choices: inferno, viridis, hot, gray, magma, coolwarm
CMAPCHOICE3D_SET = cm.coolwarm          # Colorbar choices: viridis, coolwarm, plasma
plt.rcParams["mathtext.fontset"] = 'cm' # Computer Modern mathtext font (applies when ``usetex=False'')



class MidpointNormalize(Normalize):
    """
    Reference: https://stackoverflow.com/questions/20144529/shifted-colorbar-matplotlib/20146989#20146989.
    To use, set norm_set = MidpointNormalize(midpoint=0) # set center of colorbar to 0

    """
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))
    

class Plotter:
    '''
    Class implementation of plotting suite. 
    '''

    def __init__(self, xlab_str=r'$x_1$', ylab_str=r'$x_2$', zlab_str=r'', shrink_set=SHRINK_SIZE, aspect_set=ASPECT_SIZE, ALW=1.5, FSZ=16, LW=3, MSZ=8):
        self.xlab_str = xlab_str
        self.ylab_str = ylab_str
        self.zlab_str = zlab_str
        self.shrink_size = shrink_set
        self.aspect_size = aspect_set
        self.ALW = ALW                          # AxesLineWidth
        self.FSZ = FSZ                          # Fontsize
        self.LW = LW                            # LineWidth
        self.MSZ = MSZ                          # MarkerSize
        plt.rc('font', size=FSZ)                # Controls default text sizes
        plt.rc('axes', titlesize=FSZ)           # Fontsize of the axes title
        plt.rc('axes', labelsize=FSZ)           # Fontsize of the x and y labels
        plt.rc('xtick', labelsize=FSZ - 1)          # Fontsize of the x-tick labels
        plt.rc('ytick', labelsize=FSZ - 1)          # Fontsize of the y-tick labels
        plt.rc('legend', fontsize=FSZ)          # Legend fontsize
        plt.rc('figure', titlesize=FSZ)         # Fontsize of the figure title
        plt.rcParams['axes.linewidth'] = ALW    # Sets the default axes lindewidth to ``ALW''

        
    def save_plot(self, fig_obj, fig_name_str, fig_format=FORMAT_VEC_SET, fig_dpi=DPI_SET, bbox=BBOX_SET):
        '''
        Save fig_obj to filepath fig_name_str. Default assumes 1D plot in vector graphics format. For heat maps and surface plots, change fig_format to FORMAT_IM_SET.
        '''
        if fig_format != FORMAT_VEC_SET: # save as png image
            fig_format = FORMAT_IM_SET
        fig_obj.savefig(fig_name_str, format=fig_format, dpi=fig_dpi, bbox_inches=bbox)
    
    
    def plot_SurfAndHeat(self, fig_num, X, Y, Z, title_str=r'', cmap_set2D=CMAPCHOICE2D_SET, cmap_set3D=CMAPCHOICE3D_SET, interp_set='spline16', norm_set=None):
        """
        fig_num: Figure Number for this set of plots
        X,Y,Z: X, Y are coordinate matrices in `cartesian' indexing and Z the corresponding field matrix Z(X,Y)
        ____str: Labels for plots, must be RAW STRING LITERALS, i.e., xlab_str=r'TEST-x-label'
        """
        if norm_set is not None:
            norm_set = MidpointNormalize(midpoint=norm_set)
        
        fig_str=str(fig_num)
        xmin=X.min()
        xmax=X.max()
        ymin=Y.min()
        ymax=Y.max()
    
        # 3D Surface
        fig = plt.figure(fig_num)
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, Z, 
                                   cmap=cmap_set3D,
                                   linewidth=0,
                                   norm=norm_set,
                                   antialiased=False)
        ax.set_xlabel(self.xlab_str)
        ax.set_ylabel(self.ylab_str)
        ax.set_zlabel(self.zlab_str)
        fig.colorbar(surf, shrink=self.shrink_set, aspect=self.aspect_set)
        ax.set_title(title_str)
        
        # 2D Heat Map
        lastdigit_str=str(np.mod(fig_num,10))
        figg = plt.figure(int(fig_str+lastdigit_str))
        hm = figg.add_subplot(111, xlabel=self.xlab_str, ylabel=self.ylab_str)
        im = hm.imshow(Z,
                         extent=(xmin, xmax, ymin, ymax),
                         cmap=cmap_set2D,
                         norm=norm_set,
                         interpolation=interp_set,
                         origin='lower')
        figg.colorbar(im)  
        plt.title(title_str)
        return fig, figg


    def plot_Surf(self, fig_num, X, Y, Z, title_str=r'', cmap_set3D=CMAPCHOICE3D_SET, norm_set=None):
        """
        fig_num: Figure Number for this plot
        X,Y,Z: X, Y are coordinate matrices in `cartesian' indexing and Z the corresponding field matrix Z(X,Y)
        ____str: Labels for plots, must be RAW STRING LITERALS, i.e., xlab_str=r'TEST-x-label'
        """
        if norm_set is not None:
            norm_set = MidpointNormalize(midpoint=norm_set)
            
        # 3D Surface
        fig = plt.figure(fig_num)
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, Z, 
                                   cmap=cmap_set3D,
                                   linewidth=0,
                                   norm=norm_set,
                                   antialiased=False)
        ax.set_xlabel(self.xlab_str)
        ax.set_ylabel(self.ylab_str)
        ax.set_zlabel(self.zlab_str)
        fig.colorbar(surf, shrink=self.shrink_set, aspect=self.aspect_set)
        ax.set_title(title_str)
        return fig


    def plot_Heat(self, fig_num, X, Y, Z, title_str=r'', cmap_set2D=CMAPCHOICE2D_SET, interp_set='spline16', norm_set=None, cb_ticks_sn=True):
        """
        fig_num: Figure Number for this plot
        X,Y,Z: X, Y are coordinate matrices in `cartesian' indexing and Z the corresponding field matrix Z(X,Y)
        ____str: Labels for plots, must be RAW STRING LITERALS, i.e., xlab_str=r'TEST-x-label'
        """
        if norm_set == 'lognorm':
            norm_set = LogNorm()
        elif norm_set is not None:
            norm_set = MidpointNormalize(midpoint=norm_set)
            # TODO: should this be here?
            # norm_set = DivergingNorm(norm_set) 
            
        xmin=X.min()
        xmax=X.max()
        ymin=Y.min()
        ymax=Y.max()
        
        # 2D Heat Map
        fig = plt.figure(fig_num, figsize=(FIG_MAX_WIDTH, FIG_MAX_WIDTH))
        hm = fig.add_subplot(111, xlabel=self.xlab_str, ylabel=self.ylab_str)
        im = hm.imshow(Z,
                         extent=(xmin, xmax, ymin, ymax),
                         cmap=cmap_set2D,
                         norm=norm_set,
                         interpolation=interp_set,
                         origin='lower')
        plt.title(title_str)
        divider = make_axes_locatable(hm)
        cax = divider.append_axes("right", size="6%", pad=0.15)
        cb = fig.colorbar(im, cax=cax)
        if cb_ticks_sn:
            cb.formatter.set_powerlimits((0, 0))
            cb.ax.yaxis.set_offset_position('left')
            cb.formatter.set_scientific(True)
            cb.update_ticks()
        return fig
    

    def plot_oneD(self, fig_num, x, y, xlab_str1D=r'$x$', ylab_str1D=r'', titlelab_str=r'', legendlab_str=r'', linestyle_str='ko:', LW_set=None, MSZ_set=None, semilogy=False, loglog=False, markerfc=False, fig_width=None, fig_height=None, fig_sz_default=False, linestyle=None):
        """
        fig_num: Figure Number for this plot
        x,y: input and output data to plot
        ____str: Labels for plots, must be RAW STRING LITERALS, i.e., xlab_str1D=r'TEST-x-label'
        To plot multiple 1D functions on the same plot, call plot_oneD repeatedly in sucession with the same figure number fig_num.
        """
        if fig_width is None:
            fig_width = FIG_MAX_WIDTH
        if fig_height is None:
            fig_height = FIG_MAX_WIDTH      # square is default size
        if LW_set is None:
            LW_set = self.LW
        if MSZ_set is None:
            MSZ_set = self.MSZ
        if not markerfc:
            markerfc = 'none'
        else:
            markerfc = None
        xmin = x.min()
        xmax = x.max()
        if fig_sz_default:
            # plt.close(fig)
            fig = plt.figure(fig_num)
        else:
            fig = plt.figure(fig_num, figsize=(fig_width, fig_height)) # figsize=(5.1667, 8.5166)
        if linestyle is None:
            if semilogy:
                plt.semilogy(x, y, linestyle_str, linewidth=LW_set, markersize=MSZ_set, label=legendlab_str, markerfacecolor=markerfc, markeredgewidth=LW_set)
            elif loglog:
                plt.loglog(x, y, linestyle_str, linewidth=LW_set, markersize=MSZ_set, label=legendlab_str, markerfacecolor=markerfc, markeredgewidth=LW_set)
            else:
                plt.plot(x, y, linestyle_str, linewidth=LW_set, markersize=MSZ_set, label=legendlab_str,  markerfacecolor=markerfc, markeredgewidth=LW_set)
        else:
            if semilogy:
                plt.semilogy(x, y, linestyle_str, linewidth=LW_set, markersize=MSZ_set, label=legendlab_str, markerfacecolor=markerfc, markeredgewidth=LW_set, linestyle=linestyle)
            elif loglog:
                plt.loglog(x, y, linestyle_str, linewidth=LW_set, markersize=MSZ_set, label=legendlab_str, markerfacecolor=markerfc, markeredgewidth=LW_set, linestyle=linestyle)
            else:
                plt.plot(x, y, linestyle_str, linewidth=LW_set, markersize=MSZ_set, label=legendlab_str,  markerfacecolor=markerfc, markeredgewidth=LW_set, linestyle=linestyle)
        plt.xlim(xmin, xmax)
        plt.xlabel(xlab_str1D)
        if ylab_str1D != r'':
            plt.ylabel(ylab_str1D)
        if legendlab_str != r'':
            plt.legend(loc='best')
        if titlelab_str != r'':
            plt.title(titlelab_str)
        return fig
