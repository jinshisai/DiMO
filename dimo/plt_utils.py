import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes



def add_colorbar_toaxis(ax, cim=None, 
    loc = 'right', pad = '3%', width = '3%',
    length = '100%', cbaroptions: list = None, 
    cbarlabel = '', ticks: list = None, fontsize: float = None, 
    tickcolor: str = 'k', labelcolor: str = 'k'):
    '''
    Add a colorbar to axis.

    Parameters
    ----------
    loc (str): Location of the colorbar. Must be choosen from right, left, top or bottom.
    pad (str or float): Pad between the image and colorbar. Must be given as percentage (e.g., '3%') or
        fraction (e.g, 0.03) of the full plot width.
    width (str or float): Width of the colorbar. Must be given as percentage or fraction of 
        the full plot width.
    length (str or float): Length of the colorbar. Must be given as percentage or fraction of 
        the full plot width.
    cbaroptions (list): Colorbar options which set location, pad, width all at once.
    ticks (list): Ticks of colorbar. Optional parameter.
    fontsize (float): Fontsize of the colorbar label and tick labels. Optional parameter.
    tickcolor, labelcolor (str): Set tick and label colors.
    '''
    # parameters to set orientation
    orientations = {
    'right': 'vertical',
    'left': 'vertical',
    'top': 'horizontal',
    'bottom': 'horizontal'}

    # colorbar options
    if cbaroptions is not None:
        if len(cbaroptions) == 3:
            cbar_loc, cbar_wd, cbar_pad = cbaroptions
        elif len(cbaroptions) == 4:
            cbar_loc, cbar_wd, cbar_pad, cbar_lbl = cbaroptions
        else:
            print('WARNING\tadd_colorbar_toaxis: cbaroptions must be a list object with three or four elements.')
            print('WARNING\tadd_colorbar_toaxis: Input cbaroptions are ignored.')
    else:
        cbar_loc = loc
        cbar_wd = width
        cbar_pad = pad
    # str to float
    if type(cbar_wd) == str: cbar_wd = float(cbar_wd.strip('%')) * 0.01
    if type(cbar_pad) == str: cbar_pad = float(cbar_pad.strip('%')) * 0.01
    if type(length) == str: length = float(length.strip('%')) * 0.01

    # check loc keyword
    if cbar_loc not in orientations.keys():
        print('ERROR\tadd_colorbar_toaxis: location keyword is wrong.')
        print('ERROR\tadd_colorbar_toaxis: it must be choosen from right, left, top or bottom.')
        return 0

    # color image
    if cim is not None:
        pass
    else:
        try:
            cim = ax.images[0] # assume the first one is a color map.
        except:
            print('ERROR\tadd_colorbar_toaxis: cannot find a color map.')
            return 0

    # set an inset axis
    # x0 and y0 of bounds are lower-left corner
    if cbar_loc == 'right':
        bounds = [1.0 + cbar_pad, 0., cbar_wd, length] # x0, y0, dx, dy
    elif cbar_loc == 'left':
        bounds = [0. - cbar_pad - cbar_wd, 0., cbar_wd, length]
    elif cbar_loc == 'top':
        bounds = [0., 1. + cbar_pad, length, cbar_wd]
    elif cbar_loc == 'bottom':
        bounds = [0., 0. - cbar_pad - cbar_wd, length, cbar_wd]

    # set a colorbar axis
    cax = ax.inset_axes(bounds, transform=ax.transAxes)
    cbar = plt.colorbar(cim, cax=cax, ticks=ticks, 
        orientation=orientations[cbar_loc], ticklocation=cbar_loc)
    cbar.set_label(cbarlabel)
    cbar.ax.tick_params(labelsize=fontsize, labelcolor=labelcolor, 
        color=tickcolor,)
    return cax, cbar


def add_scalebar(ax, scalebar: list, orientation='horizontal',
    loc: str = 'bottom right', barcolor: str = 'k', fontsize: float = 11.,
    lw: float = 2., zorder: float = 10., alpha: float = 1.,
    coordinate_mode = 'relative'):

    coords = {'bottom left': (0.1, 0.1),
            'bottom right': (0.9, 0.1),
            'top left': (0.1, 0.9),
            'top right': (0.9, 0.9),
            }
    offsets = {'bottom left': (0.05, -0.02),
            'bottom right': (-0.05, -0.02),
            'top left': (0.05, -0.02),
            'top right': (-0.05, -0.02),
            }

    if len(scalebar) == 5:
        barlength, bartext, loc, barcolor, fontsize = scalebar
        barlength = float(barlength)
        fontsize  = float(fontsize)

        if type(loc) == str:
            if loc in coords.keys():
                barx, bary = coords[loc]
                offx, offy = offsets[loc]
            else:
                print('CAUTION\tplot_beam: loc keyword is not correct.')
                return 0
        elif type(loc) == list:
            barx, bary = loc[0]
            txtx, txty = loc[1]
            offx = txtx - barx
            offy = txty - bary

        inv = ax.transLimits.inverted()
        if orientation == 'vertical':
            offy = 0.
            _, bary_l = ax.transLimits.transform(
                inv.transform((barx, bary)) - np.array([0., barlength*0.5]))
            _, bary_u = ax.transLimits.transform(
                inv.transform((barx, bary)) + np.array([0., barlength*0.5,]))
            ax.vlines(barx, bary_l, bary_u, 
                color=barcolor, lw=lw, zorder=zorder,
                transform=ax.transAxes, alpha=alpha)
            ax.text(barx + offx, bary + offy, bartext, fontsize=fontsize,
                color=barcolor, transform=ax.transAxes, 
                verticalalignment='center', horizontalalignment=loc.split(' ')[1])
        elif orientation == 'horizontal':
            offx = 0.
            barx_l, _ = ax.transLimits.transform(
                inv.transform((barx, bary)) - np.array([barlength*0.5, 0]))
            barx_u, _ = ax.transLimits.transform(
                inv.transform((barx, bary)) + np.array([barlength*0.5, 0]))
            ax.hlines(bary, barx_l, barx_u, 
                color=barcolor, lw=lw, zorder=zorder,
                transform=ax.transAxes, alpha=alpha)
            ax.text(barx + offx, bary + offy, bartext, fontsize=fontsize,
                color=barcolor, transform=ax.transAxes, 
                horizontalalignment='center', verticalalignment='top')
        else:
            print('ERROR\tadd_scalebar: orientation must be vertical or horizontal.')
            return 0
    elif len(scalebar) == 8:
        # read
        barx, bary, barlength, textx, texty, text, barcolor, barcsize = scalebar

        # to float
        barx      = float(barx)
        bary      = float(bary)
        barlength = float(barlength)
        textx     = float(textx)
        texty     = float(texty)

        if orientation == 'vertical':
            ax.vlines(barx, bary - barlength*0.5,bary + barlength*0.5, 
                color=barcolor, lw=lw, zorder=zorder, alpha=alpha)
        elif orientation == 'horizontal':
            ax.hlines(bary, barx - barlength*0.5,barx + barlength*0.5, 
                color=barcolor, lw=lw, zorder=zorder, alpha=alpha)
        else:
            print('ERROR\tadd_scalebar: orientation must be vertical or horizontal.')
            return 0

        ax.text(textx,texty,text,color=barcolor,fontsize=barcsize,horizontalalignment='center',verticalalignment='center')
    else:
        print ('ERROR\tadd_scalebar: scalebar must consist of 5 or 8 elements. Check scalebar.')



def change_aspect_ratio(ax, ratio, plottype='linear'):
    '''
    This function change aspect ratio of figure.
    Parameters:
        ax: ax (matplotlit.pyplot.subplots())
            Axes object
        ratio: float or int
            relative x axis width compared to y axis width.
    '''
    if plottype == 'linear':
        aspect = (1/ratio) *(ax.get_xlim()[1] - ax.get_xlim()[0]) / (ax.get_ylim()[1] - ax.get_ylim()[0])
    elif plottype == 'loglog':
        aspect = (1/ratio) *(np.log10(ax.get_xlim()[1]) - np.log10(ax.get_xlim()[0])) / (np.log10(ax.get_ylim()[1]) - np.log10(ax.get_ylim()[0]))
    elif plottype == 'linearlog':
        aspect = (1/ratio) *(ax.get_xlim()[1] - ax.get_xlim()[0]) / np.log10(ax.get_ylim()[1]/ax.get_ylim()[0])
    elif plottype == 'loglinear':
        aspect = (1/ratio) *(np.log10(ax.get_xlim()[1]) - np.log10(ax.get_xlim()[0])) / (ax.get_ylim()[1] - ax.get_ylim()[0])
    else:
        print('ERROR\tchange_aspect_ratio: plottype must be choosen from the types below.')
        print('   plottype can be linear or loglog.')
        print('   plottype=loglinear and linearlog is being developed.')
        return

    aspect = np.abs(aspect)
    aspect = float(aspect)
    ax.set_aspect(aspect)