import pylab
import numpy
from numpy import argwhere, zeros, hstack, append
import matplotlib.patches
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy.stats
import matplotlib as mpl

def fracs_inside_contours(x, y, contours):
    """
    Calculate the fraction of points x,y inside each contour level.
    """
    fracs = []
    xy = numpy.vstack([x,y]).transpose()
    for (icollection, collection) in enumerate(contours.collections):
        path = collection.get_paths()[0]
        frac = float(sum(path.contains_points(xy)))/len(x)
        fracs.append(frac)
    return fracs

def frac_label_contours(x, y, contours, format='%.2f'):
    """
    Label contours according to the fraction of points x,y inside.
    """
    fracs = fracs_inside_contours(x,y,contours)
    levels = contours.levels
    labels = {}
    for (level, frac) in zip(levels, fracs):
        labels[level] = format % frac
    contours.clabel(fmt=labels)

def contour_enclosing(x, y, fractions, xgrid, ygrid, zvals, 
                      axes, nstart = 200, 
                      *args, **kwargs):
    """
    Plot contours encompassing specified fractions of points x,y.
    """

    # Generate a large set of contours initially.
    contours = axes.contour(xgrid, ygrid, zvals, nstart,extend='both')
    # Set up fracs and levs for interpolation.
    levs = contours.levels
    fracs = numpy.array(fracs_inside_contours(x,y,contours))
    sortinds = numpy.argsort(fracs)
    levs = levs[sortinds]
    fracs = fracs[sortinds]
    # Find the levels that give the specified fractions.
    levels = scipy.interp(fractions, fracs, levs)

    # Remove the old contours from the graph.
    for coll in contours.collections:
        coll.remove()
    # Reset the contours
    contours.__init__(axes, xgrid, ygrid, zvals, levels, *args, **kwargs)
    return contours
    
def invert_boxcox(z,lam):
    return (1+z*lam)**(1.0/lam)

def plot2Ddist(variables,axeslist=None,maxvalues=None,histbinslist=[100, 100],
               labels=[r'$l$',r'$\sigma_f$'],scaleview=True,plotscatter=True,
               plothists=True,plotcontours=True,contourNGrid=200,bcx=True,
               contourFractions=[0.68, 0.95],labelcontours=True):
    """
    Plot contours of 2D distribution with marginal histograms:    
    Input:
    variables = 2d array with samples
    axes = optional pass axes to add plots to 
    maxvalues = values of hypers that maximise marginal posterior
    histbinlist = number of bins to use for the histogram
    labels = optional x and y axis labels
    scaleview = optional argument determines whether to set the axes limits according to the plotted data
    plotscatter, plothists, plotcontours = optional bool whether to plot the scatter, marginal histograms, and contours
    contourNGrid = int number of grid points to evaluate kde on
    contourFractions = optional % levels for contours
    labelcontours = bool whether to label the contours with the fraction of points enclosed
    """
    
    ### Set up figures and axes. ###
    if axeslist is None:
        fig1 = pylab.figure(figsize=(8,8))
        fig1.set_label('traces')
        ax1 = pylab.gca()

        divider = make_axes_locatable(ax1)
        ax2 = divider.append_axes("top", 1.5, pad=0.0, sharex=ax1)
        ax3 = divider.append_axes("right", 1.5, pad=0.0, sharey=ax1)
        
        for tl in (ax2.get_xticklabels() + ax2.get_yticklabels() + ax3.get_xticklabels() + ax3.get_yticklabels()):
            tl.set_visible(False)
        axeslist = (ax1, ax2, ax3)
    else:
        ax1, ax2, ax3 = axeslist
    
    #Do box-cox transform on data
    if bcx:
        x, mx = scipy.stats.boxcox(variables[0])
        y, my = scipy.stats.boxcox(variables[1])
    else:
        x = variables[0]
        y = variables[1]
    
    ### Plot the variables. ###
    # Plot 2D scatter of variables.
    if plotscatter:
        ax1.plot(x, y,ls='',marker=',',color='r',alpha=0.15)

    #Here we use kde to plot contours, might be better to use smoothing splines
    if plotcontours:
        style = {'linewidths':2.0, 'alpha':0.75,'zorder':10,'color':'k'}
        gkde = scipy.stats.gaussian_kde([x,y])
        xgrid, ygrid = numpy.mgrid[min(x):max(x):contourNGrid * 1j,min(y):max(y):contourNGrid * 1j]
        zvals = numpy.array(gkde.evaluate([xgrid.flatten(),ygrid.flatten()])).reshape(xgrid.shape)
        contours = contour_enclosing(x, y, contourFractions,xgrid, ygrid, zvals,ax1, **style)
   
    # Plot marginal histograms.
    if plothists:
        style = {'histtype':'step', 'normed':True, 'color':'k'}
        ax2.hist(x, histbinslist[0], **style)
        ax3.hist(y, histbinslist[1], orientation='horizontal', **style)

    # Plot lines to indicate max values.
    if maxvalues is not None:
        ax1.axvline(x=maxvalues[0], ls=':', c='k')
        ax1.axhline(y=maxvalues[1], ls=':', c='k')
        ax2.axvline(x=maxvalues[0], ls=':', c='k')
        ax3.axhline(y=maxvalues[1], ls=':', c='k')

    if scaleview:
        ax2.relim()
        ax3.relim()
        ax1.relim()
        ax2.autoscale_view(tight=True)
        ax3.autoscale_view(tight=True)
        ax1.autoscale_view(tight=True)
        ax2.set_ylim(bottom=0)
        ax3.set_xlim(left=0)

    #Set labels
    ax1.set_xlabel(labels[0],fontsize=35)
    ax1.set_ylabel(labels[1],fontsize=35)
        
    if plotcontours and labelcontours:
        frac_label_contours(x, y, contours)
    return fig1
    
def plot2Ddist2(x,y,ax1,contourNGrid=200,contourFractions=[0.68, 0.95],mode1='x',mode2='y'):
    """
    Plot contours of 2D distribution 
    """
    # Plot 2D scatter of variables.
    #ax1.plot(x, y,ls='',marker=',',color='r',alpha=0.15)

    #Here we use kde to plot contours, might be better to use smoothing splines
    style = {'linewidths':1.0, 'alpha':0.0,'zorder':10,'color':'k'}
    gkde = scipy.stats.gaussian_kde([x,y])
#    bw = gkde.covariance_factor()
#    gkde = scipy.stats.gaussian_kde([x,y],bw_method=bw/10.0)
    xgrid, ygrid = numpy.mgrid[min(x):max(x):contourNGrid * 1j,min(y):max(y):contourNGrid * 1j]
    zvals = numpy.array(gkde.evaluate([xgrid.flatten(),ygrid.flatten()])).reshape(xgrid.shape)
    contours = contour_enclosing(x, y, contourFractions,xgrid, ygrid, zvals,ax1, **style)
   
    p1 = contours.collections[1].get_paths()[0]
    v1 = p1.vertices  
    p2 = contours.collections[0].get_paths()[0]
    v2 = p2.vertices
    if (mode1 == 'fill'):
        t1 = v1[-1,0]
        v10 = v1[:,0]
        q1 = argwhere(v10 < t1)
        l1 = q1.size
        v10app = v10[q1]
        z1 = zeros([l1,1])
        v11app = hstack((z1,v10app))
        v1 = append(v1,v11app,axis=0)
    if (mode2=='fill'):
        t2 = v2[-1,0]
        v20 = v2[:,0]
        q2 = argwhere(v20 < t2)
        l2 = q2.size
        v20app = v20[q2]
        z2 = zeros([l2,1])
        v21app = hstack((z2,v20app))
        v2 = append(v2,v21app,axis=0)
    ax1.fill(v1[:,0],v1[:,1],'b',alpha=0.5)
    ax1.fill(v2[:,0],v2[:,1],'b',alpha=0.5)   
    #Label the contours        
    #frac_label_contours(x, y, contours)
   #ax1.fill()
#    ax1.relim()
    #ax1.autoscale_view(True,True,True)
    return contours