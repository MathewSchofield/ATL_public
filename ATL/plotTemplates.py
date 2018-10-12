"""
create functions for plot templates. DO NOT save the file in this function,
save after calling the function. Also show() and plot() the figure after calling function
"""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
import matplotlib.gridspec as gridspec


def hrPlot(title=None, labels=True, delta_scuti=True, delta_scuti_label=False):

    # plt.rc('font', size=24)
    # figHR, ax = plt.subplots(figsize=(13,10))
    plt.rc('font', size=14)
    figHR, ax = plt.subplots()
    ax.set_rasterized(True)

    # only add a title if one is given
    if title != None:
        plt.title(title)

    trackloc = '/home/mxs191/Desktop/MSc/data_files/02.09 diegos solar-like tracks from sim 2/track text files/'
    teff_08m, lum_08m = np.loadtxt(trackloc + 'm0.8.txt', skiprows = 2, usecols = (2,3), unpack = True)
    teff_10m, lum_10m = np.loadtxt(trackloc + 'm1.0.txt', skiprows = 2, usecols = (2,3), unpack = True)
    teff_12m, lum_12m = np.loadtxt(trackloc + 'm1.2.txt', skiprows = 2, usecols = (2,3), unpack = True)
    teff_14m, lum_14m = np.loadtxt(trackloc + 'm1.4.txt', skiprows = 2, usecols = (2,3), unpack = True)
    teff_16m, lum_16m = np.loadtxt(trackloc + 'm1.6.txt', skiprows = 2, usecols = (2,3), unpack = True)
    teff_18m, lum_18m = np.loadtxt(trackloc + 'm1.8.txt', skiprows = 2, usecols = (2,3), unpack = True)
    teff_20m, lum_20m = np.loadtxt(trackloc + 'm2.0.txt', skiprows = 2, usecols = (2,3), unpack = True)


    figt1 = plt.plot(teff_08m[940:5000], lum_08m[940:5000], color='k', linewidth=2.0)
    figt2 = plt.plot(teff_10m[967:5000], lum_10m[967:5000], color='k', linewidth=2.0)
    figt3 = plt.plot(teff_12m[985:5000], lum_12m[985:5000], color='k', linewidth=2.0)
    figt3 = plt.plot(teff_14m[1040:5000], lum_14m[1040:5000], color='k', linewidth=2.0)
    figt3 = plt.plot(teff_16m[1100:5000], lum_16m[1100:5000], color='k', linewidth=2.0)
    figt3 = plt.plot(teff_18m[1090:5000], lum_18m[1090:5000], color='k', linewidth=2.0)
    figt3 = plt.plot(teff_20m[1093:5000], lum_20m[1093:5000], color='k', linewidth=2.0)

    if labels==True:
        plt.annotate(r'$0.8\textrm{M}_{\odot}$', xy = (5164, 0.45))
        plt.annotate(r'$1.0\textrm{M}_{\odot}$', xy = (5700, 0.91))
        plt.annotate(r'$1.2\textrm{M}_{\odot}$', xy = (6156, 2.10))
        plt.annotate(r'$1.4\textrm{M}_{\odot}$', xy = (6550, 3.89))
        plt.annotate(r'$1.6\textrm{M}_{\odot}$', xy = (7003, 6.57))
        plt.annotate(r'$1.8\textrm{M}_{\odot}$', xy = (7410, 10.6))
        plt.annotate(r'$2.0\textrm{M}_{\odot}$', xy = (7578, 17.5))

    if delta_scuti==True:
        #using tred from p8 eqn 8 of Bill's 'predicting the detectability...' paper
        lum_array = np.linspace(-2.0, 3.0, 1000) #values for log(L) in solar units
        lum_array = 10.0**lum_array #values of L in solar units
        tred = 8907.0 * ((lum_array)**-0.093)
        plt.plot(tred, lum_array, '--', color = 'k')

    if delta_scuti_label == True:
        plt.annotate('$\delta$-Scuti instability strip', xy = (7420, 20.7), rotation=30.5, fontsize = size)

    plt.xlim(7700,4300)
    plt.ylim(0.3,50)
    plt.yscale('log')
    plt.xlabel(r'$T_{\textrm{eff}}$ / K')
    plt.gca().set_ylabel(r'$L$ / $\textrm{L}_{\odot}$')
    #ax.set_yticks([1,2,3,4,5,6,7,8,9,10,20,30,40,50])
    #ax.get_yaxis().set_major_formatter(tkr.ScalarFormatter())

    return figHR, ax


# legend for scatter points
def legendScatter(num, loc='lower left'):

    leg = plt.legend(loc=loc, scatterpoints=1, handlelength=0.5, labelspacing=0.5)
    size = 100

    if num == 1:
        leg.legendHandles[0]._sizes = [size]
    if num == 2:
        leg.legendHandles[0]._sizes = [size]
        leg.legendHandles[1]._sizes = [size]
    if num == 3:
        leg.legendHandles[0]._sizes = [size]
        leg.legendHandles[1]._sizes = [size]
        leg.legendHandles[2]._sizes = [size]
    if num == 4:
        leg.legendHandles[0]._sizes = [size]
        leg.legendHandles[1]._sizes = [size]
        leg.legendHandles[2]._sizes = [size]
        leg.legendHandles[3]._sizes = [size]


def histPlot(title=None, xaxis=None, yaxis=None):

    plt.rc('font', size=14)
    figHist, ax = plt.subplots()

    # only add a title if one is given
    if title != None:
        plt.title(title)
    if xaxis != None:
        plt.xlabel(xaxis)
    if yaxis != None:
        plt.ylabel(yaxis)

    width=2.0
    plttype='step'
    return figHist, ax, width, plttype


def generalPlot(title=None, xaxis=None, yaxis=None):

    plt.rc('font', size=30)
    gfig, ax = plt.subplots(figsize=(15.0, 16.0))

    if title != None:
        plt.title(title)
    if xaxis != None:
        plt.xlabel(xaxis)
    if yaxis != None:
        plt.ylabel(yaxis)

    width=2 # linewidth
    size=100 # scatter point size

    return gfig, ax, width, size


# 4 subplot panels
def fourSubs():


    plt.rc('font', size=30)
    gfig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey='row', figsize=(23.0, 14.0))

    width=2 # linewidth
    size=100 # scatter point size

    return gfig, ax1, ax2, ax3, ax4


# 2 subplots, stacked vertically
def twoSubs(xaxis=None, yaxis=None):

    plt.rc('font', size=30)
    f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(23., 14.))

    if xaxis != None:
        plt.xlabel(xaxis)
    if yaxis != None:
        plt.ylabel(yaxis)

    return f, ax1, ax2


def twoSubplots():

    plt.rc('font', size=30)
    gs = gridspec.GridSpec(2, 1, height_ratios=[4,1])
    fig = plt.figure(figsize = (16, 12))

    ax = fig.add_subplot(gs[0])


    ax1 = fig.add_subplot(gs[1], sharex=ax)

    return gs, fig, ax, ax1












#
