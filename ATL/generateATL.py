"""
Python 2

Produce the ATL using either the Gaia (TGAS or DR2) or XHIP archives
to make different versions of the ATL. These versions are then merged in the combine()
function.

Before using the catalogues in this code:
Download the DR2 and XHIP catalogues from
https://figshare.com/s/e62b08021fba321175d6
(TGAS catalogue also available at this link)

References:
    (1)  Anderson+ 2012. catalogue at: http://vizier.cfa.harvard.edu/viz-bin/VizieR?-source=V/137D
    (2)  Houdek, G., 1999 'On the location of the instability strip'
    (3)  Chaplin, W., 2011 'PREDICTING THE DETECTABILITY OF OSCILLATIONS IN
         SOLAR-TYPE STARS OBSERVED BY KEPLER'
    (4)  Torres, G., 2010, ApJ, 140, 1158
    (5)  Pijpers, F., 2003
    (6)  Flower, P. J., 1996, ApJ, 469, 355
    (7)  Campante et al (2016)
    (8)  https://filtergraph.com/tess
    (9)  https://github.com/jobovy/mwdust
    (10) http://chris.friedline.net/2015-12-15-rutgers/lessons/python2/04-merging-data.html
    (11) http://astronomy.swin.edu.au/cosmos/I/Interstellar+Reddening
    (12) 'The relationship between infrared, optical, and ultraviolet extinction'
    	 by J cardelli (1989)
"""

import warnings
warnings.simplefilter(action = "ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import timeit
import sys
import os
import copy
import tarfile
from scipy import stats
from plotTemplates import hrPlot, histPlot, generalPlot, legendScatter
from DR import bv2teff, gal2ecl, Teff2bc2lum, equa_2Ecl_orGal, tess_field_only, globalDetections

# for TESS observation time
from astropysics.coords.coordsys import EquatorialCoordinatesEquinox, \
EclipticCoordinatesEquinox, GalacticCoordinates
from itertools import groupby
from operator import itemgetter


def loadData(fpath, fname, sep=None, nrows=None, print_headers=True, ctl=False):
    """ Load the catalogues used to generate the ATL from. """

    extension = fname.split('.')[-1]
    if ctl == True:
        data = pd.read_csv(fpath + fname, usecols=['TICID', 'TESSmag', 'HipNo'], delimiter=r'\s+')
        data = data[data['HipNo'] > 0] # remove non-Hipparcos stars in the CTL
        data.rename(columns={'HipNo':'HIP', 'TESSmag':'Tmag', 'Imag':'ctlImag'}, inplace=True)
        data = data.drop_duplicates('HIP') # remove duplicate entries in the ctl
    elif sep != None:
        data = pd.read_csv(fpath + fname, sep=sep, nrows=nrows)
    elif extension == 'tsv':
        data = pd.read_csv(fpath + fname, sep='\t')
    elif extension == 'csv':
        data = pd.read_csv(fpath + fname)

    if print_headers==True:
        print fname.split('.')[0], list(data)
    return data

def reddening(data, file_loc):
    """ calculate reddening using the mwdust Combined15 dustmap (Green et al (2015)). """
    print data.shape, 'before reddening'

    # check if the reddening file exists
    if (os.path.isfile(file_loc + 'XHIP_reddening.txt') == True):
        print 'reddening file located.'
        reds = pd.read_csv(file_loc + 'XHIP_reddening.txt')

        # if it does exist, add extinction (Av) to the dataframe
        if len(reds) == len(data):
            reds = reds.round({'GLon': 4, 'GLat': 4})
            data = data.round({'GLon': 4, 'GLat': 4})
            data = pd.merge(data, reds[list(['GLon', 'GLat', 'E(B-V)', 'Av'])], how='inner')

            # define the extinction coefficient in I band.
            # from (12) eqn 1 and table 3 a(x) + b(x)/Rv value for Imag
            data['Ai'] = data['Av'] * 0.479
            print data.shape, 'after reddening'

        else:
            print 'the reddening file is a different length to XHIP! recalculate reddening.'
            # calculate reddening values using the dataframe inputs
            # mwdust info at (9). E(B-V) TO Av info at (11)
            import mwdust
            combined15 = mwdust.Combined15(filter='2MASS H')
            a = timeit.default_timer()
            print 'calculating extinction coefficients...'
            GLon, GLat, Dist = np.split(data[['GLon', 'GLat', 'Dist']].as_matrix(), 3, 1)
            comb15 = np.full(len(GLon), -5.)
            Av = np.full(len(GLon), -5.)

            # calculate extinction using galactic longitude, latitude and distance (in pc)
            for i in range(len(GLon)):
                comb15[i] = combined15(float(GLon[i]),float(GLat[i]),float(Dist[i]))
                if (i%1000.==0): print i

            Av = comb15 * 3.2 # convert from reddening E(B-V) to extinction Av
            reds = np.c_[GLon, GLat, Dist, comb15, Av]
            h = 'GLon,GLat,Dist,E(B-V),Av'
            np.savetxt(file_loc + 'XHIP_reddening.txt', reds, comments='', header=h, delimiter=',')
            b = timeit.default_timer()
            print b-a, 'seconds'

    else:
        print 'calculating reddening.'
        # calculate reddening values using the dataframe inputs
        # mwdust info at (9). E(B-V) TO Av info at (11)
        import mwdust
        combined15 = mwdust.Combined15(filter='2MASS H')
        a = timeit.default_timer()
        print 'calculating extinction coefficients...'
        GLon, GLat, Dist = np.split(data[['GLon', 'GLat', 'Dist']].as_matrix(), 3, 1)
        comb15 = np.full(len(GLon), -5.)
        Av = np.full(len(GLon), -5.)

        # calculate extinction using galactic longitude, latitude and distance (in pc)
        for i in range(len(GLon)):
            comb15[i] = combined15(float(GLon[i]),float(GLat[i]),float(Dist[i]))
            if (i%1000.==0): print i # only print the code's progress every 100 stars

        Av = comb15 * 3.2 # convert from reddening E(B-V) to extinction Av
        reds = np.c_[GLon, GLat, Dist, comb15, Av]
        h = 'GLon,GLat,Dist,E(B-V),Av'
        np.savetxt(file_loc + 'XHIP_reddening.txt', reds, comments='', header=h, delimiter=',')
        b = timeit.default_timer()
        print b-a, 'seconds'
        sys.exit()

    return data

def Teff2lum(teff, parallax, d, vmag, Av=0):
    """ Calculate luminosities from effective temperatures. From F Pijpers+ (2003).
    BCv values from Flower 1996 polynomials presented in Torres+ (2010).
    Av is a keword argument. If reddening values not available, ignore it's effect.
    Equation modified to include distance instead of parallax. """

    lteff = np.log10(teff)
    BCv = np.full(len(lteff), -100.5)

    BCv[lteff<3.70] = (-0.190537291496456*10.0**5) + \
    (0.155144866764412*10.0**5*lteff[lteff<3.70]) + \
    (-0.421278819301717*10.0**4.0*lteff[lteff<3.70]**2.0) + \
    (0.381476328422343*10.0**3*lteff[lteff<3.70]**3.0)

    BCv[(3.70<lteff) & (lteff<3.90)] = (-0.370510203809015*10.0**5) + \
    (0.385672629965804*10.0**5*lteff[(3.70<lteff) & (lteff<3.90)]) + \
    (-0.150651486316025*10.0**5*lteff[(3.70<lteff) & (lteff<3.90)]**2.0) + \
    (0.261724637119416*10.0**4*lteff[(3.70<lteff) & (lteff<3.90)]**3.0) + \
    (-0.170623810323864*10.0**3*lteff[(3.70<lteff) & (lteff<3.90)]**4.0)

    BCv[lteff>3.90] = (-0.118115450538963*10.0**6) + \
    (0.137145973583929*10.0**6*lteff[lteff > 3.90]) + \
    (-0.636233812100225*10.0**5*lteff[lteff > 3.90]**2.0) + \
    (0.147412923562646*10.0**5*lteff[lteff > 3.90]**3.0) + \
    (-0.170587278406872*10.0**4*lteff[lteff > 3.90]**4.0) + \
    (0.788731721804990*10.0**2*lteff[lteff > 3.90]**5.0)

    Mbol_sol = 4.73
    #lum = 10**(4.0 + 0.4*Mbol_sol - 2.0*np.log10(parallax) - 0.4*(vmag - Av + BCv))
    llum = 0.4*Mbol_sol - 0.4*(vmag-Av+BCv) + 2.*np.log10(d/10.) # in solar units
    lum = 10**llum

    return lum

def Parameters(data, reddening_floc):
    """ Calculate Teff and Luminosity from (B-V) and parallax. Correct B-V using
    reddening E(B-V). Make Teff, Lum, (B-V) and parallax cuts. """

    data = data.apply(pd.to_numeric, errors='coerce') # make array all floats
    data = data[data.Plx > 0] # remove stars with negative parallaxes
    print data.shape, 'after removing -VE Plx'

    #huber(data); print '1.'  # check if 'celebrity' stars are in the list

    # mwdust needs distances in Kpc for all stars. The parallax in XHIP is in mas
    data['Dist'] = 1. / (data['Plx'])

    data = reddening(data, reddening_floc) # use mwdust to calculate reddening

    # remove stars with fractional parallax uncertainties > 0.5
    data = data[(data.e_Plx / data.Plx) < 0.5]
    print data.shape, 'after Frac Plx cut'

    #huber(data); print '2.'  # check if 'celebrity' stars are in the list

    # account for reddening E(B-V) and extinction Av
    data['Vmag_reddened'] = data['Vmag']
    data['Imag_reddened'] = data['Imag']
    data['B-V_observed'] = data['B-V']
    data['B-V'] = data['B-V_observed'] - data['E(B-V)']
    data['Vmag'] = data['Vmag_reddened'] - data['Av']
    data['Imag'] = data['Imag_reddened'] - data['Ai']
    if not saveall:  data = data.drop(['Av', 'Ai'], axis=1) # remove extinction from array

    # calculate teff from (4) table 2. Applies to MS, SGB and giant stars
    # B-V limits from (6) fig 5
    data = data[(data['B-V'] > -0.4) & (data['B-V'] < 1.7)]
    data['teff'] = bv2teff(data['B-V'])

    #huber(data); print '3.'  # check if 'celebrity' stars are in the list

    # Write over the luminosities given in the XHIP from (1); calculate L from parallax instead.
    # Luminosity relation taken from (5). BCv values from (6), polynomials presented in (4).
    data['Lum'] = Teff2lum(data['teff'].as_matrix(), data['Plx'].as_matrix(),\
        data['Dist'].as_matrix()*1e3, data['Vmag'].as_matrix())

    #wg4(data)  # compare XHIP list to WG$ list
    #sys.exit()

    # make Teff and luminosity cuts to the data
    data = data[(data['teff'] < 7700) & (data['teff'] > 4300) &
        (data['Lum'] > 0.3) & (data['Lum'] < 50)]
    print data.shape, 'after Teff/L cuts'

    #huber(data); print '4.'  # check if 'celebrity' stars are in the list

    # the stars that lie close to the ecliptic plane are unobsereved by TESS
    data['ELon'], data['ELat'] = equa_2Ecl_orGal(ra=data['RAJ2000'].as_matrix(),\
    dec=data['DEJ2000'].as_matrix(), ecl=True)
    #huber(data); print '5.'  # check if 'celebrity' stars are in the list
    data = data[(data['ELat']<=-6.) | (data['ELat']>=6.)]
    print data.shape, 'after Ecl plane cut'

    #huber(data); print '6.'  # check if 'celebrity' stars are in the list

    return data

def regions(data):
    """ Break up the HR diagram into 3 regions in order to rank targets. (depreciated) """

    numaxs = np.array((240., 600.)) # numax limits for region 2 (mu Hz)

    # the corresponding radius and lum values at numax limits
    data['region2_rad'] = ( (data['teff']/5777.)**(-0.92) * (numaxs[0]/3090.) )**(1./-1.85)
    data['region1_rad'] = ( (data['teff']/5777.)**(-0.92) * (numaxs[1]/3090.) )**(1./-1.85)
    data['region1_lum'] = data['region1_rad']**2 * (data['teff']/5777.)**4
    data['region2_lum'] = data['region2_rad']**2 * (data['teff']/5777.)**4

    # define the corresponding temperatures for the catalogue stars
    data['hedge'] = 8650.0 * ((data['Lum'])**-0.093)
    data['tred'] = 8907.0 * ((data['Lum'])**-0.093)

    # split the data into separate regions
    # give each region an identifier (1,2 or 3)
    data['region'] = '0' # give every star this identifier. Write over this value if a star lies in a region
    data['region'][(data['Lum'] < data['region1_lum']) & (data['teff'] < data['hedge'])] = '1'
    data['region'][(data['Lum'] > data['region1_lum']) & (data['Lum'] < data['region2_lum']) & (data['teff'] < data['hedge'])] = '2'
    data['region'][(data['teff'] > data['hedge']) & (data['teff'] < data['tred']) & (data['Lum'] < data['region2_lum'])] = '3'
    print 'regions:', '\n', data['region'].value_counts() # print how many stars are in each region

    # remove the stars outside of a region
    data = data[data['region'] != '0']
    print data.shape, 'after removing region=0'
    data = data.drop(['hedge', 'region1_rad', 'region1_lum', 'region2_rad', 'region2_lum'], axis=1)

    return data

def regions2(data):
    """ Remove 2-minute cadence stars outside the red-edge, and above a numax cut-off frequency. """

    data['tred'] = 8907.0 * ((data['Lum'])**-0.093)

    cut = 240.
    data['region2_rad'] = ( (data['teff']/5777.)**(-0.92) * (cut/3090.) )**(1./-1.85)
    data['region2_lum'] = data['region2_rad']**2 * (data['teff']/5777.)**4

    data['region'] = '0'
    data['region'][(data['teff'] < data['tred']) & (data['Lum'] < data['region2_lum'])] = '2'
    data = data[data['region'] != '0']
    print data.shape, 'after removing region=0'
    data = data.drop(['region2_rad', 'region2_lum'], axis=1)
    return data

def seismicParameters(xhip, teff, lum):
    """ Calculate seismic parameters that are used in globalDetections() """

    # solar parameters
    teff_solar = 5777.0 # Kelvin
    teffred_solar = 8907.0 #in Kelvin
    numax_solar = 3090.0 # in micro Hz
    dnu_solar = 135.1 # in micro Hz

    cadence = 120 # in s
    vnyq = (1.0 / (2.0*cadence)) * 10**6 # in micro Hz
    rad = lum**0.5 * ((teff/teff_solar)**-2) # Steffan-Boltzmann law
    numax = numax_solar*(rad**-1.85)*((teff/teff_solar)**0.92) # from (7)

    return cadence, vnyq, rad, numax, teff_solar, teffred_solar, numax_solar, dnu_solar

def ranking(data):
    """ Sort the data by Ic and Pdet value, to investigate how the detection recipe
    can be replicated with observational data only """

    # define the ranking columns. for each column, rank based on each region in temporary arrays,
    # then merge the temp arrays together afterwards
    data['PdetRank_fixedBeta'] = -99
    data['PdetRank_varyBeta'] = -99
    data['SNRrank_fixedBeta'] = -99
    data['SNRrank_varyBeta'] = -99
    data['IcRank'] = -99
    data['PdetRank_SNR'] = -99

    temp1 = data[data['region']=='1'].sort_values(by='Pdet_fixedBeta', axis=0, ascending=True, kind='quicksort')
    temp1['PdetRank_fixedBeta'] = temp1['Pdet_fixedBeta'].rank(ascending=False)
    temp2 = data[data['region']=='2'].sort_values(by='Pdet_fixedBeta', axis=0, ascending=True, kind='quicksort')
    temp2['PdetRank_fixedBeta'] = temp2['Pdet_fixedBeta'].rank(ascending=False)
    temp3 = data[data['region']=='3'].sort_values(by='Pdet_fixedBeta', axis=0, ascending=True, kind='quicksort')
    temp3['PdetRank_fixedBeta'] = temp3['Pdet_fixedBeta'].rank(ascending=False)
    data = data.sort_values(by='Pdet_fixedBeta', axis=0, ascending=True, kind='quicksort')
    data = pd.concat([temp1, temp2, temp3])

    temp1 = data[data['region']=='1'].sort_values(by='Pdet_varyBeta', axis=0, ascending=True, kind='quicksort')
    temp1['PdetRank_varyBeta'] = temp1['Pdet_varyBeta'].rank(ascending=False)
    temp2 = data[data['region']=='2'].sort_values(by='Pdet_varyBeta', axis=0, ascending=True, kind='quicksort')
    temp2['PdetRank_varyBeta'] = temp2['Pdet_varyBeta'].rank(ascending=False)
    temp3 = data[data['region']=='3'].sort_values(by='Pdet_varyBeta', axis=0, ascending=True, kind='quicksort')
    temp3['PdetRank_varyBeta'] = temp3['Pdet_varyBeta'].rank(ascending=False)
    data = pd.concat([temp1, temp2, temp3])
    data = data.sort_values(by='Pdet_varyBeta', axis=0, ascending=True, kind='quicksort')

    temp1 = data[data['region']=='1'].sort_values(by='SNR_fixedBeta', axis=0, ascending=True, kind='quicksort')
    temp1['SNRrank_fixedBeta'] = temp1['SNR_fixedBeta'].rank(ascending=False)
    temp2 = data[data['region']=='2'].sort_values(by='SNR_fixedBeta', axis=0, ascending=True, kind='quicksort')
    temp2['SNRrank_fixedBeta'] = temp2['SNR_fixedBeta'].rank(ascending=False)
    temp3 = data[data['region']=='3'].sort_values(by='SNR_fixedBeta', axis=0, ascending=True, kind='quicksort')
    temp3['SNRrank_fixedBeta'] = temp3['SNR_fixedBeta'].rank(ascending=False)
    data = pd.concat([temp1, temp2, temp3])
    data = data.sort_values(by='SNR_fixedBeta', axis=0, ascending=True, kind='quicksort')

    temp1 = data[data['region']=='1'].sort_values(by='SNR_varyBeta', axis=0, ascending=True, kind='quicksort')
    temp1['SNRrank_varyBeta'] = temp1['SNR_varyBeta'].rank(ascending=False)
    temp2 = data[data['region']=='2'].sort_values(by='SNR_varyBeta', axis=0, ascending=True, kind='quicksort')
    temp2['SNRrank_varyBeta'] = temp2['SNR_varyBeta'].rank(ascending=False)
    temp3 = data[data['region']=='3'].sort_values(by='SNR_varyBeta', axis=0, ascending=True, kind='quicksort')
    temp3['SNRrank_varyBeta'] = temp3['SNR_varyBeta'].rank(ascending=False)
    data = pd.concat([temp1, temp2, temp3])
    data = data.sort_values(by='SNR_varyBeta', axis=0, ascending=True, kind='quicksort')

    temp1 = data[data['region']=='1'].sort_values(by='Imag', axis=0, ascending=True, kind='quicksort')
    temp1['IcRank'] = temp1['Imag'].rank()
    temp2 = data[data['region']=='2'].sort_values(by='Imag', axis=0, ascending=True, kind='quicksort')
    temp2['IcRank'] = temp2['Imag'].rank()
    temp3 = data[data['region']=='3'].sort_values(by='Imag', axis=0, ascending=True, kind='quicksort')
    temp3['IcRank'] = temp3['Imag'].rank()
    data = pd.concat([temp1, temp2, temp3])
    data = data.sort_values(by='Imag', axis=0, ascending=True, kind='quicksort')

    #print data.shape
    #print data[['Imag', 'IcRank']].head(5)
    #sys.exit()


    """ this makes PdetRank_fixedBeta """
    temp1 = data[data['region']=='1'].sort_values(by=['Pdet_fixedBeta'], axis=0, ascending=True, kind='quicksort')

    # create another temporary array, temp_top, which stores the Pdet=1 stars.
    # Sort this by SNR and rank on index. Replace 'the Pdet=1' part of temp1 with these rows.
    temp_top = temp1[temp1['Pdet_fixedBeta']>=0.999999999953]
    temp_top = temp_top.sort_values(by='SNR_fixedBeta', ascending=False, kind='quicksort')
    temp_top['RANK'] = -99 # define this data column, to replace PdetRank_fixedBeta with
    temp_top['RANK'] = temp_top.reset_index().index.values + 1 # set this rank as simply the index number

    # swap the degenerate Pdet ranks for ranks based on SNR
    temp1['PdetRank_fixedBeta'][temp1['Pdet_fixedBeta']>=0.999999999953] = temp_top['RANK']
    temp1 = temp1.sort_values(by='PdetRank_fixedBeta', ascending=False, kind='quicksort')
    #temp_top[['Pdet_fixedBeta', 'SNR_fixedBeta', 'PdetRank_fixedBeta', 'RANK']].to_csv(saveLoc + 'temp_top.csv', index=False)
    temp_top = []

    # check that the imporved PdetRank ranking has worked correctly
    #temp1[['Pdet_fixedBeta', 'SNR_fixedBeta', 'PdetRank_fixedBeta', 'region', 'PdetRank_SNR']].to_csv(saveLoc + 'temp1.csv', index=False)
    #fig, ax, width, size = generalPlot(xaxis=r'$P_{\rm det}$ Rank', yaxis='SNR')
    #plt.scatter(temp1['PdetRank_fixedBeta'], temp1['SNR_fixedBeta'], color='r', edgecolors='none')
    #plt.show()


    temp2 = data[data['region']=='2'].sort_values(by=['Pdet_fixedBeta'], axis=0, ascending=True, kind='quicksort')

    # create another temporary array, temp_top, which stores the Pdet=1 stars.
    # Sort this by SNR and rank on index. Replace 'the Pdet=1' part of temp2 with these rows.
    temp_top = temp2[temp2['Pdet_fixedBeta']>=0.999999999953]
    temp_top = temp_top.sort_values(by='SNR_fixedBeta', ascending=False, kind='quicksort')
    temp_top['RANK'] = -99
    temp_top['RANK'] = temp_top.reset_index().index.values + 1

    # swap the degenerate Pdet ranks for ranks based on SNR
    temp2['PdetRank_fixedBeta'][temp2['Pdet_fixedBeta']>=0.999999999953] = temp_top['RANK']
    temp2 = temp2.sort_values(by='PdetRank_fixedBeta', ascending=False, kind='quicksort')
    #temp_top[['Pdet_fixedBeta', 'SNR_fixedBeta', 'PdetRank_fixedBeta', 'RANK']].to_csv(saveLoc + 'temp_top.csv', index=False)
    temp_top = []

    # check that the imporved PdetRank ranking has worked correctly
    #temp2[['Pdet_fixedBeta', 'SNR_fixedBeta', 'PdetRank_fixedBeta', 'region', 'PdetRank_SNR']].to_csv(saveLoc + 'temp2.csv', index=False)
    #fig, ax, width, size = generalPlot(xaxis=r'$P_{\rm det}$ Rank', yaxis='SNR')
    #plt.scatter(temp2['PdetRank_fixedBeta'], temp2['SNR_fixedBeta'], edgecolors='none')
    #plt.show()
    #fig.savefig(saveLoc + 'ranking_region2.pdf', dpi=120)
    #plt.close()


    temp3 = data[data['region']=='3'].sort_values(by=['Pdet_fixedBeta'], axis=0, ascending=True, kind='quicksort')

    # create another temporary array, temp_top, which stores the Pdet=1 stars.
    # Sort this by SNR and rank on index. Replace 'the Pdet=1' part of temp3 with these rows.
    temp_top = temp3[temp3['Pdet_fixedBeta']>=0.999999999953]
    temp_top = temp_top.sort_values(by='SNR_fixedBeta', ascending=False, kind='quicksort')
    temp_top['RANK'] = -99 # define this data column, to replace PdetRank_fixedBeta with
    temp_top['RANK'] = temp_top.reset_index().index.values + 1 # set this rank as simply the index number

    # swap the degenerate Pdet ranks for ranks based on SNR
    temp3['PdetRank_fixedBeta'][temp3['Pdet_fixedBeta']>=0.999999999953] = temp_top['RANK']
    temp3 = temp3.sort_values(by='PdetRank_fixedBeta', ascending=False, kind='quicksort')
    #temp_top[['Pdet_fixedBeta', 'SNR_fixedBeta', 'PdetRank_fixedBeta', 'RANK']].to_csv(saveLoc + 'temp_top.csv', index=False)
    temp_top = []

    # check that the imporved PdetRank ranking has worked correctly
    #temp3[['Pdet_fixedBeta', 'SNR_fixedBeta', 'PdetRank_fixedBeta', 'region', 'PdetRank_SNR']].to_csv(saveLoc + 'temp3.csv', index=False)


    data = pd.concat([temp1, temp2, temp3])
    data = data.sort_values(by='PdetRank_fixedBeta', axis=0, ascending=True, kind='quicksort')
    #data[['Pdet_fixedBeta', 'SNR_fixedBeta', 'PdetRank_fixedBeta', 'region', 'PdetRank_SNR']].to_csv(saveLoc + 'temps.csv', index=False)
    data = data.drop(['PdetRank_SNR'], axis=1) # this is incorporated into 'PdetRank_fixedBeta'
    temp1 = []
    temp2 = []
    temp3 = []
    """ this makes PdetRank_fixedBeta """


    """ this makes PdetRank_varyBeta """
    data['PdetRank_SNR'] = -99
    temp1 = data[data['region']=='1'].sort_values(by=['Pdet_varyBeta'], axis=0, ascending=True, kind='quicksort')

    # create another temporary array, temp_top, which stores the Pdet=1 stars.
    # Sort this by SNR and rank on index. Replace 'the Pdet=1' part of temp1 with these rows.
    temp_top = temp1[temp1['Pdet_varyBeta']>=0.999999999953]
    temp_top = temp_top.sort_values(by='SNR_fixedBeta', ascending=False, kind='quicksort')
    temp_top['RANK'] = -99 # define this data column, to replace PdetRank_varyBeta with
    temp_top['RANK'] = temp_top.reset_index().index.values + 1 # set this rank as simply the index number

    # swap the degenerate Pdet ranks for ranks based on SNR
    temp1['PdetRank_varyBeta'][temp1['Pdet_varyBeta']>=0.999999999953] = temp_top['RANK']
    temp1 = temp1.sort_values(by='PdetRank_varyBeta', ascending=False, kind='quicksort')
    #temp_top[['Pdet_varyBeta', 'SNR_varyBeta', 'PdetRank_varyBeta', 'RANK']].to_csv(saveLoc + 'temp_top.csv', index=False)
    temp_top = []

    # check that the imporved PdetRank ranking has worked correctly
    #temp1[['Pdet_varyBeta', 'SNR_varyBeta', 'PdetRank_varyBeta', 'region', 'PdetRank_SNR']].to_csv(saveLoc + 'temp1.csv', index=False)
    #fig, ax, width, size = generalPlot(xaxis=r'$P_{\rm det}$ Rank', yaxis='SNR')
    #plt.scatter(temp1['PdetRank_varyBeta'], temp1['SNR_varyBeta'], color='r', edgecolors='none')
    #plt.show()


    temp2 = data[data['region']=='2'].sort_values(by=['Pdet_varyBeta'], axis=0, ascending=True, kind='quicksort')

    # create another temporary array, temp_top, which stores the Pdet=1 stars.
    # Sort this by SNR and rank on index. Replace 'the Pdet=1' part of temp2 with these rows.
    temp_top = temp2[temp2['Pdet_varyBeta']>=0.999999999953]
    temp_top = temp_top.sort_values(by='SNR_varyBeta', ascending=False, kind='quicksort')
    temp_top['RANK'] = -99
    temp_top['RANK'] = temp_top.reset_index().index.values + 1

    # swap the degenerate Pdet ranks for ranks based on SNR
    temp2['PdetRank_varyBeta'][temp2['Pdet_varyBeta']>=0.999999999953] = temp_top['RANK']
    temp2 = temp2.sort_values(by='PdetRank_varyBeta', ascending=False, kind='quicksort')
    #temp_top[['Pdet_varyBeta', 'SNR_varyBeta', 'PdetRank_varyBeta', 'RANK']].to_csv(saveLoc + 'temp_top.csv', index=False)
    temp_top = []

    # check that the imporved PdetRank ranking has worked correctly
    #temp2[['Pdet_varyBeta', 'SNR_varyBeta', 'PdetRank_varyBeta', 'region', 'PdetRank_SNR']].to_csv(saveLoc + 'temp2.csv', index=False)
    #fig, ax, width, size = generalPlot(xaxis=r'$P_{\rm det}$ Rank', yaxis='SNR')
    #plt.scatter(temp2['PdetRank_varyBeta'], temp2['SNR_varyBeta'], edgecolors='none')
    #plt.show()
    #fig.savefig(saveLoc + 'ranking_region2.pdf', dpi=120)
    #plt.close()


    temp3 = data[data['region']=='3'].sort_values(by=['Pdet_varyBeta'], axis=0, ascending=True, kind='quicksort')

    # create another temporary array, temp_top, which stores the Pdet=1 stars.
    # Sort this by SNR and rank on index. Replace 'the Pdet=1' part of temp3 with these rows.
    temp_top = temp3[temp3['Pdet_varyBeta']>=0.999999999953]
    temp_top = temp_top.sort_values(by='SNR_varyBeta', ascending=False, kind='quicksort')
    temp_top['RANK'] = -99 # define this data column, to replace PdetRank_varyBeta with
    temp_top['RANK'] = temp_top.reset_index().index.values + 1 # set this rank as simply the index number

    # swap the degenerate Pdet ranks for ranks based on SNR
    temp3['PdetRank_varyBeta'][temp3['Pdet_varyBeta']>=0.999999999953] = temp_top['RANK']
    temp3 = temp3.sort_values(by='PdetRank_varyBeta', ascending=False, kind='quicksort')
    #temp_top[['Pdet_varyBeta', 'SNR_varyBeta', 'PdetRank_varyBeta', 'RANK']].to_csv(saveLoc + 'temp_top.csv', index=False)
    temp_top = []

    # check that the imporved PdetRank ranking has worked correctly
    #temp3[['Pdet_varyBeta', 'SNR_varyBeta', 'PdetRank_varyBeta', 'region', 'PdetRank_SNR']].to_csv(saveLoc + 'temp3.csv', index=False)


    data = pd.concat([temp1, temp2, temp3])
    data = data.sort_values(by='PdetRank_fixedBeta', axis=0, ascending=True, kind='quicksort')
    #data[['Pdet_varyBeta', 'SNR_varyBeta', 'PdetRank_varyBeta', 'region', 'PdetRank_SNR']].to_csv(saveLoc + 'temps.csv', index=False)
    data = data.drop(['PdetRank_SNR'], axis=1) # this is incorporated into 'PdetRank_varyBeta'
    """ this makes PdetRank_varyBeta """

    return data

def ranking2(data, var, rank, dgn=[], v=False):
    """
    Sort the data using the 'var' variable. Name the column to assign ranks
    to with 'rank'. Remove degeneracy using 'dgn' (optional).
    var, rank and dgn are strings for the column names in 'data'.
    """

    data[rank] = -99
    if v:  print data[var].head(10)

    data.sort_values(by=var, axis=0, ascending=False, inplace=True, kind='quicksort')
    data[rank] = data[var].rank(ascending=False)
    data.reset_index(inplace=True, drop=True)
    if (dgn == []) & v:  print data[[var, rank]].head(10)

    # break the degeneracy in the top end of 'var' with 'dgn'
    if dgn != []:
        if v:  print data[[var, dgn, rank]].head(10)

        top = data[data[var]>0.999999]
        top.sort_values(by=dgn, axis=0, ascending=False, inplace=True, kind='quicksort')

        top['RANK'] = top.reset_index().index.values + 1 # set this data rank as the index number
        data[rank][data[var]>=0.999999] = top['RANK']

    data.sort_values(by=rank, axis=0, ascending=True, inplace=True, kind='quicksort')
    data.reset_index(inplace=True, drop=True)
    if v:  print data[[var, rank]].head(5000)

    return data

def pmix_plot1(data, alpha, save, wg=[], vmag=[]):
    """
    Plots the results of ranking3(). Subplots of 2000 stars per panel, sorted
    by 'P_mix'. Showing stars with ranks 0-18000.

    Inputs
    data:   Pandas dataframe to plot.
    alpha:  The alpha value used in P_mix inside ranking3().
    save:   (bool) Whether to save the plot or not.
    wg:     (kewarg) if wg!=[], ALSO overplot the stars in TGAS+XHIP lists and WG4
    vmag:   (kewarg) If vmag!=[], ONLY plot the stars with V<'vmag'.
    """

    plt.rc('font', size=14)
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) =\
        plt.subplots(3, 3, sharex=True, sharey=True)

    s = 2000  # the number of stars in each subplot
    m = 18000 # the maximum star rank to plot
    r = np.arange(0, m, step=s)  # the range of each ranked-star subplot
    plots = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9]


    if isinstance(wg, pd.DataFrame):
        """ Prepare the WG4 DataFrame before plotting. """
        data.dropna(inplace=True, subset=['TICID'])
        both = pd.merge(left=data[['TICID', 'P_mix']], right=wg, how='left', left_on='TICID', right_on='TIC')
        both.dropna(inplace=True, subset=['TICID'])
        #both.rename(columns={'Lum':'lum'}, inplace=True)
        #print both.shape
        both.sort_values(by='P_mix', axis=0, ascending=False, inplace=True, kind='quicksort')
        print both.head()
        both = both[:m]
        print both.shape
        #sys.exit()

    # plot each subplot in a loop
    for ix, i in enumerate(r):

        if vmag!=[]:
            """ Make a deepcopy of the 2000 ranked stars to plot before
            selecting Vmag<6.5 stars. """
            p = copy.deepcopy(data[i:i+s])
            #print p[['Vmag', 'P_mix']].head(10)
            q = p.where(data['Vmag']<6.5)
            #print q[['Vmag', 'P_mix']].head(10)
            plots[ix].scatter(q['teff'], q['lum'], s=0.1)

        elif isinstance(both, pd.DataFrame):
            """ Check if both is given as a DataFrame. Then overplot the WG4 stars
            for each rank-chunk. """
            print i, i+s
            #cond = (both['Rank']>i) & (both['Rank']<=i+s)
            print len(both[i:i+s][both['teff']==both['teff']])
            #print both['Rank'][cond], '\n'
            plots[ix].scatter(data['teff'][i:i+s], data['lum'][i:i+s], s=0.1, c='gray')
            plots[ix].scatter(both['teff'][i:i+s], both['lum'][i:i+s], s=2, c='r')

        else:
            plots[ix].scatter(data['teff'][i:i+s], data['lum'][i:i+s], s=0.1)


        plots[ix].annotate('%s:%s' % (i, i+len(data['teff'][i:i+s])), xy = (7500, 0.45))
        if vmag !=[]:
            plots[ix].annotate('%s' % len(p[p['Vmag']<6.5]), xy = (5100, 0.45))
        elif isinstance(both, pd.DataFrame):
            plots[ix].annotate('%s' % len(both[i:i+s][both['teff']==both['teff']]), xy = (5100, 0.45))

        if i%3 == 0:  plots[ix].set_ylabel(r'$L / L_{\odot}$')
        if ix >= 6:   plots[ix].set_xlabel(r'$T_{\rm eff} / K$')

    if vmag==[]:
        title = plt.suptitle(r'ATL Sorted by Pmix(alpha=%s)' % alpha)
    else:
        title = plt.suptitle(r'ATL Sorted by Pmix(alpha=%s); Vmag$<$%s' % (alpha,vmag))

    title.set_position([.5, 1])
    plt.ylim(0.3,50)
    plt.xlim([7700,4300])
    plt.xticks([5000, 7000])
    plt.yticks([1, 10])
    plt.yscale('log')
    plt.tight_layout()
    plt.show()

    if save:
        print 'saving pmix_plot1'

        if plx_source != 'DR2_newmask':
            extension = '/home/mxs191/Desktop/MathewSchofield/ATL/TESS_telecon3/P_mix/'
        elif plx_source == 'DR2_newmask':
            extension = '/home/mxs191/Desktop/MathewSchofield/ATL/TESS_telecon3/DR2_results/TESS_telecon3_plots/'

        if vmag!=[]:
            fig.savefig(extension + 'pmix_plot1_%s_V<%s.pdf' % (alpha, vmag))
        elif isinstance(both, pd.DataFrame):
            fig.savefig(extension + 'pmix_plot1_wg_%s.pdf' % alpha)
        else:
            fig.savefig(extension + 'pmix_plot1_%s.pdf' % alpha)

def pmix_plot2(data, alpha, save, wg=[], vmag=[]):
    """
    Plots the results of ranking3().
    1 HR plot of the entire combined list (~300,000 stars).

    Inputs
    data:   Pandas dataframe to plot.
    alpha:  The alpha value used in P_mix inside ranking3().
    save:   (bool) Whether to save the plot or not.
    wg:     (kewarg) If wg!=[], plot All of the WG4 stars instead of the ATL.
    vmag:   (kewarg) If vmag!=[], only plot the stars with V<'vmag'.
    """


    #fig, ax = plt.subplots()
    fig, ax = hrPlot()
    #plt.rc('font', size=12)
    s = 10.

    if vmag!=[]:
        plt.scatter(data['teff'][data['Vmag']<vmag],\
            data['lum'][data['Vmag']<vmag], s=s, c=data['Vmag'][data['Vmag']<vmag])

    elif isinstance(wg, pd.DataFrame):
        plt.scatter(data['teff'], data['lum'], s=0.01)

        wg['tred'] = 8907.0 * ((wg['lum'])**-0.093)
        wg['region2_rad'] = ( (wg['teff']/5777.)**(-0.92) * (240./3090.) )**(1./-1.85)
        wg['region2_lum'] = wg['region2_rad']**2 * (wg['teff']/5777.)**4
        #wg['region'][(wg['teff'] < wg['tred']) & (wg['Lum'] < wg['region2_lum'])] = '2'
        cond = (wg['teff'] < wg['tred']) & (wg['lum'] < wg['region2_lum']) & (wg['teff']<7700)
        wg.drop(['tred', 'region2_lum', 'region2_rad'], axis=1, inplace=True)
        plt.scatter(wg['teff'][cond], wg['lum'][cond], s=s, c='r')

    else:
        plt.scatter(data['teff'], data['lum'], s=s, c=data['Vmag'].as_matrix())


    if isinstance(wg, list): plt.colorbar(label=r'$V_{\rm mag}$')
    plt.xlim([8000,4000])
    #plt.yscale('log')
    #plt.xlabel(r'$T_{\rm eff} / K$')
    #plt.ylabel(r'$L / L_{\odot}$')
    if vmag != []:  plt.annotate('%s Stars' % len(data[data['Vmag']<vmag]), xy = (7800, 0.29))
    elif isinstance(wg, pd.DataFrame):
        plt.annotate('%s WG4 Stars' % len(wg[cond]), xy = (7843, 0.46))
    plt.show()

    if save:
        print 'saving pmix_plot2'

        if vmag != []:
            fig.savefig(os.getcwd() + '/TESS_telecon3/P_mix/pmix_plot2_V<%s.pdf' % vmag)
        elif isinstance(wg, pd.DataFrame):
            fig.savefig(os.getcwd() + '/TESS_telecon3/P_mix/pmix_plot2_wg.pdf')
        else:
            fig.savefig(os.getcwd() + '/TESS_telecon3/P_mix/pmix_plot2.pdf')

def ranking3(data, alpha, v=False):
    """
    Rank the data using 'P_mix', the probability made by combining
    Pdet_fixedBeta and Pdet_varyBeta. Break degeneracy at the top end of
    'Rank_Pmix' using SNR_varyBeta.

    Input
    data:  the Pandas dataframe to rank.
    alpha: the weighting to apply in Pmix (0<=alpha<=1).
    v:     (bool) prints the output from this function.
    """

    # remove the old columns made by KDE()
    if 'KDE1' in data:   data.drop(['KDE1', 'KDE2', 'x'], inplace=True, axis=1)

    data['P_mix'] = (1.-alpha)*data['Pdet_varyBeta'] + alpha*data['Pdet_fixedBeta']

    if v: print data[['P_mix', 'Pdet_fixedBeta', 'Pdet_varyBeta']].head()
    data.sort_values(by='P_mix', axis=0, ascending=False, inplace=True, kind='quicksort')
    data['Rank_Pmix'] = data['P_mix'].rank(ascending=False)
    if v: print data[['Rank_Pmix', 'P_mix', 'Pdet_fixedBeta', 'Pdet_varyBeta', 'SNR_varyBeta']].head()

    # break the degeneracy in the top end of Rank_Pmix with SNR_varyBeta
    top = data[data['P_mix']>0.999999]  # the 'top end' of the ATL
    top.sort_values(by='SNR_varyBeta', axis=0, ascending=False, inplace=True, kind='quicksort')
    top['RANK'] = top.reset_index().index.values + 1 # set this data rank as the index number
    data['Rank_Pmix'][data['P_mix']>=0.999999] = top['RANK']

    data.sort_values(by='Rank_Pmix', axis=0, ascending=True, inplace=True, kind='quicksort')
    data.reset_index(inplace=True, drop=True)
    if v: print data[['Rank_Pmix', 'P_mix', 'Pdet_fixedBeta', 'Pdet_varyBeta', 'SNR_varyBeta']].head(3000)

    # just plot the dataly ranked ATL
    #pmix_plot1(data, alpha, save=True, wg=[], vmag=[])

    # plot Vmag<6.5 cross-over with the ATL
    #pmix_plot1(data, alpha, save=True, vmag=6.5)
    #pmix_plot2(data, alpha, save=True, vmag=6.5)

    # overplot WG4 cross-over with the ATL
    # WGpath = saveLoc + 'WG4/' + 'WG4_TGASandXHIP.csv'  # common TGAS/XHIP & WG4 stars
    # if os.path.exists(WGpath):  wg = pd.read_csv(WGpath)
    wg = pd.read_csv('/home/mxs191/Desktop/MathewSchofield/ATL/TESS_telecon3/WG4/targets_2min_DSCTstars_WG4.csv')
    wg = wg[wg['TIC']!='-']
    wg['TIC'] = wg['TIC'].astype(float)
    wg = pd.merge(left=data[['TICID', 'teff', 'lum']], right=wg[['TIC']], left_on='TICID', right_on='TIC', how='inner')
    wg.drop(inplace=True, labels=['TICID'], axis=1)
    print wg.shape
    pmix_plot1(data, alpha, save=True, wg=wg)
    # pmix_plot2(data, alpha, save=True, wg=wg)
    sys.exit()

    return data

def make_tarfile(tarName, snames, extension):
    """ make a tarball, given a tarball name, savenames for the files and a save directory """

    tar = tarfile.open(extension + tarName, "w:gz")
    for name in snames:
        tar.add(name, arcname=os.path.basename(name))
    tar.close()

def saveData(data):
    """ save the entire ranked list, with the correct columns, in the correct order. """

    if choice == '4':  # for TRILEGAL simulations
        data.to_csv('/home/mxs191/Desktop/MathewSchofield/ATL/TRILEGAL/trilegal_results.csv', index=False)
        #data.to_csv('/home/mxs191/Desktop/MathewSchofield/ATL/TRILEGAL/trilegal_results_test.csv', index=False)
        stop = timeit.default_timer()
        print stop-start, 'seconds.'
        sys.exit()


    # combine the data with the CTL catalogue. The CTL is from (8)
    ctl = loadData('/home/mxs191/Desktop/phd_y2/CTL/', 'tessdwarfs_fix.txt',
        ctl=True, print_headers=False)
    data = pd.merge(left=data, right=ctl[['HIP', 'Tmag', 'TICID']], how='left', left_on='HIP', right_on='HIP')

    # Add Tycho IDs to the ATL. the Tycho catalogue is from vizier. see (10)
    tycho = pd.read_csv('/home/mxs191/Desktop/phd_y2/Tycho/asu.tsv',
        names=['TYC1', 'TYC2', 'TYC3', 'HIP'], skiprows=40, delimiter='\t')
    tycho['TYC'] = tycho['TYC1'].map(str) + '-' + tycho['TYC2'].map(str) + '-' + tycho['TYC3'].map(str)
    tycho = tycho[['HIP', 'TYC']][tycho['HIP'] == tycho['HIP']]
    tycho = tycho.drop_duplicates('HIP')
    data = pd.merge(left=data, right=tycho, how='left', left_on='HIP', right_on='HIP')


    # assign high priority flag to MS stars
    data['HP'] = 0
    data['HP'][data['numax'] >= 2000.] = 1


    if choice == '1': # for Gaia
        data.rename(columns={'PdetRank_fixedBeta':'Rank', 'Lum':'lum', 'hip_e_b_v':'e_B-V'}, inplace=True)

        # replace Nans where available
        #print data[['TICID', 'TIC', 'TYC2_id', 'TYC']].head(20)
        data['TICID'][pd.isnull(data['TICID'])] = data['TIC'][pd.isnull(data['TICID'])]
        data['TYC2_id'][data['TYC2_id']=='TYC nan'] = data['TYC'][data['TYC2_id']=='TYC nan']
        #print data[['TICID', 'TIC', 'TYC2_id', 'TYC']].head(20)

        if plx_source == 'oldplx_oldmask':
            data.drop(['ref_epoch', 'tred', 'phot_g_mean_mag', 'TIC', 'TYC', \
                'hip_b_v', 'hip0_vmag', 'tycho2_b_v', 'tycho2_vmag',\
                'urat_b_v', 'urat_vmag', 'source_id',], axis=1, inplace=True)

        if (plx_source == 'DR2_oldmask') or (plx_source == 'DR2_newmask'):
            data.drop(['ref_epoch', 'tred', 'phot_g_mean_mag', 'TIC', 'TYC', \
                'hip_b_v', 'hip0_vmag', 'tycho2_b_v', 'tycho2_vmag',\
                'urat_b_v', 'urat_vmag'], axis=1, inplace=True)


        if plx_source == 'oldplx_oldmask':  subFolder = 'Gaia_ATL/'
        if plx_source == 'oldplx_newmask':  subFolder = ''
        if plx_source == 'DR2_oldmask':  subFolder = ''
        if plx_source == 'DR2_newmask':  subFolder = ''

        name = 'Gaia_'


    if choice == '2': # for XHIP

        #print list(data)
        #print data[['HIP', 'TYC2_id', 'TYC', 'TIC', 'TICID']].head(200)

        # remove NaNs from Tycho2 and TIC id columns
        data['TYC2_id'] = pd.concat([data['TYC'].dropna(), \
            data['TYC2_id'].dropna()]).reindex_like(data)
        data['TICID'][pd.isnull(data['TICID'])] = data['TIC'][pd.isnull(data['TICID'])]

        data = data.drop(['Dist', 'e_Dist', 'e_V-I', '[Fe/H]',
            'e_[Fe/H]', 'tred', 'TYC', 'TIC'], axis=1)

        #print list(data)
        #print data[['HIP', 'TYC2_id', 'TICID']].head(200)

        # rename the columns to match those from Gaia ATL so the Gaia and XHIP
        # ATLs can be merged easily in combine()
        data.rename(columns={'PdetRank_fixedBeta':'Rank', 'Lum':'lum',
            'RAJ2000':'ra', 'DEJ2000':'dec'}, inplace=True)

        # reorder columns before saving
        if not saveall:
            cols = ['Rank', 'PdetRank_varyBeta', 'SNRrank_fixedBeta',
            'SNRrank_varyBeta', 'HIP', 'TYC2_id', 'TICID' ,'ra',
            'dec', 'ELon', 'ELat', 'GLon', 'GLat', 'Lc', 'HP', 'max_T',
            'Tmag', 'Imag', 'Vmag_reddened', 'Vmag',
            'region', 'B-V', 'B-V_observed', 'e_B-V', 'V-I', 'E(B-V)', 'Plx',
            'e_Plx', 'lum', 'teff',
            'rad', 'numax', 'SNR_fixedBeta', 'Pdet_fixedBeta',
            'SNR_varyBeta', 'Pdet_varyBeta']
            data = data.ix[:, cols]

        if plx_source == 'oldplx_oldmask':  subFolder = 'XHIP_ATL/'
        if plx_source == 'oldplx_newmask':  subFolder = ''
        if plx_source == 'DR2_oldmask':  subFolder = ''
        if plx_source == 'DR2_newmask':  subFolder = ''

        name = 'XHIP_'


    print name + 'atl:', data.shape
    print list(data)
    print 'saving data...'

    data.to_csv(saveLoc + subFolder + name + 'ATL.csv', index=False)
    # data[data['region']=='1'].to_csv(saveLoc + subFolder + name + 'ATL_region1.csv', index=False)
    # data[data['region']=='2'].to_csv(saveLoc + subFolder + name + 'ATL_region2.csv', index=False)
    # data[data['region']=='3'].to_csv(saveLoc + subFolder + name + 'ATL_region3.csv', index=False)
    #
    # # also save the files in a tarball
    # snames = [saveLoc + subFolder + name + 'ATL_region1.csv',
    #     saveLoc + subFolder + name + 'ATL_region2.csv',
    #     saveLoc + subFolder + name + 'ATL_region3.csv',
    #     saveLoc + subFolder + name + 'ATL.csv']
    # make_tarfile(tarName='ATL.tar.gz', snames=snames, extension=saveLoc + subFolder)

def varyCadence(data, v=False):
    """ For the High Priority combined ATL stars, vary the cadence between 20s and
    120s to find the difference in the number of detected stars """

    # define parameters & variables needed for the detection recipe
    sys_limit = 0 # in ppm
    dilution = 1
    cadence = np.array([20, 120]) # in s
    teff_solar = 5777.0 # Kelvin
    teffred_solar = 8907.0 #in Kelvin
    numax_solar = 3090.0 # in micro Hz
    dnu_solar = 135.1 # in micro Hz
    vnyq = (1.0 / (2.0*cadence[:,np.newaxis])) * 10**6 # in micro Hz
    data['tred'] = 8907.0 * ((data['lum'])**-0.093) # hot-edge temperature

    # 'Imag' is the reddened Imag value
    data['Pdet_20s'], data['SNR_20s'] = globalDetections(data['GLon'].as_matrix(),\
    data['GLat'].as_matrix(), data['ELon'].as_matrix(), data['ELat'].as_matrix(),\
    data['Imag'].as_matrix(), data['lum'].as_matrix(), data['rad'].as_matrix(),\
    data['teff'].as_matrix(), data['numax'].as_matrix(), data['max_T'].as_matrix(),\
    data['tred'].as_matrix(), teff_solar, teffred_solar, numax_solar, dnu_solar,\
    sys_limit, dilution, float(vnyq[0]), cadence[0], vary_beta=False)
    if v: print len(data['Pdet_20s'][(data['Pdet_20s']>0.5)]),\
        'with detections, in the 3 regions. Beta=1, cadence=20s'
    if v: print 'regions with det:', '\n', data['region'][data['Pdet_20s']>0.5].value_counts()

    # originally the detection recipe was run at 120s cadence
    data.rename(columns={'Pdet_fixedBeta':'Pdet_120s', 'SNR_fixedBeta':'SNR_120s'}, inplace=True)
    data['diff'] = data['Pdet_20s'] - data['Pdet_120s']


    # set the conditions for high priority stars, and plot results
    if plx_source == 'DR2_newmask':
        saveCad = '/home/mxs191/Desktop/MathewSchofield/ATL/TESS_telecon3/DR2_results/varyCadence_plots/'
    else:
        print 'check'
        saveCad = saveLoc + 'plots/varyCadence_plots/'

    cond1 = (data['Pdet_20s']>0.5) & (data['Pdet_120s']<0.5) # stars only detected at 20s cadence
    cond2 = (data['numax']>=1500) & (data['Pdet_120s']>0.4)
    cond3 = np.logical_or(cond1, cond2) # conditions 1 and/or 2
    cond4 = (data['Pdet_20s']>0.4) & (data['Pdet_120s']<0.4) # stars only detected at 20s cadence
    cond5 = np.logical_or(cond2, cond4)
    cond6 = (data['diff']>0.1) #& (data['Pdet_20s']>0.5)
    cond7 = (data['Pdet_120s']>0.5) & (data['numax']>=1300)
    cond8 = (data['P_mix']>0.52) & (data['numax']>950)
    print len(cond8)
    if v: print
    if v: print len(cond1[cond1==True]), 'Pdet 20s > 0.5, Pdet 120s < 0.5'
    if v: print len(cond2[cond2==True]), 'numax > 1500, Pdet 120s > 0.4'
    if v: print len(cond3[cond3==True]), '(Pdet 20s > 0.5 & Pdet 120s < 0.5) and/or numax > 1500 & Pdet 120s > 0.4'
    if v: print len(cond4[cond4==True]), 'Pdet 20s > 0.4 & Pdet 120s < 0.4'
    if v: print len(cond5[cond5==True]), 'numax > 1500, Pdet 120s > 0.4 and/or Pdet 20s > 0.4 & Pdet 120s < 0.4'
    if v: print len(cond6[cond6==True]), 'stars with a Pdet difference > 0.1'

    fig1 = False
    fig2 = False
    fig3 = False
    fig4 = False
    fig5 = False
    fig6 = True

    if fig1 == True:
        fig, ax = hrPlot()
        plt.scatter(data['teff'][cond1], data['lum'][cond1], s=50, c=data['Imag'][cond1])
        ax.text(0.01, 0.01, r'%s stars with $P_{\rm det; \Delta T = 20s} > 0.5$ $\&$ $P_{\rm det; \Delta T = 120s} < 0.5$' %
            len(cond1[cond1==True]), ha='left', va='bottom', transform=ax.transAxes)
        cbar = plt.colorbar()
        cbar.set_label(r'$I_{\rm mag}$ / mag')
        plt.show()
        fig.savefig(saveCad + 'Plot1_varyCad_HR.pdf', dpi=120)

    if fig2 == True:
        fig, ax = hrPlot()
        plt.scatter(data['teff'][cond3], data['lum'][cond3], s=50, c=data['Imag'][cond3])
        ax.text(0.01, 0.01, r'%s stars with $P_{\rm det; \Delta T = 20s} > 0.5$ or' %
            len(cond3[cond3==True]) + '\n' + r'$\nu_{\rm max} > 1500 \mu$Hz $\&$ $P_{\rm det; \Delta T = 120s} > 0.4$',
            ha='left', va='bottom', transform=ax.transAxes)
        cbar = plt.colorbar()
        cbar.set_label(r'$I_{\rm mag}$ / mag')
        plt.show()
        fig.savefig(saveCad + 'Plot2_varyCondition_HR.pdf', dpi=120)

    if fig3 == True:
        fig, ax, width, plttype = histPlot(xaxis=r'$\nu_{\rm max} / \mu$Hz',
            yaxis='Number of Stars')
        plt.hist(data['numax'][cond1].as_matrix(), bins=50, histtype='stepfilled',
            label=r'%s stars with $P_{\rm det; \Delta T = 20s} > 0.5$' %
            len(cond1[cond1==True]) + '\n' + r'$\&$ $P_{\rm det; \Delta T = 120s} < 0.5$', edgecolor='none', alpha=0.7)
        plt.hist(data['numax'][cond3].as_matrix(), bins=50, histtype=plttype,
            label=r'%s stars with $P_{\rm det; \Delta T = 20s} > 0.5$' %
            len(cond1[cond3==True]) + '\n' + r'$\&$ $P_{\rm det; \Delta T = 120s} < 0.5$ or' +
            '\n' + r'$\nu_{\rm max} > 1500 \mu$Hz $\&$ $P_{\rm det; \Delta T = 120s} > 0.4$',
            color='g')
        plt.legend(loc='upper right')
        plt.show()
        fig.savefig(saveCad + 'Plot3_varyCondition_numaxHist.pdf', dpi=120)

    if fig4 == True:
        binwidth = 0.2
        b = np.arange(min(data['Imag'][cond3]), max(data['Imag'][cond3])+binwidth, binwidth)

        fig, ax, width, plttype = histPlot(xaxis=r'$\nu_{\rm max} / \mu$ Hz',
            yaxis='Number of Stars')
        plt.hist(data['Imag'][cond1].as_matrix(), bins=b, histtype='stepfilled',
            label=r'%s stars with $P_{\rm det; \Delta T = 20s} > 0.5$' %
            len(cond1[cond1==True]), edgecolor='none', alpha=0.7)
        plt.hist(data['Imag'][cond3].as_matrix(), bins=b, histtype=plttype,
            label=r'%s stars with $P_{\rm det; \Delta T = 20s} > 0.5$ or' %
            len(cond3[cond3==True]) + '\n' + r'$\nu_{\rm max} > 1500 \mu$Hz $\&$ $P_{\rm det; \Delta T = 120s} > 0.4$',
            color='g')
        plt.legend(loc='upper left')
        plt.xlim(-4, 12)
        plt.show()
        fig.savefig(saveCad + 'Plot4_varyCondition_ImagHist.pdf', dpi=120)

    if fig5 == True:
        #fig, ax, width, size = generalPlot(xaxis=r'$P_{\rm det; \Delta T = 20s}$',
        #    yaxis=r'$P_{\rm det; \Delta T = 20s} - P_{\rm det; \Delta T = 120s}$')
        plt.rc('font', size=14)
        fig, ax = plt.subplots()
        plt.scatter(data['Pdet_20s'], data['diff'], c='gray', edgecolors='none', label='')
        plt.scatter(data['Pdet_20s'][cond6], data['diff'][cond6], c='r', s=4,
            label=r'%s Stars with $\Delta P_{\rm det} \geq 0.1$' % len(cond6[cond6==True]))
        plt.legend(loc='upper left')
        plt.xlabel(r'$P_{\rm det; \Delta T = 20s}$')
        plt.ylabel(r'$P_{\rm det; \Delta T = 20s} - P_{\rm det; \Delta T = 120s}$')
        plt.xlim(0, 1)
        plt.ylim(0, max(data['diff'])*1.1)
        plt.show()
        fig.savefig(saveCad + 'Plot5_PdetDiff.pdf', dpi=120)

    if fig6 == True:
        fig, ax = hrPlot(labels=False)
        plt.scatter(data['teff'][cond8], data['lum'][cond8], s=25,
            c=data['Imag'][cond8], cmap=cm.jet, edgecolors='k')
        ax.text(0.01, 0.01, r'%s 20-sec cadence targets' %
            len(cond8[cond8==True]), ha='left', va='bottom', transform=ax.transAxes)
        cbar = plt.colorbar()
        cbar.set_label(r'$I_{\rm mag}$ / mag')
        plt.show()
        fig.savefig(saveCad + 'Plot6_HP_HR.pdf', dpi=120)

    plt.close('all')

    sys.exit()

def Plot_ImagTmag(total):
    """ Compare Imag and Tmag values of the ATL stars with observed Imags (i.e from XHIP) """

    lx = r'$I_{\rm mag}$'
    ly = r'$T_{\rm mag}$'
    c = 'k'
    bw = 0.1
    l = 'upper left'
    lab=r'$T_{\rm mag}$'
    sname = '/home/mxs191/Desktop/phd_y2/TESS_telecon/TESS_telecon3/plots/Plot_ATLImagTmag'
    xaxis = [1.2, 10]
    yaxis = [-2, 2.01, 1]
    y3axis = [-0.5, 0.5]

    cond = (total['cond']==True) & (total['Tmag_y']==total['Tmag_y'])
    x = total['Imag_reddened_y'][cond]
    y = total['Tmag_y'][cond]

    fig, ax, width, size = generalPlot(yaxis=r'$T_{\rm mag}$', xaxis=r'$I_{\rm mag}$')
    plt.scatter(x, y, c='k', s=5)
    plt.plot(np.arange(2, 11), np.arange(2, 11), c='r')
    plt.show()
    fig.savefig(sname + '_a.pdf')


    import matplotlib.gridspec as gridspec
    fig = plt.figure(figsize=(14, 16))
    plt.rc('font', size=26)
    G = gridspec.GridSpec(3, 1)
    widths = np.arange(0, max(y) + bw, bw)
    size = 20

    # show distribustions
    ax1 = plt.subplot(G[0, 0])
    ax1.hist(x.as_matrix(), bins=widths, histtype='step', color='g', label=lx)
    ax1.hist(y.as_matrix(), bins=widths, histtype='step', color='b', label=ly)
    ax1.set_ylabel('Number of Stars')
    ax1.yaxis.set_label_coords(-0.09,0.5)
    ax1.legend(loc=l)


    # show absolute differences
    ax2 = plt.subplot(G[1, 0], sharex=ax1)
    plt.scatter(y, (x-y), s=size, zorder=0, edgecolors='none', c=c)
    ax2.axhline(y=0, xmin=0, xmax=10000, c='r', lidataidth=1.5,
        linestyle='--', zorder=1)  # straight line

    ax2.set_xlim(xaxis[0], xaxis[1])  # for both plots
    ax2.set_ylim(yaxis[0], yaxis[1])  # only for scatter plot
    ax2.set_yticks(np.arange(yaxis[0], yaxis[1], yaxis[2]))  # only for scatter plot
    ax2.set_ylabel(r'$I_{\rm mag}$ - $T_{\rm mag}$')
    ax2.yaxis.set_label_coords(-0.09,0.5)

    # relative difference
    ax3 = plt.subplot(G[2, 0], sharex=ax1)
    ax3.scatter(y, 1-(x/y), s=size, edgecolors='none', c=c)
    ax3.axhline(y=0, xmin=0, xmax=10000, c='r', lidataidth=1.5,
        linestyle='--', zorder=1)  # straight line
    ax3.set_xlabel(lab)
    ax3.set_xlim(xaxis)
    ax3.set_ylim(y3axis)
    ax3.set_ylabel(r'1 - ($I_{\rm mag}$ / $T_{\rm mag}$)')
    ax3.yaxis.set_label_coords(-0.09,0.5)

    plt.tight_layout()
    plt.show()
    fig.savefig(sname + '_b.pdf')

def KDE(data, GD=False, DH=True, llog=False, npts=60, pplot=False, v=True):
    """ Calculate a KDE value for each star, using either DH or GD methods.

    Inputs
    GD, DH: which KDE method to use.
    llog:   whether to use a log-scale for the KDE or not.
    npts:   the number of contours to use for the KDE (in GD method).
    pplot:  make plots.
    v:      verbose.

    Outputs
    data[x]    = random uniform variable for each star
    data[KDE1] = Initial KDe for all stars
    data[KDE2] = re-calculated KDE for stars with x>KDE1. If x<KDE1, data[KDE2] = -99
    """

    if 'Lum' not in data.columns:
        data['Lum'] = data['lum']

    llog = False  # use a log scale for the KDE
    npts = 60  # the number of different contours of the KDE

    # set the KDE model limits
    if llog:
        values = np.vstack([np.log10(data['teff']), np.log10(data['Lum'])])
        t = np.linspace(np.log10(4300), np.log10(7700), npts)
        l = np.linspace(np.log10(0.3), np.log10(50), npts)
    else:

        #kde_atl = data.iloc[:25000,:]  # calculate the kde values of a small subset of the ATL
        kde_atl = data[data['P_mix']>0.5]
        values = np.vstack([kde_atl['teff'], kde_atl['Lum']])

        t = np.linspace(4300, 7700, npts)
        l = np.linspace(0.3, 50, npts)

    kde_model = stats.gaussian_kde(values)  # the kernel

    if DH:      # use the DH method

        res = kde_model(values)  # the result of the kernel at these teffs and lums

        # normalise the kernel values between 0 and 1
        normfac=np.max(res)/0.99
        resnorm=res/normfac
        if v:  print resnorm

        keep=np.zeros(len(res))
        x = np.random.uniform(size=len(resnorm))
        if v:  print x

        cond = (x>resnorm)  # keep stars if the random variable is larger than the KDE value
        keep[cond] = 1
        if v:  print keep, len(keep), len(keep[keep==1])

        # re-calculate the KDE using stars more evenly distributed across HR
        um=np.where(keep)[0]
        values2 = np.vstack([kde_atl['teff'][um], kde_atl['Lum'][um]])
        kde_model2 = stats.gaussian_kde(values2)
        res2 = kde_model2(values2)
        normfac = np.max(res2)/0.99
        resnorm2 = res2/normfac  # re-normalise the KDE values between 0 and 1
        if v:  print len(kde_atl['teff']), len(kde_atl['teff'][keep==1]), len(kde_atl['teff'][um])

        #print kde_atl['KDE2'].head()

        kde_atl['x']    = x
        kde_atl['KDE1'] = resnorm
        kde_atl['KDE2'] = -99  # write over these values for the stars where x>KDE1
        kde_atl['KDE2'][um] = resnorm2
        if v:  print kde_atl[['x', 'KDE1', 'KDE2']].head(10)

        #print kde_atl['KDE2'].head()

        if pplot:
            if choice == '1':    sname = 'TGAS'
            elif choice == '2':  sname = 'XHIP'
            elif choice == '3':  sname = 'Combined'

            KDE_plot1(kde_atl=kde_atl, kde=resnorm, save=True, sname=sname + '_KDE.pdf')
            KDE_plot1(kde_atl=kde_atl, idx=(um), kde=resnorm2, save=True, sname=sname + '_KDE2.pdf')


    elif GD:      # use the GD method
        T, L = np.meshgrid(t, l)
        d = np.vstack([T.flatten(), L.flatten()])

        simpop = kde_model(d)
        simpop = simpop.reshape(npts, npts)

        if pplot:

            fig, ax = hrPlot()
            plt.scatter(kde_atl['teff'], kde_atl['Lum'], color='gray', s=1)
            CS = ax.contour(T, L, simpop, npts, \
                            cmap='Reds', \
                            label='Temperature: {:.2f}'.format(1.))
            cbar = fig.colorbar(CS)
            cbar.ax.set_ylabel('Probability Density')

            if llog:
                # change plotting limits
                plt.xlim([np.log10(7700), np.log10(4300)])
                plt.ylim([np.log10(0.3), np.log10(50)])

                plt.xlim([max(np.log10(kde_atl['teff'])), min(np.log10(kde_atl['teff']))])
                plt.ylim([min(np.log10(kde_atl['Lum'])), max(np.log10(kde_atl['Lum']))])

            plt.show()


    if ('lum' in kde_atl.columns) & ('Lum' in kde_atl.columns):
        kde_atl.drop(['Lum'], axis=1, inplace=True)

    return kde_atl

def KDE_plot1(data, kde, idx=[], save=False, sname=[]):
    """ Produce a plot of the KDE. Note: unless specified, use all data points.

    Inputs
    data:  plot the data['teff'] and data['Lum'] columns.
    kde:   the KDE Kernel at the data['teff'] and data['Lum'] values.
    idx:   the indices to plot. Default: all indices.
    save:  Whether to save the plot or not.
    sname: the name to save the figure with, inside the /KDE folder.
    """

    # 'Lum' is renamed to 'lum' while the XHIP ATL is made. This ensure that
    # the plot can always be made
    if 'Lum' not in data.columns:
        data['Lum'] = data['lum']

    plt.rc('font', size=16)
    fig, ax = plt.subplots()

    if idx==[]:  plt.scatter(data['teff'],data['Lum'],c=kde)
    else:        plt.scatter(data['teff'][idx],data['Lum'][idx],c=kde)

    plt.colorbar(label='KDE')
    plt.xlim([7900,4000])
    plt.yscale('log')
    plt.xlabel(r'$T_{\rm eff}$ / K')
    plt.ylabel(r'$L / L_{\odot}$')
    plt.show()
    if save:
        if plx_source != 'DR2_newmask':
            pass#fig.savefig(os.getcwd() + '/TESS_telecon3/KDE/' + sname)
        elif plx_source == 'DR2_newmask':
            fig.savefig('/home/mxs191/Desktop/MathewSchofield/ATL/TESS_telecon3/DR2_results/kde_plots/' + sname)


    if 'lum' in data.columns:
        data.drop(['Lum'], axis=1, inplace=True)

def KDE_plot2(data, save=False, sname=[]):
    """ Plot the Pdet value vs number of stars in the ATL, after using
    different Kernels to smooth the data.

    Inputs
    data:  A .csv file containing the Pdet threshold values and number of stars.
    save:  Whether to save the plot or not.
    sname: the name to save the figure with, inside the /KDE folder.
    """

    plt.rc('font', size=16)
    fig, ax = plt.subplots()
    plt.plot(data['clumpy'],  data['Pdet'], label='v1:Clumpy',  c='k')
    plt.plot(data['smooth'],  data['Pdet'], label='v2:Smooth',  c='b')
    plt.plot(data['mixture'], data['Pdet'], label='v3:Mixture', c='g')
    plt.legend()
    plt.xlabel(r'Number of stars in ATL')
    plt.ylabel(r'$P_{\rm det}$')
    plt.show()
    if plx_source != 'DR2_newmask':
        if save:  fig.savefig(os.getcwd() + '/TESS_telecon3/KDE/' + sname + '_compare_KDEs.pdf')
    elif plx_source == 'DR2_newmask':
        if save:  fig.savefig('/home/mxs191/Desktop/MathewSchofield/ATL/TESS_telecon3/DR2_results/kde_plots/' + sname + '_compare_KDEs.pdf')

def KDE_plot3(data, sname):
    """ Plot a histogram of the KDE distributions. """

    plt.rc('font', size=16)
    fig, ax = plt.subplots()
    plt.hist(data['KDE1'], histtype='step', bins=30, label=r'KDE1 (all stars with $P_{\rm det}>0.5$)')
    plt.hist(data['KDE2'][data['KDE2']!=-99], histtype='step', bins=30, label=r'KDE2 (only stars with $x<$KDE1)')
    plt.xlabel(r'KDE')
    plt.ylabel('Number of Stars')
    #plt.ylim([0, 1600])
    plt.legend()
    plt.show()

    if plx_source != 'DR2_newmask':
        fig.savefig(os.getcwd() + '/TESS_telecon3/KDE/' + sname)
    elif plx_source == 'DR2_newmask':
        fig.savefig('/home/mxs191/Desktop/MathewSchofield/ATL/TESS_telecon3/DR2_results/kde_plots/' + sname)

def combined_KDE(data, dataset=[], make_csv=True, pplot=True, v=True):
    """ Test the effect of using a KDE to select stars on the entire ATL.

    Inputs
    data:      Data with Pdet and KDE values to compare.
    make_csv:  Choose whether to save a csv file with the data to plot.
    pplot:     Choose whether to plot the data from the csv file.
    """

    if make_csv:
        steps = np.arange(0.5, 1.01, 0.1)[::-1]  # define Pdet values
        csv = np.full((len(steps), 4), np.NaN)

        for ix, i in enumerate(steps):
            v1 = (data['Pdet_fixedBeta'] > i)  # clumpy
            v2 = ((data['Pdet_fixedBeta']>i) & (data['KDE2']!=-99))  # smooth
            v3 = ((data['x']<data['Pdet_fixedBeta']) & (data['Pdet_fixedBeta']>i))  # mixture

            csv[ix,:] = i, len(data[v1]), len(data[v2]), len(data[v3])
            if v:  print ix, i, len(data[v1]), len(data[v2]), len(data[v3])

            # create columns in data to identify stars in v1, v2 and v3 for HR plots/comparisons
            if i==0.5:
                data['v1'], data['v2'], data['v3'] = [np.zeros(len(data)), np.zeros(len(data)), np.zeros(len(data))]
                data['v1'] = 1.
                data['v2'][v2] = 1.
                data['v3'][v3] = 1.
                if v:  print data[['v1', 'v2', 'v3']].head()

        if plx_source != 'DR2_newmask':
            pass#np.savetxt(os.getcwd() + '/TESS_telecon3/KDE/' + dataset + '_combined_KDE.csv', csv, header='Pdet,clumpy,smooth,mixture', comments='', delimiter=',')
        elif plx_source == 'DR2_newmask':
            np.savetxt('/home/mxs191/Desktop/MathewSchofield/ATL/TESS_telecon3/DR2_results/kde_plots/' + dataset + '_combined_KDE.csv', csv, header='Pdet,clumpy,smooth,mixture', comments='', delimiter=',')

    if pplot:
        if plx_source != 'DR2_newmask':
            csv = pd.read_csv(os.getcwd() + '/TESS_telecon3/KDE/' + dataset + '_combined_KDE.csv')
        elif plx_source == 'DR2_newmask':
            csv = pd.read_csv('/home/mxs191/Desktop/MathewSchofield/ATL/TESS_telecon3/DR2_results/kde_plots/' + dataset + '_combined_KDE.csv')

        KDE_plot2(csv, save=True, sname=dataset)

def weighting_plot1(data, w1, w2, var, rank, save=True, sname=''):

    plt.rc('font', size=16)
    fig, ax = plt.subplots()
    plt.plot(data['Rank'], data[var], c='K', lidataidth=2, label=var.replace("_", " "))
    plt.scatter(data['Rank'], data['pdet + (med-kde)*x'], c='g', s=0.5, label='pdet + (med-kde)*x')
    plt.scatter(data['Rank'], data['pdet + (med-kde)*0.2'], c='b', s=0.5, label='pdet + (med-kde)*0.2')
    legendScatter(num=3)
    plt.xlabel('Rank')
    plt.ylabel('Ranking Metric')
    plt.show()
    if save:
        print 'saving weighting_plot1'
        fig.savefig(os.getcwd() + '/TESS_telecon3/KDE/weighting_plot1_' + sname + '.pdf')

def weighting_plot2(data, var, rank, save=True, sname=''):

    """ Inside this function... sort by the variable to plot in steps of 2000. do not re rank by this variable.
    make sure that the list is unchanged outside of this function. check rank order / sorting of list outsdie
    """

    print sname, var, rank
    print data[[rank, sname, 'Pdet_fixedBeta']].head(10)
    #print data[['Rank', var]].head(10)
    data.sort_values(by=[rank], ascending=True, axis=0, inplace=True)
    print data[[rank, sname, 'Pdet_fixedBeta']].head(10)
    #sys.exit()


    plt.rc('font', size=14)
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) =\
        plt.subplots(3, 3, sharex=True, sharey=True)

    s = 2000  # the number of stars in each subplot
    r = np.arange(0, 18000, step=s)  # the range of each ranked-star subplot
    plots = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9]

    # plot each subplot in a loop
    for ix, i in enumerate(r):
        plots[ix].scatter(data['teff'][i:i+s], data['lum'][i:i+s], s=0.1)
        plots[ix].annotate('%s:%s' % (i, i+len(data['teff'][i:i+s])), xy = (7500, 0.45))

        if i%3 == 0:  plots[ix].set_ylabel(r'$L / L_{\odot}$')
        if ix >= 6:   plots[ix].set_xlabel(r'$T_{\rm eff} / K$')

    title = plt.suptitle(r'ATL Sorted by %s' % sname.replace("_", " "))
    title.set_position([.5, 1])
    plt.ylim(0.3,50)
    plt.xlim([7700,4300])
    plt.xticks([5000, 7000])
    plt.yticks([1, 10])
    plt.yscale('log')
    plt.tight_layout()
    plt.show()
    if save:
        print 'saving weighting_plot2'

        if plx_source != 'DR2_newmask':
            pass#fig.savefig(os.getcwd() + '/TESS_telecon3/KDE/weighting_plot2_' + rank + '-' + sname + '.pdf')
        elif plx_source == 'DR2_newmask':
            fig.savefig('/home/mxs191/Desktop/MathewSchofield/ATL/TESS_telecon3/DR2_results/kde_plots/weighting_plot2_' + rank + '-' + sname + '.pdf')

def weighting(data, var, rank, save=True, v=False):
    """ Test ranking strategies based upon Pdet_fixedBeta, Pdet_varyBeta, KDE1. """

    pdet = data['Pdet_fixedBeta']
    kde = data['KDE1']
    med = np.median(kde)
    x = data['x']
    #eta = 0.2
    if v:  print ((med-kde)*eta).head(10)
    if v:  print ((med-kde)*x).head(10)

    w1 = pdet + (med-kde)*x  # the weighting
    w1 = w1/(max(w1)/0.999)  # the normalised weighting
    w2 = pdet + (med-kde)*0.2
    w2 = w2/(max(w2)/0.999)
    data['pdet + (med-kde)*x'] = w1
    data['pdet + (med-kde)*0.2'] = w2

    #weighting_plot1(data, w1, w2, var, rank, save=save, sname=var)
    #weighting_plot2(data, var, rank, save=save, sname=var)

    return data

def combine(saveLoc, v=True, loop=False):
    """
    Combine the XHIP and Gaia versions of the ATL, and save the files.

    Input
    'v': (bool) if True, prints output
    'loop' prints the output from the TGAS/XHIP columns merging
    """

    if plx_source == 'oldplx_oldmask':
        gaia = pd.read_csv(saveLoc + 'Gaia_ATL/Gaia_ATL.csv')
        xhip = pd.read_csv(saveLoc + 'XHIP_ATL/XHIP_ATL.csv')
    if plx_source == 'DR2_oldmask':
        gaia = pd.read_csv(saveLoc + 'Gaia_ATL.csv')
        xhip = pd.read_csv(saveLoc + 'XHIP_ATL.csv')
    if plx_source == 'DR2_newmask':
        gaia = pd.read_csv(saveLoc + 'Gaia_ATL.csv')
        xhip = pd.read_csv(saveLoc + 'XHIP_ATL.csv')
    #xhip.drop(['PdetRank_varyBeta', 'SNRrank_fixedBeta',
    #    'SNRrank_varyBeta'], axis=1, inplace=True)

    if v: print 'gaia:', gaia.shape, 'xhip:', xhip.shape
    if v: print 'Gaia with det:', len(gaia[gaia['Pdet_fixedBeta']>0.5])
    if v: print 'XHIP with det:', len(xhip[xhip['Pdet_fixedBeta']>0.5])

    # merge the similar stars in both ATLs
    total = pd.merge(left=gaia, right=xhip, how='left', left_on=['HIP'], right_on=['HIP'])
    if v: print 'number of XHIP stars to merge:', len(total[total['ELon_y']==total['ELon_y']])
    if v: print 'total', total.shape, '\n'

    # calculate the condition to use XHIP values over TGAS values, for stars where
    # both values are available. Use the fractional parallax ratio
    total['fracPlx_x'] = total['e_Plx_x']/total['Plx_x'] # TGAS frac plx
    total['fracPlx_y'] = total['e_Plx_y']/total['Plx_y'] # XHIP frac plx
    total['fracPlx_ratio'] = total['fracPlx_y'] / total['fracPlx_x'] # ratio of xhip to tgas parallax fractions
    total['cond'] = (total['fracPlx_ratio'] <= 1)
    total['cond'][total['cond']==0] = np.nan # set columns to nan for non-xhip stars
    total.drop(['fracPlx_x', 'fracPlx_y', 'fracPlx_ratio', 'HIP_x', 'HIP_y'], axis=1, inplace=True)
    if v: print 'XHIP stars\' values to merge:', len(total['cond'][total['cond']==True])

    # change all TGAS columns to XHIP values for the stars that satisfy 'cond'
    cols = list(total)
    for idx, TGAScol in enumerate(cols):

        # change the TGAS columns except for Tycho2 ID (values already correct)
        if (cols[idx][-1] == 'x') & (TGAScol != 'TYC2_id_x') & (TGAScol != 'TICID_x'):

            if loop: print total[TGAScol].head(10)

            XHIPcol = TGAScol[0:-2] + '_y'  # the equivalent XHIP column
            if loop: print XHIPcol

            # set the XHIP rows that do not satisfy condition to Nan
            total[XHIPcol] = total[XHIPcol] * total['cond']
            if loop: print total[XHIPcol].head(10) # the equivalent XHIP column

            # the indices where the equivalent XHIP value is Nan
            index = np.isnan(total[XHIPcol])
            if loop: print index.head(10)

            # change the TGAS columns where the equivalent XHIP value satisfies 'cond'
            # i.e where 'cond' is not a Nan
            total[TGAScol][~index] = total[XHIPcol]
            if loop: print total[TGAScol].head(10)
            if loop: sys.exit()

            # rename the TGAS column by removing '_x'. Delete the equivalent XHIP column
            total.rename(columns={TGAScol:TGAScol[0:-2]}, inplace=True)
            total.drop(XHIPcol, axis=1, inplace=True)


        # remove the depreciated Tycho2 ID column
        elif (TGAScol == 'TYC2_id_x'):
            total.rename(columns={TGAScol:TGAScol[0:-2]}, inplace=True)
            total.drop('TYC2_id_y', axis=1, inplace=True)


        # always use TIC IDs where available in XHIP (as there are no 'NaN' TICIDs in XHIP ATL)
        elif (TGAScol == 'TICID_x'):
            XHIPcol = 'TICID_y'

            #print total[['TICID_x', 'TICID_y']], '\n'
            total[TGAScol][pd.isnull(total[TGAScol])] = total[XHIPcol]
            #print total[['TICID_x', 'TICID_y']], '\n'
            #print len(total[pd.isnull(total[TGAScol])])

            total.rename(columns={TGAScol:TGAScol[0:-2]}, inplace=True)
            total.drop(XHIPcol, axis=1, inplace=True)


    if v: print 'total after column merge', total.shape

    # append on the xhip stars that are NOT already in Gaia DR1
    xhip['cond'] = 1 # set the flag to 1 for the XHIP stars not already in TGAS
    merged = total['HIP'].isin(xhip['HIP']) # the indices of the merged HIP numbers
    mergedHIP = total['HIP'][merged] # the merged Hipparcos numbers
    notmerged = ~xhip['HIP'].isin(mergedHIP) # the indices of XHIP NOT merged with TGAS
    if v: print 'XHIP stars not yet merged with TGAS:', len(notmerged[notmerged==True])

    total = pd.concat([total, xhip[notmerged]])
    total.reset_index(inplace=True, drop=True)
    if v: print 'total after adding extra XHIP stars', total.shape
    if v: print 'total with Pdet > 0.5:', len(total[total['P_mix']>0.5])

    # re-rank the combined ATL
    # var  = 'Pdet_fixedBeta'      # the variable to rank by
    # rank = 'PdetRank_fixedBeta'  # the name of the ranking column
    # dgn  = 'SNR_fixedBeta'       # the column to break degeneracy in 'var'
    # total = ranking2(total, var=var, rank=rank, dgn=dgn, v=False)
    # if v: print total[[var, rank, dgn]].head(100)
    #
    # keep all stars with Pdet>0.5
    # if var == 'Pdet_fixedBeta':
    #     total = total[total[var] >= 0.5]
    # else:
    #     total = total.head(17680)

    # replace the separate rank values from TGAS and XHIP with the recalculated rank
    # based on Pdet_mix from ranking3()
    total.drop(['region', 'Lc', 'Rank_Pmix', 'Imag', 'Vmag',
    'g_mag_abs', 'B-V', 'B-V_flag', 'V_flag', 'V-I'], axis=1, inplace=True)
    total.rename(columns={'cond':'HIP_vals',
        'Imag_reddened':'Imag', 'Vmag_reddened':'Vmag', 'B-V_observed':'B-V'}, inplace=True)

    total = ranking3(total, alpha=0.5)  # re-rank the full combined ATL using 'P_mix'

    #total.sort_values(by=['Rank'], axis=0, inplace=True)
    if v: print 'stars where HIP values are used:', len(total[total['HIP_vals']==True])
    if v: print 'stars kept:', len(total)

    run_KDE = False
    if run_KDE:

        # var  = 'P_mix'
        # rank = 'Rank_Pmix'
        var  = 'Pdet_fixedBeta'      # the variable to rank by
        rank = 'PdetRank_fixedBeta'
        # var  = 'Pdet_varyBeta'      # the variable to rank by
        # rank = 'PdetRank_varyBeta'

        kde_atl = KDE(total, GD=False, DH=True, llog=False, npts=60, pplot=False, v=False)
        kde_atl.sort_values(by=var, axis=0, ascending=False, inplace=True)
        kde_atl[rank] = kde_atl[var].rank(ascending=False)
        #print kde_atl[[rank, var]]
        # kde_atl.sort_values(by=var, axis=0, ascending=True, inplace=True)
        # print kde_atl[[rank, var]]
        #sys.exit()

        combined_KDE(kde_atl, dataset='Combined', pplot=False, v=False)
        #print kde_atl[['Pdet_fixedBeta', 'x', 'KDE1', 'KDE2', 'v1', 'v2', 'v3']].head()

        kde_atl = weighting(kde_atl, var, rank, save=True)
        #print kde_atl[['Rank', var]].head()

        #KDE_plot1(kde_atl[kde_atl['v2']==1], kde_atl['KDE2'][kde_atl['v2']==1], idx=[], save=False, sname='Combined_KDE2.pdf')
        #KDE_plot1(kde_atl[kde_atl['v3']==1], kde_atl['KDE1'][kde_atl['v3']==1], idx=[], save=True, sname='Combined_KDE3.pdf')
        KDE_plot3(kde_atl, sname='Combined_KDEhist.pdf')
        #if v:  print len(kde_atl), len(kde_atl[kde_atl['v2']==1]), len(kde_atl[kde_atl['v3']==1])
        #print np.median(kde_atl['KDE1']), np.median(kde_atl['KDE2'][kde_atl['KDE2']!=-99])
        sys.exit()

    # high priority targets for 20s cadence
    # total['HP'] = 0
    # total['HP'][(total['numax']>=1300) & (total['Pdet_fixedBeta']>0.5)] = 1
    # print 'Number of High priority stars:', len(total[total['HP']==1])


    # make a call to TASOC wiki to get missing TICIDs and TESSmags
    get_missing = False
    if get_missing == True:

        cond = ((pd.isnull(total['TICID'])) | (pd.isnull(total['Tmag'])))
        temp = total[['HIP', 'TYC2_id']][cond]
        temp['TYC2_id'] = 'TYC ' + temp['TYC2_id']

        # give hipparcos numbers for the stars without Tycho 2 IDs
        a = pd.DataFrame(['HIP ' + x.rstrip(".0") for x in temp['HIP'][pd.isnull(temp['TYC2_id'])].astype(str)])
        m = pd.concat([temp['TYC2_id'][pd.notnull(temp['TYC2_id'])], a])
        m.to_csv(saveLoc + 'combinedATL/missing_TICs.csv', index=False, header=False)


    merge_missing = False
    if merge_missing == True:
        # merge missing TIC iDs from TASOC wiki
        #print total['TYC2_id'].head()
        total['TYC2_id'][pd.notnull(total['TYC2_id'])] = total['TYC2_id'][pd.notnull(total['TYC2_id'])].astype(str)
        s = total['TYC2_id'][pd.notnull(total['TYC2_id'])].str.split('-')
        s0 = [x[0] for x in s]
        s1 = [x[1] for x in s]
        s2 = [x[2] for x in s]

        s0 = [x.rjust(4, '0') for x in s0] # add zeros before the first part of string
        s1 = [x.rjust(5, '0') for x in s1] # add zeros before the 2nd part of string

        s0 = np.char.array(s0)
        s1 = np.char.array(s1)
        s2 = np.char.array(s2)
        total['TYC2_id'][pd.notnull(total['TYC2_id'])] = s0 + '-' + s1 + '-' + s2
        #print total['TYC2_id'].head()

        ids = pd.read_csv(saveLoc + 'combinedATL/table.csv', names=['TICID', 'Tmag', 'HIP', 'TYC2_id'])
        t = (pd.notnull(ids['TYC2_id']))
        total = pd.merge(left=total, right=ids[['Tmag', 'TYC2_id', 'TICID']][t], how='left', left_on=['TYC2_id'], right_on=['TYC2_id'])

        # fill in missing Tmags and TIC IDs
        total['Tmag'] = pd.concat([total['Tmag_x'].dropna(), total['Tmag_y'].dropna()]).reindex_like(total)
        total['TICID_x'][np.isnan(total['TICID_x'])] = total['TICID_y'][np.isnan(total['TICID_x'])]
        total.drop(['Tmag_x', 'Tmag_y', 'TICID_y'], axis=1, inplace=True)
        total.rename(columns={'TICID_x':'TICID'}, inplace=True)

        # add a couple of extra TYC2 IDs
        total['TYC2_id'][total['HIP']==43225] = '212-1537-1'
        total['TYC2_id'][total['HIP']==69481] = '3471-1251-1'

        #print len(total[ (total['HP']==1) & (pd.notnull(total['TICID'])) ])
        #print total[['HIP', 'TYC2_id']][pd.isnull(total['TICID'])] # these stars still do not have TIC IDs
        #total[['HIP', 'TYC2_id']][pd.isnull(total['TICID'])].to_csv(saveLoc + 'combinedATL/' + 'missing_TICs2.csv', index=False)

        # missing TIC iDs to send to Josh pepper and Rasmus
        #print total['HIP'][pd.isnull(total['TICID']) & pd.notnull(total['HIP'])]
        #total['TYC2_id'][(pd.isnull(total['TICID'])) & (pd.notnull(total['TYC2_id']))].to_csv(saveLoc + 'combinedATL/' + 'missingTICs2_TYC.csv', index=False)


    # re-check tic
    #ctl = pd.read_csv('/home/mxs191/Desktop/phd_y2/CTL/tessdwarfs_fix.txt', names=['TICID', 'HipNo'])
    #print list(ctl)
    #print ctl[ctl['HipNo'].isin(total['HIP'][pd.isnull(total['TICID'])])]

    #print list(total)
    # print len(total[total['r_est']!=total['r_est']])
    # print len(total[total['Dist_MW']!=total['Dist_MW']])
    #sys.exit()

    if (plx_source == 'DR2_oldmask') or (plx_source == 'DR2_newmask'):
        # change distance column name to match equivalent when plx_source=='oldplx_oldmask' catalogue
        total.rename(columns={'r_est':'Dist_MW', 'r_sigma':'e_Dist_MW'}, inplace=True)
        # total['Dist_MW'] *= 1e3  # convert from kpc to pc
        # total['e_Dist_MW'] *= 1e3  # convert from kpc to pc


    # calculate distances for XHIP stars
    xhip = (total['Dist_MW']!=total['Dist_MW'])
    total['Dist_MW'][xhip] = 1./total['Plx'][xhip]
    total['e_Dist_MW'][xhip] = total['Dist_MW'][xhip] * total['e_Plx'][xhip]/total['Plx'][xhip]


    # calculate lower and upper sigma Pdet bounds
    run_MC = False
    if run_MC:
        MC = 1000  # number of Monte Carlo trials
        b = int(np.ceil((MC*0.67)/2.))  # the 1 sigma upper and lower MC bounds
        trials = np.full((len(total), 4, MC), -99)  # store fixed and varied Beta Pdet and SNR values from all MC trials
        print 'Calculate Pdet bounds for %s stars' % len(total)

        for i in range(MC):

            if i==MC/2:  print 'halfway'
            if i%100 == 0: print i
            d2 = deepcopy(total)  # make a deepcopy so pristine ATL values are not overwritten
            d2 = Param(d2, vary=False)

            trials[:, 0, i] = d2['Pdet_fixedBeta2']
            trials[:, 1, i] = d2['SNR_fixedBeta2']
            #trials[:, 2, i] = d2['Pdet_varyBeta2']  # to get this, set vary==True in Param()
            #trials[:, 3, i] = d2['SNR_varyBeta2']   # to get this, set vary==True in Param()

        # sort the Pdet and SNR values to get the standard deviation
        trials, median, upper, lower, sd = Sorted(trials, MC, b)
        total['l_Pdet_Ranking'] = lower[:,0]
        total['u_Pdet_Ranking'] = upper[:,0]


    # re-order columns. 'saveall' is a global variable defined in __main__
    if not saveall:
        total.rename(columns={'Dist_MW': 'Dist', 'e_Dist_MW': 'e_Dist',
        'Pdet_fixedBeta': 'Pdet_Ranking'}, inplace=True)

        cols = ['Rank', 'HIP', 'TYC2_id', 'TICID' ,'ra', 'dec', 'ELon', 'ELat',
        'GLon', 'GLat', 'HIP_vals', 'HP', 'max_T', 'Tmag', 'Imag',
        'Vmag', 'region', 'B-V', 'e_B-V', 'Dist', 'e_Dist', 'E(B-V)',
        'lum', 'teff', 'rad', 'numax', 'SNR_fixedBeta', 'Pdet_Ranking',
        'l_Pdet_Ranking', 'u_Pdet_Ranking', 'SNR_varyBeta', 'Pdet_varyBeta']
        total = total.ix[:, cols]
    #if v: print list(total)

    if plx_source == 'oldplx_oldmask':  subFolder = 'combinedATL/'
    if plx_source == 'oldplx_newmask':  subFolder = ''
    if plx_source == 'DR2_oldmask':  subFolder = ''
    if plx_source == 'DR2_newmask':  subFolder = ''


    # seperate high priority stars. save just TIC IDs (with TYC IDs where not available), and all columns
    HP_sep = False
    if HP_sep == True:
        for i in range(2):
            if i==0: star_list = '120s'
            if i==1: star_list = '20s'
            print len(total[total['HP']==i]), '%s cadence targets' % star_list
            total[total['HP']==i].to_csv(saveLoc + subFolder + '%s_ATL.csv' % star_list, index=False)


            a = total['TICID'][total['HP']==i]

            # put Tycho2 IDs in rows that do not have TIC IDs
            a.fillna('TYC ' + total['TYC2_id'][total['HP']==i], inplace=True)

            # put HIP IDs in rows that do not have Tycho2 IDs.
            # strip '0' and '.' separately to avoid removing too many zeros
            # http://stackoverflow.com/questions/13682044/pandas-dataframe-remove-unwanted-parts-from-strings-in-a-column
            a.fillna('HIP ' + total['HIP'][total['HP']==i].astype(str).map(lambda x: x.rstrip('0').rstrip('.')), inplace=True)

            # add 'TIC ' to the stars with TIC IDs, and remove '.0'
            cond = (pd.notnull(total['TICID'][total['HP']==i]))
            a[cond] = 'TIC ' + a[cond].astype(str).map(lambda x: x.rstrip('0').rstrip('.'))

            a.to_csv(saveLoc + subFolder + '%s_IDs.csv' % star_list, index=False)


    top_25k = True
    if top_25k == True:

        # NOTE: save the 'full' array for sigpdet1.py
        total.to_csv(saveLoc + subFolder + 'ATL.csv', index=False)


        """ Only save the highest ranked 25k stars. """
        total = total.iloc[:25000, :]

        # total.to_csv(saveLoc + subFolder + 'ATL_top25k.csv', index=False)
        # sys.exit()

        # NOTE: save out the star ID files to cross match with MAST
        # tyc2 = total[['TYC2_id', 'ra', 'dec']]
        # tyc2.to_csv(saveLoc + subFolder + 'TYC2_top25k.csv', index=False)
        # xhip = total[['HIP', 'ra', 'dec']]
        # xhip.to_csv(saveLoc + subFolder + 'XHIP_top25k.csv', index=False)
        # print len(xhip), 'XHIP IDs to save'
        # print len(tyc2), 'TYC2 IDs to save'

        # NOTE: save out the missing star ID files to cross check with MAST
        # print len(total[total['TICID']!=total['TICID']]), 'stars without TIC IDs'
        # missing = total[['HIP', 'TYC2_id', 'ra', 'dec']][(total['TICID']!=total['TICID'])]
        # missing['TYC2_id'] = 'TYC ' + missing['TYC2_id']
        # missing[['TYC2_id', 'ra', 'dec']][(total['TYC2_id']==total['TYC2_id'])].to_csv('/home/mxs191/Desktop/MathewSchofield/ATL/TESS_telecon3/DR2_results/ATL_DR2newmask/MAST_missingIDs/ATL_top25k_missingTYCIDs.csv', index=False)
        # print missing[missing['TYC2_id']!=missing['TYC2_id']]
        #
        # # give hipparcos numbers for the stars without Tycho 2 IDs
        # print missing['HIP'][missing['TYC2_id']!=missing['TYC2_id']]
        # missing[['HIP', 'ra', 'dec']][missing['TYC2_id']!=missing['TYC2_id']].to_csv('/home/mxs191/Desktop/MathewSchofield/ATL/TESS_telecon3/DR2_results/ATL_DR2newmask/MAST_missingIDs/ATL_top25k_missingHIPIDs.csv', index=False)


        # NOTE: add in missing TIC IDs in top_25k list
        print 'adding TICIDs for the remaining stars...'
        tyc = pd.read_csv('/home/mxs191/Desktop/MathewSchofield/ATL/TESS_telecon3/DR2_results/ATL_DR2newmask/MAST_missingIDs/MAST_Crossmatch_top25k_TYCID.csv')
        hip = pd.read_csv('/home/mxs191/Desktop/MathewSchofield/ATL/TESS_telecon3/DR2_results/ATL_DR2newmask/MAST_missingIDs/MAST_Crossmatch_top25k_HIPID.csv')
        missing_hip = (total['TICID']!=total['TICID']) & (total['TYC2_id']!=total['TYC2_id'])  # stars that only have HIP IDs
        missing_tyc = (total['TICID']!=total['TICID']) & (total['HIP']!=total['HIP'])  # stars that only have TYC2 IDs
        #print tyc.shape, hip.shape

        #print total[['HIP', 'TICID']][missing_hip]
        total = pd.merge(left=total, right=hip[['HIP', 'MatchID']], left_on=['HIP'], right_on=['HIP'], how='left')
        total['TICID'][missing_hip] = total['MatchID'][missing_hip]
        total.drop(['MatchID'], axis=1, inplace=True)
        #print total[['HIP', 'TICID']][missing_hip]

        tyc.dropna(how='any', inplace=True, subset=['TYC'])
        tyc['TYC2_id'] = tyc['TYC2_id'].str[4:]  # remove 'TYC ' from ID name

        #print total[['TICID', 'HIP', 'TYC2_id']][total['TICID']!=total['TICID']].shape
        total = pd.merge(left=total, right=tyc[['TYC2_id', 'MatchID']], left_on=['TYC2_id'], right_on=['TYC2_id'], how='left')
        total['TICID'][missing_tyc] = total['MatchID'][missing_tyc]
        total.drop(['MatchID'], axis=1, inplace=True)
        #print total[['TICID', 'HIP', 'TYC2_id']][(total['TICID']!=total['TICID'])]
        #total[['TYC2_id', 'ra', 'dec']][(total['TICID']!=total['TICID'])].to_csv('/home/mxs191/Desktop/MathewSchofield/ATL/TESS_telecon3/DR2_results/ATL_DR2newmask/MAST_missingIDs/ATL_top25k_missingTYCIDs2.csv', index=False)
        #print total.shape

        tyc2 = pd.read_csv('/home/mxs191/Desktop/MathewSchofield/ATL/TESS_telecon3/DR2_results/ATL_DR2newmask/MAST_missingIDs/MAST_Crossmatch_top25k_TYCID2.csv')
        tyc2.dropna(how='any', inplace=True, subset=['TYC'])
        #print tyc2.shape

        total = pd.merge(left=total, right=tyc2[['TYC2_id', 'MatchID']], left_on=['TYC2_id'], right_on=['TYC2_id'], how='left')
        total['TICID'][total['MatchID']==total['MatchID']] = total['MatchID'][total['MatchID']==total['MatchID']]
        total.drop(['MatchID'], axis=1, inplace=True)
        #print total[['TICID', 'HIP', 'TYC2_id']][(total['TICID']!=total['TICID'])]

        # total.to_csv(saveLoc + subFolder + 'ATL_top25k.csv', index=False)
        # sys.exit()


        # NOTE: manually add extra stars with TYc2 IDs to ATL
        extra = pd.DataFrame(columns=['TYC2_id', 'MatchID'])
        extra_tyc = ['9128-1083-1', '8870-1384-1']
        extra_tic = [394125166, 262843771]
        extra['TYC2_id'] = extra_tyc
        extra['MatchID'] = extra_tic
        total = pd.merge(left=total, right=extra, left_on=['TYC2_id'], right_on=['TYC2_id'], how='left')
        total['TICID'][total['MatchID']==total['MatchID']] = total['MatchID'][total['MatchID']==total['MatchID']]
        total.drop(['MatchID'], axis=1, inplace=True)
        #print total[['TICID', 'HIP', 'TYC2_id']][(total['TICID']!=total['TICID'])]

        # NOTE: add celebrity stars to the ATL at rank 5000. See my email on 22.05.18
        # checked Vizier for their TICIDs
        celeb = pd.DataFrame(columns=['HIP', 'TICID'])
        celeb_hip = [79672, 33719, 32851, np.NaN]
        celeb_tic = [135656809, 268565917, 281812116, 158711270]
        celeb['HIP'] = celeb_hip
        celeb['TICID'] = celeb_tic
        celeb['celeb'] = 1
        #print celeb

        total['celeb'] = np.NaN  # add flag for celebrity stars
        #print total[['TICID', 'Rank_Pmix', 'celeb']][4995:5010]

        total = pd.concat([total.ix[:4998], celeb, total.ix[4999:]]).reset_index(drop=True)
        #print total[['TICID', 'Rank_Pmix', 'celeb']][4995:5010]

        total.drop_duplicates(subset=['TICID'], keep='first', inplace=True)
        total.reset_index(inplace=True, drop=True)
        total['Rank_Pmix'] = total.index + 1  # reset the Pmix ranks to include celeb stars
        #print total[['TICID', 'Rank_Pmix', 'celeb']][4995:5010]

        # NOTE: get the DR2 IDs+ra,dec,pm for the stars without TICIDs using ARI. Sent to Rasmus on 22.05.18
        # total[['ra', 'dec']][(total['TICID']!=total['TICID'])].to_csv('/home/mxs191/Desktop/MathewSchofield/ATL/TESS_telecon3/DR2_results/ATL_DR2newmask/no_TICID_exists/get_DR2ID.csv',
        #     header=False, index=False)
        # print total[['DR2 ID', 'ra', 'dec']][(total['TICID']!=total['TICID'])]

        # NOTE: vary the cadence. How many extra stars get detections with 20s cadence?
        # total = varyCadence(total)
        # sys.exit()

        # NOTE: save out the 120s and 20s ID files for TESS Science team
        print 'saving the top 25k of the ATL...'
        a = total['TICID']
        #a[a==a] = a[a==a].map(lambda x: 'TIC ' + str(int(x)))
        a[a==a] = a[a==a].map(lambda x: str(int(x)))  # don't add TIC to each ID
        #print a.head(10)

        b = a.fillna(value=total['TYC2_id'])
        b[b==b] = b[b==b].map(lambda x: 'TYC ' + str(x) if str(x[-2]) == '-' else np.NaN)
        #print b.head(10)
        a = a.fillna(value=b)
        #print a.head(10)

        c = a.fillna(value=total['HIP'])
        c[c==c] = c[c==c].map(lambda x: 'HIP ' + str(int(x)) if len(str(x)) <= 7 else np.NaN)
        a = a.fillna(value=c)

        cond = (total['P_mix']>0.52) & (total['numax']>950.)
        print 'saving %s stars as 20-sec targets...' % (len(total[cond]))    # (see Bill email 24.05.18)
        total['HP'] = 0
        total['HP'][cond] = 1

        total['TICID'][cond].to_csv(saveLoc + subFolder + 'ATL_top25k_20sIDs.csv', index=False)
        total[cond].to_csv(saveLoc + subFolder + 'ATL_top25k_20s.csv', index=False)
        a.to_csv(saveLoc + subFolder + 'ATL_top25k_IDs.csv', index=False)
        total.to_csv(saveLoc + subFolder + 'ATL_top25k.csv', index=False)

        sys.exit()



    print 'not saving the %s stars!' % len(total); sys.exit()
    print 'saving all %s 120/20s stars together' % len(total)
    total.to_csv(saveLoc + subFolder + 'ATL.csv', index=False)

    # total[total['region']=='1'].to_csv(saveLoc + subFolder + 'ATL_region1.csv', index=False)
    # total[total['region']=='2'].to_csv(saveLoc + subFolder + 'ATL_region2.csv', index=False)
    # total[total['region']=='3'].to_csv(saveLoc + subFolder + 'ATL_region3.csv', index=False)
    #
    # # also save the files in a tarball
    # snames = [saveLoc + subFolder + 'ATL.csv',
    #     saveLoc + subFolder + 'ATL_region1.csv',
    #     saveLoc + subFolder + 'ATL_region2.csv',
    #     saveLoc + subFolder + 'ATL_region3.csv']
    # make_tarfile(tarName='ATL.tar.gz', snames=snames, extension=saveLoc + subFolder)

def wg4(data, common=False):
    """ Compare the full TGAS and XHIP lists (before temp/luminosity/ecliptic
    latitude/red edge cuts) with the WG4 list on TASOC. Get a list of Teff/lum
    values for the common stars in WG4 and TGAS/XHIP. Combine the lists.

    Inputs
    data:   the Pandas DataFrame to merge with the WG4 list
    common: (bool) find and save common stars between 'data' and 'wg'
    """

    if common:
        wg = pd.read_csv(saveLoc + 'WG4/targets_2min_DSCTstars_WG4.csv',
            usecols=['TIC', 'magV'])
        print wg.shape
        wg = wg.loc[wg.loc[wg['TIC'] != '-'].index.values]  # drop '-' TIC values
        print wg.shape
        wg.drop_duplicates(subset='TIC', inplace=True)
        wg['TIC'] = wg['TIC'].astype(float)
        print wg.shape

        a = pd.merge(left=data[data['TIC']==data['TIC']], right=wg, how='inner',
            left_on='TIC', right_on='TIC')
        print len(a), 'common stars'

        if choice == '1':  sname = 'TGAS'
        if choice == '2':  sname = 'XHIP'
        a.to_csv(saveLoc + 'WG4/' + 'WG4_' + sname + '.csv', index=False)


    if os.path.exists(saveLoc + 'WG4/' + 'WG4_TGAS.csv') and\
       os.path.exists(saveLoc + 'WG4/' + 'WG4_XHIP.csv'):

       t = pd.read_csv(saveLoc + 'WG4/' + 'WG4_TGAS.csv')
       x = pd.read_csv(saveLoc + 'WG4/' + 'WG4_XHIP.csv')
       #print t.shape, x.shape
       both = pd.concat([t[['TIC', 'teff', 'Lum']], x[['TIC', 'teff', 'Lum']]])
       print both.shape
       both.rename(columns={'Lum':'lum'}, inplace=True)
       both.to_csv(saveLoc + 'WG4/' + 'WG4_TGASandXHIP.csv', index=False)

    sys.exit()

def TIC(data, dataset):
    """ get TIC ID and Tycho2 ID information from the TIC at https://tasoc.dk/search_tic/ """

    if dataset == 'XHIP':
        # get Tycho2 IDs and TIC IDs from TASOC Wiki
        data['HIP'] = data['HIP'].astype(str)
        data['HIP'] = 'HIP ' + data['HIP']
        data['HIP'].to_csv('/home/mxs191/Desktop/phd_y2/TESS_telecon/XHIP_TASC_list/XHIP_HIPnumbers.csv',
            index=False)
        tic = pd.read_csv('/home/mxs191/Desktop/phd_y2/TESS_telecon/XHIP_TASC_list/XHIP_TICnumbers.dat',
            skiprows=6, names=['TIC', 'HIP', 'TYC2_id']) # from https://tasoc.dk/search_tic/
        tic = tic.drop_duplicates(subset=['HIP']) # remove duplicate TIC values before merging

        # merge Tycho2 and TIC IDs with input catalogue
        data['HIP'] = data['HIP'].map(lambda x: int(str(x)[3:])) # remove 'HIP ' before merging
        data = pd.merge(left=data, right=tic[['HIP', 'TIC', 'TYC2_id']], how='left', left_on='HIP', right_on='HIP')
        print data.shape

    elif dataset == 'DR1':

        # get TIC IDs from TASOC Wiki. first: save out Tycho2 IDs
        data.rename(columns={'tycho2_id': 'TYC2_id'}, inplace=True)
        data['TYC2_id'] = data['TYC2_id'].astype(str)
        data['TYC2_id'] = 'TYC ' + data['TYC2_id']
        nans = data[data['TYC2_id']=='TYC nan'] # re-add rows with NaN Tycho2 IDs after merging
        data = data[data['TYC2_id']!='TYC nan'] # remove NaN's before getting Tycho2 IDs
        #print data.shape, nans.shape

        # save the Tycho 2 IDs to files, to be put into TASOC wiki
        # tyc2_ids = '/home/mxs191/Desktop/phd_y2/Gaia/gaia_archive/DR1_TYC2ids/'
        # flengths = np.round(np.linspace(0, len(data), 5)).astype(int)
        # for idx, val in enumerate(flengths[:-1]): # don't loop through the last element
        #     data['TYC2_id'][val:flengths[idx+1]].to_csv(tyc2_ids + 'tyc2_ids%s.csv' % idx,
        #         index=False)

        tic0 = pd.read_csv('/home/mxs191/Desktop/phd_y2/Gaia/gaia_archive/DR1_TICids/TICids0.dat',
            skiprows=6, names=['TIC', 'HIP', 'TYC2_id']) # from https://tasoc.dk/search_tic/
        tic1 = pd.read_csv('/home/mxs191/Desktop/phd_y2/Gaia/gaia_archive/DR1_TICids/TICids1.dat',
            skiprows=6, names=['TIC', 'HIP', 'TYC2_id']) # from https://tasoc.dk/search_tic/
        tic2 = pd.read_csv('/home/mxs191/Desktop/phd_y2/Gaia/gaia_archive/DR1_TICids/TICids2.dat',
            skiprows=6, names=['TIC', 'HIP', 'TYC2_id']) # from https://tasoc.dk/search_tic/
        tic3 = pd.read_csv('/home/mxs191/Desktop/phd_y2/Gaia/gaia_archive/DR1_TICids/TICids3.dat',
            skiprows=6, names=['TIC', 'HIP', 'TYC2_id']) # from https://tasoc.dk/search_tic/
        tic = pd.concat([tic0, tic1, tic2, tic3])
        tic.drop_duplicates(subset=['HIP'])

        # make the TYC2 id column the same format as on the TASOC wiki before merging:
        # 4 digits-5 digits-1 digit (with zeros before)
        s = data['TYC2_id'].str.split('-')
        s0 = [x[0].split(' ') for x in s]
        s0 = [x[1] for x in s0] # remove 'TYC ' from the first part of the identifier
        s0 = [x.rjust(4, '0') for x in s0] # add zeros before the first part of string

        s1 = [x[1] for x in s]
        s1 = [x.rjust(5, '0') for x in s1] # add zeros before the 2nd part of string
        s2 = [x[2] for x in s]
        s0 = np.char.array(s0)
        s1 = np.char.array(s1)
        s2 = np.char.array(s2)
        data['TYC2_id'] = s0 + '-' + s1 + '-' + s2

        data = pd.merge(left=data, right=tic[['TIC', 'HIP', 'TYC2_id']],
            how='left', left_on=['TYC2_id'], right_on=['TYC2_id'])
        data = pd.concat([data, nans])

    return data

def McDonald_teffs(data):
    """ Correct the XHIP and TGAS Teffs. Use McDonald Teffs rather than (B-V)
    estimates. 'both.csv' is made in bv_teff/bv_teff.py """

    mcd = pd.read_csv('/home/mxs191/Desktop/MathewSchofield/ATL/bv_teff/bv_teff1/both.csv')
    # print list(mcd)
    # print list(data)
    #print data.shape, len(mcd)
    #print data[['HIP', 'Vmag']]

    if choice == '1':
        """ match on 'teff' for the Gaia stars """
        data = pd.merge(left=data, right=mcd[['HIP', 'teff', 'Teff']], how='left')

    if choice == '2':
        """ match on 'HIP' for the XHIP stars """
        data = pd.merge(left=data, right=mcd[['HIP', 'Teff']], left_on='HIP', right_on='HIP', how='left')

    data.drop_duplicates(inplace=True)
    #print data.shape, list(data)
    #print data[['teff', 'Teff']]
    #print data[['HIP', 'Teff']][data['Teff']==data['Teff']]
    #sys.exit()


    z = [6.17303887e-01, -3.57906306e+03]  # from bv_teff/bv_teff.py Fit()
    data['corrected_teff'] = data['teff'] + data['teff']*z[0]+z[1]

    # where McDonald has a Teff value, use it instead of the relation from Fit()
    #print data[['teff', 'Teff', 'corrected_teff']].tail()
    data['corrected_teff'][data['Teff']==data['Teff']] = data['Teff'][data['Teff']==data['Teff']]
    #print data[['teff', 'Teff', 'corrected_teff']].tail()
    #print data.shape
    data.drop(['teff', 'Teff'], inplace=True, axis=1)
    data.rename(columns={'corrected_teff':'teff'}, inplace=True)
    #print data['teff'].tail()

    print data.shape, 'after changing Teffs'
    # data = pd.merge(left=data, right=mcd[['TYC2_id', 'Teff']], how='left')
    # print data.shape, list(data)
    # print data['Teff'][data['Teff']==True]
    # sys.exit()
    return data



if __name__ == '__main__':
    start = timeit.default_timer()
    saveLoc = '/home/mxs191/Desktop/MathewSchofield/ATL_public/ATL/'
    choice = '3'  # which dataset to run (1: DR2/TGAS, 2: XHIP, 3: comibned, 4: TRILEGAL)
    plx_source = 'DR2_newmask'  # The source of parallax and pixel mask size to use to produce the ATL.
                                # 'oldplx_oldmask', 'DR2_oldmask', 'DR2_newmask'.
    saveall = True  # save all parameters calculated for the stars

    if plx_source == 'oldplx_oldmask':
        saveLoc = '/home/mxs191/Desktop/MathewSchofield/ATL_public/ATL/'
    elif plx_source == 'DR2_oldmask':
        saveLoc = '/home/mxs191/Desktop/MathewSchofield/ATL_public/ATL/'
    elif plx_source == 'DR2_newmask':
        saveLoc = '/home/mxs191/Desktop/MathewSchofield/ATL_public/ATL/'
    else:
        print 'Set \"plx_source\" as old, DR2_oldmask or DR2_newmask'; sys.exit()


    if choice=='1':

        if (plx_source == 'oldplx_oldmask') or (plx_source == 'oldplx_newmask'):
            # the modified Gaia catalogue from https://gea.esac.esa.int/archive/ (made in forMat.py)
            data = pd.read_csv('/home/mxs191/Desktop/MathewSchofield/ATL/TGAS/prepTGASfiles/Gaia_final.csv')
            print data[['tycho2_id', 'Plx', 'Dist_MW']].head()

        if (plx_source == 'DR2_oldmask') or (plx_source == 'DR2_newmask'):
            data = pd.read_csv('/home/mxs191/Desktop/MathewSchofield/ATL/TGAS/prepTGASfiles_DR2/Gaia_final_DR2.csv')
            print data[['tycho2_id', 'Plx', 'r_est']].head()

        print 'Gaia', data.shape
        data = TIC(data, dataset='DR1') # get TIC and Tycho2 IDs
        #wg4()  # compare TGAS and WG4 lists

    elif choice=='2':
        """
        if (plx_source == 'oldplx_oldmask') or (plx_source == 'oldplx_newmask'):
            # the XHIP catalogue from V/137D
            data = loadData(os.getcwd() + os.sep + 'XHIP' + os.sep, \
                'XHIP_catalogue.csv', print_headers=False)
            print 'XHIP', data.shape

            data = TIC(data, dataset='XHIP') # get TIC and Tycho2 IDs
            data = Parameters(data, reddening_floc='/home/mxs191/Desktop/MathewSchofield/ATL/TESS_telecon3/reddening/')  # calculate Teff and L from (B-V) and plx. Apply cuts.

        if plx_source == 'DR2_oldmask':
            # the XHIP catalogue from V/137D with DR2 parallaxes made in
            # Large ATL files/Dr2_crossmatch/DR2_crossmatch.py
            data = pd.read_csv('/home/mxs191/Desktop/Large ATL files/DR2_crossmatch/XHIP_catalogue_DR2.csv')
            print 'XHIP', data.shape

            data = TIC(data, dataset='XHIP') # get TIC and Tycho2 IDs
            data = Parameters(data, reddening_floc='/home/mxs191/Desktop/MathewSchofield/ATL/TESS_telecon3/DR2_results/ATL_DR2oldmask/')  # calculate Teff and L from (B-V) and plx. Apply cuts.

        if plx_source == 'DR2_newmask':
            # the XHIP catalogue from V/137D with DR2 parallaxes made in
            # Large ATL files/Dr2_crossmatch/DR2_crossmatch.py
            data = pd.read_csv('/home/mxs191/Desktop/Large ATL files/DR2_crossmatch/XHIP_catalogue_DR2.csv')
            print 'XHIP', data.shape

            data = TIC(data, dataset='XHIP') # get TIC and Tycho2 IDs
            data = Parameters(data, reddening_floc='/home/mxs191/Desktop/MathewSchofield/ATL/TESS_telecon3/DR2_results/ATL_DR2newmask/')  # calculate Teff and L from (B-V) and plx. Apply cuts.
        """

        # NOTE: regardless of the plx_choice, the same XHIP parallaxes should be used
        # the XHIP catalogue from V/137D
        data = loadData(os.getcwd() + os.sep + 'XHIP' + os.sep, \
            'XHIP_catalogue.csv', print_headers=False)
        print 'XHIP', data.shape

        data = TIC(data, dataset='XHIP') # get TIC and Tycho2 IDs
        data = Parameters(data, reddening_floc='/home/mxs191/Desktop/MathewSchofield/ATL/TESS_telecon3/reddening/')  # calculate Teff and L from (B-V) and plx. Apply cuts.

    elif choice == '3':
        # combine the 2 ATLs together
        combine(saveLoc)
        stop = timeit.default_timer()
        print stop - start, "seconds."
        sys.exit()

    elif choice == '4':
        # TRILEGAL simulations for Warrick
        #data = pd.read_csv('/home/mxs191/Desktop/MathewSchofield/ATL/TRILEGAL/atl_test.csv', sep='\s+')
        data = pd.read_csv('/home/mxs191/Desktop/MathewSchofield/ATL/TRILEGAL/trilegal_for_atl.dat', sep='\s+')
        #data = pd.read_csv('/home/mxs191/Desktop/MathewSchofield/ATL/TRILEGAL/trilegal_for_atl_test.dat', sep='\s+')
        data.rename(columns={'R':'rad', 'L':'Lum', 'imag':'Imag_reddened', 'Teff':'teff',
                             'gall':'GLon', 'galb':'GLat'}, inplace=True)
        data['ELon'], data['ELat'] = gal2ecl(data['GLon'].as_matrix(), data['GLat'].as_matrix())
        #a, b = equa_2Ecl_orGal(data['gall'].as_matrix(), data['galb'].as_matrix(), ecl=True)
        #print e_lng, e_lat

        #print data.shape, list(data)
        #sys.exit()


    #data = McDonald_teffs(data)  # replace Teffs with values from McDonald


    #data = regions(data)  # separate the data into 3 regions
    data = regions2(data)  # remove stars beyond the hot edge, and numax cut-off


    # calculate seismic parameters that are not needed until the detection test is run
    cadence, vnyq, data['rad'], data['numax'], teff_solar, teffred_solar,\
        numax_solar, dnu_solar = seismicParameters(data, teff=data['teff'], lum=data['Lum'])

    # calculate observing time. For stars that lie between observing sectors,
    # set the observing time from 0 to 1 so as to not miss these stars
    T, data['max_T'] = tess_field_only(e_lng=data['ELon'].as_matrix(),
        e_lat=data['ELat'].as_matrix())
    data['max_T'][data['max_T']==0] = 1

    # calculate the detection probability for solar-like oscillations for fixed and varied beta.
    # then rank based on detection probability Pdet (when beta is fixed to 1).
    sys_limit = 0 # in ppm
    dilution = 1
    data['Pdet_fixedBeta'], data['SNR_fixedBeta'] = globalDetections(data['GLon'].as_matrix(),\
    data['GLat'].as_matrix(), data['ELon'].as_matrix(), data['ELat'].as_matrix(),\
    data['Imag_reddened'].as_matrix(), data['Lum'].as_matrix(), data['rad'].as_matrix(),\
    data['teff'].as_matrix(), data['numax'].as_matrix(), data['max_T'].as_matrix(),\
    data['tred'].as_matrix(), teff_solar, teffred_solar, numax_solar, dnu_solar,\
    sys_limit, dilution, vnyq, cadence, vary_beta=False, mask_size=plx_source)
    print len(data['Pdet_fixedBeta'][(data['Pdet_fixedBeta']>0.5)]),\
        'with detections. Beta=1'

    data['Pdet_varyBeta'], data['SNR_varyBeta'] = globalDetections(data['GLon'].as_matrix(), data['GLat'].as_matrix(), \
    data['ELon'].as_matrix(), data['ELat'].as_matrix(), data['Imag_reddened'].as_matrix(), \
    data['Lum'].as_matrix(), data['rad'].as_matrix(), data['teff'].as_matrix(), \
    data['numax'].as_matrix(), data['max_T'].as_matrix(), data['tred'].as_matrix(), \
    teff_solar, teffred_solar, numax_solar, dnu_solar, sys_limit, dilution, \
    vnyq, cadence, vary_beta=True, mask_size=plx_source)
    print len(data['Pdet_varyBeta'][(data['Pdet_varyBeta']>0.5)]),\
        'with detections. Beta=varied'

    #data = ranking(data) # sort the data. Rank by Pdet, SNR and Ic (all with different Beta values).
    data = ranking3(data, alpha=0.5)  # Rank by Pdet. This is redone in combine()

    #data = KDE(data)

    saveData(data)

    stop = timeit.default_timer()
    print stop-start, 'seconds.'











#
