#!/usr/bin/env python

# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 10:25:06 2015

@author: landman

Simulate SKA and reddrift data

"""
from numpy import linspace, array, zeros, ones, loadtxt,append, column_stack,argsort, savetxt, abs
from genFLRW import FLRW
from scipy.interpolate import UnivariateSpline as uvs
import matplotlib.pyplot as plt

if __name__ == "__main__":
    print "Getting LCDM vals"
    #Get FLRW funcs for comparison
    zmax = 2.4054
    np = 500
    nret = 250
    LCDM = FLRW(0.3,0.7,0.2335,zmax,np)
    DzF = LCDM.Dz
    HzF = LCDM.Hz
    zF = LCDM.z
    rhozF = LCDM.getrho()
    nuzF = LCDM.getnuz()
    dzdwzF = LCDM.getdzdw()


    #Do integration of FLRW funcs
    zp = linspace(0,zmax,np)

    LamF = 3*0.7*0.2335**2

    
    #Now the three functions we need are D, H and dzdw so create splines for these
    Do = uvs(zp,DzF,k=3,s=0.0)
    Ho = uvs(zp,HzF,k=3,s=0.0)
    dzdwo = uvs(zp,dzdwzF,k=3,s=0.0)
    
    #Set redshift points for Dz and Hz
    nDat = 12
    zDat = zeros(nDat)
    delz = 0.037
    zDat0 = 1.0/3
    zDat[0] = zDat0
    zDat[1] = zDat0 + 2.2*delz
    zDat[2] = zDat0 + 4.5*delz
    zDat[3] = zDat0 + 7.3*delz
    zDat[4] = zDat0 + 10.5*delz
    zDat[5] = zDat0 + 14.0*delz
    zDat[6] = zDat0 + 18.1*delz
    zDat[7] = zDat0 + 23.1*delz
    zDat[8] = zDat0 + 29.2*delz
    zDat[9] = zDat0 + 36.5*delz
    zDat[10] = zDat0 + 45.8*delz
    zDat[11] = zDat0 + 56.0*delz

    # Set redshift points for dzdw
    zDatdzdw = zeros(4)
    zDatdzdw[0] = 1.0
    zDatdzdw[1] = 1.4
    zDatdzdw[2] = 1.85
    zDatdzdw[3] = 2.3

    #Get func values at these points
    D = Do(zDat)
    H = Ho(zDat)
    dzdw = dzdwo(zDatdzdw)
    errdzdw = zeros(4)
    errdzdw[0] = dzdw[0] * 0.4
    errdzdw[1] = dzdw[1] * 0.65
    errdzdw[2] = dzdw[2] * 1.75
    errdzdw[3] = abs(dzdw[3]) * 2.2
    
    #Set error at these points
    delz = zmax - zDat0
    errDf = 0.07
    errD0 = 0.02
    mD = (errDf - errD0)/delz
    errD = (errD0 + mD*(zDat - zDat0))*D
    errHf = 0.02
    errH0 = 0.01    
    mH = (errHf - errH0)/delz*H
    errH = errH0 + mH*(zDat - zDat0)

    #Load the data sets to append these to
    #zD0, Dz0, sDz0 = loadtxt('/home/landman/Projects/Data/CP_data/RawData/Unionrz.txt',unpack=True)
    #zH0, Hz0, sHz0 = loadtxt('/home/landman/Projects/Data/CP_data/RawData/CChz.txt',unpack=True)
    
    #Append the data
    zD = zDat #append(zD0,zDat)
    Dz = D #append(Dz0,D)
    sDz = errD #append(sDz0,errD)
    zH = zDat #append(zH0,zDat)
    Hz = H #append(Hz0,H)
    sHz = errH #append(sHz0,errH)
    
    #Sort and save the data
    I = argsort(zH)
    zH = zH[I]
    Hz = Hz[I]
    sHz = sHz[I]
    saveH = column_stack((zH,Hz,sHz))
    Hf = open('/home/landman/Projects/Data/CP_data/SKAH.txt','w')
    savetxt(Hf,saveH,fmt='%s')
    Hf.close()
    I = argsort(zD)
    zD = zD[I]
    Dz = Dz[I]
    sDz = sDz[I]
    saveD = column_stack((zD,Dz,sDz))
    Df = open('/home/landman/Projects/Data/CP_data/SKAD.txt','w')
    savetxt(Df,saveD,fmt='%s')
    Df.close()
    savedzdw = column_stack((zDatdzdw,dzdw,errdzdw))
    dzdwf = open('/home/landman/Projects/Data/CP_data/SKAdzdw.txt','w')
    savetxt(dzdwf,savedzdw,fmt='%s')
    dzdwf.close()

    #Plot
    plt.figure('D')
    plt.plot(zp,DzF)
    plt.errorbar(zD,Dz,sDz,fmt='xr')
    plt.show()
    plt.figure('H')
    plt.plot(zp,HzF)
    plt.errorbar(zH,Hz,sHz,fmt='xr')
    plt.show()
    plt.figure('dzdw')
    plt.plot(zp,dzdwzF)
    plt.errorbar(zDatdzdw,dzdw,errdzdw,fmt='xr')
    plt.show()