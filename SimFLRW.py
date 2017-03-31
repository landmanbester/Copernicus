# -*- coding: utf-8 -*-
"""
Created on Thu Aug 21 09:33:52 2014

@author: landman

This program simulates data for an LCDM model

"""

from numpy import linspace, zeros, sort, random, eye,column_stack,concatenate, savetxt,loadtxt,histogram,diag,array,ones,log10
from scipy.interpolate import UnivariateSpline as uvs
import matplotlib.pyplot as plt
import matplotlib as mpl
from genFLRW import FLRW

from Copernicus.Parset import MyOptParse

if __name__=="__main__":
    # Get input args
    GD = MyOptParse.readargs()

    #Determine how many samplers to spawn
    fname = GD["fname"]
    zmax = GD["zmax"]
    h_sigma = GD["h_sigma"]
    rho_sigma = GD["rho_sigma"]


    #set z values for data points
    nD = 500
    nH = 50
    nrho = 10

    # zD = sort(0.005 + random.ranf(nD)*(zmax-0.005))
    # zH = sort(random.ranf(nH)*zmax)
    # zH[0] = 0
    zrho = linspace(0,zmax,nrho) #sort(random.ranf(nrho)*zmax)

    #zD = sort(random.beta(2,5,nD)*zmax)
    #zD[0] = 0  #make sure its zero at the origin
    #zH = sort(random.beta(2,5,nH)*zmax)
    #zH[0] = 0
    #zrho = sort(random.beta(2,5,nrho)*zmax)

    #zmax1 = max(zD)
    #zmax2 = max(zH)
    #zmax = max(zmax1,zmax2)

    z = linspace(0,zmax,nD)

    #Set how the error grows
    alpha = 0.225
    #errD = (1+zD)**alpha
    #errH = (1+zH)**alpha
    errrho = (1+zrho)**alpha

    #set universe parameters
    Om = 0.3
    OL = 0.7
    H0 = 0.2335
    #create FLRW object to get H and D splines
    LCDM = FLRW(Om,OL,H0,zmax,500)
    #Ho = LCDM.getHo()
    #Do = LCDM.getDo()
    rhoo = LCDM.getrhoo()

    #get functions values at these points
    #H = Ho(zH)
    #D = Do(zD)
    rho = rhoo(zrho)

    #set number of trials (controls Gaussianity of distribution)
    N = 21
    #delD = 0.05
    #delH = 0.1
    delrho = 0.75

    #Do simulation
    # SimD = zeros([N,nD])
    # SimH = zeros([N,nH])
    Simrho = zeros([N,nrho])
    # zND = zeros(nD)
    # zNH = zeros(nH)
    zNrho = zeros(nrho)
    # eyeND = eye(nD)
    # eyeNH = eye(nH)
    eyeNrho = eye(nrho)
    #eyeH = diag(concatenate((array([0.05]),ones(nH-1))))
    for i in range(N):
    #    SimD[i,:] = D + errD*delD*D*random.multivariate_normal(zND,eyeND)
    #    SimH[i,:] = H + errH*delH*H*random.multivariate_normal(zNH,eyeNH)
        Simrho[i, :] = rho + errrho*delrho*rho*random.multivariate_normal(zNrho, eyeNrho)

    ##sort columns in ascending order
    #SimD.sort(axis=0)
    #SimH.sort(axis=0)
    #Simrho.sort(axis=0)
    ##get mean and 1-sigma error bars
    #meanD = SimD[N/2,:]
    #sigD = SimD[0.16*N,:]
    #meanmu = 5*log10(1.0e8*(1+zD)**2*meanD)
    #sigmu = 5*log10(1.0e8*(1+zD)**2*sigD)
    #mu = 5*log10(1.0e8*(1+z)**2*Do(z))
    #meanH = SimH[N/2,:]
    #sigH = SimH[0.16*N,:]
    meanrho = rhoo(zrho)
    sigrho = delrho*errrho*rho #Simrho[0.16*N,:]

    ###save data
    #savemu = column_stack((zD,meanmu,(meanmu-sigmu)))
    #muf = open('RawData/Simmu.txt','w')
    #savetxt(muf,savemu,fmt='%s')
    #muf.close()
    #saveH = column_stack((zH,meanH,(meanH-sigH)))
    #Hf = open('RawData/SimH.txt','w')
    #savetxt(Hf,saveH,fmt='%s')
    #Hf.close()
    saverho = column_stack((zrho,meanrho,sigrho))
    rhof = open(fname+'Data/rho.txt','w')
    savetxt(rhof,saverho,fmt='%s')
    rhof.close()

    #mpl.rcParams.update({'font.size': 20, 'font.family': 'serif'})
    #
    #plt.figure('Dsim')
    #plt.errorbar(zD,meanmu,(meanmu-sigmu),fmt='xr')
    #plt.plot(z,mu,linewidth=2)
    #
    #plt.figure('Hsim')
    #plt.errorbar(zH,meanH,(meanH-sigH),fmt='xr')
    #plt.plot(z,Ho(z),linewidth=2)

    plt.figure('rhosim')
    plt.errorbar(zrho,meanrho,sigrho,fmt='xr')
    plt.plot(z,rhoo(z),linewidth=2)
    plt.show()

    ##%%
    #plt.figure('HistD')
    #plt.xlabel(r'$\frac{\delta  D}{D}$',fontsize=30)
    #plt.ylabel(r'$ Freq $',fontsize=30)
    #histD = (D[1::] - meanD[1::])/D[1::]
    #plt.hist(histD,bins=nD/25,normed=True)
    #
    #plt.figure('HistH')
    #plt.xlabel(r'$\frac{\delta  H}{H}$',fontsize=30)
    #plt.ylabel(r'$ Freq $',fontsize=30)
    #histH = (H - meanH)/H
    #plt.hist(histH,bins=nH/25,normed=True)
    ##%%
