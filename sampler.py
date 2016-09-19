# -*- coding: utf-8 -*-
"""
Created on Mon Feb  9 11:57:16 2015

@author: landman

The 2nd Driver routine

"""

import sys
#sys.path.insert(0, '/home/bester/Algo_Pap') #On cluster
#sys.path.insert(0, '/home/landman/Algorithm') #At home
import time
from numpy import linspace, array,zeros, savez, exp, sqrt, pi, delete
from Master import SSU

def sampler(zmax,np,nret,nsamps,nburn,tmin,DoPLCF,DoTransform,err,i):
    #Set sparams
    Xrho = array([0.25,2.8]) 
    XH = array([0.6,3.5])

    #set characteristic variance of Lambda prior (here 60%)
    sigmaLam = 0.6*3*0.7*(70.0/299.79)**2
    
    #Set z domain
    zp = linspace(0,zmax,np)
    
    #Instantiate universe object
    U = SSU(zmax,tmin,np,err,XH,Xrho,sigmaLam,nret)

    #Get starting sample
    Hz = U.Hm
    rhoz = U.rhom
    Lam = U.Lam
    logLik = U.logLik
    
    #Set sample sizes
    nsamp = 5000  #Number of MCMC samples to draw

    #Array storage for posterior samples
    Hsamps = zeros([np,nsamp])
    rhosamps = zeros([np,nsamp])
    Lamsamps = zeros(nsamp)
    musamps = zeros([np,nsamp])
    Dsamps = zeros([np,nsamp])
    dzdwsamps = zeros([np,nsamp])
    T2i = zeros([nret,nsamp])
    T2f = zeros([nret,nsamp])
    T1i = zeros([nret,nsamp])
    T1f = zeros([nret,nsamp])
    LLTBConsi = zeros([nret,nsamp])
    LLTBConsf = zeros([nret,nsamp])
    rhostar = zeros([nret,nsamp])
    Dstar = zeros([nret,nsamp])
    Xstar = zeros([nret,nsamp])
    Hperpstar = zeros([nret,nsamp])
    rmax = zeros([nsamp])
    Omsamps = zeros([nsamp])
    OLsamps = zeros([nsamp])
    t0samps = zeros([nsamp])

    accrate = zeros(2)
    
    #Do the burnin period
    print "Burning"
    t1 = time.time()
    for i in range(nsamp):
        Hz,rhoz,Lam,logLik,F,a = U.MCMCstep(logLik,Hz,rhoz,Lam)
        accrate = accrate + array([a,1])
        Hsamps[:,i] = Hz
        rhosamps[:,i] = rhoz
        Lamsamps[i] = Lam
        Dsamps[:,i],musamps[:,i],dzdwsamps[:,i],T1i[:,i], T1f[:,i],T2i[:,i],T2f[:,i],LLTBConsi[:,i],LLTBConsf[:,i],rhostar[:,i],Dstar[:,i],Xstar[:,i],Hperpstar[:,i],rmax[i],Omsamps[i],OLsamps[i],t0samps[i] = U.get_funcs(F)
        print i
        if F == 1:
            print "Flag raised"
    print 'It took ',(time.time() - t1)/60.0,'min to draw ',nsamp,' samples'
    savez("Samps.npz",Hsamps=Hsamps,rhosamps=rhosamps,Lamsamps=Lamsamps,Dsamps=Dsamps,musamps=musamps,dzdwsamps=dzdwsamps,T1i=T1i,T1f=T1f,T2i=T2i,T2f=T2f,LLTBConsi=LLTBConsi,LLTBConsf=LLTBConsf,rhostar=rhostar,Dstar=Dstar,Xstar=Xstar,rmax=rmax,Omsamps=Omsamps,OLsamps=OLsamps,t0samps=t0samps,Hperpstar=Hperpstar) 