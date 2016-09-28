# -*- coding: utf-8 -*-
"""
Created on Mon Feb  9 11:57:16 2015

@author: landman

The 2nd Driver routine

"""

import time
import numpy as np
from Copernicus.Master import SSU

def sampler(zmax,Np,Nret,Nsamp,Nburn,tmin,data_prior,data_lik,DoPLCF,DoTransform,err,j,fname):
    #Set sparams
    Xrho = np.array([0.25,2.8])
    XH = np.array([0.6,3.5])

    #set characteristic variance of Lambda prior (here 60%)
    sigmaLam = 0.6*3*0.7*(70.0/299.79)**2
    
    #Set z domain
    zp = np.linspace(0.0, zmax, Np)
    
    #Instantiate universe object
    U = SSU(zmax,tmin,Np,err,XH,Xrho,sigmaLam,Nret,data_prior,data_lik,fname)

    #Get starting sample
    Hz = U.Hm
    rhoz = U.rhom
    Lam = U.Lam
    logLik = U.logLik

    #Array storage for posterior samples
    Hsamps = np.zeros([Np,Nsamp])
    rhosamps = np.zeros([Np,Nsamp])
    Lamsamps = np.zeros(Nsamp)
    # musamps = np.zeros([Np,Nsamp])
    # Dsamps = np.zeros([Np,Nsamp])
    # dzdwsamps = np.zeros([Np,Nsamp])
    T2i = np.zeros([Nret,Nsamp])
    T2f = np.zeros([Nret,Nsamp])
    T1i = np.zeros([Nret,Nsamp])
    T1f = np.zeros([Nret,Nsamp])
    LLTBConsi = np.zeros([Nret,Nsamp])
    LLTBConsf = np.zeros([Nret,Nsamp])
    # rhostar = np.zeros([Nret,Nsamp])
    # Dstar = np.zeros([Nret,Nsamp])
    # Xstar = np.zeros([Nret,Nsamp])
    # Hperpstar = np.zeros([Nret,Nsamp])
    # rmax = np.zeros([Nsamp])
    # Omsamps = np.zeros([Nsamp])
    # OLsamps = np.zeros([Nsamp])
    # t0samps = np.zeros([Nsamp])

    accrate = np.zeros(2)
    
    #Do the burnin period
    t1 = time.time()
    for i in range(Nburn):
        Hz,rhoz,Lam,logLik,F,a = U.MCMCstep(logLik,Hz,rhoz,Lam)

    print 'It took sampler'+str(j),(time.time() - t1)/60.0,'min to draw ',Nburn,' samples'

    t1 = time.time()
    for i in range(Nsamp):
        Hz,rhoz,Lam,logLik,F,a = U.MCMCstep(logLik,Hz,rhoz,Lam)
        accrate += np.array([a,1])
        Hsamps[:,i] = Hz
        rhosamps[:,i] = rhoz
        Lamsamps[i] = Lam
        #Dsamps[:,i],musamps[:,i],dzdwsamps[:,i],T1i[:,i], T1f[:,i],T2i[:,i],T2f[:,i],LLTBConsi[:,i],LLTBConsf[:,i],rhostar[:,i],Dstar[:,i],Xstar[:,i],Hperpstar[:,i],rmax[i],Omsamps[i],OLsamps[i],t0samps[i] = U.get_funcs(F)
        T1i[:, i], T1f[:, i], T2i[:, i], T2f[:, i], LLTBConsi[:, i], LLTBConsf[:, i] = U.get_funcs(F)
        # print i
        # if F == 1:
        #     print "Flag raised"

    print 'It took sampler' + str(j), (time.time() - t1) / 60.0, 'min to draw ', Nsamp, ' samples'

    #print fname + "Samps"+str(j)+".npz"
    np.savez(fname + "Samps"+str(j)+".npz", Hsamps=Hsamps, rhosamps=rhosamps, Lamsamps=Lamsamps,
             T1i=T1i, T1f=T1f, T2i=T2i, T2f=T2f, LLTBConsi=LLTBConsi, LLTBConsf=LLTBConsf)


    # np.savez("Samps.npz",Hsamps=Hsamps,rhosamps=rhosamps,Lamsamps=Lamsamps,Dsamps=Dsamps,musamps=musamps,dzdwsamps=dzdwsamps,
    #          T1i=T1i,T1f=T1f,T2i=T2i,T2f=T2f,LLTBConsi=LLTBConsi,LLTBConsf=LLTBConsf,rhostar=rhostar,Dstar=Dstar,Xstar=Xstar,
    #             rmax=rmax,Omsamps=Omsamps,OLsamps=OLsamps,t0samps=t0samps,Hperpstar=Hperpstar)