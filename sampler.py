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
    Xrho = np.array([0.5, 2.8])
    XH = np.array([0.6, 3.5])

    #set characteristic variance of Lambda prior (here 60%)
    sigmaLam = 0.5*3*0.7*(70.0/299.79)**2
    
    #Set z domain
    zp = np.linspace(0.0, zmax, Np)
    
    #Instantiate universe object
    U = SSU(zmax, tmin, Np, err, XH, Xrho, sigmaLam, Nret, data_prior, data_lik, fname)

    #Get starting sample
    Hz = U.Hm
    rhoz = U.rhom
    Lam = U.Lam
    logLik = U.logLik

    #Array storage for posterior samples
    Hzsamps = np.zeros([Np,Nsamp])
    rhozsamps = np.zeros([Np,Nsamp])
    Lamsamps = np.zeros(Nsamp)
    # musamps = np.zeros([Np,Nsamp])
    Di = np.zeros([Nret,Nsamp])
    Df = np.zeros([Nret,Nsamp])
    Si = np.zeros([Nret,Nsamp])
    Sf = np.zeros([Nret, Nsamp])
    Qi = np.zeros([Nret,Nsamp])
    Qf = np.zeros([Nret, Nsamp])
    Ai = np.zeros([Nret,Nsamp])
    Af = np.zeros([Nret, Nsamp])
    Zi = np.zeros([Nret,Nsamp])
    Zf = np.zeros([Nret, Nsamp])
    Spi = np.zeros([Nret,Nsamp])
    Spf = np.zeros([Nret, Nsamp])
    Qpi = np.zeros([Nret, Nsamp])
    Qpf = np.zeros([Nret, Nsamp])
    Zpi = np.zeros([Nret, Nsamp])
    Zpf = np.zeros([Nret, Nsamp])
    ui = np.zeros([Nret,Nsamp])
    uf = np.zeros([Nret,Nsamp])
    upi = np.zeros([Nret,Nsamp])
    upf = np.zeros([Nret,Nsamp])
    uppi = np.zeros([Nret,Nsamp])
    uppf = np.zeros([Nret,Nsamp])
    udoti = np.zeros([Nret,Nsamp])
    udotf = np.zeros([Nret,Nsamp])
    rhoi = np.zeros([Nret,Nsamp])
    rhof = np.zeros([Nret,Nsamp])
    rhopi = np.zeros([Nret,Nsamp])
    rhopf = np.zeros([Nret,Nsamp])
    rhodoti = np.zeros([Nret,Nsamp])
    rhodotf = np.zeros([Nret,Nsamp])
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
        Hzsamps[:,i] = Hz
        rhozsamps[:,i] = rhoz
        Lamsamps[i] = Lam
        #print "Sampler" + str(j), i, Lam, Hz[0], np.random.randn(1)
        #Dsamps[:,i],musamps[:,i],dzdwsamps[:,i],T1i[:,i], T1f[:,i],T2i[:,i],T2f[:,i],LLTBConsi[:,i],LLTBConsf[:,i],rhostar[:,i],Dstar[:,i],Xstar[:,i],Hperpstar[:,i],rmax[i],Omsamps[i],OLsamps[i],t0samps[i] = U.get_funcs(F)
        T1i[:, i], T1f[:, i], T2i[:, i], T2f[:, i], LLTBConsi[:, i], LLTBConsf[:, i], Di[:, i], Df[:, i], Si[:, i], \
        Sf[:, i], Qi[:, i], Qf[:, i], Ai[:, i], Af[:, i], Zi[:, i], Zf[:, i], Spi[:, i], Spf[:, i], Qpi[:, i], Qpf[:, i], \
        Zpi[:, i], Zpf[:, i], ui[:, i], uf[:, i], upi[:, i], upf[:, i], uppi[:, i], uppf[:, i], udoti[:, i], udotf[:, i], \
        rhoi[:, i], rhof[:, i], rhopi[:, i], rhopf[:, i], rhodoti[:, i], rhodotf[:, i] = U.get_funcs()
        # print i
        # if F == 1:
        #     print "Flag raised"

    print 'It took sampler' + str(j), (time.time() - t1) / 60.0, 'min to draw ', Nsamp, ' samples with an acceptance rate of ', accrate[0]/accrate[1]

    return Hzsamps, rhozsamps, Lamsamps, T1i, T1f, T2i, T2f, LLTBConsi, LLTBConsf, Di, Df, Si, Sf, Qi, Qf, Ai, Af, Zi, \
           Zf, Spi, Spf, Qpi, Qpf, Zpi, Zpf, ui, uf, upi, upf, uppi, uppf, udoti, udotf, rhoi, rhof, rhopi, rhopf, \
           rhodoti, rhodotf

    #print fname + "Samps"+str(j)+".npz"
    #print Hsamps.shape, rhosamps.shape, Lamsamps.shape, T1i.shape, T1f.shape, T2i.shape, T2f.shape, LLTBConsi.shape, LLTBConsf.shape
    #np.savez(fname + "Samps"+str(j)+".npz", Hsamps=Hsamps, rhosamps=rhosamps, Lamsamps=Lamsamps,
    #         T1i=T1i, T1f=T1f, T2i=T2i, T2f=T2f, LLTBConsi=LLTBConsi, LLTBConsf=LLTBConsf)


    # np.savez("Samps.npz",Hsamps=Hsamps,rhosamps=rhosamps,Lamsamps=Lamsamps,Dsamps=Dsamps,musamps=musamps,dzdwsamps=dzdwsamps,
    #          T1i=T1i,T1f=T1f,T2i=T2i,T2f=T2f,LLTBConsi=LLTBConsi,LLTBConsf=LLTBConsf,rhostar=rhostar,Dstar=Dstar,Xstar=Xstar,
    #             rmax=rmax,Omsamps=Omsamps,OLsamps=OLsamps,t0samps=t0samps,Hperpstar=Hperpstar)