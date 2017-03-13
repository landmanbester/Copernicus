# -*- coding: utf-8 -*-
"""
Created on Mon Feb  9 11:57:16 2015

@author: landman

The 2nd Driver routine

"""

import traceback
import time
import numpy as np
from Copernicus.Master import SSU

def sampler(*args, **kwargs):
    """
    This here is to get the stack trace when an exception is thrown
    """
    try:
        return sampler_impl(*args, **kwargs)
    except Exception as e:
        traceback_str = traceback.format_exc(e)
        raise StandardError("Error occurred. Original traceback "
                            "is\n%s\n" % traceback_str)

def sampler_impl(zmax,Np,Nret,Nsamp,Nburn,tmin,data_prior,data_lik,DoPLCF,DoTransform,err,j,fname,beta,Hz,rhoz,Lam,use_meanf):
    #Set sparams
    Xrho = np.array([0.8, 3.5])
    XH = np.array([0.6, 3.5])

    #set characteristic variance of Lambda prior (here 60%)
    sigmaLam = 0.1*3*0.7*(70.0/299.79)**2
    
    #Set z domain
    zp = np.linspace(0.0, zmax, Np)

    #Instantiate universe object
    if use_meanf:
        U = SSU(zmax, tmin, Np, err, XH, Xrho, sigmaLam, Nret, data_prior, data_lik, fname, DoPLCF,
                beta=beta, Hz=Hz, rhoz=rhoz, Lam=Lam)
    else:
        U = SSU(zmax, tmin, Np, err, XH, Xrho, sigmaLam, Nret, data_prior, data_lik, fname, DoPLCF, beta=beta)

    #Get starting sample
    Hz = U.Hz
    rhoz = U.rhoz
    Lam = U.Lam
    logLik = U.logLik

    #Array storage for posterior samples
    Dzsamps = np.zeros([Np, Nsamp])
    Hzsamps = np.zeros([Np,Nsamp])
    rhozsamps = np.zeros([Np,Nsamp])
    dzdwzsamps = np.zeros([Np, Nsamp])
    Lamsamps = np.zeros(Nsamp)
    t0samps = np.zeros(Nsamp)
    # musamps = np.zeros([Np,Nsamp])
    Di = np.zeros([Nret,Nsamp])
    Si = np.zeros([Nret,Nsamp])
    Qi = np.zeros([Nret,Nsamp])
    Ai = np.zeros([Nret,Nsamp])
    Zi = np.zeros([Nret,Nsamp])
    Spi = np.zeros([Nret,Nsamp])
    Qpi = np.zeros([Nret, Nsamp])
    Zpi = np.zeros([Nret, Nsamp])
    ui = np.zeros([Nret,Nsamp])
    upi = np.zeros([Nret,Nsamp])
    uppi = np.zeros([Nret,Nsamp])
    udoti = np.zeros([Nret,Nsamp])
    rhoi = np.zeros([Nret,Nsamp])
    rhopi = np.zeros([Nret,Nsamp])
    rhodoti = np.zeros([Nret,Nsamp])
    T2i = np.zeros([Nret,Nsamp])
    T1i = np.zeros([Nret,Nsamp])
    sigmasqi = np.zeros([Nret,Nsamp])
    LLTBConsi = np.zeros([Nret,Nsamp])

    if DoPLCF:
        Df = np.zeros([Nret, Nsamp])
        Sf = np.zeros([Nret, Nsamp])
        Qf = np.zeros([Nret, Nsamp])
        Af = np.zeros([Nret, Nsamp])
        Zf = np.zeros([Nret, Nsamp])
        Spf = np.zeros([Nret, Nsamp])
        Qpf = np.zeros([Nret, Nsamp])
        Zpf = np.zeros([Nret, Nsamp])
        uf = np.zeros([Nret, Nsamp])
        upf = np.zeros([Nret, Nsamp])
        uppf = np.zeros([Nret, Nsamp])
        udotf = np.zeros([Nret, Nsamp])
        rhof = np.zeros([Nret, Nsamp])
        rhopf = np.zeros([Nret, Nsamp])
        rhodotf = np.zeros([Nret, Nsamp])
        T2f = np.zeros([Nret, Nsamp])
        T1f = np.zeros([Nret, Nsamp])
        sigmasqf = np.zeros([Nret, Nsamp])
        LLTBConsf = np.zeros([Nret, Nsamp])


    I = []
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
    print "Sampler ",j, "started burnin"
    U.set_DoPLCF(False) # don't need to compute the PLCF during burnin
    t1 = time.time()
    for i in range(Nburn):
        Hz,rhoz,Lam,logLik,F,a = U.MCMCstep(logLik,Hz,rhoz,Lam)
        U.track_max_lik(logLik,Hz,rhoz,Lam)
        accrate += np.array([a, 1])

    print 'It took sampler'+str(j),(time.time() - t1)/60.0,'min to draw ',Nburn,' samples with an accrate of ', accrate[0]/accrate[1]

    # Reset the lambda prior
    U.set_Lambda_Prior(U.Hz, U.rhoz)
    U.set_DoPLCF(DoPLCF)
    interval = Nsamp/10
    accrate = np.zeros(2)
    t1 = time.time()
    for i in range(Nsamp):
        if i%Nburn == 0:
            print "Sampler ",j," is at sample ",i
        if i%interval == 0 and i != 0:
            #Check acceptance rate
            arate = accrate[0]/accrate[1]
            if arate < 0.2:
                beta /= 1.3
                U.reset_beta(beta)
                print "Acceptance rate of ", arate," is too low. Resetting beta to ", beta, i, j
            elif arate > 0.5:
                beta *= 1.3
                U.reset_beta(beta)
                print " Acceptance rate of ", arate, " is too high. Resetting beta to ", beta, i, j
        Hz,rhoz,Lam,logLik,F,a = U.MCMCstep(logLik,Hz,rhoz,Lam)
        accrate += np.array([a,1])
        Hzsamps[:,i] = Hz
        rhozsamps[:,i] = rhoz
        Lamsamps[i] = Lam

        T1i[:, i], T2i[:, i], LLTBConsi[:, i], Di[:, i], Si[:, i], Qi[:, i], Ai[:, i], Zi[:, i], Spi[:, i], \
        Qpi[:, i], Zpi[:, i], ui[:, i], upi[:, i], uppi[:, i], udoti[:, i], rhoi[:, i], rhopi[:, i], rhodoti[:, i], \
        Dzsamps[:, i], dzdwzsamps[:, i], sigmasqi[:, i], t0samps[i] = U.get_funcsi()


        if t0samps[i] > U.tmin and U.NI > 1:
            T1f[:, i], T2f[:, i], LLTBConsf[:, i], Df[:, i], Sf[:, i], Qf[:, i], Af[:, i], Zf[:, i], Spf[:, i], \
            Qpf[:, i], Zpf[:, i], uf[:, i], upf[:, i], uppf[:, i], udotf[:, i], rhof[:, i], rhopf[:, i], rhodotf[:, i],\
            sigmasqf[:, i] = U.get_funcsf()
        else:
            I.append(i)
    # Delete the empty columns in PLCF quantities that result when t0 < tfind
    np.delete(T1f, I, axis=1)
    np.delete(T2f, I, axis=1)
    np.delete(LLTBConsf, I, axis=1)
    np.delete(Df, I, axis=1)
    np.delete(Sf, I, axis=1)
    np.delete(Qf, I, axis=1)
    np.delete(Af, I, axis=1)
    np.delete(Zf, I, axis=1)
    np.delete(Spf, I, axis=1)
    np.delete(Qpf, I, axis=1)
    np.delete(Zpf, I, axis=1)
    np.delete(uf, I, axis=1)
    np.delete(upf, I, axis=1)
    np.delete(uppf, I, axis=1)
    np.delete(udotf, I, axis=1)
    np.delete(rhof, I, axis=1)
    np.delete(rhopf, I, axis=1)
    np.delete(rhodotf, I, axis=1)
    np.delete(sigmasqf, I, axis=1)

    # Report
    print 'It took sampler' + str(j), (time.time() - t1) / 60.0, 'min to draw ', Nsamp, \
        ' samples with an acceptance rate of ', accrate[0]/accrate[1]

    if DoPLCF:
        return Hzsamps, rhozsamps, Lamsamps, T1i, T1f, T2i, T2f, LLTBConsi, LLTBConsf, Di, Df, Si, Sf, Qi, Qf, Ai, Af, Zi, \
               Zf, Spi, Spf, Qpi, Qpf, Zpi, Zpf, ui, uf, upi, upf, uppi, uppf, udoti, udotf, rhoi, rhof, rhopi, rhopf, \
               rhodoti, rhodotf, Dzsamps, dzdwzsamps, sigmasqi, sigmasqf, t0samps
    else:
        return Hzsamps, rhozsamps, Lamsamps, T1i, T2i, LLTBConsi, Di, Si, Qi, Ai, Zi, Spi, Qpi, Zpi, ui, upi, uppi, \
               udoti, rhoi, rhopi, rhodoti, Dzsamps, dzdwzsamps, sigmasqi, t0samps