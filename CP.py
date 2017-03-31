#!/usr/bin/env python
import sys
import numpy as np
import concurrent.futures as cf
from Copernicus.sampler import sampler
from Copernicus import MCMC_Tools as MCT
from genFLRW import FLRW
from Copernicus.Parset import MyOptParse


if __name__ == "__main__":
    #sys.settrace(sampler)
    # Get input args
    GD = MyOptParse.readargs()

    #Determine how many samplers to spawn
    NSamplers = GD["nwalkers"]
    Nsamp = GD["nsamples"]
    Nburn = GD["nburnin"]
    tstar = GD["tstar"]
    DoPLCF = GD["doplcf"]
    DoTransform = GD["dotransform"]
    fname = GD["fname"]
    data_prior = GD["data_prior"]
    data_lik = GD["data_lik"]
    zmax = GD["zmax"]
    Np = GD["np"]
    Nret = GD["nret"]
    err = GD["err"]
    beta = GD["beta"]
    use_meanf = GD["use_meanf"]
    samps_out_name = GD["samps_out_name"]
    h_sigma = GD["h_sigma"]
    rho_sigma = GD["rho_sigma"]
    sigma_lower = np.array([h_sigma, rho_sigma])

    # Print out parset settings
    keyslist = GD.keys()
    for it in keyslist:
        print it, GD[it]

    futures = []
    Hzlist = []
    Dzlist = []
    rhozlist = []
    dzdwzlist = []
    Lamlist = []
    T1ilist = []
    T2ilist = []
    sigmasqilist = []
    LLTBConsilist = []
    Dilist = []
    Silist = []
    Qilist = []
    Ailist = []
    Zilist = []
    Spilist = []
    Qpilist = []
    Zpilist = []
    uilist = []
    upilist = []
    uppilist = []
    udotilist = []
    rhoilist = []
    rhopilist = []
    rhodotilist = []
    t0list = []


    if DoPLCF:
        T1flist = []
        T2flist = []
        sigmasqflist = []
        LLTBConsflist = []
        Dflist = []
        Sflist = []
        Qflist = []
        Aflist = []
        Zflist = []
        Spflist = []
        Qpflist = []
        Zpflist = []
        uflist = []
        upflist = []
        uppflist = []
        udotflist = []
        rhoflist = []
        rhopflist = []
        rhodotflist = []

    # Get FLRW funcs for comparison
    Om0 = 0.3
    OL0 = 0.7
    H0 = 0.2335
    LCDM = FLRW(Om0, OL0, H0, zmax, Np)
    HzF = LCDM.Hz
    rhozF = LCDM.getrho()
    Lam = 3 * OL0 * H0 ** 2

    #sampler.sampler(zmax,Np,Nret,Nsamp,Nburn,tstar,data_prior,data_lik,DoPLCF,DoTransform,err,0,fname)
    #Create a pool for this number of samplers and submit the jobs
    Hsamps = np.zeros([NSamplers,Np,Nsamp])
    cont = True
    with cf.ProcessPoolExecutor(max_workers=NSamplers) as executor:
        for i in xrange(NSamplers):
            future = executor.submit(sampler, zmax, Np, Nret, Nsamp, Nburn, tstar, data_prior, data_lik, DoPLCF,
                                     DoTransform, err, i, fname, beta, HzF, rhozF, Lam, use_meanf, sigma_lower=sigma_lower)
            futures.append(future)
        k = 0
        for f in cf.as_completed(futures):
            if DoPLCF:
                Hz, rhoz, Lam, T1i, T1f, T2i, T2f, LLTBConsi, LLTBConsf, Di, Df, Si, Sf, Qi, Qf, Ai, Af, Zi, \
                Zf, Spi, Spf, Qpi, Qpf, Zpi, Zpf, ui, uf, upi, upf, uppi, uppf, udoti, udotf, rhoi, rhof, rhopi, rhopf, \
                rhodoti, rhodotf, Dz, dzdwz, sigmasqi, sigmasqf, t0 = f.result()
            else:
                Hz, rhoz, Lam, T1i, T2i, LLTBConsi, Di, Si, Qi, Ai, Zi, Spi, Qpi, Zpi, ui, upi, uppi, udoti, rhoi, \
                rhopi, rhodoti, Dz, dzdwz, sigmasqi, t0 = f.result()

            Dzlist.append(Dz)
            Hzlist.append(Hz)
            rhozlist.append(rhoz)
            dzdwzlist.append(dzdwz)
            Lamlist.append(Lam)
            T1ilist.append(T1i)
            T2ilist.append(T2i)
            sigmasqilist.append(sigmasqi)
            LLTBConsilist.append(LLTBConsi)
            Dilist.append(Di)
            Silist.append(Si)
            Qilist.append(Qi)
            Ailist.append(Ai)
            Zilist.append(Zi)
            Spilist.append(Spi)
            Qpilist.append(Qpi)
            Zpilist.append(Zpi)
            uilist.append(ui)
            upilist.append(upi)
            uppilist.append(uppi)
            udotilist.append(udoti)
            rhoilist.append(rhoi)
            rhopilist.append(rhopi)
            rhodotilist.append(rhodoti)
            t0list.append(t0)

            if DoPLCF:
                T1flist.append(T1f)
                T2flist.append(T2f)
                sigmasqflist.append(sigmasqf)
                LLTBConsflist.append(LLTBConsf)
                Dflist.append(Df)
                Sflist.append(Sf)
                Qflist.append(Qf)
                Aflist.append(Af)
                Zflist.append(Zf)
                Spflist.append(Spf)
                Qpflist.append(Qpf)
                Zpflist.append(Zpf)
                uflist.append(uf)
                upflist.append(upf)
                uppflist.append(uppf)
                udotflist.append(udotf)
                rhoflist.append(rhof)
                rhopflist.append(rhopf)
                rhodotflist.append(rhodotf)

        Htest = MCT.MCMC_diagnostics(NSamplers, Hzlist).get_GRC().max()
        rhotest = MCT.MCMC_diagnostics(NSamplers, rhozlist).get_GRC().max()
        Lamtest = MCT.MCMC_diagnostics(NSamplers, Lamlist).get_GRC()

        # Test for convergence
        test_GR = np.array([Htest,rhotest,Lamtest])

        try:
            cont = any(test_GR > 1.15)
        except:
            cont = True
            print "Something went wrong in GR test"

        if cont:
            print "Gelman-Rubin indicates non-convergence"

        print "GR(H) = ", Htest, "GR(rho) = ", rhotest, "GR(Lambda) = ", Lamtest

    # Save the data
    if DoPLCF:
        np.savez(fname + "Processed_Data/" + samps_out_name + ".npz", Hz=Hzlist, rhoz=rhozlist, Lam=Lamlist, T1i=T1ilist, T1f=T1flist, T2i=T2ilist,
                 T2f=T2flist, LLTBConsi=LLTBConsilist, LLTBConsf=LLTBConsflist, Di=Dilist, Df=Dflist, Si=Silist, Sf=Sflist,
                 Qi=Qilist, Qf=Qflist, Ai=Ailist, Af=Aflist, Zi=Zilist, Zf=Zflist, Spi=Spilist, Spf=Spflist, Qpi=Qpilist,
                 Qpf=Qpflist, Zpi=Zpilist, Zpf=Zpflist, ui=uilist, uf=uflist, upi=upilist, upf=upflist, uppi=uppilist,
                 uppf=uppflist, udoti=udotilist, udotf=udotflist, rhoi=rhoilist, rhof=rhoflist, rhopi=rhopilist,
                 rhopf=rhopflist, rhodoti=rhodotilist, rhodotf=rhodotflist, NSamplers=NSamplers, Dz=Dzlist, dzdwz=dzdwzlist,
                 sigmasqi=sigmasqilist, sigmasqf=sigmasqflist, t0list=t0list)
    else:
        np.savez(fname + "Processed_Data/" + samps_out_name + ".npz", Hz=Hzlist, rhoz=rhozlist, Lam=Lamlist,
                 T1i=T1ilist, T2i=T2ilist, LLTBConsi=LLTBConsilist, Di=Dilist, Si=Silist, Qi=Qilist, Ai=Ailist,
                 Zi=Zilist, Spi=Spilist, Qpi=Qpilist, Zpi=Zpilist, ui=uilist, upi=upilist, uppi=uppilist,
                 udoti=udotilist, rhoi=rhoilist, rhopi=rhopilist, rhodoti=rhodotilist, NSamplers=NSamplers, Dz=Dzlist,
                 dzdwz=dzdwzlist, sigmasqi=sigmasqilist, t0list=t0list)