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

    futures = []
    Hzlist = []
    Dzlist = []
    rhozlist = []
    dzdwzlist = []
    Lamlist = []
    T1ilist = []
    T1flist = []
    T2ilist = []
    T2flist = []
    sigmasqilist = []
    sigmasqflist = []
    LLTBConsilist = []
    LLTBConsflist = []
    Dilist = []
    Dflist = []
    Silist = []
    Sflist = []
    Qilist = []
    Qflist = []
    Ailist = []
    Aflist = []
    Zilist = []
    Zflist = []
    Spilist = []
    Spflist = []
    Qpilist = []
    Qpflist = []
    Zpilist = []
    Zpflist = []
    uilist = []
    uflist = []
    upilist = []
    upflist = []
    uppilist = []
    uppflist = []
    udotilist = []
    udotflist = []
    rhoilist = []
    rhoflist = []
    rhopilist = []
    rhopflist = []
    rhodotilist = []
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
    num_repeats = 0
    max_repeats = 1
    while cont and num_repeats < max_repeats:
        with cf.ThreadPoolExecutor(max_workers=NSamplers) as executor:
            for i in xrange(NSamplers):
                future = executor.submit(sampler, zmax, Np, Nret, Nsamp, Nburn, tstar, data_prior, data_lik, DoPLCF,
                                         DoTransform, err, i, fname, beta, HzF, rhozF, Lam, use_meanf)
                futures.append(future)
            k = 0
            for f in cf.as_completed(futures):
                Hz, rhoz, Lam, T1i, T1f, T2i, T2f, LLTBConsi, LLTBConsf, Di, Df, Si, Sf, Qi, Qf, Ai, Af, Zi, \
                Zf, Spi, Spf, Qpi, Qpf, Zpi, Zpf, ui, uf, upi, upf, uppi, uppf, udoti, udotf, rhoi, rhof, rhopi, rhopf, \
                rhodoti, rhodotf, Dz, dzdwz, sigmasqi, sigmasqf = f.result()

                Dzlist.append(Dz)
                Hzlist.append(Hz)
                rhozlist.append(rhoz)
                dzdwzlist.append(dzdwz)
                Lamlist.append(Lam)
                T1ilist.append(T1i)
                T1flist.append(T1f)
                T2ilist.append(T2i)
                T2flist.append(T2f)
                sigmasqilist.append(sigmasqi)
                sigmasqflist.append(sigmasqf)
                LLTBConsilist.append(LLTBConsi)
                LLTBConsflist.append(LLTBConsf)
                Dilist.append(Di)
                Dflist.append(Df)
                Silist.append(Si)
                Sflist.append(Sf)
                Qilist.append(Qi)
                Qflist.append(Qf)
                Ailist.append(Ai)
                Aflist.append(Af)
                Zilist.append(Zi)
                Zflist.append(Zf)
                Spilist.append(Spi)
                Spflist.append(Spf)
                Qpilist.append(Qpi)
                Qpflist.append(Qpf)
                Zpilist.append(Zpi)
                Zpflist.append(Zpf)
                uilist.append(ui)
                uflist.append(uf)
                upilist.append(upi)
                upflist.append(upf)
                uppilist.append(uppi)
                uppflist.append(uppf)
                udotilist.append(udoti)
                udotflist.append(udotf)
                rhoilist.append(rhoi)
                rhoflist.append(rhof)
                rhopilist.append(rhopi)
                rhopflist.append(rhopf)
                rhodotilist.append(rhodoti)
                rhodotflist.append(rhodotf)

        Htest = MCT.MCMC_diagnostics(NSamplers, Hzlist).get_GRC().max()
        rhotest = MCT.MCMC_diagnostics(NSamplers, rhozlist).get_GRC().max()
        T1itest = MCT.MCMC_diagnostics(NSamplers, T1ilist).get_GRC().max()
        T1ftest = MCT.MCMC_diagnostics(NSamplers, T1flist).get_GRC().max()
        T2itest = MCT.MCMC_diagnostics(NSamplers, T2ilist).get_GRC().max()
        T2ftest = MCT.MCMC_diagnostics(NSamplers, T2flist).get_GRC().max()

        test_GR = np.array([Htest,rhotest,T1itest, T1ftest, T2itest, T2ftest])

        try:
            cont = any(test_GR > 1.15)
        except:
            cont = True
            print "Something went wrong in GR test"

        if cont and num_repeats < max_repeats:
            print "Gelman-Rubin indicates non-convergence"
            Nsamp *= 2
            num_repeats += 1

    #Save the data
    np.savez(fname + "Processed_Data/" + "Samps.npz", Hz=Hzlist, rhoz=rhozlist, Lam=Lamlist, T1i=T1ilist, T1f=T1flist, T2i=T2ilist,
             T2f=T2flist, LLTBConsi=LLTBConsilist, LLTBConsf=LLTBConsflist, Di=Dilist, Df=Dflist, Si=Silist, Sf=Sflist,
             Qi=Qilist, Qf=Qflist, Ai=Ailist, Af=Aflist, Zi=Zilist, Zf=Zflist, Spi=Spilist, Spf=Spflist, Qpi=Qpilist,
             Qpf=Qpflist, Zpi=Zpilist, Zpf=Zpflist, ui=uilist, uf=uflist, upi=upilist, upf=upflist, uppi=uppilist,
             uppf=uppflist, udoti=udotilist, udotf=udotflist, rhoi=rhoilist, rhof=rhoflist, rhopi=rhopilist,
             rhopf=rhopflist, rhodoti=rhodotilist, rhodotf=rhodotflist, NSamplers=NSamplers, Dz=Dzlist, dzdwz=dzdwzlist,
             sigmasqi=sigmasqilist, sigmasqf=sigmasqflist)

    #print Hsamps.shape
    print "GR(H) = ", Htest, "GR(rho)", rhotest, "GR(T1i)", T1itest, "GR(T1f)", T1ftest, "GR(T2i)", T2itest, "GR(T2f)", T2ftest