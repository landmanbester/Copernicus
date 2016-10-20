#!/usr/bin/env python

import numpy as np
import os
import argparse
import ConfigParser
import concurrent.futures as cf
from Copernicus.sampler import sampler
from Copernicus import MCMC_Tools as MCT
import matplotlib.pyplot as plt
import Plotter

def readargs():
    conf_parser = argparse.ArgumentParser(
        # Turn off help, so we print all options in response to -h
            add_help=False
            )
    conf_parser.add_argument("-c", "--conf_file",
                             help="Specify config file", metavar="FILE")
    args, remaining_argv = conf_parser.parse_known_args()
    defaults = {
        "nwalkers" : 10,
        "nsamples" : 10000,
        "nburnin"  : 2500,
        "tstar"    : 3.25,
        "doplcf"   : True,
        "dotransform" : True,
        "fname" : "/home/bester/Projects/CP_Dir/",
        "data_prior" : ["H","rho"],
        "data_lik" : ["D","H","dzdw"],
        "zmax" : 2.0,
        "np" : 200,
        "nret" : 100,
        "err" : 1e-5
        }
    if args.conf_file:
        config = ConfigParser.SafeConfigParser()
        config.read([args.conf_file])
        defaults = dict(config.items('Defaults'))

    # Don't surpress add_help here so it will handle -h
    parser = argparse.ArgumentParser(
        # Inherit options from config_parser
        parents=[conf_parser],
        # print script description with -h/--help
        description=__doc__,
        # Don't mess with format of description
        formatter_class=argparse.RawDescriptionHelpFormatter,
        )
    parser.set_defaults(**defaults)
    parser.add_argument("--nwalkers", type=int, help="The number of samplers to spawn")
    parser.add_argument("--nsamples", type=int, help="The number of samples each sampler should draw")
    parser.add_argument("--nburnin", type=int, help="The number of samples in the burnin period")
    parser.add_argument("--tstar", type=float, help="The time up to which to integrate to [in Gpc for now]")
    parser.add_argument("--doplcf", type=bool, help="Whether to compute the interior of the PLC or not")
    parser.add_argument("--dotransform", type=bool, help="Whether to perform the coordinate transformation or not")
    parser.add_argument("--fname", type=str, help="Where to save the results")
    parser.add_argument("--data_prior", type=str, help="The data sets to use to set priors")
    parser.add_argument("--data_lik", type=str, help="The data sets to use for inference")
    parser.add_argument("--zmax", type=float, help="The maximum redshift to go out to")
    parser.add_argument("--np", type=int, help="The number of redshift points to use")
    parser.add_argument("--nret", type=int, help="The number of points at which to return quantities of interest")
    parser.add_argument("--err", type=float, help="Target error of the numerical integration scheme")
    args = parser.parse_args(remaining_argv)


    #return dict containing args
    return vars(args)

def load_samps(NSamplers,fname):
    """
    Method to load samples after MCMC
    :param NSamplers: the numbers of walkers i.e. MCMC chains
    :param fname: the path where thje results were written out to
    :return:
    """
    #Load first samples
    fpath = fname+"Samps0.npz"
    holder = np.load(fpath) #/home/landman/Documents/Research/Algo_Pap/Simulated_LCDM_prior/ProcessedData/Samps1s.npz')
    # Dsamps = np.asarray(holder['Dsamps'])
    # Dpsamps = np.asarray(holder['Dpsamps'])
    Hsamps = np.asarray(holder['Hsamps'])
    rhosamps = np.asarray(holder['rhosamps'])
    Lamsamps = np.asarray(holder['Lamsamps'])
#    rhopsamps = np.asarray(holder['rhopsamps'])
#    zfmax = np.asarray(holder['zfmax'])
    T2i = np.asarray(holder['T2i'])
    T2f = np.asarray(holder['T2f'])
    T1i = np.asarray(holder['T1i'])
    T1f = np.asarray(holder['T1f'])
    LLTBConsi = np.asarray(holder['LLTBConsi'])
    LLTBConsf = np.asarray(holder['LLTBConsf'])
#    t0 = np.asarray(holder['t0'])
#    rhostar = np.asarray(holder['rhostar'])
#    Dstar = np.asarray(holder['Dstar'])
#    Xstar = np.asarray(holder['Xstar'])
#    rmax= np.asarray(holder['rmax'])
#    vmax= np.asarray(holder['vmax'])

    #Load the rest of the data
    for i in xrange(1,NSamplers):
        dirpath = fname + "Samps" + str(i) + '.npz'
        holder = np.load(dirpath)
        # Dsamps = np.append(Dsamps,holder['Dsamps'],axis=1)
        # Dpsamps = np.append(Dpsamps,holder['Dpsamps'],axis=1)
        Hsamps = np.append(Hsamps,holder['Hsamps'],axis=1)
        rhosamps = np.append(rhosamps,holder['rhosamps'],axis=1)
#        rhopsamps = np.append(rhopsamps,holder['rhopsamps'],axis=1)
#        zfmax = np.append(zfmax,holder['zfmax'])
        T2i = np.append(T2i,holder['T2i'],axis=1)
        T2f = np.append(T2f,holder['T2f'],axis=1)
        T1i = np.append(T1i,holder['T1i'],axis=1)
        T1f = np.append(T1f,holder['T1f'],axis=1)
        LLTBConsi = np.append(LLTBConsi, holder['LLTBConsi'])
        LLTBConsf = np.append(LLTBConsf, holder['LLTBConsf'])
        Lamsamps = np.append(Lamsamps, holder['Lamsamps'])
        # t0 = np.append(t0,holder['t0'])
        # rhostar = np.append(rhostar,holder['rhostar'],axis=1)
        # Dstar = np.append(Dstar,holder['Dstar'],axis=1)
        # Xstar = np.append(Xstar,holder['Xstar'],axis=1)
        # rmax = np.append(rmax,holder['rmax'])
        # vmax = np.append(vmax,holder['vmax'])

    #return Dsamps,Dpsamps,Hsamps,rhosamps,rhopsamps,zfmax,Ki,Kf,sheari,shearf,t0,rhostar,Dstar,Xstar,rmax,vmax
    return Hsamps, rhosamps, T1i, T1f, T2i, T2f, LLTBConsi, LLTBConsf, Lamsamps


if __name__ == "__main__":
    #Get config
    GD = readargs()

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

    futures = []
    Hzlist = []
    rhozlist = []
    Lamlist = []
    T1ilist = []
    T1flist = []
    T2ilist = []
    T2flist = []
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

    #sampler.sampler(zmax,Np,Nret,Nsamp,Nburn,tstar,data_prior,data_lik,DoPLCF,DoTransform,err,0,fname)
    #Create a pool for this number of samplers and submit the jobs
    Hsamps = np.zeros([NSamplers,Np,Nsamp])
    cont = True
    num_repeats = 0
    max_repeats = 1
    while cont and num_repeats < max_repeats:
        with cf.ProcessPoolExecutor(max_workers=NSamplers) as executor:
            for i in xrange(NSamplers):
                # tmpstr = 'sampler' + str(i)
                # SamplerDICT[tmpstr] =
                future = executor.submit(sampler,zmax,Np,Nret,Nsamp,Nburn,tstar,data_prior,data_lik,DoPLCF,DoTransform,err,i,fname)
                futures.append(future)
                # cf.as_completed(SamplerDICT[tmpstr])
                #cf.as_completed(executor.submit(sampler.sampler,zmax,Np,Nret,Nsamp,Nburn,tstar,data_prior,data_lik,DoPLCF,DoTransform,err,i,fname))
            k = 0
            for f in cf.as_completed(futures):
                Hz, rhoz, Lam, T1i, T1f, T2i, T2f, LLTBConsi, LLTBConsf, Di, Df, Si, Sf, Qi, Qf, Ai, Af, Zi, \
                Zf, Spi, Spf, Qpi, Qpf, Zpi, Zpf, ui, uf, upi, upf, uppi, uppf, udoti, udotf, rhoi, rhof, rhopi, rhopf, \
                rhodoti, rhodotf = f.result()

                Hzlist.append(Hz)
                rhozlist.append(rhoz)
                Lamlist.append(Lam)
                T1ilist.append(T1i)
                T1flist.append(T1f)
                T2ilist.append(T2i)
                T2flist.append(T2f)
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

        cont = any(test_GR > 1.15)

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
             rhopf=rhopflist, rhodoti=rhodotilist, rhodotf=rhodotflist, NSamplers=NSamplers)

    #print Hsamps.shape
    print "GR(H) = ", Htest, "GR(rho)", rhotest, "GR(T1i)", T1itest, "GR(T1f)", T1ftest, "GR(T2i)", T2itest, "GR(T2f)", T2ftest

    # plt.figure('H')
    # for i in xrange(NSamplers):
    #     plt.plot(Hsampslist[i],'b',alpha=0.1)
    # plt.savefig(fname + "H.png",dpi=250)
    #
    # plt.figure('rho')
    # for i in xrange(NSamplers):
    #     plt.plot(rhosampslist[i],'b',alpha=0.1)
    # plt.savefig(fname + "rho.png", dpi=250)
    #
    # plt.figure('T1i')
    # for i in xrange(NSamplers):
    #     plt.plot(T1ilist[i],'b',alpha=0.1)
    # plt.savefig(fname + "T1i.png", dpi=250)
    #
    # plt.figure('T1f')
    # for i in xrange(NSamplers):
    #     plt.plot(T1flist[i],'b',alpha=0.1)
    # plt.savefig(fname + "T1f.png", dpi=250)
    #
    # plt.figure('T2i')
    # for i in xrange(NSamplers):
    #     plt.plot(T2ilist[i],'b',alpha=0.1)
    # plt.savefig(fname + "T2i.png", dpi=250)
    #
    # plt.figure('T2f')
    # for i in xrange(NSamplers):
    #     plt.plot(T2flist[i],'b',alpha=0.1)
    # plt.savefig(fname + "T2f.png", dpi=250)
    #
    # plt.figure('LLTBi')
    # for i in xrange(NSamplers):
    #     plt.plot(LLTBConsilist[i],'b',alpha=0.1)
    # plt.savefig(fname + "LLTBi.png", dpi=250)
    #
    # plt.figure('LLTBf')
    # for i in xrange(NSamplers):
    #     plt.plot(LLTBConsflist[i],'b',alpha=0.1)
    # plt.savefig(fname + "LLTBf.png", dpi=250)

    #print NSamplers, Nsamp, Nburn, tstar, DoPLCF, DoTransform, fname, data_prior, data_lik, zmax, Np, Nret, err
