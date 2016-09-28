#!/usr/bin/env python

import numpy as np
import os
import argparse
import ConfigParser
import concurrent.futures as cf
#from concurrent.futures import ProcessPoolExecutor
from Copernicus import sampler
from Copernicus import MCMC_Tools as MCT

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
        "fname" : "/home/landman/Projects/CP_In/Data/",
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
        defaults = dict(config.items("Defaults"))

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

    #sampler.sampler(zmax,Np,Nret,Nsamp,Nburn,tstar,data_prior,data_lik,DoPLCF,DoTransform,err,0,fname)
    #Create a pool for this number of samplers and submit the jobs
    #SamplerDICT = {}
    with cf.ProcessPoolExecutor(max_workers=NSamplers) as executor:
        for i in xrange(NSamplers):
            # tmpstr = 'sampler' + str(i)
            # SamplerDICT[tmpstr] = executor.submit(sampler.sampler,zmax,Np,Nret,Nsamp,Nburn,tstar,data_prior,data_lik,DoPLCF,DoTransform,err,i,fname)
            # cf.as_completed(SamplerDICT[tmpstr])
            cf.as_completed(executor.submit(sampler.sampler,zmax,Np,Nret,Nsamp,Nburn,tstar,data_prior,data_lik,DoPLCF,DoTransform,err,i,fname))

    # Check if the MCMC has converged
    # Load the data
    #Dsamps, Dpsamps, Hsamps, rhosamps, rhopsamps, zfmax, Ki, Kf, sheari, shearf, t0, rhostar, Dstar, Xstar, rmax, vmax = load_samps(NSamplers,fname)
    Hsamps, rhosamps, T1i, T1f, T2i, T2f, LLTBConsi, LLTBConsf, Lamsamps = load_samps(NSamplers, fname)


    #print NSamplers, Nsamp, Nburn, tstar, DoPLCF, DoTransform, fname, data_prior, data_lik, zmax, Np, Nret, err
