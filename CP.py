#!/usr/bin/env python

import numpy as np
import os
import argparse
import ConfigParser
from concurrent.futures import ProcessPoolExecutor
from Copernicus import sampler

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
        "dotransform" : True
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
    parser.add_argument("--nwalkers", help="The number of samplers to spawn")
    parser.add_argument("--nsamples", help="The number of samples each sampler should draw")
    parser.add_argument("--nburnin", help="The number of samples in the burnin period")
    parser.add_argument("--tstar", help="The time up to which to integrate to [in Gpc for now]")
    parser.add_argument("--doplcf", help="Whether to compute the interior of the PLC or not")
    parser.add_argument("--dotransform", help="Whether to perform the coordinate transformation or not")
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
    fpath = fname+"0.npz"
    holder = np.load('SampsPNC02.npz') #/home/landman/Documents/Research/Algo_Pap/Simulated_LCDM_prior/ProcessedData/Samps1s.npz')
    Dsamps = np.asarray(holder['Dsamps'])
    Dpsamps = np.asarray(holder['Dpsamps'])
    Hsamps = np.asarray(holder['Hsamps'])
    rhosamps = np.asarray(holder['rhosamps'])
    rhopsamps = np.asarray(holder['rhopsamps'])
    zfmax = np.asarray(holder['zfmax'])
    Ki = np.asarray(holder['Ki'])
    Kf = np.asarray(holder['Kf'])
    sheari = np.asarray(holder['sheari'])
    shearf = np.asarray(holder['shearf'])
    t0 = np.asarray(holder['t0'])
    rhostar = np.asarray(holder['rhostar'])
    Dstar = np.asarray(holder['Dstar'])
    Xstar = np.asarray(holder['Xstar'])
    rmax= np.asarray(holder['rmax'])
    vmax= np.asarray(holder['vmax'])

    #Load the rest of the data
    for i in xrange(1,NSamplers):
        dirpath = fname + str(i) + '.npz'
        holder = np.load(dirpath)
        Dsamps = np.append(Dsamps,holder['Dsamps'],axis=1)
        Dpsamps = np.append(Dpsamps,holder['Dpsamps'],axis=1)
        Hsamps = np.append(Hsamps,holder['Hsamps'],axis=1)
        rhosamps = np.append(rhosamps,holder['rhosamps'],axis=1)
        rhopsamps = np.append(rhopsamps,holder['rhopsamps'],axis=1)
        zfmax = np.append(zfmax,holder['zfmax'])
        Ki = np.append(Ki,holder['Ki'],axis=1)
        Kf = np.append(Kf,holder['Kf'],axis=1)
        sheari = np.append(sheari,holder['sheari'],axis=1)
        shearf = np.append(shearf,holder['shearf'],axis=1)
        t0 = np.append(t0,holder['t0'])
        rhostar = np.append(rhostar,holder['rhostar'],axis=1)
        Dstar = np.append(Dstar,holder['Dstar'],axis=1)
        Xstar = np.append(Xstar,holder['Xstar'],axis=1)
        rmax = np.append(rmax,holder['rmax'])
        vmax = np.append(vmax,holder['vmax'])

    return Dsamps,Dpsamps,Hsamps,rhosamps,rhopsamps,zfmax,Ki,Kf,sheari,shearf,t0,rhostar,Dstar,Xstar,rmax,vmax


if __name__ == "__main__":
    #Get config
    GD = readargs()

    #Determine how many samplers to spawn
    NSamplers = GD["nwalkers"]
    nsamps = GD["nsamples"]
    nburn = GD["nburnin"]
    tstar = GD["tstar"]
    DoPLCF = GD["doplcf"]
    DoTransform = GD["dotransform"]
    fname = GD["fname"]

    #Create a pool for this number of samplers and submit the jobs
    SamplerDICT = {}
    with ProcessPoolExecutor(max_workers=NSamplers) as executor:
        for i in xrange(NSamplers):
            tmpstr = 'sampler' + str(i)
            SamplerDICT[tmpstr] = executor.submit(sampler,(nsamps,nburn,tstar,DoPLCF,DoTransform,i))

    # Next we check to see if the MCMC has converged
    # Load the data
    Dsamps, Dpsamps, Hsamps, rhosamps, rhopsamps, zfmax, Ki, Kf, sheari, shearf, t0, rhostar, Dstar, Xstar, rmax, vmax = load_samps(NSamplers,fname)

    print "Converged!"
