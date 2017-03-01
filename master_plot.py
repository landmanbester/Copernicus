import numpy as np
from scipy.interpolate import UnivariateSpline as uvs
import matplotlib as mpl
mpl.use('Agg')
mpl.rcParams.update({'font.size': 14, 'font.family': 'serif'})
import matplotlib.pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF
from genFLRW import FLRW
from Master import SSU
from My2Ddist import plot2Ddist2 as pl2d
from matplotlib.patches import Rectangle
from Copernicus.Parset import MyOptParse


def Plot_Data(zmax,Np,Nret,tmin,err,data_prior,data_lik,fname,Nsamp):
    print "Getting LCDM vals"
    # Get FLRW funcs for comparison
    Om0 = 0.3
    OL0 = 0.7
    H0 = 0.2335
    LCDM = FLRW(Om0, OL0, H0, zmax, Np)
    HzF = LCDM.Hz
    rhozF = LCDM.getrho()
    sigmasqFz100 = LCDM.get_sigmasq(2.41e-9, 0.005) * HzF ** 2
    z = LCDM.z
    sigmasq100o = uvs(z / z[-1], sigmasqFz100, k=3, s=0.0)

    # Do integration of FLRW funcs
    zp = np.linspace(0, zmax, Np)
    #zp2 = np.linspace(0, zmax, 200)
    LamF = 3 * 0.7 * 0.2335 ** 2
    Xrho = np.array([0.5,2.8])
    XH = np.array([0.6,3.5])
    #set characteristic variance of Lambda prior (here 60%)
    sigmaLam = 0.6*3*0.7*(70.0/299.79)**2

    # Do LCDM integration
    UF = SSU(zmax, tmin, Np, err, XH, Xrho, sigmaLam, Nret, data_prior, data_lik, fname, Hz=HzF, rhoz=rhozF, Lam=LamF, useInputFuncs=True)

    # Get quantities of interrest
    T1iF, T1fF, T2iF, T2fF, LLTBConsiF, LLTBConsfF, DiF, DfF, SiF, \
    SfF, QiF, QfF, AiF, AfF, ZiF, ZfF, SpiF, SpfF, QpiF, QpfF, \
    ZpiF, ZpfF, uiF, ufF, upiF, upfF, uppiF, uppfF, udotiF, udotfF, \
    rhoiF, rhofF, rhopiF, rhopfF, rhodotiF, rhodotfF, DzF, dzdwzF, sigmasqiF, sigmasqfF = UF.get_funcs()
    sigmasqiF100 = sigmasq100o(np.linspace(0, 1, Nret))

    # Do LTB integration
    print "Getting LTB vals"
    #LTB_z_funcs = np.load(fname + 'Processed_Data/LTB_z_funcs.npz')
    LTB_z_funcs = np.load(fname + 'Processed_Data/ConLTBDat.npz')
    print LTB_z_funcs.keys()
    HzLT = LTB_z_funcs['Hz']
    rhozLT = LTB_z_funcs['rhoz']
    zLT = LTB_z_funcs['z']
    HzLT = uvs(zLT,HzLT,k=3,s=0.0)(zp)
    rhozLT = uvs(zLT, rhozLT, k=3, s=0.0)(zp)
    # plt.figure('Hz')
    # plt.plot(zp,HzLT,'b')
    # plt.plot(zp,HzF,'g')
    # plt.savefig('/home/landman/Projects/CP_LCDM/Figures/LTBvLCDM_Hz.png',dpi=200)
    # plt.figure('rhoz')
    # plt.plot(zp,rhozLT,'b')
    # plt.plot(zp,rhozF,'g')
    # plt.savefig('/home/landman/Projects/CP_LCDM/Figures/LTBvLCDM_rhoz.png', dpi=200)
    ULT = SSU(zmax, tmin, Np, err, XH, Xrho, sigmaLam, Nret, data_prior, data_lik, fname, Hz=HzLT, rhoz=rhozLT, Lam=0.0, useInputFuncs=True)

    # Get quantities of interrest
    print "Getting quantities of interest"
    T1iLT, T1fLT, T2iLT, T2fLT, LLTBConsiLT, LLTBConsfLT, DiLT, DfLT, SiLT, \
    SfLT, QiLT, QfLT, AiLT, AfLT, ZiLT, ZfLT, SpiLT, SpfLT, QpiLT, QpfLT, \
    ZpiLT, ZpfLT, uiLT, ufLT, upiLT, upfLT, uppiLT, uppfLT, udotiLT, udotfLT, \
    rhoiLT, rhofLT, rhopiLT, rhopfLT, rhodotiLT, rhodotfLT, DzLT, dzdwzLT, sigmasqiLT, sigmasqfLT = ULT.get_funcs()

    files = {}
    files["DHt0"] = "/home/landman/Projects/CP_LCDM_DHt0/"
    files["Ddzdw"] = "/home/landman/Projects/CP_LCDM_Ddzdw/"
    files["DHdzdw"] = "/home/landman/Projects/CP_LCDM_DHdzdw/"


    # Load first samples
    print "Loading Samps"
    holder = np.load(fname + 'Processed_Data/Samps.npz')
    Dzlist = holder['Dz']
    Hzlist = holder['Hz']
    rhozlist = holder['rhoz']
    dzdwzlist = holder['dzdwz']
    Lamlist = holder['Lam']
    T2ilist = holder['T2i']
    T2flist = holder['T2f']
    T1ilist = holder['T1i']
    T1flist = holder['T1f']
    sigmasqilist = holder['sigmasqi']
    sigmasqflist = holder['sigmasqf']
    LLTBConsilist = holder['LLTBConsi']
    LLTBConsflist = holder['LLTBConsf']
    NSamplers = holder['NSamplers']

    # Load the rest of the data
    for i in xrange(NSamplers):
        if i > 0:
            Dzsamps = np.append(Dzsamps, Dzlist[i], axis=1)
            Hzsamps = np.append(Hzsamps, Hzlist[i], axis=1)
            rhozsamps = np.append(rhozsamps, rhozlist[i], axis=1)
            dzdwzsamps = np.append(dzdwzsamps, dzdwzlist[i], axis=1)
            Lamsamps = np.append(Lamsamps, Lamlist[i])
            T2i = np.append(T2i, T2ilist[i], axis=1)
            T2f = np.append(T2f, T2flist[i], axis=1)
            T1i = np.append(T1i, T1ilist[i], axis=1)
            T1f = np.append(T1f, T1flist[i], axis=1)
            sigmasqi = np.append(sigmasqi, sigmasqilist[i], axis=1)
            sigmasqf = np.append(sigmasqf, sigmasqflist[i], axis=1)
            LLTBConsi = np.append(LLTBConsi, LLTBConsilist[i], axis=1)
            LLTBConsf = np.append(LLTBConsf, LLTBConsflist[i], axis=1)
        else:
            Dzsamps = Dzlist[0]
            Hzsamps = Hzlist[0]
            rhozsamps = rhozlist[0]
            dzdwzsamps = dzdwzlist[0]
            Lamsamps = Lamlist[0]
            T2i = T2ilist[0]
            T2f = T2flist[0]
            T1i = T1ilist[0]
            T1f = T1flist[0]
            sigmasqi = sigmasqilist[0]
            sigmasqf = sigmasqflist[0]
            LLTBConsi = LLTBConsilist[0]
            LLTBConsf = LLTBConsflist[0]

    Om0samps = 8 * np.pi * rhozsamps[0, :] / (3 * Hzsamps[0, :] ** 2)
    OL0samps = Lamsamps / (3 * Hzsamps[0, :] ** 2)

if __name__=="__main__":
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

    # Do the plots
    Plot_Data(zmax,Np,Nret,tstar,err,data_prior,data_lik,fname,Nsamp)