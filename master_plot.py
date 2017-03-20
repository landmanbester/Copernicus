#!/usr/bin/env python
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
from Plotter import plh

def Plot_Data(zmax,Np,Nret,tmin,err,data_prior,data_lik,fname,Nsamp,DoPLCF,samps_out_name):
    print "Getting LCDM vals"
    # Get FLRW funcs for comparison
    Om0 = 0.3
    OL0 = 0.7
    H0 = 0.2335
    LCDM = FLRW(Om0, OL0, H0, zmax, Np)

    DelRSq = 2.41e-9
    UV_cut = 0.005
    DzF, HzF, rhozF, dzdwzF, sigmasqiF, sigmasqfF = LCDM.give_shear_for_plotting(Om0, OL0, H0, DelRSq, UV_cut, zmax, Np,
                                                                               tstar, Nret, data_prior,
                                                                               data_lik, fname, DoPLCF, err)
    # set redshift and params required by CIVP
    zp = np.linspace(0, zmax, Np)
    Xrho = np.array([0.5, 2.8])
    XH = np.array([0.6, 3.5])
    #set characteristic variance of Lambda prior (here 60%)
    sigmaLam = 0.6*3*0.7*(70.0/299.79)**2

    # Do LTB integration
    print "Getting LTBCon vals"
    #LTB_z_funcs = np.load(fname + 'Processed_Data/LTB_z_funcs.npz')
    LTB_z_funcs = np.load(fname + 'Processed_Data/ConLTBDat.npz')
    HzLT = LTB_z_funcs['Hz']
    rhozLT = LTB_z_funcs['rhoz']
    zLT = LTB_z_funcs['z']
    HzLT = uvs(zLT,HzLT,k=3,s=0.0)(zp)
    rhozLT = uvs(zLT, rhozLT, k=3, s=0.0)(zp)

    # Do LTBCon integration
    ULT = SSU(zmax, tmin, Np, err, XH, Xrho, sigmaLam, Nret, data_prior, data_lik, fname, DoPLCF, Hz=HzLT, rhoz=rhozLT, Lam=0.0, useInputFuncs=True)

    T1iLT, T2iLT, LLTBConsiLT, DiLT, SiLT, QiLT, AiLT, ZiLT, SpiLT, QpiLT, ZpiLT, uiLT, upiLT, uppiLT, udotiLT, rhoiLT, rhopiLT, rhodotiLT, \
    DzLT, dzdwzLT, sigmasqiLT, t0LT = ULT.get_funcsi()

    if t0LT > ULT.tmin and ULT.NI > 1 and DoPLCF:
        T1fLT, T2fLT, LLTBConsfLT, DfLT, SfLT, QfLT, AfLT, ZfLT, SpfLT, QpfLT, ZpfLT, ufLT, upfLT, uppfLT, udotfLT, rhofLT, rhopfLT, \
        rhodotfLT, sigmasqfLT = ULT.get_funcsf()

    # Do LTB integration
    print "Getting LTB vals"
    LTB_z_funcs = np.load(fname + 'Processed_Data/LTB_z_funcs.npz')
    HzLT2 = LTB_z_funcs['Hz']
    rhozLT2 = LTB_z_funcs['rhoz']
    DzLT2 = LTB_z_funcs['Dz']
    #zLT2 = LTB_z_funcs['z']
    #HzLT2 = uvs(zLT2,HzLT,k=3,s=0.0)(zp)
    #rhozLT2 = uvs(zLT2, rhozLT, k=3, s=0.0)(zp)

    # Do LTBCon integration
    ULT2 = SSU(zmax, tmin, Np, err, XH, Xrho, sigmaLam, Nret, data_prior, data_lik, fname, DoPLCF, Hz=HzLT2, rhoz=rhozLT2, Lam=0.0, useInputFuncs=True)

    T1iLT2, T2iLT2, LLTBConsiLT2, DiLT2, SiLT2, QiLT2, AiLT2, ZiLT2, SpiLT2, QpiLT2, ZpiLT2, uiLT2, upiLT2, uppiLT2, udotiLT2, rhoiLT2, rhopiLT2, rhodotiLT2, \
    DzLT2num, dzdwzLT2, sigmasqiLT2, t0LT2 = ULT2.get_funcsi()

    if t0LT2 > ULT2.tmin and ULT2.NI > 1 and DoPLCF:
        T1fLT2, T2fLT2, LLTBConsfLT2, DfLT2, SfLT2, QfLT2, AfLT2, ZfLT2, SpfLT2, QpfLT2, ZpfLT2, ufLT2, upfLT2, uppfLT2, udotfLT2, rhofLT2, rhopfLT2, \
        rhodotfLT2, sigmasqfLT2 = ULT2.get_funcsf()

    # Load the data we want to plot
    files = ["DHt0/", "Ddzdw/", "DHdzdw/"]
    Dzdict = {}
    Hzdict = {}
    rhozdict = {}
    dzdwdict = {}
    Lamdict = {}
    sigmasqidict = {}
    sigmasqfdict = {}
    Om0dict = {}
    OL0dict = {}
    colourdict = {}
    colourdict[files[0]] = "blue"
    colourdict[files[1]] = "green"
    colourdict[files[2]] = "red"
    labelsdict = {}
    labelsdict[files[0]] = r'$\mathcal{D}_0$'
    labelsdict[files[1]] = r'$\mathcal{D}/\{H_\parallel\}$'
    labelsdict[files[2]] = r'$\mathcal{D}$'

    # Load first samples
    print "Loading samps"
    for s in files:
        with np.load("/home/landman/Projects/CP_LCDM_" + s + 'Processed_Data/' + samps_out_name + '.npz') as holder:
            Dzlist = holder['Dz']
            Hzlist = holder['Hz']
            rhozlist = holder['rhoz']
            dzdwzlist = holder['dzdwz']
            Lamlist = holder['Lam']
            sigmasqilist = holder['sigmasqi']
            sigmasqflist = holder['sigmasqf']
            NSamplers = holder['NSamplers']

            # Load the rest of the data
            for i in xrange(NSamplers):
                if i > 0:
                    Dzsamps = np.append(Dzsamps, Dzlist[i], axis=1)
                    Hzsamps = np.append(Hzsamps, Hzlist[i], axis=1)
                    rhozsamps = np.append(rhozsamps, rhozlist[i], axis=1)
                    dzdwzsamps = np.append(dzdwzsamps, dzdwzlist[i], axis=1)
                    Lamsamps = np.append(Lamsamps, Lamlist[i])
                    sigmasqisamps = np.append(sigmasqisamps, sigmasqilist[i], axis=1)
                    sigmasqfsamps = np.append(sigmasqfsamps, sigmasqflist[i], axis=1)
                else:
                    Dzsamps = Dzlist[0]
                    Hzsamps = Hzlist[0]
                    rhozsamps = rhozlist[0]
                    dzdwzsamps = dzdwzlist[0]
                    Lamsamps = Lamlist[0]
                    sigmasqisamps = sigmasqilist[0]
                    sigmasqfsamps = sigmasqflist[0]

            Om0samps = 8 * np.pi * rhozsamps[0, :] / (3 * Hzsamps[0, :] ** 2)
            OL0samps = Lamsamps / (3 * Hzsamps[0, :] ** 2)

            Dzdict[s] = Dzsamps
            Hzdict[s] = Hzsamps
            rhozdict[s] = rhozsamps
            dzdwdict[s] = dzdwzsamps
            Lamdict[s] = Lamsamps
            sigmasqidict[s] = sigmasqisamps
            sigmasqfdict[s] = sigmasqfsamps
            Om0dict[s] = Om0samps
            OL0dict[s] = OL0samps

            del Dzsamps, Hzsamps, rhozsamps, dzdwzsamps, Lamsamps, sigmasqisamps, Om0samps, OL0samps #sigmasqfsamps,

    # read in data
    zD, Dz, sDz = np.loadtxt(fname + 'Data/D.txt', unpack=True)
    zH, Hz, sHz = np.loadtxt(fname + 'Data/H.txt', unpack=True)
    #zrho, rhoz, srhoz = np.loadtxt(fname + 'Data/rho.txt', unpack=True)
    zdzdw, dzdwz, sdzdwz = np.loadtxt(fname + 'Data/dzdw.txt', unpack=True)

    # Create the figures we want to plot
    # PLC0
    figPLC0, axPLC0 = plt.subplots(nrows=2, ncols=2, figsize=(15, 9), sharex=True)
    # Shear
    figsigmasq, axsigmasq = plt.subplots(nrows=1, ncols=2, figsize=(15, 9), sharey=True)
    # Om vs OL contours
    figOL, axOL = plt.subplots(nrows=1, ncols=1, figsize=(9, 9))

    # # Plot PLC0
    # print "PLC0"
    # # First the contours
    # for s in files:
    #     Dplh = plh(Dzdict[s], axPLC0[0, 0])
    #     Dplh.draw_Contours(zp, only_2sig=True, colour=colourdict[s], draw_median=False)
    #     Hplh = plh(Hzdict[s], axPLC0[0, 1])
    #     Hplh.draw_Contours(zp, scale=299.8, only_2sig=True, colour=colourdict[s], draw_median=False)
    #     rhoplh = plh(rhozdict[s], axPLC0[1, 0])
    #     rhoplh.draw_Contours(zp, scale=153.66, only_2sig=True, colour=colourdict[s], draw_median=False)
    #     dzdwplh = plh(dzdwdict[s], axPLC0[1, 1])
    #     dzdwplh.draw_Contours(zp, only_2sig=True, colour=colourdict[s], draw_median=False)
    #
    # # Now add labels and background plots
    # # D
    # axPLC0[0, 0].set_ylabel(r'$ D / [Gpc]$', fontsize=20)
    # axPLC0[0, 0].set_ylim(0.0, 2.0)
    # Dplh.add_plot(zp, DzF, col='k', lab=r'$\Lambda CDM$', wid=1.5)
    # Dplh.add_plot(zp, DzLT, col='m', lab=r'$LTB \ t_B=0$', wid=1.5)
    # Dplh.add_plot(zp, DzLT2, col='c', lab=r'$LTB$', wid=1.5)
    # Dplh.add_data(zD, Dz, sDz, alp=1.0)
    # #Dplh.show_lab(4, only_2sig=True)
    #
    # # H
    # axPLC0[0, 1].set_ylabel(r'$ H_\parallel / [km s^{-1} Mpc^{-1}]$', fontsize=20)
    # axPLC0[0, 1].set_ylim(65, 220.0)
    # Hplh.add_plot(zp, HzF, col='k', scale=299.8, lab=r'$\Lambda CDM$', wid=1.5)
    # Hplh.add_plot(zp, HzLT, col='m', scale=299.8, lab=r'$LTB \ t_B=0$', wid=1.5)
    # Hplh.add_plot(zp, HzLT2, col='c', scale=299.8, lab=r'$LTB$', wid=1.5)
    # Hplh.add_data(zH, Hz, sHz, scale=299.8, alp=1.0)
    # #Hplh.show_lab(4)
    #
    # # rho
    # axPLC0[1, 0].set_xlabel(r'$z$', fontsize=20)
    # axPLC0[1, 0].set_xlim(0, zmax)
    # axPLC0[1, 0].set_ylabel(r'$\frac{\rho}{\rho_c} $', fontsize=30)
    # axPLC0[1, 0].set_ylim(0, 10.0)
    # rhoplh.add_plot(zp, rhozF, col='k', scale=153.66, lab=r'$\Lambda CDM$', wid=1.5)
    # rhoplh.add_plot(zp, rhozLT, col='m', scale=153.66, lab=r'$LTB \ t_B=0$', wid=1.5)
    # rhoplh.add_plot(zp, rhozLT2, col='c', scale=153.66, lab=r'$LTB$', wid=1.5)
    # #rhoplh.add_data(zrho, rhoz, srhoz, alp=0.5, scale=153.66)
    # #rhoplh.show_lab(2)
    #
    # # dzdw
    # axPLC0[1, 1].set_xlabel(r'$z$', fontsize=20)
    # axPLC0[1, 1].set_xlim(0, zmax)
    # axPLC0[1, 1].set_ylabel(r'$  \frac{\delta z}{\delta w} / [Gyr^{-1}] $', fontsize=20)
    # dzdwplh.add_plot(zp, dzdwzF, col='k', lab=r'$\Lambda CDM$', wid=1.5)
    # dzdwplh.add_plot(zp, dzdwzLT, col='m', lab=r'$LTB \ t_B=0$', wid=1.5)
    # dzdwplh.add_plot(zp, dzdwzLT2, col='c', lab=r'$LTB$', wid=1.5)
    # dzdwplh.add_data(zdzdw, dzdwz, sdzdwz, alp=1.0)
    # #dzdwplh.show_lab(3)
    #
    # handles, labels = axPLC0[0, 0].get_legend_handles_labels()
    # p1 = Rectangle((0, 0), 1, 1, fc=colourdict[files[0]], alpha=0.5)
    # handles.append(p1)
    # labels.append(labelsdict[files[0]])
    # p2 = Rectangle((0, 0), 1, 1, fc=colourdict[files[1]], alpha=0.5)
    # handles.append(p2)
    # labels.append(labelsdict[files[1]])
    # p3 = Rectangle((0, 0), 1, 1, fc=colourdict[files[2]], alpha=0.5)
    # handles.append(p3)
    # labels.append(labelsdict[files[2]])
    #
    # figPLC0.legend(handles=handles, labels=labels, loc=7)
    #
    # figPLC0.savefig(fname + 'Figures/PLC0.png', dpi=250)

    # Plot sigmasq
    print "sigmasqi0"
    l = np.linspace(0,1,Nret)
    sigmasqiplh = plh(sigmasqidict[files[0]], axsigmasq[0])
    sigmasqiplh.draw_Upper(l, sigmasqiF, sigmasqiLT)
    sigmasqiplh.add_plot(l, sigmasqiLT2, col='k-.', lab=r'$LTB_2$')
    sigmasqiplh.add_plot(l, sigmasqiLT, col='k:',lab=r'$LTB_1$')
    print "sigmasqi1"
    sigmasqiplh = plh(sigmasqidict[files[2]], axsigmasq[0])
    sigmasqiplh.add_plot(l, sigmasqiplh.contours[:, 4], col='k-', lab=labelsdict[files[2]])
    print "sigmasqi2"
    sigmasqiplh = plh(sigmasqidict[files[1]], axsigmasq[0])
    sigmasqiplh.add_plot(l, sigmasqiplh.contours[:, 4], col='k--', lab=labelsdict[files[1]])

    axsigmasq[0].set_ylabel(r'$  \log(\sigma^2_iD^2_i) $', fontsize=20)
    axsigmasq[0].set_xlabel(r'$ \frac{z}{z_{max}}$', fontsize=20)
    axsigmasq[0].set_yscale('log')
    axsigmasq[0].set_ylim(1e-13, 0.5)

    print "sigmasqf0"
    sigmasqfplh = plh(sigmasqfdict[files[0]], axsigmasq[1])
    sigmasqfplh.draw_Upper(l, sigmasqfF, sigmasqiLT)
    sigmasqfplh.add_plot(l, sigmasqfLT2, col='k-.', lab=r'$LTB_2$')
    sigmasqfplh.add_plot(l, sigmasqfLT, col='k:', lab=r'$LTB_1$')
    print "sigmasqf1"
    sigmasqfplh = plh(sigmasqfdict[files[1]], axsigmasq[1])
    sigmasqfplh.add_plot(l, sigmasqfplh.contours[:, 4], col='k--', lab=labelsdict[files[1]])
    print "sigmasqf2"
    sigmasqfplh = plh(sigmasqfdict[files[2]], axsigmasq[1])
    sigmasqfplh.add_plot(l, sigmasqfplh.contours[:, 4], col='k-', lab=labelsdict[files[2]])

    axsigmasq[1].set_ylabel(r'$  \log(\sigma^2_fD^2_f) $', fontsize=20)
    axsigmasq[1].set_xlabel(r'$ \frac{z}{z_{max}}$', fontsize=20)
    axsigmasq[1].set_yscale('log')
    axsigmasq[1].set_ylim(1e-13, 0.5)

    handles, labels = axsigmasq[0].get_legend_handles_labels()
    # p1 = Rectangle((0, 0), 1, 1, fc="red", alpha=0.5)
    # handles.append(p1)
    # labels.append(r'$FLRW$')
    # p2 = Rectangle((0, 0), 1, 1, fc="blue", alpha=0.5)
    # handles.append(p2)
    # labels.append(r'$\mathcal{D}_0$')

    figsigmasq.legend(handles=handles[::-1], labels=labels[::-1], loc=9, bbox_to_anchor=(0.035, -0.045, 1, 1), borderaxespad=0.)

    figsigmasq.savefig(fname + 'Figures/sigmasq.png', dpi=250)

    # print "OL vs Om"
    # i = 0
    # for s in files:
    #     pl2d(Om0dict[s], OL0dict[s], axOL, colour=colourdict[s])
    #     print colourdict[s], files[i]
    #     i += 1
    #     # p2 = Rectangle((0, 0), 1, 1, fc="blue", alpha=0.5)
    #     # handles.append(p2)
    #     # labels.append(r'$2-\sigma$')
    #
    # # Do labels and
    # axOL.plot(l, 1 - l, 'k', label='Flat', alpha=0.5)
    # axOL.set_xlabel(r'$\Omega_{m0}$', fontsize=25)
    # axOL.set_ylabel(r'$\Omega_{\Lambda 0}$', fontsize=25)
    # axOL.set_xlim(0.0, 1.0)
    # axOL.set_ylim(0.0, 1.5)
    # handles, labels = axOL.get_legend_handles_labels()
    # p1 = Rectangle((0, 0), 1, 1, fc=colourdict[files[0]], alpha=0.5)
    # handles.append(p1)
    # labels.append(labelsdict[files[0]])
    # p2 = Rectangle((0, 0), 1, 1, fc=colourdict[files[1]], alpha=0.5)
    # handles.append(p2)
    # labels.append(labelsdict[files[1]])
    # p3 = Rectangle((0, 0), 1, 1, fc=colourdict[files[2]], alpha=0.5)
    # handles.append(p3)
    # labels.append(labelsdict[files[2]])
    #
    # figOL.legend(handles=handles, labels=labels, loc=7)
    #
    # figOL.savefig(fname + 'Figures/OLvOm.png', dpi=250)

if __name__=="__main__":
    # Get input args
    GD = MyOptParse.readargs()

    # Print out parset settings
    keyslist = GD.keys()
    for it in keyslist:
        print it, GD[it]

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
    samps_out_name = GD["samps_out_name"]

    # Do the plots
    Plot_Data(zmax,Np,Nret,tstar,err,data_prior,data_lik,fname,Nsamp,DoPLCF,samps_out_name)