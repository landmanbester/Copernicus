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
from matplotlib.patches import Polygon

def Plot_Data(zmax,Np,Nret,tmin,err,data_prior,data_lik,fname,Nsamp,DoPLCF,samps_out_name):
    print "Getting LCDM vals"
    # Get FLRW funcs for comparison
    Om0 = 0.3
    OL0 = 0.7
    H0 = 0.2335
    LCDM = FLRW(Om0, OL0, H0, zmax, Np)

    DelRSq = 2.41e-9
    UV_cut = 0.01
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
    colourdict[files[1]] = "blue"
    colourdict[files[2]] = "blue"
    alphadict = {}
    alphadict[files[0]] = 0.25
    alphadict[files[1]] = 0.5
    alphadict[files[2]] = 0.75
    labelsdict = {}
    labelsdict[files[0]] = r'$\mathcal{D}_0$'
    labelsdict[files[1]] = r'$\mathcal{D}_1$'
    labelsdict[files[2]] = r'$\mathcal{D}_2$'

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
            #del sigmasqisamps, sigmasqfsamps

    # read in data
    zD, Dz, sDz = np.loadtxt(fname + 'Data/D.txt', unpack=True)
    zH, Hz, sHz = np.loadtxt(fname + 'Data/Hfull.txt', unpack=True)
    zrho, rhoz, srhoz = np.loadtxt(fname + 'Data/rho.txt', unpack=True)
    zdzdw, dzdwz, sdzdwz = np.loadtxt(fname + 'Data/dzdw.txt', unpack=True)

    # Create the figures we want to plot
    # PLC0
    figPLC0, axPLC0 = plt.subplots(nrows=2, ncols=2, figsize=(15, 9), sharex=True)
    # Shear
    figsigmasq, axsigmasq = plt.subplots(nrows=1, ncols=2, figsize=(15, 9), sharey=True)
    # Om vs OL contours
    figOL, axOL = plt.subplots(nrows=1, ncols=1, figsize=(9, 9))

    # Plot PLC0
    print "PLC0"
    # First the contours
    files_tmp = ["DHt0/", "DHdzdw/"]
    for s in files_tmp:
        Dplh = plh(Dzdict[s], axPLC0[0, 0])
        Dplh.draw_Contours(zp, only_2sig=True, alp=alphadict[s], colour=colourdict[s], draw_median=False)
        Hplh = plh(Hzdict[s], axPLC0[0, 1])
        Hplh.draw_Contours(zp, scale=299.8, only_2sig=True, alp=alphadict[s], colour=colourdict[s], draw_median=False)
        rhoplh = plh(rhozdict[s], axPLC0[1, 0])
        rhoplh.draw_Contours(zp, scale=153.66, only_2sig=True, alp=alphadict[s], colour=colourdict[s], draw_median=False)
        dzdwplh = plh(dzdwdict[s], axPLC0[1, 1])
        dzdwplh.draw_Contours(zp, only_2sig=True, alp=alphadict[s], colour=colourdict[s], draw_median=False)

    # Now add labels and background plots
    # D
    axPLC0[0, 0].set_ylabel(r'$ D / [Gpc]$', fontsize=20)
    axPLC0[0, 0].set_ylim(0.0, 2.0)
    Dplh.add_plot(zp, DzF, col='b', lab=r'$\Lambda CDM$', wid=1.0)
    Dplh.add_plot(zp, DzLT, col='k:', lab=r'$LTB_1$', wid=2)
    Dplh.add_plot(zp, DzLT2, col='k--', lab=r'$LTB_2$', wid=2)
    Dplh.add_data(zD, Dz, sDz, alp=0.25)
    #Dplh.show_lab(4, only_2sig=True)

    # H
    axPLC0[0, 1].set_ylabel(r'$ H_\parallel / [km s^{-1} Mpc^{-1}]$', fontsize=20)
    axPLC0[0, 1].set_ylim(65, 220.0)
    Hplh.add_plot(zp, HzF, col='b', scale=299.8, lab=r'$\Lambda CDM$', wid=1.0)
    Hplh.add_plot(zp, HzLT, col='k:', scale=299.8, lab=r'$LTB_1$', wid=2)
    Hplh.add_plot(zp, HzLT2, col='k--', scale=299.8, lab=r'$LTB_2$', wid=2)
    Hplh.add_data(zH, Hz, sHz, scale=299.8, alp=0.5)
    #Hplh.show_lab(4)

    # rho
    axPLC0[1, 0].set_xlabel(r'$z$', fontsize=20)
    axPLC0[1, 0].set_xlim(0, zmax)
    axPLC0[1, 0].set_ylabel(r'$\frac{\rho}{\rho_c} $', fontsize=30)
    axPLC0[1, 0].set_ylim(0, 10.0)
    rhoplh.add_plot(zp, rhozF, col='b', scale=153.66, lab=r'$\Lambda CDM$', wid=1.0)
    rhoplh.add_plot(zp, rhozLT, col='k:', scale=153.66, lab=r'$LTB_1$', wid=2)
    rhoplh.add_plot(zp, rhozLT2, col='k--', scale=153.66, lab=r'$LTB_2$', wid=2)
    #rhoplh.add_data(zrho, rhoz, srhoz, alp=0.5, scale=153.66)
    #rhoplh.show_lab(2)

    # dzdw
    axPLC0[1, 1].set_xlabel(r'$z$', fontsize=20)
    axPLC0[1, 1].set_xlim(0, zmax)
    axPLC0[1, 1].set_ylabel(r'$  \frac{\delta z}{\delta w} / [Gyr^{-1}] $', fontsize=20)
    dzdwplh.add_plot(zp, dzdwzF, col='b', lab=r'$\Lambda CDM$', wid=1.0)
    dzdwplh.add_plot(zp, dzdwzLT, col='k:', lab=r'$LTB_1$', wid=2)
    dzdwplh.add_plot(zp, dzdwzLT2, col='k--', lab=r'$LTB_2$', wid=2)
    dzdwplh.add_data(zdzdw, dzdwz, sdzdwz, alp=0.5)
    #dzdwplh.show_lab(3)

    handles, labels = axPLC0[0, 0].get_legend_handles_labels()
    p1 = Rectangle((0, 0), 1, 1, fc=colourdict[files[0]], alpha=0.25)
    handles.append(p1)
    labels.append(labelsdict[files[0]])
    #p2 = Rectangle((0, 0), 1, 1, fc=colourdict[files[1]], alpha=0.5)
    #handles.append(p2)
    #labels.append(labelsdict[files[1]])
    p3 = Rectangle((0, 0), 1, 1, fc=colourdict[files[2]], alpha=0.65)
    handles.append(p3)
    labels.append(labelsdict[files[2]])

    axPLC0[0,0].legend(handles, labels, loc=4)

    figPLC0.savefig(fname + 'Figures/PLC0.pdf', dpi=250)

    # Plot sigmasq
    print "sigmasqi0"
    l = np.linspace(0,1,Nret)
    sigmasqiplh0 = plh(sigmasqidict[files[0]], axsigmasq[0], delzeros=True)
    sigmasqiplh1 = plh(sigmasqidict[files[1]], axsigmasq[0], delzeros=True)
    sigmasqiplh2 = plh(sigmasqidict[files[2]], axsigmasq[0], delzeros=True)
    axsigmasq[0].fill_between(l, sigmasqiplh0.contours[:,4], sigmasqiplh1.contours[:,4], facecolor='blue',
                              edgecolor='blue', alpha=0.25, lw=0.0)
    axsigmasq[0].fill_between(l, sigmasqiplh1.contours[:,4], sigmasqiplh2.contours[:,4], facecolor='blue',
                              edgecolor='blue', alpha=0.5, lw=0.0)
    axsigmasq[0].fill_between(l, sigmasqiplh2.contours[:,4], np.ones(Nret)*1e-13, facecolor='blue',
                              edgecolor='blue', alpha=0.75, lw=0.0)
    # Create polygon for hatching FLRW inclusion region
    x = np.zeros(2*Nret)
    x[0:Nret] = l
    x[Nret::] = np.linspace(1,0, Nret)
    y = np.zeros(2*Nret)
    y[0:Nret] = sigmasqiF
    y[Nret::] = np.ones(Nret)*1e-13
    poly = np.vstack((x,y)).T


    # axsigmasq[0].fill_between(l, sigmasqiplh2.contours[:,4], sigmasqiF, facecolor='blue',
    #                           edgecolor='blue', alpha=0.75, lw=0.0)
    # axsigmasq[0].fill_between(l, sigmasqiF, np.ones(Nret)*1e-13, facecolor='green',
    #                           edgecolor='green', alpha=1.0, lw=0.0)

    axsigmasq[0].plot(l, sigmasqiLT2, 'k--', label=r'$LTB_2$', lw=2)
    axsigmasq[0].plot(l, sigmasqiLT, 'k:', label=r'$LTB_1$', lw=2)
    axsigmasq[0].add_patch(Polygon(poly, closed=True, fill=False, hatch='/', color='k'))

    axsigmasq[0].set_ylabel(r'$  \log(\sigma^2D^2) $', fontsize=25)
    axsigmasq[0].set_xlabel(r'$ \frac{z}{z_{max}}$', fontsize=30)
    axsigmasq[0].set_yscale('log')
    axsigmasq[0].set_ylim(1e-13, 0.5)
    axsigmasq[0].set_title(r"$PLC_0$", fontsize=30)

    print "sigmasqf0"
    sigmasqfplh0 = plh(sigmasqfdict[files[0]], axsigmasq[1], delzeros=True)
    sigmasqfplh1 = plh(sigmasqfdict[files[1]], axsigmasq[1], delzeros=True)
    sigmasqfplh2 = plh(sigmasqfdict[files[2]], axsigmasq[1], delzeros=True)
    axsigmasq[1].fill_between(l, sigmasqfplh0.contours[:,4], sigmasqfplh1.contours[:,4], facecolor='blue',
                              edgecolor='blue', alpha=0.25, lw=0.0)
    axsigmasq[1].fill_between(l, sigmasqfplh1.contours[:,4], sigmasqfplh2.contours[:,4], facecolor='blue',
                              edgecolor='blue', alpha=0.5, lw=0.0)
    axsigmasq[1].fill_between(l, sigmasqfplh2.contours[:,4], np.ones(Nret)*1e-13, facecolor='blue',
                              edgecolor='blue', alpha=0.75, lw=0.0)
    # Create polygon for hatching FLRW inclusion region
    #x = np.zeros(2*Nret)
    #x[0:Nret] = l
    #x[Nret::] = np.linspace(1,0, Nret)
    #y = np.zeros(2*Nret)
    y[0:Nret] = sigmasqfF
    #y[Nret::] = np.ones(Nret)*1e-13
    poly = np.vstack((x,y)).T
    axsigmasq[1].add_patch(Polygon(poly, closed=True, fill=False, hatch='/', color='k'))

    # axsigmasq[1].fill_between(l, sigmasqfplh2.contours[:,4], sigmasqfF, facecolor='blue',
    #                           edgecolor='blue', alpha=0.75, lw=0.0)
    # axsigmasq[1].fill_between(l, sigmasqfF, np.ones(Nret)*1e-13, facecolor='green',
    #                           edgecolor='green', alpha=1.0, lw=0.0)
    axsigmasq[1].plot(l, sigmasqfLT, 'k:', label=r'$LTB_1$', lw=2)
    axsigmasq[1].plot(l, sigmasqfLT2, 'k--', label=r'$LTB_2$', lw=2)

    #axsigmasq[1].set_ylabel(r'$  \log(\sigma^2_fD^2_f) $', fontsize=20)
    axsigmasq[1].set_xlabel(r'$ \frac{z}{z_{max}}$', fontsize=30)
    axsigmasq[1].set_yscale('log')
    axsigmasq[1].set_ylim(1e-13, 0.5)
    axsigmasq[1].set_title(r"$PLC_f$",fontsize=30)

    handles, labels = axsigmasq[0].get_legend_handles_labels()
    px = Rectangle((0, 0), 1, 1, fc="blue", alpha=0.75, hatch='/')
    handles.append(px)
    labels.append(r'$FLRW$')
    p0 = Rectangle((0, 0), 1, 1, fc="blue", alpha=0.75)
    handles.append(p0)
    labels.append(r'$\mathcal{D}_2$')
    p1 = Rectangle((0, 0), 1, 1, fc="blue", alpha=0.5)
    handles.append(p1)
    labels.append(r'$\mathcal{D}_1$')
    p2 = Rectangle((0, 0), 1, 1, fc="blue", alpha=0.25)
    handles.append(p2)
    labels.append(r'$\mathcal{D}_0$')

    #figsigmasq.legend(handles=handles[::-1], labels=labels[::-1], loc=9, bbox_to_anchor=(0.035, -0.045, 1, 1), borderaxespad=0.)
    axsigmasq[0].legend(handles[::-1], labels[::-1], loc=4, ncol=3)
    figsigmasq.tight_layout(pad=1.08, h_pad=0.1, w_pad=0.1)
    figsigmasq.savefig(fname + 'Figures/sigmasq.pdf', dpi=250)

    print "OL vs Om"
    i = 0
    for s in files:
        pl2d(Om0dict[s], OL0dict[s], axOL, colour=colourdict[s], alp=alphadict[s])
        print colourdict[s], files[i]
        i += 1
        # p2 = Rectangle((0, 0), 1, 1, fc="blue", alpha=0.5)
        # handles.append(p2)
        # labels.append(r'$2-\sigma$')

    # Do labels and
    axOL.plot(l, 1 - l, 'k', label='Flat', alpha=0.5)
    axOL.set_xlabel(r'$\Omega_{m0}$', fontsize=25)
    axOL.set_ylabel(r'$\Omega_{\Lambda 0}$', fontsize=25)
    axOL.set_xlim(0.0, 1.0)
    axOL.set_ylim(0.0, 1.5)
    handles, labels = axOL.get_legend_handles_labels()
    p1 = Rectangle((0, 0), 1, 1, fc=colourdict[files[0]], alpha=0.25)
    handles.append(p1)
    labels.append(labelsdict[files[0]])
    p2 = Rectangle((0, 0), 1, 1, fc=colourdict[files[1]], alpha=0.5)
    handles.append(p2)
    labels.append(labelsdict[files[1]])
    p3 = Rectangle((0, 0), 1, 1, fc=colourdict[files[2]], alpha=0.75)
    handles.append(p3)
    labels.append(labelsdict[files[2]])

    axOL.legend(handles, labels, loc=4)

    figOL.savefig(fname + 'Figures/OLvOm.pdf', dpi=250)

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