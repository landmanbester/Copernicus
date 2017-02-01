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

class plh(object):
    def __init__(self, samps, ax, ind=[]):
        self.ax = ax
        self.samps = samps
        # Check for nans
        if np.isnan(samps).any():
            I = np.unique(np.linspace(np.isnan(samps))[:, 1])
            self.samps = np.delete(samps, I, axis=1)
        # Delete preselected indices
        for i in ind:
            self.samps = np.delete(self.samps, i, axis=1)
        # self.set_Nodes(Ngrid,0.1)
        self.contours = self.get_Conf()

    def get_Conf(self):
        nstar, npoints = self.samps.shape
        contours = np.zeros([nstar, 5])
        for i in range(nstar):
            x = np.sort(self.samps[i, :])
            cdf = ECDF(x)
            #            xgrid = x[0] + x[-1]*self.l
            #            for j in range(Ngrid):
            #                cdf[j] = (sum(x <= xgrid[j]) + 0.0)/npoints
            Im = np.argwhere(cdf.y <= 0.5)[-1]  # Mean
            contours[i, 0] = cdf.x[Im]
            Id = np.argwhere(cdf.y <= 0.16)[-1]  # lower 1sig
            contours[i, 1] = cdf.x[Id]
            Idd = np.argwhere(cdf.y <= 0.025)[-1]  # lower 2sig
            contours[i, 3] = cdf.x[Idd]
            Iu = np.argwhere(cdf.y <= 0.84)[-1]  # upper 1sig
            contours[i, 2] = cdf.x[Iu]
            Iuu = np.argwhere(cdf.y <= 0.975)[-1]  # upper 2sig
            contours[i, 4] = cdf.x[Iuu]
        return contours

    def add_data(self, x, y, sy, alp=0.5, scale=1.0):
        self.ax.errorbar(x, y * scale, sy * scale, fmt='xr', alpha=alp)
        return

    def add_plot(self, x, y, col, lab, scale=1.0, wid=1.0):
        self.ax.plot(x, y * scale, col, label=lab, lw=wid)
        return

    def set_lims(self, xlow, xhigh, ylow, yhigh):
        self.ax.set_xlim(xlow, xhigh)
        self.ax.set_ylim(ylow, yhigh)
        return

    def set_label(self, xlab, xfnt, ylab, yfnt):
        self.ax.set_xlabel(xlab, fontsize=xfnt)
        self.ax.set_ylabel(ylab, fontsize=yfnt)
        return

    def show_lab(self, x):
        handles, labels = self.ax.get_legend_handles_labels()
        p1 = Rectangle((0, 0), 1, 1, fc="blue", alpha=0.8)
        handles.append(p1)
        labels.append(r'$1-\sigma$')
        p2 = Rectangle((0, 0), 1, 1, fc="blue", alpha=0.5)
        handles.append(p2)
        labels.append(r'$2-\sigma$')
        #        [p1, p2], [r'$1-\sigma$',r'$2-\sigma$']
        self.ax.legend(handles, labels, loc=x)
        return

    def draw_Contours(self, x, scale=1, smooth=0.0, alp=0.5, mode='Normal'):
        if (smooth != 0.0):
            Fm = uvs(x, self.contours[:, 0], k=3, s=smooth)(x)
            Flow1 = uvs(x, self.contours[:, 1], k=3, s=smooth)(x)
            Flow2 = uvs(x, self.contours[:, 3], k=3, s=smooth)(x)
            Fhigh1 = uvs(x, self.contours[:, 2], k=3, s=smooth)(x)
            Fhigh2 = uvs(x, self.contours[:, 4], k=3, s=smooth)(x)
        else:
            Fm = self.contours[:, 0]
            Flow1 = self.contours[:, 1]
            Flow2 = self.contours[:, 3]
            Fhigh1 = self.contours[:, 2]
            Fhigh2 = self.contours[:, 4]
        self.ax.fill_between(x, Fhigh2 * scale, Flow2 * scale, facecolor='blue', edgecolor='blue', alpha=alp,
                             label=r'$2-\sigma$')
        self.ax.fill_between(x, Fhigh1 * scale, Flow1 * scale, facecolor='blue', edgecolor='blue', alpha=alp,
                             label=r'$1-\sigma$')
        if mode == 'Cheat':
            xc = np.linspace(x[0], x[-2], x.size)
            Fm = uvs(x, Fm, k=3, s=smooth)(xc)
        self.ax.plot(x, Fm * scale, 'blue', label=r'$Median$', alpha=1.0)
        return

    def draw_Upper(self, x, F_cut, F_LTB, scale=1, alp=0.5):
        Fhigh2 = self.contours[:, 4]
        self.ax.fill_between(x, Fhigh2 * scale, F_cut * scale, facecolor='blue', edgecolor='blue', alpha=alp,
                             label=r'$2-\sigma$', lw=0.0)
        self.ax.fill_between(x, F_cut * scale, np.zeros(x.size)+1e-16, facecolor='red', edgecolor='red', alpha=alp,
                             label=r'$FLRW \ uv-cut=100Mpc$', lw=0.0)
        self.ax.plot(x, self.contours[:, 0] * scale, 'blue', label=r'$Median$', alpha=1.0)
        self.ax.plot(x, F_LTB, 'm', label=r'$t_B = 0 \ LTB$', lw=1.5)
        handles, labels = self.ax.get_legend_handles_labels()
        p1 = Rectangle((0, 0), 1, 1, fc="red", alpha=alp)
        handles.append(p1)
        labels.append(r'$FLRW \ uv-cut=100Mpc$')
        p2 = Rectangle((0, 0), 1, 1, fc="blue", alpha=alp)
        handles.append(p2)
        labels.append(r'$2-\sigma$')
        #        [p1, p2], [r'$1-\sigma$',r'$2-\sigma$']
        self.ax.legend(handles, labels, loc=2)
        return

def Plot_Data(zmax,Np,Nret,tmin,err,data_prior,data_lik,fname,Nsamp):
    print "Getting LCDM vals"
    # Get FLRW funcs for comparison
    Om0 = 0.3
    OL0 = 0.7
    H0 = 0.2335
    LCDM = FLRW(Om0, OL0, H0, zmax, Np)
    HzF = LCDM.Hz
    rhozF = LCDM.getrho()
    # sigmasqFz10 = LCDM.get_sigmasq(2.41e-9, 0.1)*HzF**2
    # sigmasqFz20 = LCDM.get_sigmasq(2.41e-9, 0.05) * HzF ** 2
    # sigmasqFz50 = LCDM.get_sigmasq(2.41e-9, 0.02) * HzF ** 2
    sigmasqFz100 = LCDM.get_sigmasq(2.41e-9, 0.01) * HzF ** 2
    v = LCDM.getnuz()
    # sigmasq10o = uvs(v/v[-1], sigmasqFz10, k =3, s=0.0)
    # sigmasq20o = uvs(v/v[-1], sigmasqFz20, k =3, s=0.0)
    # sigmasq50o = uvs(v/v[-1], sigmasqFz50, k =3, s=0.0)
    sigmasq100o = uvs(v / v[-1], sigmasqFz100, k=3, s=0.0)
    #sigmasqiF = sigmasqo(np.linspace(0, 1, Nret))

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
    # sigmasqiF10 = sigmasq10o(np.linspace(0, 1, Nret))
    # sigmasqiF20 = sigmasq20o(np.linspace(0, 1, Nret))
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

    # # read in data
    zD, Dz, sDz = np.loadtxt(fname + 'Data/D.txt', unpack=True)
    zH, Hz, sHz = np.loadtxt(fname + 'Data/H.txt', unpack=True)
    zrho, rhoz, srhoz = np.loadtxt(fname + 'Data/rho.txt', unpack=True)
    zdzdw, dzdwz, sdzdwz = np.loadtxt(fname + 'Data/dzdw.txt', unpack=True)

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

    Om0samps = 8 * np.pi * rhozsamps[0,:] / (3 * Hzsamps[0,:] ** 2)
    OL0samps = Lamsamps / (3 * Hzsamps[0,:] ** 2)

    # 3 2x2 figures with functions contours
    # The first is for data on the PLC0
    figPLC0, axPLC0 = plt.subplots(nrows=2, ncols=2, figsize=(15, 9), sharex=True)
    # The second for CP tests
    figCP, axCP = plt.subplots(nrows=2, ncols=2, figsize=(15, 9), sharex=True, sharey=True)
    # The third for t slice
    figsigmasq, axsigmasq = plt.subplots(nrows=1, ncols=1, figsize=(11, 11), sharex=True)

    #Get contours and set figure labels and lims
    print 'PLC0'
    Dplh = plh(Dzsamps, axPLC0[0, 0])
    axPLC0[0, 0].set_ylabel(r'$ D / [Gpc]$', fontsize=20)
    axPLC0[0, 0].set_ylim(0.0, 2.0)

    Hplh = plh(Hzsamps, axPLC0[0, 1])
    axPLC0[0, 1].set_ylabel(r'$ H_\parallel / [km s^{-1} Mpc^{-1}]$', fontsize=20)
    axPLC0[0, 1].set_ylim(65, 220.0)

    rhoplh = plh(rhozsamps, axPLC0[1, 0])
    axPLC0[1, 0].set_xlabel(r'$z$', fontsize=20)
    axPLC0[1, 0].set_xlim(0, zmax)
    axPLC0[1, 0].set_ylabel(r'$\frac{\rho}{\rho_c} $', fontsize=30)
    axPLC0[1, 0].set_ylim(0, 10.0)

    dzdwplh = plh(dzdwzsamps, axPLC0[1, 1])
    axPLC0[1, 1].set_xlabel(r'$z$', fontsize=20)
    axPLC0[1, 1].set_xlim(0, zmax)
    axPLC0[1, 1].set_ylabel(r'$  \frac{\delta z}{\delta w} / [Gyr^{-1}] $', fontsize=20)
    #axPLC0[1, 1].set_ylim(-1.25, 0.125)

    print 'CP'
    T1iplh = plh(T1i, axCP[0, 0])
    axCP[0, 0].set_ylabel(r'$ T_1 $', fontsize=20)

    T1fplh = plh(T1f, axCP[0, 1])

    T2iplh = plh(T2i, axCP[1, 0])
    axCP[1, 0].set_ylabel(r'$ T_2 $', fontsize=20)
    axCP[1, 0].set_xlabel(r'$ \frac{v}{v_{max}} $', fontsize=20)
    axCP[1, 0].set_xlim(0.0, 1.0)
    axCP[1, 0].set_ylim(-0.8, 0.3)

    T2fplh = plh(T2f, axCP[1, 1])
    axCP[1, 1].set_xlabel(r'$ \frac{v}{v_{max}} $', fontsize=20)

    print 'sigmasq'
    sigmasqiplh = plh(sigmasqi, axsigmasq)
    axsigmasq.set_ylabel(r'$  \sigma^2_iD^2_i $', fontsize=20)
    #axsigmasq[0, 0].set_ylim(0, 1.5)

    #sigmasqfplh = plh(sigmasqf, axsigmasq[1])
    #axsigmasq[1].set_ylabel(r'$  \sigma^2_fD^2_f $', fontsize=20)
    #axsigmasq[0, 1].set_ylim(0.4, 1.0)
    #
    # rhosplh = plh(rhostar, axts[1, 0])
    # axts[1, 0].set_ylabel(r'$  \frac{\rho^*}{\rho_c} $', fontsize=30)
    # axts[1, 0].set_xlabel(r'$  \frac{r}{r_{max}} $', fontsize=20)
    # axts[1, 0].set_xlim(0, 1)
    # axts[1, 0].set_ylim(0.0, 1.8)
    #
    # Hperpsplh = plh(Hperpstar, axts[1, 1])
    # axts[1, 1].set_ylabel(r'$ H_{\perp}^* / [km s^{-1} Mpc^{-1}] $', fontsize=20)
    # axts[1, 1].set_xlabel(r'$  \frac{r}{r_{max}} $', fontsize=20)
    # axts[1, 1].set_ylim(70, 100)

    # Plot contours
    print "Plotting"
    l = np.linspace(0, 1, Nret)

    # Plot mu(z) reconstruction and comparison
    Dplh.draw_Contours(zp)
    Dplh.add_plot(zp, DzF, col='k', lab=r'$\Lambda CDM$', wid=1.5)
    Dplh.add_plot(zp, DzLT,col='m',lab=r'$LTB$',wid=1.5)
    Dplh.add_data(zD, Dz, sDz, alp=0.2)
    Dplh.show_lab(4)

    # Plot H(z) reconstruction and comparison
    Hplh.draw_Contours(zp, scale=299.8)
    Hplh.add_plot(zp, HzF, col='k', scale=299.8, lab=r'$\Lambda CDM$', wid=1.5)
    Hplh.add_plot(zp,HzLT,col='k',scale=299.8,lab=r'$LTB$',wid=1.5)
    Hplh.add_data(zH, Hz, sHz, scale=299.8, alp=0.5)
    Hplh.show_lab(4)

    # Plot rho(z) reconstruction and comparison
    rhoplh.draw_Contours(zp, scale=153.66)
    rhoplh.add_plot(zp, rhozF, col='k', scale=153.66, lab=r'$\Lambda CDM$', wid=1.5)
    rhoplh.add_plot(zp,rhozLT,col='k',scale=153.66,lab=r'$LTB$',wid=1.5)
    rhoplh.add_data(zrho, rhoz, srhoz, alp=0.5, scale=153.66)
    rhoplh.show_lab(2)

    # Plot dzdw(z) reconstruction and comparison
    dzdwplh.draw_Contours(zp)
    dzdwplh.add_plot(zp, dzdwzF, col='k', lab=r'$\Lambda CDM$', wid=1.5)
    dzdwplh.add_plot(zp, dzdwzLT,col='m',lab=r'$LTB$',wid=1.5)
    dzdwplh.add_data(zdzdw,dzdwz,sdzdwz,alp=0.5)
    dzdwplh.show_lab(3)


    # Plot T2i(v) reconstruction and comparison
    T2iplh.draw_Contours(l)
    T2iplh.add_plot(l, T2iF, col='k', lab=r'$\Lambda CDM$', wid=1.5)
    T2iplh.add_plot(l, T2iLT, col='k', lab=r'$LTB$', wid=1.5)

    # Plot T2f(v) reconstruction and comparison
    T2fplh.draw_Contours(l)
    T2fplh.add_plot(l, T2fF, col='k', lab=r'$\Lambda CDM$', wid=1.5)
    T2fplh.add_plot(l, T2fLT, col='k', lab=r'$LTB$', wid=1.5)
    T2fplh.show_lab(2)

    # Plot T1i(v) reconstruction and comparison
    T1iplh.draw_Contours(l)
    T1iplh.add_plot(l, T1iF, col='k', lab=r'$\Lambda CDM$', wid=1.5)
    T1iplh.add_plot(l, T1iLT, col='k', lab=r'$LTB$', wid=1.5)

    # Plot T1f(v) reconstruction and comparison
    T1fplh.draw_Contours(l)
    T1fplh.add_plot(l, T1fF, col='k', lab=r'$\Lambda CDM$', wid=1.5)
    T1fplh.add_plot(l, T1fLT, col='k', lab=r'$LTB$', wid=1.5)


    # # Plot rhostar reconstruction and comparison
    # rhosplh.draw_Contours(l, scale=153.66)
    # rhosplh.add_plot(l, rhostarF, col='k', scale=153.66, lab=r'$\Lambda CDM$', wid=1.5)
    # # rhosplh.add_plot(l,rhostarConLTB,col='k',scale=153.66,lab=r'$LTB1$',wid=1.5)
    # # rhosplh.add_plot(l,rhostarLTB,col='m',scale=153.66,lab=r'$LTB2$',wid=1.5)
    # # rhosplh.show_lab(2)

    # Plot sigmasqi reconstruction
    sigmasqiplh.draw_Upper(l, sigmasqiF100, sigmasqiLT)
    #sigmasqiplh.add_plot(l, sigmasqiF10, col='k', lab=r'$\Lambda CDM \ uv-cut=10$', wid=1.5)
    #sigmasqiplh.add_plot(l, sigmasqiF20, col='y', lab=r'$\Lambda CDM \ uv-cut=20$', wid=1.5)
    #sigmasqiplh.add_plot(l, sigmasqiF100, col='c', lab=r'$\Lambda CDM \ uv-cut=100Mpc$', wid=1.5)
    #sigmasqiplh.add_plot(l, sigmasqiLT,col='m',lab=r'$t_B = 0 \ LTB$',wid=1.5)
    axsigmasq.set_yscale('log')
    axsigmasq.set_ylim(1e-13, 1e-2)
    #sigmasqiplh.show_lab(0)

    # # Plot Xstar reconstruction
    # sigmasqfplh.draw_Contours(l)
    # sigmasqfplh.add_plot(l, sigmasqfF, col='k', lab=r'$\Lambda CDM$', wid=1.5)
    # sigmasqfplh.add_plot(l, sigmasqfLT,col='m',lab=r'$LTB$',wid=1.5)
    # sigmasqfplh.show_lab(2)

    # # Plot Xstar reconstruction
    # Hperpsplh.draw_Contours(l, scale=299.8)
    # Hperpsplh.add_plot(l, HperpF * 299.8, col='k', lab=r'$\Lambda CDM$', wid=1.5)
    # # Xsplh.add_plot(l,XstarConLTB,col='k',lab=r'$LTB1$',wid=1.5)
    # # Xsplh.add_plot(l,XstarLTB,col='m',lab=r'$LTB2$',wid=1.5)
    # # Hperpsplh.show_lab(4)

    #figPLC0.tight_layout(pad=1.08, h_pad=0.0, w_pad=0.6)
    figCP.tight_layout(pad=1.08, h_pad=0.0, w_pad=0.0)
    #figts.tight_layout(pad=1.08, h_pad=0.0, w_pad=0.6)

    figPLC0.savefig(fname + 'Figures/PLC0.png', dpi=250)
    figCP.savefig(fname + 'Figures/CP.png', dpi=250)
    figsigmasq.savefig(fname + 'Figures/sigmasq.png', dpi=500)

    # Do contour plots
    print "Doing Om v OL contours"
    figConts, axConts = plt.subplots(nrows=1, ncols=2, figsize=(15, 9))

    # First Om v OL
    pl2d(Om0samps, OL0samps, axConts[0])
    axConts[0].plot(l, 1 - l, 'k', label='Flat', alpha=0.5)
    axConts[0].set_xlabel(r'$\Omega_{m0}$', fontsize=25)
    axConts[0].set_ylabel(r'$\Omega_{\Lambda 0}$', fontsize=25)
    axConts[0].set_xlim(0.0, 1.0)
    axConts[0].set_ylim(0.0, 1.5)
    handles, labels = axConts[0].get_legend_handles_labels()
    p1 = Rectangle((0, 0), 1, 1, fc="blue", alpha=0.8)
    handles.append(p1)
    labels.append(r'$1-\sigma$')
    p2 = Rectangle((0, 0), 1, 1, fc="blue", alpha=0.5)
    handles.append(p2)
    labels.append(r'$2-\sigma$')
    axConts[0].legend(handles, labels, loc=1)
    #
    # pl2d(t0samps / 0.3064, Lamsamps, axConts[1])
    axConts[1].hist2d(Om0samps,OL0samps)
    # axConts[1].set_xlabel(r'$t_0 /[Gyr]$', fontsize=25)
    # axConts[1].set_ylabel(r'$\Lambda$', fontsize=25)
    # axConts[1].set_xlim(10, 20)
    # axConts[1].set_ylim(0.0, 0.25)
    # handles, labels = axConts[1].get_legend_handles_labels()
    # p1 = Rectangle((0, 0), 1, 1, fc="blue", alpha=0.8)
    # handles.append(p1)
    # labels.append(r'$1-\sigma$')
    # p2 = Rectangle((0, 0), 1, 1, fc="blue", alpha=0.5)
    # handles.append(p2)
    # labels.append(r'$2-\sigma$')
    # axConts[1].legend(handles, labels, loc=1)

    figConts.savefig(fname + 'Figures/Contours.png', dpi=250)

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