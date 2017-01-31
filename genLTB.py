#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 10:38:55 2015

@author: landman

This class sets up and LTB model and find the observables on the PLC0

"""

from numpy import linspace,sqrt,sinh,arccosh,zeros,cosh,flipud,squeeze,pi,array,savez,argwhere,loadtxt,exp, tanh, isnan
from scipy.optimize import fsolve
from scipy.interpolate import RectBivariateSpline
from scipy.interpolate import UnivariateSpline as uvs
from scipy.integrate import odeint, cumtrapz, trapz, quad
import scipy.optimize as opt
from numpy.random import multivariate_normal as mvn
from numpy.random import random
import sympy as sm
import mpmath as mp
import matplotlib.pyplot as plt
from genFLRW import FLRW
from mpmath import elliprj
from Copernicus.Parset import MyOptParse
        
class LTB(object):
    def __init__(self,OmI,OmO,HI,HO,delr,r0,rmax,zmax,tmin,NJ,mode="LTB", fname=None):
        self.fname = fname
        #Set permanent pars
        self.mode=mode
        self.tmin = tmin
        self.NJ = NJ

        #Set z domain
        self.z = linspace(0,zmax,NJ)

        #Set r domain
        self.r = linspace(0,rmax,NJ)

        #Initialise for central differences
        self.c = range(1,NJ-1)
        self.cp = range(2,NJ)
        self.cb = range(0,NJ-2)

        #Set the symbolic expressions
        print "Setting symbolics"
        self.set_Symbolics()

        # Fit LTB model to data
        # Load the data
        print "Loading data"
        self.load_Dat()
        # self.zDdat = self.zDdat[1::]
        # self.Dzdat = self.Dzdat[1::]
        # self.sDzdat = self.sDzdat[1::]

        # print "Optimising"
        # # Initial guess for params
        X0 = array([OmI,OmO,HI,HO,r0,delr])
        # Set bounds for the optimizer
        bnds = [(0.0001, 0.999), (0.01, 1.0), (0.001, 1.0), (0.001, 1.0), (1.0e-5, 1.0), (1.0e-5, 10.0)]
        # Perform the optimisation
        opt_dict = {}
        opt_dict["epsilon"] = 1e-5
        #X = opt.fmin_slsqp(self.lik_as_func_of_params, X0, bounds=bnds, acc=1e-4, epsilon=1e-8)

        # # Get optimised value
        # X = Xp.x
        # if Xp.success:
        #     print "Optimised parameters = ", X
        # else:
        #     print "Failed to reach minimum"
        #     print Xp

        # Save the z funcs
        X = X0
        Om = self.Omf(self.r, X[0], X[1], X[4], X[5])
        dOm = self.dOmf(self.r, X[0], X[1], X[4], X[5])
        print "Getting HT0"
        if self.mode == "ConLTB":
            HT0 = self.HT0f(self.r, X[0], X[1], X[2], X[3], X[4], X[5])
            dHT0 = self.dHT0f(self.r, X[0], X[1], X[2], X[3], X[4], X[5])
        elif self.mode == "LTB":
            HT0 = self.HT0f(self.r, X[2], X[3], X[4], X[5])
            dHT0 = self.dHT0f(self.r, X[2], X[3], X[4], X[5])
        print "Getting t"
        tr = self.trf(self.r, X[0], X[1], X[2], X[3], X[4], X[5])
        if isnan(tr).any():
            print "Got NaN for tr", (Om<0.0).any(), (Om>1.0).any()
            print X
        dtr = self.dtrf(self.r, X[0], X[1], X[2], X[3], X[4], X[5])
        if isnan(dtr).any():
            print "Got NaN for dtr"
        print "Getting eta"
        eta, etadp, etap, etad, etao, etadpo, etapo, etado = self.get_t_and_eta(tr, dtr, HT0, dHT0, Om, dOm)
        # A, Ap, Ad, Adp = self.get_A_from_params(eta, etadp, etap, etad, self.r, X[0], X[1], X[2], X[3], X[4], X[5])
        # Hperp, Hpar = self.get_H_from_params(eta, etadp, etap, etad, self.r, X[0], X[1], X[2], X[3], X[4], X[5])
        print "Getting z funcs"
        Hz, Dz, rhoz, rz, tz, sigmasqDsqz = self.get_z_funcs(X[0], X[1], X[2], X[3], X[4], X[5], etao, etadpo, etapo, etado, tr[0],
                                                ComputeShear=True)

        savez(fname + 'Processed_Data/LTB_z_funcs.npz', z=self.z, Hz=Hz, Dz=Dz, rhoz=rhoz, sigmasqDsqz=sigmasqDsqz,
              params=X)


        plt.figure('Dz')
        plt.plot(self.z,Dz,'k')
        plt.errorbar(self.zDdat, self.Dzdat, self.sDzdat, fmt='xr')
        plt.savefig(fname + 'Figures/LTB_Dz.png',dpi=250)
        plt.figure('Hz')
        plt.plot(self.z,Hz,'k')
        plt.errorbar(self.zHdat, self.Hzdat, self.sHzdat, fmt='xr')
        plt.savefig(fname + 'Figures/LTB_Hz.png',dpi=250)
        plt.figure('rhoz')
        plt.plot(self.z,rhoz,'k')
        plt.savefig(fname + 'Figures/LTB_rhoz.png',dpi=250)
        plt.figure('rz')
        plt.plot(self.z,rz,'k')
        plt.savefig(fname + 'Figures/LTB_rz.png',dpi=250)
        plt.figure('tz')
        plt.plot(self.z,tz,'k')
        plt.savefig(fname + 'Figures/LTB_tz.png',dpi=250)
        plt.figure('sigmasqz')
        plt.plot(self.z, sigmasqDsqz, 'k')
        plt.savefig(fname + 'Figures/LTB_sigmasqDsq.png', dpi=250)

    def set_Symbolics(self):
        t,r,OmO,OmI,HO,HI,r0,delr,etap,etad,etadp = sm.symbols('t,r,Omega_O,Omega_I,HO,HI,r_0,delta_r,eta_p,eta_d,etadp')
        Om = OmO + (OmI - OmO)*(1 - sm.tanh((r - r0)/(2*delr)))/(1 + sm.tanh(r0/(2*delr)))
        dOm = sm.diff(Om,r)
        self.Omf = sm.utilities.lambdify([r,OmI,OmO,r0,delr],Om,modules="numpy")
        self.dOmf = sm.utilities.lambdify([r,OmI,OmO,r0,delr],dOm,modules="numpy")
        OK = 1 - Om
        if self.mode == "LTB":
            HT0 = HO + (HI - HO)*(1 - sm.tanh((r - r0)/(2*delr)))/(1 + sm.tanh(r0/(2*delr)))
            self.HT0f = sm.utilities.lambdify([r,HI,HO,r0,delr],HT0,modules="numpy")
            dHT0 = sm.diff(HT0,r)
            self.dHT0f = sm.utilities.lambdify([r,HI,HO,r0,delr],dHT0,modules="numpy")
        elif self.mode == "ConLTB":
            t0 = (1/(1-OmI) - OmI*sm.asinh(sm.sqrt((1-OmI)/OmI))/sm.sqrt(1-OmI)**3)/HI
            HT0 = HO*(1/OK - Om*sm.asinh(sm.sqrt(OK/Om))/sm.sqrt(OK)**3)/t0
            self.HT0f = sm.utilities.lambdify([r,OmI,OmO,HI,HO,r0,delr],HT0,modules="numpy")
            dHT0 = sm.diff(HT0,r)
            self.dHT0f = sm.utilities.lambdify([r,OmI,OmO,HI,HO,r0,delr],dHT0,modules="numpy")
        tr = (1/OK - Om*sm.asinh(sm.sqrt(OK/Om))/sm.sqrt(OK)**3)/HT0
        dtr = sm.diff(tr,r)
        self.trf = sm.utilities.lambdify([r,OmI,OmO,HI,HO,r0,delr],tr,modules="numpy")
        self.dtrf = sm.utilities.lambdify([r,OmI,OmO,HI,HO,r0,delr],dtr,modules="numpy")
        eta = sm.Function('eta')(t,r)
        A = Om*(sm.cosh(eta)-1)*r/(2*OK)
        self.Af = sm.utilities.lambdify([eta,r,OmI,OmO,HI,HO,r0,delr],A,modules="numpy")
        Ap = sm.diff(A,r).subs(sm.diff(eta,r),etap)
        self.Apf = sm.utilities.lambdify([eta,etap,r,OmI,OmO,HI,HO,r0,delr],Ap,modules="numpy")
        Ad = sm.diff(A,t).subs(sm.diff(eta,t),etad)
        self.Adf = sm.utilities.lambdify([eta,etad,r,OmI,OmO,HI,HO,r0,delr],Ad,modules="numpy")
        Adp = sm.diff(A,r,t)
        Adp1 = Adp.subs(sm.diff(eta,r,t),etadp)
        Adp2 = Adp1.subs(sm.diff(eta,r),etap)
        Adp3 = Adp2.subs(sm.diff(eta,t),etad)
        self.Adpf = sm.utilities.lambdify([eta,etadp,etap,etad,r,OmI,OmO,HI,HO,r0,delr],Adp3,modules="numpy")
        Hpar = Adp/Ap
        Hpar1 = Hpar.subs(sm.diff(eta,r,t),etadp)
        Hpar2 = Hpar1.subs(sm.diff(eta,r),etap)
        Hpar3 = Hpar2.subs(sm.diff(eta,t),etad)
        self.Hparf = sm.utilities.lambdify([eta,etadp,etap,etad,r,OmI,OmO,HI,HO,r0,delr],Hpar3,modules="numpy")
        Hperp = Ad/A
        Hperp1 = Hperp.subs(sm.diff(eta,t),etad)
        self.Hperpf = sm.utilities.lambdify([eta,etad,r,OmI,OmO,HI,HO,r0,delr],Hperp1,modules="numpy")
        M = HT0**2*Om*r**3
        self.Mf = sm.utilities.lambdify([r,OmI,OmO,HI,HO,r0,delr],M,modules="numpy")
        Mp = sm.diff(M,r)
        self.Mpf = sm.utilities.lambdify([r,OmI,OmO,HI,HO,r0,delr],Mp,modules="numpy")
        rho = Mp/(8*pi*A**2*Ap)
        self.rhof = sm.utilities.lambdify([eta,etap,r,OmI,OmO,HI,HO,r0,delr],rho,modules="numpy")
        E = OK*HT0**2*r**2
        self.Ef = sm.utilities.lambdify([r,OmI,OmO,HI,HO,r0,delr],E,modules="numpy")
        return

    def get_t_and_eta(self,tr,dtr,HT0,dHT0,Om,dOm):
        eta = zeros([self.NJ,self.NJ])
        etap = zeros([self.NJ,self.NJ])
        etad = zeros([self.NJ,self.NJ])
        etadp = zeros([self.NJ,self.NJ])
        eta[0,:] = arccosh(2.0/Om-1)
        if self.mode == "LTB":
            t0 = tr[0]
            tB = t0 - tr
        elif self.mode == "ConLTB":
            t0 = tr
            tB = 0.0
        t = linspace(t0,self.tmin,self.NJ)
        for i in range(self.NJ):
            if i == 0:#Need to get dtB
                etap[i,:] = fsolve(self.etapf,zeros(self.NJ),args=(eta[i,:],tr,dtr,HT0,dHT0,Om,dOm),xtol=1.0e-6)
                etad[i,:] = fsolve(self.etadf,zeros(self.NJ),args=(eta[i,:],HT0,Om),xtol=1.0e-6)
                etadp[i,:] = fsolve(self.etadpf,zeros(self.NJ),args=(etad[i,:],etap[i,:],eta[i,:],HT0,dHT0,Om,dOm),xtol=1.0e-6)
            else:
                eta[i,:] = fsolve(self.etaf,eta[i-1,:],args=(t[i] - tB,HT0,Om),xtol=1.0e-6)
                etap[i,:] = fsolve(self.etapf,etap[i-1,:],args=(eta[i,:],t[i] - tB,dtr,HT0,dHT0,Om,dOm),xtol=1.0e-6)
                etad[i,:] = fsolve(self.etadf,etad[i-1,:],args=(eta[i,:],HT0,Om),xtol=1.0e-6)
                etadp[i,:] = fsolve(self.etadpf,etadp[i-1,:],args=(etad[i,:],etap[i,:],eta[i,:],HT0,dHT0,Om,dOm),xtol=1.0e-6)
        #Now interpolate eta and derivs (RectBivariateSpline is fast but requires strictly ascending coordinates)
        if t0 > self.tmin:
            tup = flipud(t)
            #These objects are to be used in ode for t(z) and r(z) relations
            etao = RectBivariateSpline(tup,self.r,flipud(eta))
            etapo = RectBivariateSpline(tup,self.r,flipud(etap))
            etado = RectBivariateSpline(tup,self.r,flipud(etad))
            etadpo = RectBivariateSpline(tup,self.r,flipud(etadp))
        else:
            print "Got t0 <= tmin"
            if t0 == self.tmin:
                print "t0 == tmin"
            #These objects are to be used in ode for t(z) and r(z) relations
            etao = RectBivariateSpline(t,self.r,eta)
            etapo = RectBivariateSpline(t,self.r,etap)
            etado = RectBivariateSpline(t,self.r,etad)
            etadpo = RectBivariateSpline(t,self.r,etadp)
        return eta, etadp, etap, etad, etao, etadpo, etapo, etado

    #These are solver functions written in the form f(eta) = 0
    def etaf(self,eta,t,HT0,Om):
        return eta - sinh(eta) + (2*t*HT0*sqrt(1.0 - Om)**3.0)/Om

    def etapf(self,etap,eta,t,dt,HT0,dHT0,Om,dOm):
        return etap - cosh(eta)*etap + (2*dt*HT0*sqrt(1.0 - Om)**3.0)/Om + (2*t*dHT0*sqrt(1.0 - Om)**3.0)/Om - (3*t*HT0*sqrt(1.0 - Om)*dOm)/Om - (2*t*HT0*sqrt(1.0 - Om)**3.0*dOm)/Om**2

    def etadf(self,etad,eta,HT0,Om):
        return etad - cosh(eta)*etad + (2*HT0*sqrt(1.0 - Om)**3.0)/Om

    def etadpf(self,etadp,etad,etap,eta,HT0,dHT0,Om,dOm):
        return etadp - sinh(eta)*etad*etap - cosh(eta)*etadp + (2*dHT0*sqrt(1.0 - Om)**3.0)/Om - (3*HT0*sqrt(1.0 - Om)*dOm)/Om - (2*HT0*sqrt(1.0 - Om)**3.0*dOm)/Om**2

    def get_A_from_params(self,eta,etadp,etap,etad,r,OmI,OmO,HI,HO,r0,delr):
        A = self.Af(eta,r,OmI,OmO,HI,HO,r0,delr)
        Ap = self.Apf(eta,etap,r,OmI,OmO,HI,HO,r0,delr)
        Ad = self.Adf(eta,etad,r,OmI,OmO,HI,HO,r0,delr)
        Adp = self.Adpf(eta,etadp,etap,etad,r,OmI,OmO,HI,HO,r0,delr)
        return A, Ap, Ad, Adp

    def get_H_from_params(self,eta,etadp,etap,etad,r,OmI,OmO,HI,HO,r0,delr):
        Hperp = self.Hperpf(eta,etad,r,OmI,OmO,HI,HO,r0,delr)
        Hpar = self.Hparf(eta,etadp,etap,etad,r,OmI,OmO,HI,HO,r0,delr)
        return Hperp, Hpar

    def get_z_funcs(self, OmI, OmO, HI, HO, r0, delr, etao, etadpo, etapo, etado, t0, ComputeShear=False):
        #Set ICs
        y0 = [t0,0.0]
        y,outargs = odeint(self.LTBode,y0,self.z,args=(OmI,OmO,HI,HO,r0,delr,etao,etadpo,etapo,etado),full_output=True,rtol=1e-5,atol=1e-5)
        tz = y[:,0]
        rz = y[:,1]
        etaz = zeros(self.NJ)
        etapz = zeros(self.NJ)
        etadz = zeros(self.NJ)
        etadpz = zeros(self.NJ)
        for i in range(self.NJ):
            etaz[i] = squeeze(etao(tz[i],rz[i]))
            etapz[i] = squeeze(etapo(tz[i],rz[i]))
            etadz[i] = squeeze(etado(tz[i],rz[i]))
            etadpz[i] = squeeze(etadpo(tz[i],rz[i]))
        Hz = self.Hparf(etaz,etadpz,etapz,etadz,rz,OmI,OmO,HI,HO,r0,delr)
        Dz = self.Af(etaz,rz,OmI,OmO,HI,HO,r0,delr)
        rhoz = self.rhof(etaz,etapz,rz,OmI,OmO,HI,HO,r0,delr)
        rhoz[0] = 3*OmI*HI**2/(8*pi)
        if ComputeShear:
            Az, Apz, Adz, Adpz = self.get_A_from_params(etaz, etadpz, etapz, etadz, rz, OmI, OmO, HI, HO, r0, delr)
            sigmasqDsq = (Apz*Adz - Adpz*Az)/(3*Apz**2)
            return Hz, Dz, rhoz, rz, tz, sigmasqDsq
        else:
            return Hz, Dz, rhoz, rz, tz

    def LTBode(self,y,z,OmI,OmO,HI,HO,r0,delr,etao,etadpo,etapo,etado):
        eta = squeeze(etao(y[0],y[1]))
        etap = squeeze(etapo(y[0],y[1]))
        etad = squeeze(etado(y[0],y[1]))
        etadp = squeeze(etadpo(y[0],y[1]))
        Adp = self.Adpf(eta,etadp,etap,etad,y[1],OmI,OmO,HI,HO,r0,delr)
        Hpar = self.Hparf(eta,etadp,etap,etad,y[1],OmI,OmO,HI,HO,r0,delr)
        E = self.Ef(y[1],OmI,OmO,HI,HO,r0,delr)
        dy = zeros(2)
        dy[0] = -1.0/((1.0+z)*Hpar)
        dy[1] = sqrt(1.0 + E)/((1.0+z)*Adp)
        return [dy[0],dy[1]]

    def get_D(self,H,rho):
        """
        Solves ODE for D
        """
        nuz = cumtrapz(1.0/((1.0+self.z)**2.0*H),self.z, initial = 0.0)
        nu = linspace(0,nuz[-1],self.NJ)
        u1o = uvs(nuz,(1.0+self.z),k=3,s=0.0)
        rhoo = uvs(nuz,rho,k=3,s=0.0)
        y0 = array([0.0,1.0])
        y, odeinfo = odeint(self.D_ode,y0,nu, args=(rhoo,u1o),full_output=1)
        Dnu = y[:,0]
        Do = uvs(nu,Dnu,k=3,s=0.0)
        Dz = Do(nuz)
        Dz[0] = 0.0
        return Dz

    def D_ode(self,y,v,rhoo,u1o):
        """
        The ode for D, can be used to check if CIVP agrees with parametric LTB
        """
        rho = rhoo(v)
        u1 = u1o(v)
        dy = zeros(2)
        dy[0] = y[1]
        dy[1] = -8.0*pi*u1**2.0*rho*y[0]/2.0
        return dy

    def load_Dat(self):
        self.zDdat,self.Dzdat,self.sDzdat = loadtxt('Data/Unionrz.txt',unpack=True)
        self.zHdat,self.Hzdat,self.sHzdat = loadtxt('Data/CChz.txt',unpack=True)
        #self.zrhodat, self.rhozdat, self.srhozdat = loadtxt('Data/Simrho.txt', unpack=True)
        return

    def get_Chi2(self,Dz,Hz,rhoz):
        #Get funcs at data points
        Dzi = uvs(self.z,Dz,k=3,s=0.0)(self.zDdat)
        Hzi = uvs(self.z,Hz,k=3,s=0.0)(self.zHdat)
        #rhozi = uvs(self.z, rhoz, k=3, s=0.0)(self.zrhodat)
        chi2D = sum((self.Dzdat - Dzi)**2/(self.sDzdat)**2)
        chi2H = sum((self.Hzdat - Hzi)**2/(self.sHzdat)**2)
        #chi2rho = sum((self.rhozdat - rhozi) ** 2 / (self.srhozdat) ** 2)
        return chi2D + chi2H #+ chi2rho

    def lik_as_func_of_params(self, X):
        if (X <= 0.0).any():
            return 1.0e9
        else:
            #print "Getting Om"
            Om = self.Omf(self.r, X[0], X[1], X[4], X[5])
            dOm = self.dOmf(self.r, X[0], X[1], X[4], X[5])
            #print "Getting HT0"
            if self.mode == "ConLTB":
                HT0 = self.HT0f(self.r, X[0], X[1], X[2], X[3], X[4], X[5])
                dHT0 = self.dHT0f(self.r, X[0], X[1], X[2], X[3], X[4], X[5])
            elif self.mode == "LTB":
                HT0 = self.HT0f(self.r, X[2], X[3], X[4], X[5])
                dHT0 = self.dHT0f(self.r, X[2], X[3], X[4], X[5])
            #print "Getting t"
            tr = self.trf(self.r, X[0], X[1], X[2], X[3], X[4], X[5])
            if isnan(tr).any():
                print "Got NaN for tr", (Om<0.0).any(), (Om>1.0).any()
                print X
            dtr = self.dtrf(self.r, X[0], X[1], X[2], X[3], X[4], X[5])
            if isnan(dtr).any():
                print "Got NaN for dtr"
            #print "Getting eta"
            eta, etadp, etap, etad, etao, etadpo, etapo, etado = self.get_t_and_eta(tr, dtr, HT0, dHT0, Om, dOm)
            # A, Ap, Ad, Adp = self.get_A_from_params(eta, etadp, etap, etad, self.r, X[0], X[1], X[2], X[3], X[4], X[5])
            # Hperp, Hpar = self.get_H_from_params(eta, etadp, etap, etad, self.r, X[0], X[1], X[2], X[3], X[4], X[5])
            #print "Getting z funcs"
            Hz, Dz, rhoz , tz, rz = self.get_z_funcs(X[0], X[1], X[2], X[3], X[4], X[5], etao, etadpo, etapo, etado, tr[0])
            #print "Getting lik"
            lik = self.get_Chi2(Dz, Hz, rhoz)
            print lik, X
            return lik



if __name__ == "__main__":
    # Get input args
    GD = MyOptParse.readargs()

    fname = GD["fname"]
    zmax = GD["zmax"]
    NJ = GD['np']

    # Set LTB params
    OmO = 7.21983806e-01
    OmI = 4.47595647e-03
    HI = 2.34416934e-01
    HO = 1.77978280e-01
    r0 = 9.99999994e-06
    delr = 2.07029825e+00
    rmax = 10.0
    tmin = 0.05

    M = LTB(OmI, OmO, HI, HO, delr, r0, rmax, zmax, tmin, NJ, mode='ConLTB', fname=fname)