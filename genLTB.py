# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 10:38:55 2015

@author: landman

This class sets up and LTB model and find the observables on the PLC0

"""

from numpy import linspace,sqrt,sinh,arccosh,zeros,cosh,flipud,squeeze,pi,array,savez,argwhere,loadtxt,exp, tanh
from scipy.optimize import fsolve
from scipy.interpolate import RectBivariateSpline
from scipy.interpolate import UnivariateSpline as uvs
from scipy.integrate import odeint, cumtrapz, trapz, quad
from numpy.random import multivariate_normal as mvn
from numpy.random import random
import sympy as sm
import mpmath as mp
import matplotlib.pyplot as plt
from genFLRW import FLRW
from mpmath import elliprj

class LLTB(object):
    def __init__(self,OmI,OmO,HI,HO,delr,r0,rmax,zmax,NJ,Lambda = 0.0,mode="LTB"):
        self.NJ = NJ
        #Set z domain
        self.z = linspace(0,zmax,NJ)
        #Set r domain
        self.r = linspace(0,rmax,NJ)
        #Set elliptic function to get t(R)
        self.set_age_symb()
        #Set R grid
        self.R = zeros([self.NJ,self.NJ])
        #Here we choose the gauge R0 = r
        for i in range(1,self.NJ):
            self.R[:,i] = linspace(self.r[i],0.1*self.r[i],self.NJ)
        #Set R0/R
        self.c = self.R/self.r
        self.c[:,0] = self.c[:,1]


        #Init storage array for MCMC parameters
    def get_tR(self,OmI,OmO,HI,HO,delr,r0):
        #Set dim dens at t0
        Om0 = self.Omf(OmI,OmO,delr,r0)
        Ok0 = self.Okf(Om0)
        HT0 = self.HTf(HI,HO,delr,r0)
        tr = zeros([self.NJ,self.NJ])
        tRf = lambda x,a,b,c,d: sqrt(x)/(d*sqrt(a + b*x + c*x**3))
        for i in range(self.NJ):
            for j in range(self.NJ):
                tr[i,j] = quad(tRf,0,self.c[i,j],args=(Om0[j],Ok0[j],0.0,HT0[j]))[0]
        return tr
        
    def get_age(self,Om0,Ok0,OL0,H0,c):
        qi = self.zroots(Om0,1.0-Om0,0.0,1.0)
        print Om0, Ok0
        print qi
        t0tmp = 2*elliprj(-qi[0],-qi[1],-qi[2],c)/(3*H0*sqrt(Om0))
        t0 = float(t0tmp.real)        
#        try:
#            qi = self.zroots(Om0,Ok0,OL0,c)
#            t0tmp = 2*elliprj(-qi[0],-qi[1],-qi[2],c)/(3*H0*sqrt(Om0))
#            t0 = float(t0tmp.real)
#        except:
#            a = linspace(1e-8,1,10000)
#            t0 = trapz(sqrt(a)/(H0*sqrt(Om0 + Ok0*a + OL0*a**3)),a)
#            print "Resorted to finding t0 numerically", t0
        return t0            
                
        
    def Omf(self,OmI,OmO,delr,r0):
        return OmO + (OmI - OmO)*(1 - tanh((self.r - r0)/(2*delr)))/(1 + tanh(r0/(2*delr)))

    def Okf(self,Om):
        return 1 - Om
        
    def HTf(self,HI,HO,delr,r0):
        return HO + (HI - HO)*(1 - tanh((self.r - r0)/(2*delr)))/(1 + tanh(r0/(2*delr)))        

    def set_age_symb(self):
        #Set the symbolic vars required to get age 
        x,Omo,OKo,OLo,c = sm.symbols('t_0,Omega_m,Omega_K,Omega_Lambda,c')
        f = (x + c)**3 + OKo*(x + c)**2/Omo + OLo/Omo
        q = sm.roots(f,x,multiple=True)
        self.zroots = sm.utilities.lambdify([Omo,OKo,OLo,c],q,modules="numpy")
        return
        
class LTB(object):
    def __init__(self,OmI,OmO,HI,HO,delr,r0,rmax,zmax,tmin,NJ,Lambda = 0.0,mode="LTB"):
        #Set free parameters
        self.OmI = OmI
        self.OmO = OmO
        self.HI = HI
        self.HO = HO
        self.delr = delr
        self.r0 = r0
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

        #Set density and expansion profiles and get t_B(r)
        self.set_Symbolics(mode)
        self.Om = self.Omf(self.r,self.OmI,self.OmO,self.r0,self.delr)
        self.dOm = self.dOmf(self.r,self.OmI,self.OmO,self.r0,self.delr)
        if mode == "ConLTB":
            #Make sure HO is set to one (dirty bypass to include constrained model as submodel)
            self.HO = 1.0
            self.HT0 = self.HT0f(self.r,self.OmI,self.OmO,self.HI,self.HO,self.r0,self.delr)
            self.dHT0 = self.dHT0f(self.r,self.OmI,self.OmO,self.HI,self.HO,self.r0,self.delr)
        elif mode == "LTB":
            self.HT0 = self.HT0f(self.r,self.HI,self.HO,self.r0,self.delr)
            self.dHT0 = self.dHT0f(self.r,self.HI,self.HO,self.r0,self.delr)
        elif mode =="ConLLTB":
            self.params = [OmI,OmO,delr,r0,Lambda]
        elif mode == "LLTB":
            self.params = [OmI,OmO,HI,HO,delr,r0,Lambda]
        self.tr = self.trf(self.r,self.OmI,self.OmO,self.HI,self.HO,self.r0,self.delr)
        self.dtr = self.dtrf(self.r,self.OmI,self.OmO,self.HI,self.HO,self.r0,self.delr)

        #Get other profiles
        if mode == "LTB" or mode == "ConLTB":
            self.OK = 1 - self.Om
        else:
            print "Not implemented"


        #Now get parameter eta
        eta, etadp, etap, etad = self.get_t_and_eta(mode)
        #Test etap
        delx = self.r[1] - self.r[0]
        self.etapn = self.get_num_diff(self.eta,delx)
        #Test etad
        delt = self.t[1] - self.t[0]
        self.etadn = self.get_num_diff(self.eta,delt,mode='t')
        #Test etadp
        self.etadpn = self.get_num_diff(self.etap,delt,mode='t')

        #Now get A and derivs
        self.A, self.Ap, self.Ad, self.Adp = self.get_A_from_params(eta,etadp,etap,etad,self.r,self.OmI,self.OmO,self.HI,self.HO,self.r0,self.delr)
        self.Apn = self.get_num_diff(self.A,delx)
        self.Adn = self.get_num_diff(self.A,delt,mode='t')
        self.Adpn = self.get_num_diff(self.Ad,delx)

        #Get rho
        self.rhot = self.rhof(eta,etap,self.r,self.OmI,self.OmO,self.HI,self.HO,self.r0,self.delr)
        self.rhot[:,0] = 3*OmI*HI**2/(8*pi*self.Ap[:,0]**3)

        #Next get Hperp and Hpar from params
        self.Hperp, self.Hpar = self.get_H_from_params(eta,etadp,etap,etad,self.r,self.OmI,self.OmO,self.HI,self.HO,self.r0,self.delr)

        #Finally get z rels
        self.Hz, self.Dz, self.rhoz = self.get_z_funcs(self.OmI,self.OmO,self.HI,self.HO,self.r0,self.delr)

        #Load the data
        self.load_Dat()
        self.zDdat = self.zDdat[1::]
        self.Dzdat = self.Dzdat[1::]
        self.sDzdat = self.sDzdat[1::]

        #Get likelihood
        self.lik = self.get_Chi2(self.Dz,self.Hz)
        self.maxlik = self.lik
        self.bestX = array([self.OmI,self.OmO,self.HI,self.HO,self.r0,self.delr])

        #Test D with obs ode
        #self.Dz2 = self.get_D(self.Hz,self.rhoz)
        
        self.save_z_funcs(3.25)


    def set_Symbolics(self,mode):
        t,r,OmO,OmI,HO,HI,r0,delr,etap,etad,etadp = sm.symbols('t,r,Omega_O,Omega_I,HO,HI,r_0,delta_r,eta_p,eta_d,etadp')
        Om = OmO + (OmI - OmO)*(1 - sm.tanh((r - r0)/(2*delr)))/(1 + sm.tanh(r0/(2*delr)))
        dOm = sm.diff(Om,r)
        self.Omf = sm.utilities.lambdify([r,OmI,OmO,r0,delr],Om,modules="numpy")
        self.dOmf = sm.utilities.lambdify([r,OmI,OmO,r0,delr],dOm,modules="numpy")
        OK = 1 - Om
        if mode == "LTB":
            HT0 = HO + (HI - HO)*(1 - sm.tanh((r - r0)/(2*delr)))/(1 + sm.tanh(r0/(2*delr)))
            self.HT0f = sm.utilities.lambdify([r,HI,HO,r0,delr],HT0,modules="numpy")
            dHT0 = sm.diff(HT0,r)
            self.dHT0f = sm.utilities.lambdify([r,HI,HO,r0,delr],dHT0,modules="numpy")
        elif mode == "ConLTB":
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

    def get_t_and_eta(self,mode):
        self.eta = zeros([self.NJ,self.NJ])
        self.etap = zeros([self.NJ,self.NJ])
        self.etad = zeros([self.NJ,self.NJ])
        self.etadp = zeros([self.NJ,self.NJ])
        self.eta[0,:] = arccosh(2.0/self.Om-1)
        if mode == "LTB":
            self.t0 = self.tr[0]
            self.tB = self.t0 - self.tr
        elif mode == "ConLTB":
            self.t0 = self.tr
            self.tB = 0.0
        self.t = linspace(self.t0,self.tmin,self.NJ)
        for i in range(self.NJ):
            if i%25 == 0:
                print i
            if i == 0:
                self.etap[i,:] = fsolve(self.etapf,zeros(self.NJ),args=(self.eta[i,:],self.tr,self.dtr,self.HT0,self.dHT0,self.Om,self.dOm),xtol=1.0e-10)
                self.etad[i,:] = fsolve(self.etadf,zeros(self.NJ),args=(self.eta[i,:],self.HT0,self.Om),xtol=1.0e-10)
                self.etadp[i,:] = fsolve(self.etadpf,zeros(self.NJ),args=(self.etad[i,:],self.etap[i,:],self.eta[i,:],self.HT0,self.dHT0,self.Om,self.dOm),xtol=1.0e-10)
            else:
                self.eta[i,:] = fsolve(self.etaf,self.eta[i-1,:],args=(self.t[i] - self.tB,self.HT0,self.Om),xtol=1.0e-10)
                self.etap[i,:] = fsolve(self.etapf,self.etap[i-1,:],args=(self.eta[i,:],self.t[i] - self.tB,self.dtr,self.HT0,self.dHT0,self.Om,self.dOm),xtol=1.0e-10)
                self.etad[i,:] = fsolve(self.etadf,self.etad[i-1,:],args=(self.eta[i,:],self.HT0,self.Om),xtol=1.0e-10)
                self.etadp[i,:] = fsolve(self.etadpf,self.etadp[i-1,:],args=(self.etad[i,:],self.etap[i,:],self.eta[i,:],self.HT0,self.dHT0,self.Om,self.dOm),xtol=1.0e-10)
        #Now interpolate eta and derivs (RectBivariateSpline is fast but requires strictly ascending coordinates)
        tup = flipud(self.t)
        #These objects are to be used in ode for t(z) and r(z) relations
        self.etao = RectBivariateSpline(tup,self.r,flipud(self.eta))
        self.etapo = RectBivariateSpline(tup,self.r,flipud(self.etap))
        self.etado = RectBivariateSpline(tup,self.r,flipud(self.etad))
        self.etadpo = RectBivariateSpline(tup,self.r,flipud(self.etadp))
        return self.eta, self.etadp, self.etap, self.etad

    def get_t_and_eta_MCMC(self,mode,tr,dtr,HT0,dHT0,Om,dOm):
        eta = zeros([self.NJ,self.NJ])
        etap = zeros([self.NJ,self.NJ])
        etad = zeros([self.NJ,self.NJ])
        etadp = zeros([self.NJ,self.NJ])
        eta[0,:] = arccosh(2.0/Om-1)
        if mode == "LTB":
            t0 = tr[0]
            tB = t0 - tr
        elif mode == "ConLTB":
            t0 = tr
            tB = 0.0
        t = linspace(t0,self.tmin,self.NJ)
        for i in range(self.NJ):
            if i == 0:#Need to get dtB
                etap[i,:] = fsolve(self.etapf,zeros(self.NJ),args=(eta[i,:],tr,dtr,HT0,dHT0,Om,dOm),xtol=1.0e-10)
                etad[i,:] = fsolve(self.etadf,zeros(self.NJ),args=(eta[i,:],HT0,Om),xtol=1.0e-10)
                etadp[i,:] = fsolve(self.etadpf,zeros(self.NJ),args=(etad[i,:],etap[i,:],eta[i,:],HT0,dHT0,Om,dOm),xtol=1.0e-10)
            else:
                eta[i,:] = fsolve(self.etaf,eta[i-1,:],args=(t[i] - tB,HT0,Om),xtol=1.0e-10)
                etap[i,:] = fsolve(self.etapf,etap[i-1,:],args=(eta[i,:],t[i] - tB,dtr,HT0,dHT0,Om,dOm),xtol=1.0e-10)
                etad[i,:] = fsolve(self.etadf,etad[i-1,:],args=(eta[i,:],HT0,Om),xtol=1.0e-10)
                etadp[i,:] = fsolve(self.etadpf,etadp[i-1,:],args=(etad[i,:],etap[i,:],eta[i,:],HT0,dHT0,Om,dOm),xtol=1.0e-10)
        #Now interpolate eta and derivs (RectBivariateSpline is fast but requires strictly ascending coordinates)
        tup = flipud(t)
        #These objects are to be used in ode for t(z) and r(z) relations
        etao = RectBivariateSpline(tup,self.r,flipud(eta))
        etapo = RectBivariateSpline(tup,self.r,flipud(etap))
        etado = RectBivariateSpline(tup,self.r,flipud(etad))
        etadpo = RectBivariateSpline(tup,self.r,flipud(etadp))
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

    def get_z_funcs(self,OmI,OmO,HI,HO,r0,delr):
        #Set ICs
        y0 = [self.t0,self.r[0]]
        y,outargs = odeint(self.LTBode,y0,self.z,args=(OmI,OmO,HI,HO,r0,delr),full_output=True)
        self.tz = y[:,0]
        self.rz = y[:,1]
        etaz = zeros(self.NJ)
        etapz = zeros(self.NJ)
        etadz = zeros(self.NJ)
        etadpz = zeros(self.NJ)
        for i in range(self.NJ):
            etaz[i] = squeeze(self.etao(self.tz[i],self.rz[i]))
            etapz[i] = squeeze(self.etapo(self.tz[i],self.rz[i]))
            etadz[i] = squeeze(self.etado(self.tz[i],self.rz[i]))
            etadpz[i] = squeeze(self.etadpo(self.tz[i],self.rz[i]))
        Hz = self.Hparf(etaz,etadpz,etapz,etadz,self.rz,OmI,OmO,HI,HO,r0,delr)
        Dz = self.Af(etaz,self.rz,OmI,OmO,HI,HO,r0,delr)
        rhoz = self.rhof(etaz,etapz,self.rz,OmI,OmO,HI,HO,r0,delr)
        rhoz[0] = 3*OmI*HI**2/(8*pi)
        return Hz, Dz, rhoz

    def LTBode(self,y,z,OmI,OmO,HI,HO,r0,delr):
        eta = squeeze(self.etao(y[0],y[1]))
        etap = squeeze(self.etapo(y[0],y[1]))
        etad = squeeze(self.etado(y[0],y[1]))
        etadp = squeeze(self.etadpo(y[0],y[1]))
        Adp = self.Adpf(eta,etadp,etap,etad,y[1],OmI,OmO,HI,HO,r0,delr)
        Hpar = self.Hparf(eta,etadp,etap,etad,y[1],OmI,OmO,HI,HO,r0,delr)
        E = self.Ef(y[1],OmI,OmO,HI,HO,r0,delr)
        dy = zeros(2)
        dy[0] = -1.0/((1.0+z)*Hpar)
        dy[1] = sqrt(1.0 + E)/((1.0+z)*Adp)
        return [dy[0],dy[1]]

    def get_z_funcs_MCMC(self,OmI,OmO,HI,HO,r0,delr,etao,etadpo,etapo,etado):
        #Set ICs
        y0 = [self.t0,self.r[0]]
        y,outargs = odeint(self.LTBode_MCMC,y0,self.z,args=(OmI,OmO,HI,HO,r0,delr,etao,etadpo,etapo,etado),full_output=True,rtol=1e-5,atol=1e-5)
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
        return Hz, Dz, rhoz

    def LTBode_MCMC(self,y,z,OmI,OmO,HI,HO,r0,delr,etao,etadpo,etapo,etado):
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
        This function gets the chisquared of the reconstructed D = D(H,rho)
        with the data. Input should be H = H(zp) and rho = rho(zp).
        Output is D(zp), Dprob and Flag
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

    def save_z_funcs(self,tstar):
        #Get tslice for comparison
        Iz = argwhere(self.tz >= tstar)[-1]
        rzmax = self.rz[Iz]
        It = argwhere(self.t >= tstar)[-1]
        Ir = argwhere(self.r <= rzmax)
        rstar = self.r[Ir]
        E = self.Ef(rstar,self.OmI,self.OmO,self.HI,self.HO,self.r0,self.delr)
        Xstar = self.Ap[It,Ir]/sqrt(1 + E)
        Dstar = self.A[It,Ir]
        rhostar = self.rhot[It,Ir]
        l = rstar/rstar[-1]
        savez("RawData/LTBDat.npz",z=self.z,Dz=self.Dz,Hz=self.Hz,rhoz=self.rhoz,NJ=self.NJ,l=l,rstar=rstar,Dstar=Dstar,Xstar=Xstar,rhostar=rhostar)
        #In case we want to save params
        #,OmI=self.OmI,OmO=self.OmO,HI=self.HI,HO=self.HO,r0=self.r0,delr=self.delr)
        plt.figure('Xstar')
        plt.plot(l,Xstar)
        plt.figure('rhostar')
        plt.plot(l,rhostar)
        plt.figure('Dstar')
        plt.plot(l,Dstar)
        return

    def D_ode(self,y,v,rhoo,u1o):
        """
        The ode for D
        """
        rho = rhoo(v)
        u1 = u1o(v)
        dy = zeros(2)
        dy[0] = y[1]
        dy[1] = -8.0*pi*u1**2.0*rho*y[0]/2.0
        return dy

    def load_Dat(self):
        self.zDdat,self.Dzdat,self.sDzdat = loadtxt('RawData/Unionrz.txt',unpack=True)
        self.zHdat,self.Hzdat,self.sHzdat = loadtxt('RawData/CChz.txt',unpack=True)
        return

    def get_Chi2(self,Dz,Hz):
        #Get funcs at data points
        Dzi = uvs(self.z,Dz,k=3,s=0.0)(self.zDdat)
        Hzi = uvs(self.z,Hz,k=3,s=0.0)(self.zHdat)
        chi2D = sum((self.Dzdat - Dzi)**2/(self.sDzdat)**2)
        chi2H = sum((self.Hzdat - Hzi)**2/(self.sHzdat)**2)
        return chi2D + chi2H

    def propose_params(self,X0,SIGMA,mode):
        X = mvn(X0,SIGMA)
        X[1] = 1.0
        if mode == "ConLTB":
            #Make sure HO is set to one (dirty bypass to include constrained model as submodel)
            X[3] = 1.0
        return X
        
    def MCMCStep(self,X0,lik0,SIGMA,mode):
        """
        X = [OmI,OmO,HI,HO,r0,delr]
        """
        X = self.propose_params(X0,SIGMA,mode)
        if (X<=0.0).any():
            return X0, lik0, 0.0
        else:
            Om = self.Omf(self.r,X[0],X[1],X[4],X[5])
            dOm = self.dOmf(self.r,X[0],X[1],X[4],X[5])
            if mode == "ConLTB":
                HT0 = self.HT0f(self.r,X[0],X[1],X[2],X[3],X[4],X[5])
                dHT0 = self.dHT0f(self.r,X[0],X[1],X[2],X[3],X[4],X[5])
            elif mode == "LTB":
                HT0 = self.HT0f(self.r,X[2],X[3],X[4],X[5])
                dHT0 = self.dHT0f(self.r,X[2],X[3],X[4],X[5])
            tr = self.trf(self.r,X[0],X[1],X[2],X[3],X[4],X[5])
            dtr = self.dtrf(self.r,X[0],X[1],X[2],X[3],X[4],X[5])
            eta, etadp, etap, etad, etao, etadpo, etapo, etado = self.get_t_and_eta_MCMC(mode,tr,dtr,HT0,dHT0,Om,dOm)
            A, Ap, Ad, Adp = self.get_A_from_params(eta,etadp,etap,etad,self.r,X[0],X[1],X[2],X[3],X[4],X[5])
            Hperp, Hpar = self.get_H_from_params(eta,etadp,etap,etad,self.r,X[0],X[1],X[2],X[3],X[4],X[5])
            Hz, Dz, rhoz = self.get_z_funcs_MCMC(X[0],X[1],X[2],X[3],X[4],X[5],etao,etadpo,etapo,etado)
            lik = self.get_Chi2(Dz,Hz)
        logr = lik - lik0
        accprob = exp(-logr/2.0)
        u = random(1)
        if u < accprob:
            self.track_max_lik(lik,X)
            return X, lik, 1.0
        else:
            return X0, lik0, 0.0
        return

    def track_max_lik(self,lik,X):
        if lik < self.maxlik:
            self.maxlik = lik
            self.bestX = X
        return

    def get_num_diff(self,f,h,mode='r'):
        df = zeros([self.NJ,self.NJ])
        if mode == 'r':
            df[:,self.c] = (f[:,self.cp] - f[:,self.cb])/(2*h)
            df[:,0] = (-f[:,2] + 4*f[:,1] - 3*f[:,0])/(2*h)
            df[:,-1] = (f[:,-3] - 4*f[:,-2] + 3*f[:,-1])/(2*h)
        elif mode == 't':
            df[self.c,:] = (f[self.cp,:] - f[self.cb,:])/(2*h)
            df[0,:] = (-f[2,:] + 4*f[1,:] - 3*f[0,:])/(2*h)
            df[-1,:] = (f[-3,:] - 4*f[-2,:] + 3*f[-1,:])/(2*h)
        return df


if __name__ == "__main__":
#    
#    
#    #Set params
#    OmO = 1.0
#    OmI = 0.1278715
#    HI = 0.23282359
#    HO = 0.16561116
#    r0 = 0.04455701
#    delr = 2.15397066
#    rmax = 10.0
#    zmax = 2.0
#    NJ = 250
#    
#    M = LLTB(OmI,OmO,HI,HO,delr,r0,rmax,zmax,NJ)
#    tR = M.get_tR(OmI,OmO,HI,HO,delr,r0)
#    
#    
###################################################################################################################
# These run a preliminary MCMC    
    #Load Data
    zD,Dz,sDz = loadtxt('RawData/SimSKAD.txt',unpack=True)
    zH,Hz,sHz = loadtxt('RawData/SimSKAH.txt',unpack=True)
    zrhom,rhom,srhom =loadtxt("RawData/Simzero.txt",unpack=True)
    
    #Load LCDM model
    zmax = 2.0
    NJ = 250
    LCDM = FLRW(0.3,0.7,0.2334889926617745,zmax,NJ)
    DzF = LCDM.Dz
    HzF = LCDM.Hz
    zF = LCDM.z
    rhozF = LCDM.getrho()
    nuzF = LCDM.getnuz()    

    #Set params
    OmO = 1.0
    OmI = 0.1278715
    HI = 0.23282359
    HO = 0.16561116
    r0 = 0.04455701
    delr = 2.15397066
    rmax = 10
    tmin = 0.5

    #Initiale LTB object
    mode = "LTB"
    M = LTB(OmI,OmO,HI,HO,delr,r0,rmax,zmax,tmin,NJ,mode=mode)
    
#    #Set MCMC params
#    nsamp = 1000
#    nburn = nsamp/10
#    
#    #Starting vals and characteristic variance
#    X = array([OmI,OmO,HI,HO,r0,delr])
#    SIGMA = array([[0.0001*OmI,0.0,0.0,0.0,0.0,0.0],[0.0,OmO,0.0,0.0,0.0,0.0],[0.0,0.0,0.0005*HI,0.0,0.0,0.0],[0.0,0.0,0.0,0.00025*HO,0.0,0.0],[0.0,0.0,0.0,0.0,0.01*r0,0.0],[0.0,0.0,0.0,0.0,0.0,0.005*delr]])    
#    
#    #Storage arrays
#    npar = 6 
#    Xs = zeros([nsamp,npar])
#
#    #Get initial likelihood
#    lik = M.lik
#
#    for i in range(nburn):
#        X, lik, a = M.MCMCStep(X,lik,SIGMA,mode)
#        
#    accrate = zeros(2)
#    
#    for i in range(nsamp):
#        X, lik, a = M.MCMCStep(X,lik,SIGMA,mode)
#        Xs[i,:] = X
#        accrate += array([a,1.0])
#        
#    print "accrate = ", accrate[0]*100/accrate[1]
#
#
#
######################### These were for testing###############################    
#    eta = M.eta
#    etapn = M.etapn
#    etap = M.etap
#    etadn = M.etadn
#    etad = M.etad
#    etadp = M.etadp
#    etadpn = M.etadpn
#
#    tr = M.tr
#    dtr = M.dtr
#    r = M.r
#
#    plt.figure('etap')
#    plt.plot(r,etap[0,:],'k')
#    plt.plot(r,etapn[0,:],'b')
#
#    plt.figure('etad')
#    plt.plot(r,etad[0,:],'k')
#    plt.plot(r,etadn[0,:],'b')
#
#    plt.figure('etadp')
#    plt.plot(r,etadp[0,:],'k')
#    plt.plot(r,etadpn[0,:],'b')
#
#    plt.figure('Om')
#    plt.plot(r,M.Om,'b')
#    plt.plot(r,M.dOm,'k')
#
#    plt.figure('HT0')
#    plt.plot(r,M.HT0,'b')
#    plt.plot(r,M.dHT0,'k')
#
#    plt.figure('tr')
#    plt.plot(r,tr,'b')
#    plt.plot(r,dtr,'k')

#    A = M.A
#    Adp = M.Adp
#    Adpn = M.Adpn
#    Ap = M.Ap
#    Apn = M.Apn
#    Ad = M.Ad
#    Adn = M.Adn
#
#    plt.figure('Ap')
#    plt.plot(r,Apn[0,:],'b')
#    plt.plot(r,Ap[0,:],'k')
#
#    plt.figure('Ad')
#    plt.plot(r,Adn[0,:],'b')
#    plt.plot(r,Ad[0,:],'k')
#
#    plt.figure('Adp')
#    plt.plot(r,Adpn[0,:],'b')
#    plt.plot(r,Adp[0,:],'k')

#    Hperp = M.Hperp
#    HT0 = M.HT0
#
#    plt.figure('Hperp')
#    plt.plot(r,Hperp[0,:],'b')
#    plt.plot(r,HT0,'k')
#
#    Hpar1 = M.Hpar
#    Hpar2 = Adp/Ap
#
#    plt.figure('Hpar')
#    plt.plot(r,Hpar1[0,:],'b')
#    plt.plot(r,Hpar2[0,:],'k')
#
#    rho = M.rhot
#    plt.figure('rho')
#    plt.plot(r,rho[0,:])
#
#    #Test z relations
#    D = M.Dz
#    H = M.Hz
#    rhoz = M.rhoz
#    tz = M.tz
#    rz = M.rz
#    z = M.z
#
#    plt.figure('Dz')
#    plt.errorbar(zD,Dz,sDz,fmt='xr',alpha=0.2)
#    plt.plot(z,D,'b')
#    plt.plot(zF,DzF,'k')
#
#    plt.figure('Hz')
#    plt.errorbar(zH,Hz,sHz,fmt='xr',alpha=0.5)
#    plt.plot(z,H,'b')
#    plt.plot(zF,HzF,'k')
#
#    plt.figure('rhoz')
#    plt.plot(z,rhoz,'b')
#    plt.plot(zF,rhozF,'k')
#
#    plt.figure('tnrz')
#    plt.plot(z,tz)
#    plt.plot(z,rz)
