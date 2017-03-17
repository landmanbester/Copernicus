#!/usr/bin/env python
"""
Created on Thu Aug 21 09:40:27 2014

@author: landman

Class to create FLRW universe object

"""
import numpy as np
from numpy import linspace, sqrt, sin, sinh, pi, ones, zeros
from scipy.interpolate import UnivariateSpline as uvs
from scipy.integrate import quad
from Copernicus import Master
import matplotlib.pyplot as plt
from Copernicus.Parset import MyOptParse

class FLRW(object):
    def __init__(self,Om0,OL0,H0,zmax,NJ):
        self.Om0 = Om0
        self.Ok0 = 1- Om0 - OL0
        self.OL0 = OL0
        self.Lambda = 3*OL0*H0**2
        self.H0 = H0
        self.NJ = NJ
        self.z = linspace(0,zmax,self.NJ)
        self.delz = self.z[1]-self.z[0]
        self.Hz = self.getHz()
        self.Om = self.getOmz()
        self.OL = self.Lambda/(3*self.Hz**2)
        self.Ok = 1 - self.Om - self.OL        
        self.Dz = self.getDz()
        self.nuz = self.getnuz()
        self.nu = linspace(self.nuz[0],self.nuz[-1],self.NJ)
        self.rhoz = self.getrho()

        # Set function to get t0
        self.t0f = lambda x, a, b, c, d: np.sqrt(x) / (d * np.sqrt(a + b * x + c * x ** 3))

    def getHz(self, z=None, Om0=None, OL0 = None, H0=None):
        if Om0 is None and OL0 is None and H0 is None and z is None:
            return self.H0*sqrt(self.Om0*(1+self.z)**3 + self.Ok0*(1+self.z)**2 + self.OL0)
        else:
            Ok0 = 1.0 - Om0 - OL0
            return H0 * sqrt(Om0 * (1 + z) ** 3 + Ok0 * (1 + z) ** 2 + OL0)
            
    def getHo(self):
        Ho = uvs(self.z,self.Hz,k=5,s=0.0)
        return Ho
        
    def getDz(self, z=None, Om0=None, OL0 = None, H0=None):
        if Om0 is None and OL0 is None and H0 is None and z is None:
            Om0 = self.Om0
            OL0 = self.OL0
            Ok0 = self.Ok0
            H0 = self.H0
            Hz = self.Hz
            z = self.z
        else:
            Ok0 = 1.0 - OL0 - Om0
            Hz = self.getHz(z=z, Om0=Om0, OL0=OL0, H0=H0)
        K = -Ok0*H0**2
        dDc = uvs(z, 1.0/Hz, k=3, s=0.0)
        Dc = dDc.antiderivative()(z) #comoving distance
        if (K>0.0):
            Dz = sin(sqrt(K)*Dc)/sqrt(K)/(1+z)
        elif (K<0.0):
            Dz = sinh(sqrt(-K)*Dc)/sqrt(-K)/(1+z)
        else:
            Dz = Dc/(1+z)
        return Dz

        
    def getDo(self):
        Do = uvs(self.z,self.Dz,k=5,s=0.0)
        return Do
    
    def getOmz(self, z=None, Om0=None, H0=None, OL0=None):
        if z is None and Om0 is None and H0 is None and OL0 is None:
            return self.Om0*(1+self.z)**3*(self.H0/self.Hz)**2
        else:
            Hz = self.getHz(z, Om0=Om0, OL0=OL0, H0=H0)
            return Om0 * (1 + z) ** 3 * (H0 / Hz) ** 2
        
    def getrho(self, z=None, Om0=None, H0=None, OL0=None):
        if z is None and Om0 is None and H0 is None and OL0 is None:
            return 3.0*self.Om*self.Hz**2.0/(8.0*pi)
        else:
            Hz = self.getHz(z, Om0=Om0, OL0=OL0, H0=H0)
            Om = self.getOmz(z, Om0=Om0, OL0=OL0, H0=H0)
            return 3.0*Om*Hz**2.0/(8.0*pi)
        
    def getrhoo(self):
        rho = 3.0*self.Om*self.Hz**2.0/(8.0*pi)
        rhoo = uvs(self.z,rho,k=5,s=0.0)
        return rhoo
        
    def getnuz(self):
        dvdzo = uvs(self.z,1/((1+self.z)**2*self.Hz),k=3,s=0.0)
        v = dvdzo.antiderivative()(self.z)
        return v

    def getdzdw(self):
        return self.H0*(1+self.z) - self.Hz

    def get_sigmasq(self, DelRSq, UV_cut, ac, Hz, Dz, z):
        """
        Compute the magnitude of sigma^2*D^2/H^2 in perturbed FLRW
        DelRSq  = amplitude of primordial fluctuations
        UV_cut  = the UV cut-off
        ac      = scale factor at central worldline
        Hz      = the Hubble rate down PLC
        Dz      = the angular diameter distance down PLC
        z       = the redshift down PLC
        """
        # rescale (1 + z) > (1 + z')/ac
        uz_rescaled = (1 + z)/ac

        # Get normalisation of growth suppression function
        ginf = 2*(self.Om0**(4.0/7) - self.OL0 + (1+self.Om0/2.0)*(1+self.OL0/70.0))/(5*self.Om0)

        #Hubble scale
        h = Hz[0]*299.8/100 #the 299.8 is a unit conversion
        kH0 = h/3.0e3

        # Normalised densities (remember to check against Julien's result
        Omz = self.Om0*uz_rescaled**3/(self.OL0 + self.Om0*uz_rescaled**3)
        OLz = self.OL0/(self.OL0 + self.Om0*uz_rescaled**3)

        # Growth suppression function and derived evolution function for the shear
        #g = 5*ginf*self.Om/(2*(self.Om**(4.0/7) - self.OL + (1+self.Om/2)*(1+self.OL/70)))
        g = 5 * ginf * Omz / (2 * (Omz ** (4.0 / 7) - OLz + (1 + Omz / 2) * (1 + OLz / 70)))
        go = uvs(z, g, k=3, s=0.0)
        dgo = go.derivative()
        dg = dgo(z)

        G = (dg**2)**uz_rescaled**2 - 2*uz_rescaled*dg + g**2

        # Get the UV cut-off
        kUV = UV_cut/kH0

        # Transfer function
        k = linspace(0,kUV,1000)
        q = k*kH0/(self.Om0*h**2)
        T = np.zeros_like(k)
        T[1::] = np.log(1 + 2.34*q[1::])/(2.34*q[1::]*(1+3.89*q[1::] + (16.1*q[1::])**2 + (5.46*q[1::])**3 + (6.71*q[1::])**4))**0.25
        T[0] = 0.0

        dSo = uvs(k, k**3*T**2, k=3, s=0.0)
        So = dSo.antiderivative() #Maybe do this with quad!!!!
        S = So(kUV)

        # Expectation value of sigmasq*D**2/H**2
        sigmasqDzsqoHzsq = 4*DelRSq*G*S*Dz**2/(uz_rescaled**2*75*self.Om0*ginf**2)

        return sigmasqDzsqoHzsq

    def get_aot(self, Om0, OL0, H0):
        """
        Since the Universe is FLRW along the central worldline (viz. C) we have analytic expressions for the input
        functions along C. These can be used as boundary data in the CIVP (not currently implemented)
        """
        Ok0 = 1.0 - Om0 - OL0
        #First get current age of universe
        t0 = quad(self.t0f, 0, 1.0, args=(Om0,Ok0,OL0,H0))[0]
        #Get t(a) when a_0 = 1 (for some reason spline does not return correct result if we use a = np.linspace(1.0,0.2,1000)?)
        a = np.linspace(0, 1.0, 5000)
        tmp = self.t0f(a, Om0, Ok0, OL0, H0)
        dto = uvs(a, tmp, k=3, s=0)
        to = dto.antiderivative()
        t = to(a)
        #Invert to get a(t)
        aoto = uvs(t, a, k=3, s=0.0)
        return aoto, t0

        # aow = aoto(self.w0)
        # #Now get rho(t)
        # rho0 = Om0*3*H0**2/(kappa)
        # rhoow = rho0*aow**(-3.0)
        # #Get How (this is up_0)
        # How = H0*np.sqrt(Om0*aow**(-3) + Ok0*aow**(-2) + OL0)
        # upow = How
        # #Now get dHz and hence uppow
        # #First dimensionless densities
        # Omow = kappa*rhoow/(3*How**2)
        # OLow = self.Lam/(3*How**2)
        # OKow = 1 - Omow - OLow
        # dHzow = How*(3*Omow + 2*OKow)/2 #(because the terms in the np.sqrt adds up to 1)
        # uppow = (dHzow + 2*upow)*upow
        # return rhoow, upow, uppow

    def give_shear_for_plotting(self, Om0, OL0, H0, DelRSq, UV_cut, zmax, Np, tstar, Nret, data_prior, data_lik, fname,
                                DoPLCF, err):
        """
        This gives the FLRW shear on the PLC0 and the PLCF
        :param Om0:
        :param OL0:
        :param H0:
        :param DelRSq:
        :param UV_cut:
        :return:
        """
        zi = np.linspace(0, zmax, Np)
        Hzi = self.getHz(zi, Om0=Om0, OL0=OL0, H0=H0)
        Dzi = self.getDz(zi, Om0=Om0, OL0=OL0, H0=H0)
        sigmasqi = self.get_sigmasq(DelRSq, UV_cut, 1.0, Hzi, Dzi, zi)
        sigmasqio = uvs(zi/zi[-1], sigmasqi, k=3, s=0.0)

        LamF = 3 * OL0 * H0 ** 2
        rhozi = self.getrho(zi, Om0=Om0, OL0=OL0, H0=H0)
        Xrho = np.array([0.5, 2.8])
        XH = np.array([0.6, 3.5])
        sigmaLam = 0.6 * 3 * 0.7 * (70.0 / 299.79) ** 2
        UF = Master.SSU(zmax, tstar, Np, err, XH, Xrho, sigmaLam, Nret, data_prior, data_lik, fname, DoPLCF, Hz=Hzi, rhoz=rhozi,
                        Lam=LamF, useInputFuncs=True)

        T1iF, T2iF, LLTBConsiF, DiF, SiF, QiF, AiF, ZiF, SpiF, QpiF, ZpiF, uiF, upiF, uppiF, udotiF, rhoiF, rhopiF, rhodotiF, \
        DzF, dzdwzF, sigmasqiF, t0F = UF.get_funcsi()

        if t0F > UF.tmin and UF.NI > 1 and DoPLCF:
            T1fF, T2fF, LLTBConsfF, DfF, SfF, QfF, AfF, ZfF, SpfF, QpfF, ZpfF, ufF, upfF, uppfF, udotfF, rhofF, rhopfF, \
            rhodotfF, sigmasqfF = UF.get_funcsf()

        # Get background z, Dz, Hz for FLRW at t = tstar
        zonuf = ufF - 1.0
        zf = np.linspace(0, zonuf[-1], Nret)
        DzFf = uvs(zonuf, DfF, k=3, s=0.0)(zf)
        DzFf[0] = 0.0
        HzFf = uvs(zonuf, upfF / ufF ** 2, k=3, s=0.0)(zf)

        # Get aot
        aot, t0 = LCDM.get_aot(Om0, OL0, H0)

        ac = aot(tstar)

        sigmasqf = LCDM.get_sigmasq(2.41e-9, 0.005, ac, HzFf, DzFf, zf)
        sigmasqfo = uvs(zf / zf[-1], sigmasqf, k=3, s=0.0)

        l = np.linspace(0, 1, Nret)

        return Dzi, Hzi, rhozi, dzdwzF, sigmasqio(l), sigmasqfo(l)

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

    OL0 = 0.7
    Om0 = 0.3
    H0 = 70 / 299.8 #The 299.8 is a conversion factor to keep units consistent
    DelRSq = 2.41e-9
    UV_cut = 0.005



    LamF = 3 * OL0 * H0 ** 2

    LCDM = FLRW(Om0, OL0, H0, zmax, Np)
    # zi = LCDM.z
    #
    # Hzi = LCDM.Hz
    # Dzi = LCDM.Dz
    # rhozi = LCDM.rhoz

    Dzi, Hzi, rhozi, dzdwzF, sigmasqi, sigmasqf = LCDM.give_shear_for_plotting(Om0, OL0, H0, DelRSq, UV_cut, zmax, Np, tstar, Nret, data_prior,
                                                      data_lik, fname, DoPLCF, err)

    # #print aot(t0), t0, aot(3.25)
    #
    # sigmasqi = LCDM.get_sigmasq(2.41e-9, 0.005, 1.0, Hzi, Dzi, zi)

    # plot shear on PLC0
    l = linspace(0, 1, Nret)
    ymax = 1.1*sigmasqi.max()
    plt.figure('PLC0')
    plt.plot(l, sigmasqi, label='PLC0')
    plt.ylim(1e-12, ymax)
    #plt.yscale('log')
    #plt.savefig(fname+'/Figures/sigmasqiF.png', dpi=250)

    # # Do LCDM integration
    # Xrho = np.array([0.5,2.8])
    # XH = np.array([0.6,3.5])
    # sigmaLam = 0.6 * 3 * 0.7 * (70.0 / 299.79) ** 2
    # UF = Master.SSU(zmax, tstar, Np, err, XH, Xrho, sigmaLam, Nret, data_prior, data_lik, fname, DoPLCF, Hz=Hzi, rhoz=rhozi,
    #          Lam=LamF, useInputFuncs=True)
    #
    # T1iF, T2iF, LLTBConsiF, DiF, SiF, QiF, AiF, ZiF, SpiF, QpiF, ZpiF, uiF, upiF, uppiF, udotiF, rhoiF, rhopiF, rhodotiF, \
    # DzF, dzdwzF, sigmasqiF, t0F = UF.get_funcsi()
    #
    # if t0F > UF.tmin and UF.NI > 1 and DoPLCF:
    #     T1fF, T2fF, LLTBConsfF, DfF, SfF, QfF, AfF, ZfF, SpfF, QpfF, ZpfF, ufF, upfF, uppfF, udotfF, rhofF, rhopfF, \
    #     rhodotfF, sigmasqfF = UF.get_funcsf()
    #
    # # Get background z, Dz, Hz for FLRW at t = tstar
    # zonuf = ufF - 1.0
    # zf = np.linspace(0, zonuf[-1], Nret)
    # DzFf = uvs(zonuf, DfF, k=3, s=0.0)(zf)
    # DzFf[0] = 0.0
    # HzFf = uvs(zonuf, upfF / ufF ** 2, k=3, s=0.0)(zf)
    #
    # # Get aot
    # aot, t0 = LCDM.get_aot(Om0, OL0, H0)
    #
    # ac = aot(tstar)
    #
    # sigmasqf = LCDM.get_sigmasq(2.41e-9, 0.005, ac, HzFf, DzFf, zf)

    #plt.figure('PLCF')
    plt.plot(l, sigmasqf, label='PLCF')
    plt.ylabel(r'$\log\left(\frac{\sigma^2 D^2}{H^2}\right)$')
    plt.xlabel(r'$z$')
    plt.legend()
    #plt.ylim(1.0e-11, 1.0e-6)
    plt.yscale('log')
    plt.savefig(fname + '/Figures/logsigmasqfF.png', dpi=250)