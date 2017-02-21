# -*- coding: utf-8 -*-
"""
Created on Thu Aug 21 09:40:27 2014

@author: landman

Class to create FLRW universe object

"""
import numpy as np
from numpy import linspace, sqrt, sin, sinh, pi, ones, zeros
from scipy.interpolate import UnivariateSpline as uvs
import matplotlib.pyplot as plt

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
        self.Hz = self.getHz(self.z)
        self.Om = self.getOmz()
        self.OL = self.Lambda/(3*self.Hz**2)
        self.Ok = 1 - self.Om - self.OL        
        self.getDz(self.z)
        self.nuz = self.getnuz()
        self.nu = linspace(self.nuz[0],self.nuz[-1],self.NJ)
        self.rhoz = self.getrho()
        

    def getHz(self,z):
        return self.H0*sqrt(self.Om0*(1+z)**3 + self.Ok0*(1+z)**2 + self.OL0)
            
    def getHo(self):
        Ho = uvs(self.z,self.Hz,k=5,s=0.0)
        return Ho
        
    def getDz(self,z):
        K = -self.Ok0*self.Hz[0]**2
        dDc = uvs(self.z,1.0/self.Hz,k=3,s=0.0)
        Dc = dDc.antiderivative()(self.z) #comoving distance
        if (K>0.0):
            self.Dz = sin(sqrt(K)*Dc)/sqrt(K)/(1+z)
        elif (K<0.0):
            self.Dz = sinh(sqrt(-K)*Dc)/sqrt(-K)/(1+z)
        else:
            self.Dz = Dc/(1+self.z)
        return
        
    def getDo(self):
        Do = uvs(self.z,self.Dz,k=5,s=0.0)
        return Do
    
    def getOmz(self):
        return self.Om0*(1+self.z)**3*(self.H0/self.Hz)**2
        
    def getrho(self):
        return 3.0*self.Om*self.Hz**2.0/(8.0*pi)
        
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

    def get_sigmasq(self,DelRSq,UV_cut):
        """
        Compute the magnitude of sigma^2*D^2/H^2 in perturbed FLRW
        DelRSq = amplitude of primordial fluctuations
        UV_cut = the UV cut-off
        """
        # Get normalisation of growth suppression function
        ginf = 2*(self.Om0**(4.0/7) - self.OL0 + (1+self.Om0/2.0)*(1+self.OL0/70.0))/(5*self.Om0)

        #Hubble scale
        h = self.Hz[0]*299.8/100 #the 299.8 is a unit conversion
        kH0 = h/3.0e3

        # Normalised densities (remember to check against Julien's result
        Omz = self.Om0*(1+self.z)**3/(self.OL0 + self.Om0*(1+self.z)**3)
        OLz = self.OL0/(self.OL0 + self.Om0*(1+self.z)**3)

        # Growth suppression function and derived evolution function for the shear
        g = 5*ginf*self.Om/(2*(self.Om**(4.0/7) - self.OL + (1+self.Om/2)*(1+self.OL/70)))
        go = uvs(self.z, g, k=3, s=0.0)
        dgo = go.derivative()
        dg = dgo(self.z)

        G = (dg**2)**(1+self.z)**2 - 2*(1+self.z)*dg + g**2

        # Get the UV cut-off
        kUV = UV_cut/kH0

        # Transfer function
        k = linspace(0,kUV,1000)
        q = k*kH0/(self.Om0*h**2)
        T = np.log(1 + 2.34*q)/(2.34*q*(1+3.89*q + (16.1*q)**2 + (5.46*q)**3 + (6.71*q)**4))**0.25
        T[0] = 0.0

        dSo = uvs(k, k**3*T**2, k=3, s=0.0)
        So = dSo.antiderivative() #Maybe do this with quad!!!!
        S = So(kUV)

        # Expectation value of sigmasq*D**2/H**2
        sigmasq = 4*DelRSq*G*S*self.Dz**2/((1+self.z)**2*75*self.Om0*ginf**2)

        return sigmasq


if __name__=="__main__":
    OL = 0.7
    Om = 0.3
    H0 = 70 / 299.8 #The 299.8 is a conversion factor to keep units consistent
    zmax = 2.0
    NJ = 500
    
    LCDM = FLRW(Om,OL,H0,zmax,NJ)
    
    #Units of rho are in Gpc^{-2} 
    plt.plot(LCDM.z,LCDM.rhoz)
    
    
    
    