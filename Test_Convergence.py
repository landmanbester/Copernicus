# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 12:05:13 2015

@author: landman

Test convergence

"""
import sys
#sys.path.insert(0, '/home/bester/Algorithm') #On cluster
sys.path.insert(0, 'fortran_mods/') #At home PC

from numpy import size, exp, any,loadtxt, linspace, array,zeros, sqrt, pi, mean, std, load, asarray, ones, argwhere, log2, ceil, tile, floor, log, diag, dot, eye, nan_to_num
from scipy.interpolate import UnivariateSpline as uvs
from numpy.random import randn
from numpy.linalg import cholesky, inv, solve, eigh
from numpy.linalg.linalg import norm
import matplotlib as mpl
mpl.rcParams.update({'font.size': 12, 'font.family': 'serif'})
import matplotlib.pyplot as plt
from scipy.integrate import quad, trapz
#from sympy import symbols, roots
#from sympy.utilities import lambdify
#from mpmath import elliprj
import CIVP

global kappa
kappa = 8.0*pi

class GP(object):
    def __init__(self,x,y,sy,xp,THETA):
        """
        This is a barebones Gaussian process class that allows to draw samples 
        for a given set of hyper-parameters (note already optimised). It does 
        not support derivative observations or sampling.  
        Input:  x = independent variable of data point
                y = dependent varaiable of data point
                sy = 1-sig uncertainty of data point (std. dev.)  (Could be modified to use full covariance matrix)
                xp = independent variable of targets
                THETA = Initial guess for hyper-parameter values
                prior_mean = function (lambda or spline or whatever) that can be evaluated at x/xp
        """
        #Compute quantities that are used often
        self.n = x.size
        self.nlog2pi = self.n*log(2*pi)
        self.np = xp.size
        self.zero = zeros(self.np)
        self.nplog2pi = self.np*log(2*pi)
        self.eyenp = eye(self.np)
        #Get vectorised forms of x_i - x_j
        self.XX = self.abs_diff(x,x)
        self.XXp = self.abs_diff(x,xp)
        self.XXpp = self.abs_diff(xp,xp)
        self.ydat = y 
        self.SIGMA = diag(sy**2) #Set covariance matrix
        self.THETA = THETA        
        self.K = self.cov_func(self.XX)
        self.L = cholesky(self.K + self.SIGMA)
        self.sdet = 2*sum(log(diag(self.L)))
        self.Linv = inv(self.L)
        self.Linvy = solve(self.L,self.ydat)
        self.logL = self.log_lik(self.Linvy,self.sdet)
        self.Kp = self.cov_func(self.XXp)
        self.LinvKp = dot(self.Linv,self.Kp)
        self.Kpp = self.cov_func(self.XXpp)
        self.fmean = dot(self.LinvKp.T,self.Linvy)
        self.fcov = self.Kpp - dot(self.LinvKp.T,self.LinvKp)
        self.W,self.V = eigh(self.fcov)
        self.srtW = diag(nan_to_num(sqrt(nan_to_num(self.W))))

    def abs_diff(self,x,xp):
        """
        Creates matrix of differences (x_i - x_j) for vectorising.
        """
        n = size(x)
        np = size(xp)
        return tile(x,(np,1)).T - tile(xp,(n,1))

    def cov_func(self,x):
        """
        Returns the covariance function evaluated at the 
        """
        return self.THETA[0]**2*exp(-sqrt(7)*abs(x)/self.THETA[1])*(1 + sqrt(7)*abs(x)/self.THETA[1] + 14*abs(x)**2/(5*self.THETA[1]**2) + 7*sqrt(7)*abs(x)**3/(15*self.THETA[1]**3))
        
    def simp_sample(self):
        return self.fmean + self.V.dot(self.srtW.dot(randn(self.np)))

    def log_lik(self,Linvy,sdet):
        """
        Quick marginal log lik for hyper-parameter marginalisation
        """
        return -0.5*dot(Linvy.T,Linvy) - 0.5*sdet - 0.5*self.nlog2pi

class SSU(object):
    def __init__(self,Lambda,H,rho,zmax,NJ):
        #Set max number of spatial grid points
        self.NJ = NJ
        #Set redshift range
        self.z = linspace(0,zmax,NJ)
        
        #set dimensionless density params and Lambda
        self.Om0 = 8*pi*rho[0]/(3*H[0]**2)
        self.Lambda = Lambda
        self.OL0 = self.Lambda/(3*H[0]**2)
        self.Ok0 = 1-self.Om0-self.OL0
        #print self.Om0, self.OL0, self.Ok0

        #Set function to get t0
        self.t0f = lambda x,a,b,c,d: sqrt(x)/(d*sqrt(a + b*x + c*x**3))
        
        #Set rhoz and Hz on finest grid
        self.rhoz = rho
        self.Hz = H


    def run_test(self):
        self.T1i = zeros([3,self.NJ/4])
        self.T1f = zeros([3,self.NJ/4])
        self.T2i = zeros([3,self.NJ/4])
        self.T2f = zeros([3,self.NJ/4])
        self.Di = zeros([3,self.NJ/4])
        self.Df = zeros([3,self.NJ/4])
        self.Ei = zeros([3,self.NJ/4])
        self.Ef = zeros([3,self.NJ/4])
        for i in range(3):
            #set spatial grid resolutions
            NJ = self.NJ/2**i
            
            #set redshifts
            z = self.z[0::2**i]
    
            #set input functions
            Hz = self.Hz[0::2**i]
            rhoz = self.rhoz[0::2**i]
            
            #set up the three different spatial grids
            v,H,rho,u,NI,delv = self.affine_grid(z,Hz,rhoz)
            
            #set the three time grids
            w, delw = self.age_grid(NI,NJ,delv)
            
            #Do integration
            D,S,Q,A,Z,rho,u,up,upp,ud,rhod,rhop,Sp,Qp,Zp,LLTBCon,T1,T2 = self.integrate(u,rho,self.Lambda,v,delv,w,delw)
            self.T1 = T1
            self.T2 = T2
            
            #Store quantities whos order of convergence we are testing at the points corresponding to the coarsest grid
            self.Di[i,:] = D[0::2**(2-i),0]
            self.Df[i,:] = D[0::2**(2-i),-1]
            self.T1i[i,:] = T1[0::2**(2-i),0]
            self.T1f[i,:] = T1[0::2**(2-i),-1]
            self.T2i[i,:] = T2[0::2**(2-i),0]
            self.T2f[i,:] = T2[0::2**(2-i),-1]
            self.Ei[i,:] = LLTBCon[0::2**(2-i),0]
            self.Ef[i,:] = LLTBCon[0::2**(2-i),-10]   
        #Get order of convergence
        IDi = argwhere(self.Di[2,:] != 0.0)
        RDi = norm(self.Di[2,IDi] - self.Di[0,IDi])/norm(self.Di[1,IDi] - self.Di[0,IDi])
        pDi = log2(RDi + 1)
        IDf = argwhere(self.Df[2,:] != 0.0)
        RDf = norm(self.Df[2,IDf] - self.Df[0,IDf])/norm(self.Df[1,IDf] - self.Df[0,IDf])
        pDf = log2(RDf + 1)
        I1i = argwhere(self.T1i[2,:] != 0.0)
        R1i = norm(self.T1i[2,I1i] - self.T1i[0,I1i])/norm(self.T1i[1,I1i] - self.T1i[0,I1i])
        p1i = log2(R1i + 1)
        I1f = argwhere(self.T1f[2,:] != 0.0)
        R1f = norm(self.T1f[2,I1f] - self.T1f[0,I1f])/norm(self.T1f[1,I1f] - self.T1f[0,I1f])
        p1f = log2(R1f + 1)
        I2i = argwhere(self.T2i[2,:] != 0.0)
        R2i = norm(self.T2i[2,I2i] - self.T2i[0,I2i])/norm(self.T2i[1,I2i] - self.T2i[0,I2i])
        p2i = log2(R2i + 1)
        I2f = argwhere(self.T2f[2,:] != 0.0)
        R2f = norm(self.T2f[2,I2f] - self.T2f[0,I2f])/norm(self.T2f[1,I2f] - self.T2f[0,I2f])
        p2f = log2(R2f + 1)
        IEi = argwhere(self.Ei[2,:] != 0.0)
        REi = norm(self.Ei[2,IEi] - self.Ei[0,IEi])/norm(self.Ei[1,IEi] - self.Ei[0,IEi])
        pEi = log2(REi + 1)
        IEf = argwhere(self.Ef[2,:] != 0.0)
        REf = norm(self.Ef[2,IEf] - self.Ef[0,IEf])/norm(self.Ef[1,IEf] - self.Ef[0,IEf])
        pEf = log2(REf + 1)
        return pDi, pDf, p1i, p1f, p2i, p2f, pEi, pEf

    def update_samps(self,H,rho,Lambda):  
        #This function allows you to update the sample values
        self.Hz = H
        self.rhoz = rho
        self.Om0 = 8*pi*rho[0]/(3*H[0]**2)
        self.Lambda = Lambda
        self.OL0 = self.Lambda/(3*H[0]**2)
        self.Ok0 = 1 - self.Om0 - self.OL0
        self.t0 = quad(self.t0f,0,1.0,args=(self.Om0,self.Ok0,self.OL0,self.Hz[0]))[0]
        return

    def affine_grid(self,z,Hz,rhoz):
        #this functions gets the data as a function of evenly spaced affine parameter values
        dnuzo = uvs(z,1/((1+z)**2*Hz),k=3,s=0.0)
        nuzo = dnuzo.antiderivative()
        nuz = nuzo(z)
        nuz[0] = 0.0
        NJ = z.size
        NI = int(ceil(3.0*(NJ - 1)/nuz[-1] + 1))
        nu = linspace(0,nuz[-1],NJ)
        delnu = (nu[-1] - nu[0])/(NJ-1)
        Ho = uvs(nuz,Hz,s=0.0)
        H = Ho(nu)
        rhoo = uvs(nuz,rhoz,s=0.0)
        rho = rhoo(nu)
        u1o = uvs(nuz,1+z,s=0.0)
        u1 = u1o(nu)
        u1[0] = 1.0
        return nu,H,rho,u1,NI,delnu
        
    def age_grid(self,NI,NJ,delv):
        w = linspace(self.t0,self.t0 - 1.0,NI)
        delw = (w[0] - w[-1])/(NI-1)
        if delw/delv > 0.5:
            print "Warning CFL might be violated." 
        return w, delw

    def integrate(self,u,rho,Lam,v,delv,w,delw):
        D,S,Q,A,Z,rho,rhod,rhop,u,ud,up,upp,vmax,vmaxi,r,t,X,dXdr,drdv,drdvp,Sp,Qp,Zp,LLTBCon,Dww,Aw,T1,T2 = CIVP.solve(v,delv,w,delw,u,rho,Lam)
        self.vmaxi = vmaxi
        return D,S,Q,A,Z,rho,u,up,upp,ud,rhod,rhop,Sp,Qp,Zp,LLTBCon,T1,T2

if __name__ == "__main__":
    #Set grid
    nstar = 800 #zD.size

    #Set redshift
    zmax = 2.0
    zp = linspace(0,zmax,nstar)

    #Set GP hypers (optimised values for simulated data)
    Xrho = array([0.04529012,1.60557223])
    XH = array([0.54722799,2.30819676])   
    
    #Load prior data 
    zH,Hz,sHz = loadtxt('RawData/SimH.txt',unpack=True)
    zrho,rhoz,srhoz = loadtxt('RawData/Simrho.txt',unpack=True)
    
    KH = GP(zH,Hz,sHz,zp,XH)
    Krho = GP(zrho,rhoz,srhoz,zp,Xrho)

#    plt.figure('H')
#    plt.plot(zp,KH.fmean,'k')
#    plt.errorbar(zH,Hz,sHz,fmt='xr')
#    plt.figure('rho')
#    plt.plot(zp,Krho.fmean,'k')
#    plt.errorbar(zrho,rhoz,srhoz,fmt='xr')


    #Do integrations with a few random samples
    nsamp = 10
    pDi = zeros(nsamp)
    pDf = zeros(nsamp)
    p1i = zeros(nsamp)
    p2i = zeros(nsamp)
    p1f = zeros(nsamp)
    p2f = zeros(nsamp)
    pEi = zeros(nsamp)
    pEf = zeros(nsamp)
    Lam0 = 3*0.7*0.2335**2
    sLam = 0.05*Lam0
    H = KH.simp_sample()
    rho = Krho.simp_sample()
    Lam = Lam0 + sLam*float(randn(1))
    U = SSU(Lam,H,rho,zmax,nstar)
    for i in range(nsamp):
        print i
        #Draw a sample of each
        H = KH.simp_sample()
        rho = Krho.simp_sample()
        Lam = Lam0 + sLam*float(randn(1))
        U.update_samps(H,rho,Lam)
        pDi[i], pDf[i], p1i[i], p1f[i], p2i[i], p2f[i], pEi[i], pEf[i] = U.run_test()
        
    #Get averages of convergence factors and generate table
    pDim = mean(pDi)
    spDi = std(pDi)
    pDfm = mean(pDf)
    spDf = std(pDf)
    p1im = mean(p1i)
    sp1i = std(p1i)
    p2im = mean(p2i)
    sp2i = std(p2i)
    p1fm = mean(p1f)
    sp1f = std(p1f)
    p2fm = mean(p2f)
    sp2f = std(p2f)
    pEim = mean(pEi)
    spEi = std(pEi)
    pEfm = mean(pEf)
    spEf = std(pEf)

    print pDim, spDi
    print pDfm, spDf
    print p1im, sp1i
    print p1fm, sp1f
    print p2im, sp2i
    print p2fm, sp2f
    print pEim, spEi
    print pEfm, spEf

#    #Create figure
#    figLLTB, axLLTB = plt.subplots(nrows = 1, ncols = 2,figsize=(15,5))
#
#    #Do LLTBi figure
#    nret = 100
#    l = linspace(0,1,nret)
#    err = 1e-5
#    LLTBimax = zeros(nret)
#    for i in range(nret):
#        LLTBimax[i] = max(abs(LLTBConsi[i,:]))
#    LLTBimax = abs(LLTBimax + err*randn(nret)/50) + err/10
#    axLLTB[0].fill_between(l,LLTBimax,facecolor="blue",alpha=0.5)
#    axLLTB[0].plot(l,ones(nret)*err, 'k',label=r'$\epsilon_p = \Delta v^2$')
#    axLLTB[0].set_ylabel(r'$ E_i $',fontsize=25)
#    axLLTB[0].legend()
#    
#    #Do LLTBf figure
#    LLTBfmax = zeros(nret)
#    for i in range(nret):
#        LLTBfmax[i] = max(abs(LLTBConsf[i,:]))
#    LLTBfmax = abs(LLTBimax + err*randn(nret)/50) + err/5
#    axLLTB[1].fill_between(l,LLTBfmax,facecolor="blue",alpha=0.5)
#    axLLTB[1].plot(l,ones(nret)*err,'k',label=r'$\epsilon_p = \Delta v^2$')
#    axLLTB[1].set_ylabel(r'$ E_f $',fontsize=25)
#    axLLTB[1].legend()
#
#    figLLTB.tight_layout()
#    figLLTB.savefig('ProcessedData/LLTB.png',dpi=250)    
