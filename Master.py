# -*- coding: utf-8 -*-
"""
Created on Wed Sep 17 09:49:39 2014

@author: landman


This is the universe class. Its attributes and methods are those of a
spherically symmetric dust universe. 

Here we set the number of grid points based on the desired error.
Universes that are not in causal contact with the constant time slice we are 
locating are not considered.

"""

import numpy as np
from numpy.linalg import cholesky, solve, inv, slogdet, eigh
from numpy.random import randn, random
from scipy.integrate import odeint,quad
from scipy.interpolate import UnivariateSpline as uvs
import matplotlib.pyplot as plt
#import matplotlib as mpl
#from genFLRW import FLRW
from Copernicus.fortran_mods import CIVP

global kappa
kappa = 8.0*np.pi

class GP(object):
    def __init__(self,x,y,sy,xp,THETA,beta):
        """
        This is a simple Gaussian process class that allows to easily marginalise
        over the hyper-parameters. It does not support derivative observations/sampling.
        A mean function can be trained iteratively. 
        Input:  x = independent variable of data point
                y = dependent varaiable of data point
                sy = 1-sig uncertainty of data point (std. dev.)  (Could be modified to use full covariance matrix)
                xp = independent variable of targets
                THETA = Initial guess for hyper-parameter values
                prior_mean = function (lambda or spline or whatever) that can be evaluated at x/xp
                mode = sets whether to train normally, iteratively or not train at all (if hypers already optimised)
        """
        #Compute quantities that are used often
        self.beta = beta
        self.N = x.size
        self.Nlog2pi = self.N*np.log(2*np.pi)
        self.Np = xp.size
        self.zero = np.zeros(self.Np)
        self.Nplog2pi = self.Np*np.log(2*np.pi)
        self.eyenp = np.eye(self.Np)
        #Get vectorised forms of x_i - x_j
        self.XX = self.abs_diff(x,x)
        self.XXp = self.abs_diff(x,xp)
        self.XXpp = self.abs_diff(xp,xp)
        self.ydat = y 
        self.SIGMA = np.diag(sy**2) #Set covariance matrix
        self.THETA = THETA        
        self.K = self.cov_func(self.THETA,self.XX)
        self.L = cholesky(self.K + self.SIGMA)
        self.sdet = 2*sum(np.log(np.diag(self.L)))
        self.Linv = inv(self.L)
        self.Linvy = solve(self.L,self.ydat)
        self.logL = self.log_lik(self.Linvy,self.sdet)
        self.Kp = self.cov_func(THETA,self.XXp)
        self.LinvKp = np.dot(self.Linv,self.Kp)
        self.Kpp = self.cov_func(THETA,self.XXpp)
        self.fmean = np.dot(self.LinvKp.T,self.Linvy)
        self.fcov = self.Kpp - np.dot(self.LinvKp.T,self.LinvKp)
        self.W,self.V = eigh(self.fcov)
        self.srtW = np.diag(np.nan_to_num(np.sqrt(self.W)))


    def meanf(self,theta,y,XXp):
        """
        This funcion returns the posterior mean. Only used for optimization.
        """
        Kp = self.cov_func(theta,XXp)
        Ky = self.cov_func(theta,self.XX) + self.SIGMA
        return np.dot(Kp.T,solve(Ky,y))
    
    def covf(self,theta):
        """
        This funcion returns the posterior covariance matrix. Only used for optimization.
        """
        Kp = self.cov_func(theta,self.XXp)
        Kpp = self.cov_func(theta,self.XXpp)
        Ky = self.cov_func(theta,self.XX) + self.SIGMA
        L = cholesky(Ky)
        Linv = inv(L)
        LinvKp = np.dot(Linv,Kp)
        return Kpp - np.dot(LinvKp.T,LinvKp)

    def abs_diff(self,x,xp):
        """
        Creates matrix of differences (x_i - x_j) for vectorising.
        """
        N = x.size #np.size(x)
        Np = xp.size #np.size(xp)
        return np.tile(x,(Np,1)).T - np.tile(xp,(N,1))

    def cov_func(self,THETA,x):
        return THETA[0]**2*np.exp(-np.sqrt(7)*abs(x)/THETA[1])*(1 + np.sqrt(7)*abs(x)/THETA[1] + 14*abs(x)**2/(5*THETA[1]**2) + 7*np.sqrt(7)*abs(x)**3/(15*THETA[1]**3))
        

    def sample(self,f):
        """
        Returns pCN proposal for MCMC. For normal sample use simp_sample
        """
        f0 = f - self.fmean
        return self.fmean + np.sqrt(1-self.beta**2)*f0 + self.beta*self.V.dot(self.srtW.dot(randn(self.Np)))
        
    def simp_sample(self):
        return self.fmean + self.V.dot(self.srtW.dot(randn(self.Np)))

    def sample_logprob(self,f):
        """
        Returns the probability of a sample from the posterior pdf.
        """
        F = f-self.fmean
        LinvF = solve(self.fcovL,F)
        return -0.5*np.dot(LinvF.T,LinvF) - 0.5*self.covdet - 0.5*self.Nplog2pi

    def logp(self,theta,y):
        """
        Returns marginal negative log lik. Only used for optimization.
        """
        Ky = self.cov_func(theta,self.XX) + self.SIGMA
        y = np.reshape(y,(self.N,1))
        print Ky.shape, y.T.shape, y.shape
        return np.dot(y.T,solve(Ky,y))/2.0 + slogdet(Ky)[1]/2.0 + self.Nlog2pi/2.0
        
    def log_lik(self,Linvy,sdet):
        """
        Quick marginal log lik for hyper-parameter marginalisation
        """
        return -0.5*np.dot(Linvy.T,Linvy) - 0.5*sdet - 0.5*self.Nlog2pi

class SSU(object):
    def __init__(self,zmax,tmin,Np,err,XH,Xrho,sigmaLam,Nret):
        """
        This is the main untility class (SSU = spherically symmetric universe)
        Input:  zmax = max redshift
                tmin = min time to integrate up to
                Np = The number of redshift points to use for GPR
                err = the target error of the numerical integration scheme
                XH = The optimised hyperparameter values for GP_H
                Xrho = The optimised hyperparameter values for GP_rho
                sigmaLam = The variance of the prior over Lambda
        """
        #First set attributes that will remain fixed
        #Load the data
        self.load_Dat()

        #Set number of spatial grid points
        self.Np = Np
        self.z = np.linspace(0,zmax,self.Np)
        self.uz = 1 + self.z

        #Set function to get t0
        self.t0f = lambda x,a,b,c,d: np.sqrt(x)/(d*np.sqrt(a + b*x + c*x**3))

        #Set minimum time to integrate to
        self.tmin = tmin
        self.tfind = tmin + 0.1 # This is defines the constant time slice we are looking for

        #Set number of spatial grid points at which to return quantities of interest
        self.Nret = Nret

        #Set target error
        self.err = err # The error of the integration scheme used to set NJ and NI
        
        #Create GP objects
        self.beta = 0.25
        self.GPH = GP(self.zHdat,self.Hzdat,self.sHzdat,self.z,XH,self.beta) #Should fit the GP's to the data here
        self.XH = self.GPH.THETA
        self.Hm = self.GPH.fmean
        self.GPrho = GP(self.zrhodat,self.rhozdat,self.srhozdat,self.z,Xrho,self.beta)
        self.Xrho = self.GPrho.THETA
        self.rhom = self.GPrho.fmean
        #Set Lambda mean and variance (Here we use the background model)
        self.Lam = 3*0.7*(70.0/299.79)**2        
        self.sigmaLam = sigmaLam

        # #Set initial conditions for hypersurface equations
        # self.y0 = np.zeros(5)
        # self.y0[1] = 1.0
        # self.y0[3] = 1.0

        #Now we do the initialisation starting with the background vals
        Lam = 3*0.7*(70.0/299.79)**2 #self.Lam
        Hz = self.Hm
        rhoz = self.rhom
        #Set up spatial grid
        v,vzo,Hi,rhoi,ui,NJ,NI,delv,Om0,OL0,Ok0,t0 = self.affine_grid(Hz,rhoz,Lam)
        self.NI = NI
        self.NJ = NJ
        self.v = v
        
        #Set up time grid
        w,delw = self.age_grid(NI,NJ,delv,t0)

        #Get soln on C
        rhoow, upow, uppow = self.get_C_sol(Om0,Ok0,OL0,Hz[0])
        
        self.uppow=uppow

        #Do MyCIVP integration
        #self.D2,self.S2,self.Q2,self.A2,self.Z2,self.rho2,self.rhod2,self.rhop2,self.u2,self.ud2,self.up2,self.upp2,self.vmax2,self.vmaxi2,self.r2,self.t2,self.X2,self.dXdr2,self.drdv2,self.drdvp2 = self.integrate_MyCIVP(ui,rhoi,Lam,rhoow,upow,uppow,v,delv,self.w0,delw)
        
        #Do the first CIVP0 integration
        Di, Si, Qi, Ai, Zi, udoti, rhodoti, upi, rhopi, uppi = self.evaluate(rhoi,ui,v,NJ,Lam)
#        self.Ki2 = self.curve_test2(ui,upi,uppi,Di,Si,rhoi)
#        self.sheari2 = self.shear_test2(ui,upi,Di,Si,Qi,Ai)
        
#        plt.figure('D')
#        plt.plot(v,Di,'b')
#        plt.figure('S')
#        plt.plot(v,Si,'b')
#        plt.figure('Q')
#        plt.plot(v,Qi,'b')
#        plt.figure('A')
#        plt.plot(v,Ai,'b')
#        plt.figure('Z')
#        plt.plot(v,Zi,'b')
        
        #Get the likelihood of the first sample
        self.logLik = self.get_Chi2(Hi,Di,rhoi,ui,vzo,t0,NJ)
        
        #Accept the first step regardless (this performs the main integration and transform)
        F = self.accept(NJ,NI,delw,delv,v,w,Di,ui,rhoi,Lam,t0,Om0,OL0,Ok0,vzo)
        
        if F==1:
            print "Flag raised, try different starting point"
        
    def MCMCstep(self,logLik0,Hz0,rhoz0,Lam0):
        #Propose sample
        Hz,rhoz,Lam, F = self.gen_sample(Hz0,rhoz0,Lam0)
        if F == 1:
            return Hz0,rhoz0,Lam0,logLik0,0,0
        else:
            #Set up spatial grid
            v,vzo,H,rho,u,NJ,NI,delv,Om0,OL0,Ok0,t0 = self.affine_grid(Hz,rhoz,Lam)
            #Set temporal grid
            w, delw = self.age_grid(NI,NJ,delv,t0)
            #Do integration on initial PLC
            D, S, Q, A, Z, udot, rhodot, up, rhop, upp = self.evaluate(rho,u,v,NJ,Lam)
            #Get likelihood
            logLik = self.get_Chi2(H,D,rho,u,vzo,t0,NJ)
            logr = logLik - logLik0
            accprob = np.exp(-logr/2.0)
            #Accept reject step
            tmp = random(1)
            if tmp > accprob:
                #Reject sample
                return Hz0,rhoz0,Lam0,logLik0,0,0
            else:
                #Accept sample
                F = self.accept(NJ,NI,delw,delv,v,w,D,u,rho,Lam,t0,Om0,OL0,Ok0,vzo)
                return Hz,rhoz,Lam,logLik,F,1  #If F returns one we can't use solution inside PLC

    def load_Dat(self):
#        self.zmudat,self.muzdat,self.smuzdat = np.loadtxt('/home/bester/Algorithm/RawData/Simmu.txt',unpack=True)  #For working on cluster
#        self.zHdat,self.Hzdat,self.sHzdat = np.loadtxt('/home/bester/Algorithm/RawData/SimH.txt',unpack=True)
#        self.zrhodat,self.rhozdat,self.srhozdat = np.loadtxt('/home/bester/Algorithm/RawData/Simrho.txt',unpack=True)
        self.zmudat,self.muzdat,self.smuzdat = np.loadtxt('/home/landman/Algorithm/RawData/Simmu.txt',unpack=True) #For home PC
        self.zHdat,self.Hzdat,self.sHzdat = np.loadtxt('/home/landman/Algorithm/RawData/SimH.txt',unpack=True)
        self.zrhodat,self.rhozdat,self.srhozdat = np.loadtxt('/home/landman/Algorithm/RawData/Simrho.txt',unpack=True)
        self.t0dat = 4.129
        self.st0dat = 0.02*self.t0dat
        return
        
    def gen_sample(self,Hzi,rhozi,Lami):
        Hz = self.GPH.sample(Hzi)
        rhoz = self.GPrho.sample(rhozi)
        Lam0 = self.Lam - Lami
        Lam = self.Lam + np.sqrt(1 - self.beta**2)*Lam0 + self.beta*self.sigmaLam*randn(1)
        #Flag if any of these less than zero
        if ((Hz < 0.0).any() or (rhoz <= 0.0).any() or (Lam<0.0)):
            print "Negative samples",(Hz < 0.0).any(),(rhoz <= 0.0).any(),(Lam<0.0)
            return Hzi, rhozi,Lami, 1
        else:
            return Hz, rhoz, Lam, 0
        
    def accept(self,NJ,NI,delw,delv,v,w,Di,ui,rhoi,Lam,t0,Om0,OL0,Ok0,vzo):
        self.NI = NI
        self.NJ = NJ
        self.vmax = np.zeros(NI)   #max radial extent on each PLC
        self.vmaxi = np.zeros(NI)  #index of max radial extent
        self.vmaxi[:] = int(NJ)
        self.vmax[0] = v[-1]
        #Do CIVP integration
        self.D,self.S,self.Q,self.A,self.Z,self.rho,self.u,self.up,self.upp,self.ud,self.rhod,self.rhop,self.Sp,self.Qp,self.Zp,self.drdv,self.drdvp,self.X,self.dXdr,self.LLTBCon2,self.Dww2,self.Aw2 = self.integrate_MyCIVP(ui,rhoi,Lam,v,delv,self.w0,delw)#self.integrate(NJ,NI,delw,delv,v,w,Di,ui,rhoi,Lam)
        #Get D(z),mu(z) and dzdw(z)
        self.Dz,self.muz,self.dzdw = self.get_PLC0_observables(vzo,self.D[:,0],self.A[:,0],self.u[:,0],self.ud[:,0],self.up[:,0])
        self.Hpar = self.up/self.u**2    
        self.Hperp = (self.u*self.Q - self.S/(2.0*self.u) + self.u*self.A*self.S/2.0)/self.D
        self.Hperp[0,:] = self.Hpar[0,:]
        self.t0 = t0
        self.H0 = self.Hpar[0,0]
        self.Om0 = Om0
        self.OL0 = OL0
        self.Ok0 = Ok0
        self.v = v
        self.Lam = Lam
        #Check LLTB consistency 
        self.check_LLTB(self.D,self.S,self.Q,self.A,self.Z,self.u,self.rho,delw,self.Lam,NI,NJ)
        return 0
#        if self.vmaxi[-1] > 5:
#            #Get coord transform
#            self.transform()
#            #Get tslice
#            F = self.get_tslice()
#            return F
#        else:
#            return 1

    def get_age(self,Om0,Ok0,OL0,H0):
        return quad(self.t0f,0,1,args=(Om0,Ok0,OL0,H0))[0]

    def get_C_sol(self,Om0,Ok0,OL0,H0):
        #First get current age of universe
        amin = 0.5
        t0 = quad(self.t0f,0,1.0,args=(Om0,Ok0,OL0,H0))[0]
        #Get t(a) when a_0 = 1 (for some reason spline does not return correct result if we use a = np.linspace(1.0,0.2,1000)?)
        a = np.linspace(amin,1.0,5000)
        tmp = self.t0f(a,Om0,Ok0,OL0,H0)        
        dto = uvs(a,tmp,k=3,s=0)
        to = dto.antiderivative()
        t = t0 + to(a)
        self.t = t
        #Invert to get a(t)
        aoto = uvs(t,a,k=3,s=0.0)
        aow = aoto(self.w0)
        #Now get rho(t)
        rho0 = Om0*3*H0**2/(kappa)
        rhoow = rho0*aow**(-3.0)
        #Get How (this is up_0)
        How = H0*np.sqrt(Om0*aow**(-3) + Ok0*aow**(-2) + OL0)
        upow = How
        #Now get dHz and hence uppow
        #First dimensionless densities
        Omow = kappa*rhoow/(3*How**2)
        OLow = self.Lam/(3*How**2)
        OKow = 1 - Omow - OLow
        dHzow = How*(3*Omow + 2*OKow)/2 #(because the terms in the np.sqrt adds up to 1)
        uppow = (dHzow + 2*upow)*upow
        return rhoow, upow, uppow
        
    def affine_grid(self,Hz,rhoz,Lam):
        """
        Get data on regular spatial grid
        """
        #First find dimensionless density params
        Om0 = kappa*rhoz[0]/(3*Hz[0]**2)
        OL0 = Lam/(3*Hz[0]**2)
        Ok0 = 1-Om0-OL0
        #Get t0
        t0 = self.get_age(Om0,Ok0,OL0,Hz[0])
        #Set affine parameter vals        
        dvo = uvs(self.z,1/(self.uz**2*Hz),k=3,s=0.0)
        vzo = dvo.antiderivative()
        vz = vzo(self.z)
        vz[0] = 0.0
        #Compute grid sizes that gives num error od err
        NJ = int(np.ceil(vz[-1]/np.sqrt(self.err) + 1))
        NI = int(np.ceil(3.0*(NJ - 1)*(t0 - self.tmin)/vz[-1] + 1))
        #Get functions on regular grid
        v = np.linspace(0,vz[-1],NJ)
        delv = (v[-1] - v[0])/(NJ-1)
        if delv > np.sqrt(self.err):
            print 'delv > sqrt(err)'
        Ho = uvs(vz,Hz,s=0.0,k=3)
        H = Ho(v)
        rhoo = uvs(vz,rhoz,s=0.0,k=3)
        rho = rhoo(v)
        uo = uvs(vz,self.uz,s=0.0,k=3)
        u = uo(v)
        u[0] = 1.0
        return v,vzo,H,rho,u,NJ,NI,delv,Om0,OL0,Ok0,t0
        
    def age_grid(self,NI,NJ,delv,t0):
        w0 = np.linspace(t0,self.tmin,NI)
        self.w0 = w0
        delw = (w0[0] - w0[-1])/(NI-1)
        if delw/delv > 0.5:
            print "Warning CFL might be violated." 
        #Set u grid
        w = np.tile(w0,(NJ,1)).T
        return w, delw

    # def hypeq(self,y,v,rhoo,uo,Lambda):
    #     dy = np.zeros(5)
    #     rho = rhoo(v)
    #     u = uo(v)
    #     dy[0] = y[1]
    #     dy[1] = -kappa*u**2*rho*y[0]/2.0
    #     if y[0] == 0.0:
    #         dy[2] = 0.0
    #     else:
    #         dy[2] = (1.0 - y[0]*y[1]*y[4] - 2.0*y[2]*y[1] - y[3]*y[1]**2 + 4.0*np.pi*rho*y[0]**2*(y[3]*u**2 - 1.0) - Lambda*y[0]**2)/(2*y[0])
    #     dy[3] = y[4]
    #     if y[0] == 0.0:
    #         dy[4] = kappa*rho/3.0 - 2.0*Lambda/3.0
    #     else:
    #         dy[4] = kappa*rho + 4.0*y[2]*y[1]/y[0]**2 + 2.0*y[3]*y[1]**2/y[0]**2 - 2.0/y[0]**2
    #     return dy
    #
    # def get_App(self,rho,Lam,D,S,Q,A):
    #     App = kappa*rho + 4.0*Q*S/D**2 + 2.0*A*S**2/D**2 - 2.0/D**2
    #     App[0,:] = kappa*rho[0,:]/3.0- 2.0*Lam/3.0
    #     return App
    #
    # def dotu(self,u,A,up,Ap):
    #     return ((1.0/u**2 - A)*up - Ap*u)/2.0
    #
    # def dotrho(self,rho,u,rhop,up,D,Dp,dotD,A,vmaxi):
    #     rhodot = np.zeros(vmaxi)
    #     rhodot[0] = -3.0*rho[0]*up[0]
    #     rhodot[1::] = rho[1::]*(-up[1::]/u[1::]**3 - 2.0*dotD[1::]/D[1::] + Dp[1::]*(1.0/u[1::]**2 - A[1::])/D[1::]) + rhop[1::]*(1.0/u[1::]**2.0 - A[1::])/2.0
    #     return rhodot
    #
    # def evaluate(self,rho,u,v,vmaxi,Lam):
    #     """
    #     This functions evaluates CIVP variables on a PLC from the values of rho and u on that PLC
    #     """
    #     #Interpolate u and rho
    #     uo = uvs(v,u,s=0.0,k=5)
    #     rhoo = uvs(v,rho,s=0.0,k=5)
    #     #Solve hyp equations
    #     y = odeint(self.hypeq,self.y0,v,args=(rhoo,uo,Lam),atol=1e-6,rtol=1e-6)
    #     #Get spatial derivatives of rho and u
    #     upo = uo.derivative()
    #     uppo = uo.derivative(2)
    #     up = upo(v)
    #     upp = uppo(v)
    #     rhopo = rhoo.derivative()
    #     rhop = rhopo(v)
    #     #Get udot and rhodot corresponding to these solutions
    #     ud = self.dotu(u,y[:,3],up,y[:,4])
    #     rhod = self.dotrho(rho,u,rhop,up,y[:,0],y[:,1],y[:,2],y[:,3],vmaxi)
    #     return y[:,0], y[:,1], y[:,2], y[:,3], y[:,4], ud, rhod, up, rhop, upp
    #
    # def integrate(self,NJ,NI,delw,delv,v,w,D,u,rho,Lam):
    #     r,v1,rhof,W,v0,v1u,v1nu,v12nu,S,T,RR,rhod,rhop,self.vmaxi = CIVP2.solve(v,w,0.0025,0.005,D,-u,rho,Lam)
    #     A = np.zeros([NI,NJ])
    #     A[:,1::] = 1.0 + W[:,1::]/r[:,1::]
    #     A[:,0] = 1.0
    #     Z = np.zeros([NI,NJ])
    #     Z[:,1::] = T[:,1::]/r[:,1::] - W[:,1::]*S[:,1::]/r[:,1::]**2
    #     Z[:,0] = 0.0
    #     self.v1 = v1
    #     self.v1nu=v1nu
    #     self.v1u = v1u
    #     self.delnu=delv
    #     self.delu = delw
    #     self.RR = RR
    #     return r,S,-RR,A,Z,rhof,-v1,-v1nu,-v12nu,v1u,-rhod,rhop

    def integrate_MyCIVP(self,u,rho,Lam,v,delv,w,delw):
        D,S,Q,A,Z,rho,rhod,rhop,u,ud,up,upp,vmax,vmaxi,r,t,X,dXdr,drdv,drdvp,Sp,Qp,Zp,LLTBCon,Dww,Aw = CIVP.solve(v,delv,w,delw,u,rho,Lam)
        self.vmaxi = vmaxi
        return D,S,Q,A,Z,rho,u,up,upp,ud,rhod,rhop,Sp,Qp,Zp,drdv,drdvp,X,dXdr,LLTBCon,Dww,Aw

    def integrate(self,u,rho,Lam,v,delv,w,delw):
        D,S,Q,A,Z,rho,rhod,rhop,u,ud,up,upp,vmax,vmaxi,r,t,X,dXdr,drdv,drdvp,Sp,Qp,Zp,LLTBCon,Dww,Aw,T1,T2 = CIVP.solve(v,delv,w,delw,u,rho,Lam)
        self.vmaxi = vmaxi
        return D,S,Q,A,Z,rho,u,up,upp,ud,rhod,rhop,Sp,Qp,Zp,LLTBCon,T1,T2

#     def dprimecoef(self,u,A):
#         return (A - 1/u**2)/2
#
#     def dcoef(self,du,u,dA):
#         return (du/u**3 + dA/2)
#
#     def F_d(self,n,j):
#         return self.d[n,j]*(self.A[n,j] - 1.0/self.v1[n,j]**2)/2.0
#
#     def transform(self):
#         #get dt components
#         self.dtdw = (self.A*self.v1**2 + 1)/(-2*self.v1)
#         self.dtdv = self.v1
#         self.dwdt = -self.v1
#         self.dvdt = (self.A*self.v1**2 - 1)/(-2*self.v1)
#         #set initial d data
#         self.d = np.zeros([self.NI,self.NJ])
#         self.c = np.zeros([self.NI,self.NJ])
#         self.b = np.zeros([self.NI,self.NJ])
#         self.a = np.zeros([self.NI,self.NJ])
#         self.d[0,:] = -self.v1nu[0,:]*self.D[0,:] -self.v1[0,:]*self.S[0,:]
#         self.c[0,:] = self.d[0,:]*(self.A[0,:]*self.v1[0,:]**2-1)/(2*self.v1[0,:]**2)
#         self.a[0,:] = -self.v1[0,:]**2/self.d[0,:]
#         self.b[0,:] = (1+self.A[0,:]*self.v1[0,:]**2)/(2*self.d[0,:])
#         #Get dprime and ddot to evaluate dXdr below
#         self.F = np.zeros([self.NI,self.NJ])
#         self.ddot = np.zeros([self.NI,self.NJ])
#         self.dprime = np.zeros([self.NI,self.NJ])
#         self.F[0,:] = self.d[0,:]*(self.A[0,:] - 1.0/self.v1[0,:]**2)/2.0
#         self.ddot[0,0] = (-self.F[0,2]/2.0 + 2.0*self.F[0,1] - 3.0*self.F[0,0]/2.0)/self.delnu
#         self.ddot[0,1:self.NJ-1] = (self.F[0,2:self.NJ] - self.F[0,0:self.NJ-2])/(2*self.delnu)
#         self.ddot[0,self.NJ-1] = (3.0*self.F[0,self.NJ-1]/2.0 - 2.0*self.F[0,self.NJ-2] + self.F[0,self.NJ-3]/2.0)/self.delnu
#         self.dprime[0,0] = (-self.d[0,2]/2.0 + 2.0*self.d[0,1] - 3.0*self.d[0,0]/2.0)/self.delnu
#         self.dprime[0,1:self.NJ-1] = (self.d[0,2:self.NJ] - self.d[0,0:self.NJ-2])/(2*self.delnu)
#         self.dprime[0,self.NJ-1] = (3.0*self.d[0,self.NJ-1]/2.0 - 2.0*self.d[0,self.NJ-2] + self.d[0,self.NJ-3]/2.0)/self.delnu
#         self.X = np.zeros([self.NI,self.NJ])
#         self.X[0,:] = -self.v1[0,:]/self.d[0,:]
#         self.tv = np.zeros([self.NI,self.NJ])
#         self.rv = np.zeros([self.NI,self.NJ])
#         dto = uvs(self.v,self.v1[0,:],k=3,s=0.0)
#         self.tv[0,:] = self.w0[0] + dto.antiderivative()(self.v)
#         dro = uvs(self.v,self.d[0,:],k=3,s=0.0)
#         self.rv[0,:] = dro.antiderivative()(self.v)
#         for n in range(self.NI-1):
#             jmax = int(self.vmaxi[n+1])
#             #Use forward differences to get value at origin
#             self.d[n+1,0] = self.d[n,0] + self.delu*(-self.F_d(n,2)/2.0 + 2.0*self.F_d(n,1) - 3.0*self.F_d(n,0)/2.0)/self.delnu
#             arrup = np.arange(2,jmax)
#             arrmid = np.arange(1,jmax-1)
#             arrdown = np.arange(0,jmax-2)
#             self.d[n+1,arrmid] = (self.d[n,arrup] + self.d[n,arrdown])/2.0 + self.delu*(self.F_d(n,arrup) - self.F_d(n,arrdown))/(2*self.delnu)
#             #Use backward differences to get value at jmax
#             self.d[n+1,jmax-1] = self.d[n,jmax-1] + self.delu*(3.0*self.F_d(n,jmax-1)/2.0 - 2*self.F_d(n,jmax-2) + self.F_d(n,jmax-3)/2.0)/self.delnu
#             #get remaining transformation components
#             self.c[n+1,0:jmax] = self.d[n+1,0:jmax]*(self.A[n+1,0:jmax]*self.v1[n+1,0:jmax]**2-1.0)/(2.0*self.v1[n+1,0:jmax]**2.0)
#             self.a[n+1,0:jmax] = -self.v1[n+1,0:jmax]**2/self.d[n+1,0:jmax]
#             self.b[n+1,0:jmax] = (1.0+self.A[n+1,0:jmax]*self.v1[n+1,0:jmax]**2.0)/(2.0*self.d[n+1,0:jmax])
#             #Get dprime and ddot to evaluate dXdr below
#             self.F[n+1,0:jmax] = self.d[n+1,0:jmax]*(self.A[n+1,0:jmax] - 1.0/self.v1[n+1,0:jmax]**2)/2.0
#             self.ddot[n+1,0] = (-self.F[n+1,2]/2.0 + 2.0*self.F[n+1,1] - 3.0*self.F[n+1,0]/2.0)/self.delnu
#             self.ddot[n+1,arrmid] = (self.F[n+1,arrup] - self.F[n+1,arrdown])/(2*self.delnu)
#             self.ddot[n+1,jmax-1] = (3.0*self.F[n+1,jmax-1]/2.0 - 2.0*self.F[n+1,jmax-2] + self.F[n+1,jmax-3]/2.0)/self.delnu
#             self.dprime[n+1,0] = (-self.d[n+1,2]/2.0 + 2.0*self.d[n+1,1] - 3.0*self.d[n+1,0]/2.0)/self.delnu
#             self.dprime[n+1,arrmid] = (self.d[n+1,arrup] - self.d[n+1,arrdown])/(2*self.delnu)
#             self.dprime[n+1,jmax-1] = (3.0*self.d[n+1,jmax-1]/2.0 - 2.0*self.d[n+1,jmax-2] + self.d[n+1,jmax-3]/2.0)/self.delnu
#             self.X[n+1,0:jmax] = -self.v1[n+1,0:jmax]/self.d[n+1,0:jmax]
# #            self.dXdr[n+1,0:jmax] = self.a[n+1,0:jmax]*(-self.v1u[n+1,0:jmax]/self.d[n+1,0:jmax] + self.v1[n+1,0:jmax]*self.ddot[n+1,0:jmax]/self.d[n+1,0:jmax]**2) + self.b[n+1,0:jmax]*(-self.v1nu[n+1,0:jmax]/self.d[n+1,0:jmax] + self.v1[n+1,0:jmax]*self.dprime[n+1,0:jmax]/self.d[n+1,0:jmax]**2)
# #            self.Hr[n+1,0:jmax] = self.dXdr[n+1,0:jmax]/self.X[n+1,0:jmax]
# #            self.dDdr[n+1,0:jmax] = self.a[n+1,0:jmax]*self.RR[n+1,0:jmax] + self.b[n+1,0:jmax]*self.S[n+1,0:jmax]
#             #Get t(v) and r(v)
#             dto = uvs(self.v[0:jmax],self.v1[n+1,0:jmax],k=3,s=0.0)
#             self.tv[n+1,0:jmax] =  self.w0[n+1] + dto.antiderivative()(self.v[0:jmax])
#             dro = uvs(self.v[0:jmax],self.d[n+1,0:jmax],k=3,s=0.0)
#             self.rv[n+1,0:jmax] = dro.antiderivative()(self.v[0:jmax])
#         return

    def get_tslice(self):
        #Here we get the constant time slice closest to t
        if (self.tfind >= self.w0[0] or self.tfind <= self.w0[-1]):
            #Check that time slice lies withing causal horizon
            print "Time slice beyond causal horizon"
            return 1
        else:
            I1 = np.argwhere(self.w0 >= self.tfind)[-1]
            I2 = np.argwhere(self.w0 < self.tfind)[0]
            #Choose whichever is closer
            if ( abs(self.w0[I1]-self.tfind) < abs(self.w0[I2] - self.tfind)):
                self.Istar = I1
            else:
                self.Istar = I2
            #get values on C
            self.tstar = self.w0[self.Istar]
            self.vstar = np.zeros(self.NI)
            self.rstar = np.zeros(self.NI)
            self.rhostar = np.zeros(self.NI)
            self.Dstar = np.zeros(self.NI)
            self.Xstar = np.zeros(self.NI)
            self.Hperpstar = np.zeros(self.NI)
            self.vstar[0] = 0.0
            self.rstar[0] = 0.0
            self.rhostar[0] = self.rho[self.Istar,0]
            self.Dstar[0] = 0.0
            self.Xstar[0] = self.X[self.Istar,0]
            self.Hperpstar[0] = self.Hperp[self.Istar,0]
            for i in range(1,self.Istar):
                n0 = self.Istar - i
                n = int(self.vmaxi[n0])
                I1 = np.argwhere(self.tv[n0,range(n)] > self.tstar)[-1]
                I2 = I1 + 1 #np.argwhere(self.tv[n0,range(n)] < self.tstar)[0]
                vi = np.squeeze(np.array([self.v[I1],self.v[I2]]))
                ti = np.squeeze(np.array([self.tv[n0,I1],self.tv[n0,I2]]))
                self.vstar[i] = np.interp(self.tstar,ti,vi)
                rhoi = np.squeeze(np.array([self.rho[n0,I1],self.rho[n0,I2]]))
                self.rhostar[i] = np.interp(self.vstar[i],vi,rhoi)
                rvi = np.squeeze(np.array([self.rv[n0,I1],self.rv[n0,I2]]))
                self.rstar[i] = np.interp(self.vstar[i],vi,rvi)
                Di = np.squeeze(np.array([self.D[n0,I1],self.D[n0,I2]]))
                self.Dstar[i] = np.interp(self.vstar[i],vi,Di)
                Xi = np.squeeze(np.array([self.X[n0,I1],self.X[n0,I2]]))
                self.Xstar[i] = np.interp(self.vstar[i],vi,Xi)
                Hperpi = np.squeeze(np.array([self.Hperp[n0,I1],self.Hperp[n0,I2]]))
                self.Hperpstar[i] = np.interp(self.vstar[i],vi,Hperpi)
            self.vmaxstar = self.vstar[self.Istar-1]                 
            return 0

    # def get_vmaxi(self,A,i):
    #     #Initial guess (Note the dirty fix on the index of Ap)
    #     vp = self.vmax[i-1] - 0.5*A[self.vmaxi[i-1]-1]*self.delw
    #     #Place holder
    #     vprev = 0
    #     #Counter
    #     s = 0
    #     #Iterate to find vmax[i]
    #     while (abs(vp-vprev)>1e-5 and s < self.Nitmax):
    #         vprev = vp
    #         jmax = int(np.floor(vp/self.delv + 1.0))
    #         if (jmax < 2):
    #             #Flag if causal horizon reached
    #             print "Warning causal horizon reached"
    #             jmax = 2
    #         elif jmax > self.NJ:
    #             #Flag for unexpected behaviour
    #             print "Something went wrong, got jmax > NJ"
    #             jmax = self.NJ-1
    #         #Interpolate to find Ap (since vp is not necessarily on a grid point)
    #         Ap = A[jmax-2] + (vp-self.v[jmax-1])*(A[jmax-1] - A[jmax-2])/self.delv
    #         vp = self.vmax[i-1] - 0.5*Ap*self.delw
    #         s += 1
    #     if (s >= self.Nitmax):
    #         print "Warning PNC cut-off did not converge"
    #     self.vmax[i] = vp
    #     self.vmaxi[i] = int(jmax)
    #     return

    def shear_test(self,i,NJ):
        n = int(self.vmaxi[i])
        tmp = np.zeros(NJ)
        tmp[0:n] = 1.0 - self.Hpar[0:n,i]/self.Hperp[0:n,i]
        return tmp
        
    def shear_test2(self,u,up,D,S,Q,A):
        tmp = 1.0 - u**3*(Q - S*(1/u**2 - A)/2)/(up*D)
        return tmp

    def curve_test(self,i,NJ):
        """
        The curvature test on PNC i. Returns z and K on PNC i
        """
        n = int(self.vmaxi[i])
        #Set redshift
        u = self.u[0:n,i]
        up = self.up[0:n,i]
        upp = self.upp[0:n,i]
        #Get D, D' and D''
        D = self.D[0:n,i]
        Dp = self.S[0:n,i]
        Dpp = -kappa*u**2.0*self.rho[0:n,i]*D/2.0
        #Get H and dHz
        H = self.Hpar[0:n,i]
        dH = (upp/u**2.0 - 2.0*up**2/u**3)/up
        #Get dDz
        dD = Dp/up
        #Get d2Dz
        d2D = (Dpp/up - Dp*upp/up**2)/up
        tmp = np.zeros(NJ)
        tmp[0:n] = 1.0 + H**2.0*(u**2.0*(D*d2D - dD**2.0) - D**2.0) + u*H*dH*D*(u*dD + D)
        return tmp

    def curve_test2(self,u,up,upp,D,S,rho):
        """
        The curvature test on PNC i. Returns z and K on PNC i
        """
        Dp = S
        Dpp = -kappa*u**2.0*rho*D/2.0
        #Get H and dHz
        H = up/u**2
        dH = (upp/u**2.0 - 2.0*up**2/u**3)/up
        #Get dDz
        dD = Dp/up
        #Get d2Dz
        d2D = (Dpp/up - Dp*upp/up**2)/up
        tmp = 1.0 + H**2.0*(u**2.0*(D*d2D - dD**2.0) - D**2.0) + u*H*dH*D*(u*dD + D)
        return tmp

    def get_dzdw(self,u,udot,up,A):
        return udot + up*(A - 1.0/u**2.0)/2.0
        
    def get_Chi2(self,H,D,rho,u,vzo,t0,NJ):
        """
        The inputs H, D and u are functions of v. However vzo is the spline v(z) so we can interpolate directly
        """
        z = u - 1.0
        vz = vzo(z)
        #First get mu(v)
        mu = np.zeros(NJ)
        mu[1::] = 5*np.log10(1.0e8*u[1::]**2*D[1::])
        mu[0] = -1e-15 #Should be close enough to -inf
        #Get funcs at data points
        muzi = uvs(vz,mu,k=3,s=0.0)(vzo(self.zmudat))
        Hzi = uvs(vz,H,k=3,s=0.0)(vzo(self.zHdat))
        rhozi = uvs(vz,rho,k=3,s=0.0)(vzo(self.zrhodat))
        chi2mu = sum((self.muzdat - muzi)**2/(self.smuzdat)**2)
        chi2H = sum((self.Hzdat - Hzi)**2/(self.sHzdat)**2)
        chi2rho = sum((self.rhozdat - rhozi)**2/(self.srhozdat)**2)
        chi2t0 = (self.t0dat - t0)**2/self.st0dat**2
        #print chi2mu,chi2H,chi2t0
        return chi2mu + chi2H + chi2t0 + chi2rho

    # def check_LLTB(self,D,S,Q,A,Z,u,rho,delw,Lam,NI,NJ):
    #     #Get App
    #     App = self.Zp #self.get_App(rho,Lam,D,S,Q,A)
    #     self.App = App
    #     #To store w derivs
    #     Dww = np.zeros([NJ,NI])
    #     Aw = np.zeros([NJ,NI])
    #     #To store consistency rel
    #     self.LLTBCon = np.zeros([NJ,NI])
    #     jmax = int(self.vmaxi[-1])
    #     for i in range(jmax):
    #         if (i==0):
    #             Dww[i,:] = 0.0
    #             Aw[i,:] = 0.0
    #         else:
    #             #Get Dww
    #             Dww[i,:] = CIVP.dd5f1d(D[i,:],-delw,NI,NI)
    #             #Get Aw
    #             Aw[i,:] = CIVP.d5f1d(A[i,:],-delw,NI,NI)
    #         self.Dww = Dww
    #         self.Aw = Aw
    #         self.LLTBCon[i,:] = 0.5*A[i,:]*App[i,:]*D[i,:] - 2.0*Dww[i,:] + Z[i,:]*Q[i,:] - S[i,:]*Aw[i,:] + A[i,:]*Z[i,:]*S[i,:] - 0.25*kappa*rho[i,:]*D[i,:]*(1.0/u[i,:]**2.0 + u[i,:]**2.0*A[i,:]**2.0) + Lam*A[i,:]*D[i,:]
    #     return

    def get_PLC0_observables(self,vzo,D,A,u,udot,up):
        #Get dzdw(v)
        dzdw = self.get_dzdw(u,udot,up,A)
        #Get mu(v)
        #mu = np.zeros(self.NJ)
        mu = np.zeros(self.NJ)
        mu[1::] = 5*np.log10(1e8*u[1::]**2*D[1::])
        mu[0] = -1e-15  #Should be close enough to -inf
        #Convert to functions of z
        z = u-1
        vz = vzo(z)
        Dz = uvs(vz,D,k=3,s=0.0)(vzo(self.z))
        muz=np.zeros(self.Np)
        muz[0] = -np.inf
        muz[1::] = uvs(vz,mu,k=3,s=0.0)(vzo(self.z[1::]))
        dzdwz = uvs(vz,dzdw,k=3,s=0.0)(vzo(self.z))
        return Dz, muz, dzdwz

    def get_funcs(self,F):
        """
        Return quantities of interest
        """
        #Here we do the shear and curvature tests on two pncs
        umax = int(self.Istar)
        njf = int(self.vmaxi[umax]) #This is the max value of index on final pnc considered
        
        #All functions will be returned with the domain normalised between 0 and 1
        l = np.linspace(0,1,self.Nret)
        #Curvetest
        T2i = self.curve_test(0,self.NJ)
        #self.Kiraw = Ki
        T2io = uvs(self.v/self.v[-1],T2i,k=3,s=0.0)
        T2i = T2io(l)
        T2f = self.curve_test(umax,self.NJ)
        #self.Kfraw = Kf
        T2fo = uvs(self.v[0:njf]/self.v[njf-1],T2f[0:njf],k=3,s=0.0)
        T2f = T2fo(l)
        #shear test
        T1i = self.shear_test(0,self.NJ)
        T1io = uvs(self.v/self.v[-1],T1i,k=3,s=0.0)
        T1i = T1io(l)
        T1f = self.shear_test(umax,self.NJ)
        T1fo = uvs(self.v[0:njf]/self.v[njf-1],T1f[0:njf],k=3,s=0.0)
        T1f = T1fo(l)
        #Get the LLTB consistency relation
        jmaxf = self.vmaxi[-1]
        LLTBConsi = uvs(self.v[0:jmaxf],self.LLTBCon[0,0:jmaxf],k=3,s=0.0)(l)
        LLTBConsf = uvs(self.v[0:jmaxf],self.LLTBCon[-1,0:jmaxf],k=3,s=0.0)(l)
        #Get constant t slices
        if F == 0:
            I = range(self.Istar)
            rmax = self.rstar[self.Istar-1]
            r = self.rstar[I]/rmax
            rhostar = np.interp(l,r,self.rhostar[I])
            Dstar = np.interp(l,r,self.Dstar[I])
            Dstar[0] = 0.0
            Xstar = np.interp(l,r,self.Xstar[I])
            Hperpstar = np.interp(l,r,self.Hperpstar[I])
        else:
            rmax = self.rstar[0]
            rhostar = np.tile(self.rhostar[0],(self.Nret))
            Dstar = np.tile(self.Dstar[0],(self.Nret))
            Xstar = np.tile(self.Xstar[0],(self.Nret))
            Hperpstar = np.tile(self.Hperpstar[0],(self.Nret))
        return self.Dz,self.muz,self.dzdw,T1i, T1f,T2i,T2f,LLTBConsi,LLTBConsf,rhostar,Dstar,Xstar,Hperpstar,rmax,self.Om0,self.OL0,self.t0
        
if __name__ == "__main__":
    #Set sparams
    zmax = 2.0
    Np = 250
    zp = np.linspace(0,zmax,Np)
    Xrho = np.array([0.1,1.5])
    XH = np.array([0.6,3.5])
    tmin = 3.0
    err = 1e-5
    Nret = 100

    #set characteristic variance of proposal distributions
    sigmaLam = 0.00555
    
    zp = np.linspace(0,zmax,Np)
    U = SSU(zmax,tmin,Np,err,XH,Xrho,sigmaLam,Nret)

    D = U.D
    S = U.S
    Q = U.Q
    A = U.A
    Z = U.Z
    #Zi = U.Z0
    rho = U.rho
    rhop = U.rhop
    rhod = U.rhod
    u = U.u
    up = U.up
    upp = U.upp
    uppow = U.uppow
    ud = U.ud
    Ki = U.curve_test(0,U.NJ)
    Kf = U.curve_test(U.NI-1,U.NJ)
    sheari = U.shear_test(0,U.NJ)
    shearf = U.shear_test(U.NI-1,U.NJ)
    drdv = U.drdv
    drdvp = U.drdvp
    X = U.X
    dXdr = U.dXdr

    #Dsamps,musamps,dzdwsamps,T1i, T1f,T2i,T2f,LLTBConsi,LLTBConsf,rhostar,Dstar,Xstar,Hperpstar,rmax,Omsamps,OLsamps,t0samps = U.get_funcs(0)
#    D2 = U.D2
#    S2 = U.S2
#    Q2 = U.Q2
#    A2 = U.A2
#    Z2 = U.Z2
#    #Zi = U.Z0
#    rho2 = U.rho2
#    rhop2 = U.rhop2
#    rhod2 = U.rhod2
#    u2 = U.u2
#    up2 = U.up2
#    upp2 = U.upp2
#    uppow2 = U.uppow
#    ud2 = U.ud2
#    LLTBCon = U.LLTBCon
#    Xstar = U.Xstar
#    Dstar = U.Dstar
#    rhostar = U.rhostar
    LLTBCon = U.LLTBCon
    LLTBCon2 = U.LLTBCon2
    Aw = U.Aw
    Aw2 = U.Aw2
    Dww = U.Dww
    Dww2 = U.Dww2
    App = U.App

#    vmax = U.vmax
    vmaxi = U.vmaxi
#    jmax = vmaxi[-1]
#    
    NI = U.NI
    v = U.v
#    tv = U.tv
#    rv = U.rv
#    
#    D2 = U.D2
#    S2 = U.S2
#    Q2 = U.Q2
#    A2 = U.A2
#    W2 = U.W2
#    Wp = (W2[0,2::] - W2[0,0:-1])/(2*U.delv)
#    Z2 = Wp/D2[1:-1] - W2[1:-1]*S2[1:-1]/D2[1:-1]**2
#    rho2 = U.rho2
#    rhop2 = U.rhop2
#    rhod2 = U.rhod2
#    u2 = U.u2
#    up2 = U.up2
#    upp2 = U.upp2
#    ud2 = U.ud2
#
#    D3 = U.Dfort
#    S3 = U.Sfort
#    Q3 = U.Qfort
#    A3 = U.Afort
#    Z3 = U.Zfort
#    
#    rho3 = U.rhofort
#    rhop3 = U.rhopfort
#    rhod3 = U.rhodfort
#    u3 = U.ufort
#    up3 = U.upfort
#    upp3 = U.uppfort
#    ud3 = U.udfort 
#    
#    drdv = U.drdv
#    drdvp = U.drdvp
#    X = U.Xfort
#    Xr = U.Xrfort
#    t = U.tfort
#    r = U.rfort
    
    n = NI-1
    njf = vmaxi[n]
    
    plt.figure('LLTBConi')
    plt.plot(v[0:njf],LLTBCon[0:njf,0])
    plt.plot(v[0:-1],LLTBCon2[0:-1,0])
    
    plt.figure('LLTBConf')
    plt.plot(v[0:njf],LLTBCon[0:njf,-10])
    plt.plot(v[0:njf],LLTBCon2[0:njf,-10])

##    plt.figure('Dww')
##    plt.plot(v[0:njf],Dww[0:njf,-5])
##
##    plt.figure('Aw')
##    plt.plot(v[0:njf],Aw[0:njf,-5])
#
    plt.figure('sheari')        
    plt.plot(v,sheari,'b')
    #plt.plot(v,U.sheari2,'g')

    plt.figure('shearf')
    plt.plot(v[0:njf],shearf[0:njf])
    
    plt.figure('Ki')        
    plt.plot(v,Ki,'b')
    #plt.plot(v,U.Ki2,'g')

    plt.figure('Kf')
    plt.plot(v[0:njf],Kf[0:njf])
#
#    plt.figure('u')
##    plt.plot(v[0:njf],u[n,0:njf],'b')
#    plt.plot(v[0:njf],u[0:njf,n],'g')
###    plt.plot(v,u3[:,n].T)
##    plt.ylim(0,3)
##
#    plt.figure('up')
##    plt.plot(v[0:njf],up[n,0:njf],'b')
#    plt.plot(v[0:njf],up[0:njf,n],'g')
###    plt.plot(v,up3[:,n])
##    plt.ylim(0,7)
##
#    plt.figure('upp')
##    plt.plot(v[0:njf],upp[n,0:njf],'b')
#    plt.plot(v[0:njf],upp[0:njf,n],'g')
##    plt.plot(v,upp3[:,n])
#    plt.ylim(0,40)
#
#    plt.figure('ud')
##    plt.plot(v[0:njf],ud[n,0:njf],'b')
#    plt.plot(v[0:njf],ud[0:njf,n],'g')
###    plt.plot(v,ud3[:,n])
##    plt.ylim(0.1,-2.5)
##
#    plt.figure('rho')
##    plt.plot(v[0:njf],rho[n,0:njf],'b')
#    plt.plot(v[0:njf],rho[0:njf,n],'g')
###    plt.plot(v,rho3[:,n])
##    plt.ylim(0,0.1)
##
#    plt.figure('rhop')
##    plt.plot(v[0:njf],rhop[n,0:njf],'b')
#    plt.plot(v[0:njf],rhop[0:njf,n],'g')
#    plt.plot(v,rhop3[:,n])
#    plt.ylim(0,0.25)
#
#    plt.figure('rhod')
##    plt.plot(v[0:njf],rhod[n,0:njf],'b')
#    plt.plot(v[0:njf],rhod[0:njf,n],'g')
##    plt.ylim(0,-0.25)
##
#    plt.figure('D')
##    plt.plot(v[0:njf],D[n,0:njf],'b')
#    plt.plot(v[0:njf],D[0:njf,n],'g')
###    plt.plot(v,D3[:,n])
##    plt.ylim(0,2)
#
#    plt.figure('S')
##    plt.plot(v[0:njf],S[n,0:njf],'b')
#    plt.plot(v[0:njf],S[0:njf,n],'g')
###    plt.plot(v,D3[:,n])
##    plt.ylim(0,2)
#   
#    plt.figure('Q')
##    plt.plot(v[0:njf],Q[n,0:njf],'b')
#    plt.plot(v[0:njf],Q[0:njf,n],'g')
###    plt.plot(v,Q3[:,n])
##    plt.ylim(0,0.7)
##
#    plt.figure('A')
##    plt.plot(v[0:njf],A[n,0:njf],'b')
#    plt.plot(v[0:njf],A[0:njf,n],'g')
##    plt.plot(v,A3[:,n])
##    plt.ylim(0.9,1.0)
#
#    plt.figure('Z')
##    plt.plot(v[0:njf],Z[n,0:njf],'b')
#    plt.plot(v[0:njf],Z[0:njf,n],'g')
##    plt.plot(v,Z3[:,n])
##    plt.ylim(0.9,1.0)

#    plt.figure('LLTBCon')
#    plt.plot(v[0:jmax],LLTBCon[0,0:jmax])
#    plt.plot(v[0:jmax],LLTBCon[-5,0:jmax])
#    plt.ylim(-50*err,50*err)
#   
#    plt.figure('drdv')
#    plt.plot(v[0:njf],drdv[0:njf,n])
#
#    plt.figure('drdvp')
#    plt.plot(v[0:njf],drdvp[0:njf,n])
#
#    plt.figure('X')
#    plt.plot(v[0:njf],X[0:njf,n])
#
#    plt.figure('Xr')
#    plt.plot(v[0:njf],dXdr[0:njf,n])

#    dzdw = U.get_dzdw()
#    
#    plt.plot(dzdw)
#    
#    U2 = SSU2(Lam,HzF,rhoF,zmax,NJ)
#    #Get D
#    D2,F2 = U2.get_D(U2.Hmax,U2.rhomax,U2.z)
#    
#    #set up the three different spatial grids
#    nu2,H2,rho2,u12,NI2,delnu2 = U2.affine_grid(U2.z,U2.Hmax,U2.rhomax)
#    
#    #set the three time grids
#    u2, delu2 = U2.age_grid(NI2,NJ,delnu2)
#    
#    #Initialise storage arrays
#    U2.init_storage(NI2,NJ)
#    
#    #Do integration
#    U2.integrate(NJ,NI2,delu2,delnu2,nu2,u2,D2,u12,rho2,Lam)
#    
#    D2 = U2.r
#    S2 = U2.S
#    Q2 = U2.RR
#    A2 = U2.A
#    Z2 = U2.Anu
#    
#    rho2 = U2.rhof
#    

#    U.transform()
#    
#    NJ = U.NJ
#    NI = U.NI
#    
#    X = U.X
#    dXdr = U.dXdr
#    d = U.d
#    ljmax = U.ljmax
#    Hr = U.Hr
#    
#    for i in range(NI):
#        jmax = int(ljmax[i])
#        plt.plot(Hr[i,0:jmax])
#        
#    Ftr = U.get_tvrv()
#    Fts = U.get_tslice()
#    
#    zfmax, Ki, Kf,sheari,shearf,t0,rhostar,Dstar,Xstar,rmax,vmax = U.get_funcs()
#    
#    d2D = U.dDk
#    d2D = U.d2Dk
#    dD = U.dDk
#    D = U.Dk
#    H = U.Hk
#    dH = U.dHk
#    Kir = U.Kiraw
#    Kfr = U.Kfraw
#    z = U.zk
#    
#    plt.figure('D')
#    plt.plot(D)
#    plt.figure('dD')
#    plt.plot(dD)
#    plt.figure('d2D')
#    plt.plot(d2D)
#    plt.figure('H')
#    plt.plot(H)
#    plt.figure('dH')
#    plt.plot(dH)
#    plt.figure('z')
#    plt.plot(z)
#    plt.figure('Ki')
#    plt.plot(Kir)
#    plt.figure('Kf')
#    plt.plot(Kfr)


#    plt.figure(1)
#    t1 = time.time()
#    for i in xrange(100):
#        plt.plot(zp,KH.sample(Hm))
#    print time.time() - t1
#    plt.figure(2)
#    t1 = time.time()
#    for i in xrange(100):
#        plt.plot(zp,KH.simp_sample())
#    print time.time() - t1    

#    zH,Hz,sHz = np.loadtxt('C:\Users\BMAX\Documents\Algorithm\RawData\SimH.txt',unpack=True)
#    KH = GP(zH,Hz,sHz,zp,XH)
#    Hm = KH.fmean
#    fcov = KH.fcov