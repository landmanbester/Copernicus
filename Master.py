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

import os
import sys
import numpy as np
from numpy.linalg import cholesky, solve, inv, slogdet, eigh
from numpy.random import randn, random, seed
from scipy.integrate import odeint,quad
import scipy.optimize as opt
from scipy.linalg import solve_triangular as soltri
from scipy.interpolate import UnivariateSpline as uvs
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from Copernicus.fortran_mods import CIVP

global kappa
kappa = 8.0*np.pi

class GP(object):
    def __init__(self, x, y, sy, xp, THETA, beta):
        """
        This is a simple Gaussian process class. It just trains the GP on the data
        Input:  x = independent variable of data point
                y = dependent varaiable of data point
                sy = 1-sig uncertainty of data point (std. dev.)  (Could be modified to use full covariance matrix)
                xp = independent variable of targets
                THETA = Initial guess for hyper-parameter values
                prior_mean = function (lambda or spline or whatever) that can be evaluated at x/xp
                mode = sets whether to train normally, iteratively or not train at all (if hypers already optimised)
        """
        #Re-seed the random number generator
        seed()
        #Compute quantities that are used often
        self.beta = beta
        self.N = x.size
        self.Nlog2pi = self.N*np.log(2.0*np.pi)
        self.Np = xp.size
        self.zero = np.zeros(self.Np)
        self.Nplog2pi = self.Np*np.log(2.0*np.pi)
        self.eyenp = np.eye(self.Np)
        #Get vectorised forms of x_i - x_j
        self.XX = self.abs_diff(x, x)
        self.XXp = self.abs_diff(x, xp)
        self.XXpp = self.abs_diff(xp, xp)
        self.ydat = y 
        self.SIGMA = np.diag(sy**2) #Set data covariance matrix

        # Train the GP
        self.train(THETA)

        self.K = self.cov_func(self.THETA, self.XX)
        self.L = cholesky(self.K + self.SIGMA)
        self.sdet = 2*sum(np.log(np.diag(self.L)))
        self.Linv = soltri(self.L.T, np.eye(self.N)).T
        self.Linvy = np.dot(self.Linv, self.ydat)
        self.logL = self.log_lik(self.Linvy, self.sdet)
        self.Kp = self.cov_func(self.THETA, self.XXp)
        self.LinvKp = np.dot(self.Linv, self.Kp)
        self.Kpp = self.cov_func(self.THETA, self.XXpp)
        self.fmean = np.dot(self.LinvKp.T, self.Linvy)
        self.fcov = self.Kpp - np.dot(self.LinvKp.T, self.LinvKp)
        self.W, self.V = eigh(self.fcov)
        #print any(self.W < 0.0)
        self.srtW = np.diag(np.nan_to_num(np.sqrt(np.nan_to_num(self.W))))


    # def meanf(self, theta, y, XXp):
    #     """
    #     This funcion returns the posterior mean. Only used for optimization.
    #     """
    #     Kp = self.cov_func(theta,XXp)
    #     Ky = self.cov_func(theta,self.XX) + self.SIGMA
    #     return np.dot(Kp.T,solve(Ky,y))
    #
    # def covf(self, theta):
    #     """
    #     This funcion returns the posterior covariance matrix. Only used for optimization.
    #     """
    #     Kp = self.cov_func(theta, self.XXp)
    #     Kpp = self.cov_func(theta, self.XXpp)
    #     Ky = self.cov_func(theta, self.XX) + self.SIGMA
    #     L = cholesky(Ky)
    #     Linv = inv(L)
    #     LinvKp = np.dot(Linv, Kp)
    #     return Kpp - np.dot(LinvKp.T, LinvKp)

    def diag_dot(self, A, B):
        D = np.zeros(A.shape[0])
        for i in range(A.shape[0]):
            D[i] = np.dot(A[i, :], B[:, i])
        return D

    def logp_and_gradlogp(self, theta, XX, y, n):
        """
        Returns the negative log (marginal) likelihood (the function to be optimised) and its gradient
        """
        # tmp is Ky
        tmp = self.cov_func(theta, XX) + self.SIGMA
        #Ky = self.cov_func(theta, XX) + self.SIGMA
        # tmp is L
        tmp = cholesky(tmp)
        #L = cholesky(Ky)
        detK = 2.0 * sum(np.log(np.diag(tmp)))
        #detK = 2.0 * sum(np.log(np.diag(L)))
        # tmp is Linv
        #Linv = inv(L)
        tmp = soltri(tmp.T, np.eye(n)).T
        # tmp2 is Linvy
        tmp2 = np.dot(tmp, y)
        #Linvy = np.dot(Linv,y)
        logp = np.dot(tmp2.T, tmp2) / 2.0 + detK / 2.0 + self.Nlog2pi / 2.0
        #logp = np.dot(Linvy.T, Linvy) / 2.0 + detK / 2.0 + n * self.Nlog2pi / 2.0
        nhypers = theta.size
        dlogp = np.zeros(nhypers)
        # tmp is Kinv
        tmp = np.dot(tmp.T, tmp)
        #Kinv = np.dot(Linv.T, Linv)
        # tmp2 becomes Kinvy
        tmp2 = np.reshape(np.dot(tmp, y), (n, 1))
        #Kinvy = np.reshape(np.dot(Kinv, y), (n, 1))
        # tmp2 becomes aaT
        tmp2 = np.dot(tmp2, tmp2.T)
        #aaT = np.dot(Kinvy, Kinvy.T)
        # tmp2 becomes Kinv - aaT
        tmp2 = tmp - tmp2
        #tmp2 = Kinv - aaT
        dKdtheta = self.dcov_func(theta, XX, mode=0)
        dlogp[0] = sum(self.diag_dot(tmp2, dKdtheta)) / 2.0
        dKdtheta = self.dcov_func(theta, XX, mode=1)
        dlogp[1] = sum(self.diag_dot(tmp2, dKdtheta)) / 2.0
        return logp, dlogp

    def dcov_func(self, theta, x, mode=0):
        if mode == 0:
            return 2.0 * theta[0] * np.exp(-x ** 2.0 / (2.0 * theta[1] ** 2.0))
        elif mode == 1:
            return x ** 2.0 * theta[0] ** 2.0 * np.exp(-x ** 2.0 / (2.0 * theta[1] ** 2.0)) / theta[1] ** 3.0

    def abs_diff(self, x, xp):
        """
        Creates matrix of differences (x_i - x_j) for vectorising.
        """
        N = x.size
        Np = xp.size
        return np.tile(x, (Np, 1)).T - np.tile(xp, (N, 1))

    def cov_func(self, theta, x):
        """
        Covariance function
        """
        return theta[0] ** 2.0 * np.exp(-x ** 2.0 / (2.0 * theta[1] ** 2.0))

    def sample(self, f):
        """
        Returns pCN proposal for MCMC. For normal sample use simp_sample
        """
        f0 = f - self.fmean
        return self.fmean + np.sqrt(1-self.beta**2)*f0 + self.beta*self.V.dot(self.srtW.dot(randn(self.Np)))

    # def simp_sample(self):
    #     return self.fmean + self.V.dot(self.srtW.dot(randn(self.Np)))
    #
    # def sample_logprob(self, f):
    #     """
    #     Returns the probability of a sample from the posterior pdf.
    #     """
    #     F = f-self.fmean
    #     LinvF = solve(self.fcovL, F)
    #     return -0.5*np.dot(LinvF.T, LinvF) - 0.5*self.covdet - 0.5*self.Nplog2pi
    #
    # def logp(self, theta, y):
    #     """
    #     Returns marginal negative log lik. Only used for optimization.
    #     """
    #     Ky = self.cov_func(theta,self.XX) + self.SIGMA
    #     y = np.reshape(y, (self.N, 1))
    #     print Ky.shape, y.T.shape, y.shape
    #     return np.dot(y.T, solve(Ky, y))/2.0 + slogdet(Ky)[1]/2.0 + self.Nlog2pi/2.0

    def train(self, THETA0):
        bnds = ((1e-7, None), (1e-7, None))
        thetap = opt.fmin_l_bfgs_b(self.logp_and_gradlogp, THETA0, fprime=None, args=(self.XX, self.ydat, self.N), bounds=bnds)
        if thetap[2]['warnflag']:
            print "There was a problem with the GPR. Please try again."
        else:
            #print "Optimised hypers = ", thetap[0]
            self.THETA = thetap[0]


    def log_lik(self, Linvy, sdet):
        """
        Quick marginal log lik for hyper-parameter marginalisation
        """
        return -0.5*np.dot(Linvy.T, Linvy) - 0.5*sdet - 0.5*self.Nlog2pi

class SSU(object):
    def __init__(self, zmax, tmin, Np, err, XH, Xrho, sigmaLam, Nret, data_prior, data_lik, fname, Hz = None, rhoz = None, Lam = None, beta = None):
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
        #Re-seed the random number generator
        seed()
        # Load the data
        self.fname = fname
        self.data_prior = data_prior.strip('[').strip(']').split(',')
        self.data_lik = data_lik.strip('[').strip(']').split(',')
        self.load_Dat()

        # Set number of spatial grid points
        self.Np = Np
        self.z = np.linspace(0, zmax, self.Np)
        self.uz = 1.0 + self.z

        # Set function to get t0
        self.t0f = lambda x, a, b, c, d: np.sqrt(x)/(d*np.sqrt(a + b*x + c*x**3))

        # Set minimum time to integrate to
        self.tmin = tmin
        self.tfind = tmin + 0.1 # This is defines the constant time slice we are looking for

        # Set number of spatial grid points at which to return quantities of interest
        self.Nret = Nret

        # Set target error
        self.err = err # The error of the integration scheme used to set NJ and NI

        # Set beta (the parameter controlling acceptance rate
        if beta is not None:
            self.beta = beta
        else:
            self.beta = 0.01

        # Create GP objects
        #print "Fitting H GP"
        self.GPH = GP(self.my_z_prior["H"], self.my_F_prior["H"], self.my_sF_prior["H"], self.z, XH, self.beta)
        self.XH = self.GPH.THETA
        self.Hm = self.GPH.fmean
        # plt.figure('H')
        # plt.plot(self.z,self.Hm)
        # plt.errorbar(self.my_z_prior["SimH"],self.my_F_prior["SimH"],self.my_sF_prior["SimH"],fmt='xr')
        # plt.show()

        #print "Fitting rho GP"
        self.GPrho = GP(self.my_z_prior["rho"], self.my_F_prior["rho"], self.my_sF_prior["rho"], self.z,Xrho, self.beta)
        self.Xrho = self.GPrho.THETA
        self.rhom = self.GPrho.fmean
        # plt.figure('rho')
        # plt.plot(self.z,self.rhom)
        # plt.errorbar(self.my_z_prior["Simrho"],self.my_F_prior["Simrho"],self.my_sF_prior["Simrho"],fmt='xr')
        # plt.show()

        # Set Lambda mean and variance (Here we use the background model)
        self.Lam = 0.0 #3*0.7*(70.0/299.79)**2
        self.sigmaLam = sigmaLam

        # Now we do the initialisation starting with the background vals
        if Lam is None:
            Lam = 3*0.7*(70.0/299.79)**2
        if Hz is None:
            Hz = self.Hm
        if rhoz is None:
            rhoz = self.rhom
            while any(rhoz<0.0):
                rhoz = self.GPrho.sample(rhoz)
        #Set up spatial grid
        v, vzo, Hi, rhoi, ui, NJ, NI, delv, Om0, OL0, Ok0, t0 = self.affine_grid(Hz, rhoz, Lam)
        self.NI = NI
        self.NJ = NJ
        self.v = v
        
        # Set up time grid
        w,delw = self.age_grid(NI, NJ, delv, t0)

        # Get soln on C
        #rhoow, upow, uppow = self.get_C_sol(Om0, Ok0, OL0, Hz[0])
        
        #self.uppow=uppow

        #Do the first CIVP0 integration
        D, S, Q, A, Z, rho, u, up, upp, udot, rhodot, rhop, Sp, Qp, Zp, LLTBCon, T1, T2, vmaxi = self.integrate(ui, rhoi, Lam, v, delv, w, delw)
        
        # Get the likelihood of the first sample Hz,D,rho,u,vzo,t0,NJ,udot,up
        self.logLik = self.get_Chi2(Hz=Hz, D=D, rhoz=rhoz, u=u, vzo=vzo, t0=t0, NJ=NJ, udot=udot, up=up, A=A)

        Dz, dzdwz = self.get_Dz_and_dzdwz(vzo=vzo, D=D[:, 0], A=A[:, 0], u=u[:, 0], udot=udot[:, 0], up=up[:, 0])

        # Accept the first step regardless (this performs the main integration and transform)
        self.accept(Dz=Dz, dzdwz=dzdwz, D=D, S=S, Q=Q, A=A, Z=Z, rho=rho, u=u, up=up, upp=upp, udot=udot,
                    rhodot=rhodot, rhop=rhop, Sp=Sp, Qp=Qp, Zp=Zp, LLTBCon=LLTBCon, T1=T1, T2=T2,
                    vmaxi=vmaxi, v=v, w0=w[:, 0], NJ=NJ, NI=NI)


    def reset_beta(self,beta):
        self.beta = beta
        self.GPH.beta = beta
        self.GPrho.beta = beta
        return

    def MCMCstep(self, logLik0, Hz0, rhoz0, Lam0):
        #Propose sample
        Hz, rhoz, Lam, F = self.gen_sample(Hz0, rhoz0, Lam0)
        if F == 1:
            return Hz0, rhoz0, Lam0, logLik0, 0, 0
        else:
            #Set up spatial grid
            v, vzo, H, rho, u, NJ, NI, delv, Om0, OL0, Ok0, t0 = self.affine_grid(Hz, rhoz, Lam)
            #Set temporal grid
            w, delw = self.age_grid(NI, NJ, delv, t0)
            #Do integration on initial PLC
            D, S, Q, A, Z, rho, u, up, upp, udot, rhodot, rhop, Sp, Qp, Zp, LLTBCon, T1, T2, vmaxi = self.integrate(u, rho, Lam, v, delv, w, delw)
            #Get likelihood
            logLik = self.get_Chi2(Hz=Hz, D=D, rhoz=rhoz, u=u, vzo=vzo, t0=t0, NJ=NJ, udot=udot, up=up, A=A) #Make sure to pass Hz and rhoz to avoid the interpolation
            logr = logLik - logLik0
            accprob = np.exp(-logr/2.0)
            #Accept reject step
            tmp = random(1)
            if tmp > accprob:
                #Reject sample
                return Hz0, rhoz0, Lam0, logLik0, 0, 0
            else:
                #Accept sample
                Dz, dzdwz = self.get_Dz_and_dzdwz(vzo=vzo, D=D[:,0], A=A[:,0], u=u[:,0], udot=udot[:,0], up=up[:,0])
                self.accept(Dz=Dz, dzdwz=dzdwz, D = D, S = S, Q = Q, A = A, Z = Z, rho = rho, u = u, up = up, upp = upp, udot = udot,
                                rhodot = rhodot, rhop = rhop, Sp = Sp, Qp = Qp, Zp = Zp, LLTBCon = LLTBCon, T1 = T1,
                                T2 = T2, vmaxi=vmaxi, v = v, w0 = w[:,0], NJ=NJ, NI=NI)
                return Hz,rhoz,Lam,logLik,F,1  #If F returns one we can't use solution inside PLC

    def load_Dat(self):
        """
        Here we read in the data that should be used for inference. Currently the data files are stored as .txt files in
        the format [z,F,sF] where z is a column of redshift values, F is a column of function values and sF a column of
        1-sigma uncertainties in F. Need to figure out a more sophisticated way to do this.
        """
        # Create dict containing data for inference
        self.my_z_data = {}
        self.my_F_data = {}
        self.my_sF_data = {}
        for s in self.data_lik:
            self.my_z_data[s], self.my_F_data[s], self.my_sF_data[s] = np.loadtxt(self.fname + "Data/" + s + '.txt', dtype=float, unpack=True)

        self.my_z_prior = {}
        self.my_F_prior = {}
        self.my_sF_prior = {}
        for s in self.data_prior:
            self.my_z_prior[s], self.my_F_prior[s], self.my_sF_prior[s] = np.loadtxt(self.fname + "Data/" + s + '.txt', dtype=float, unpack=True)
        return
        
    def gen_sample(self, Hzi, rhozi, Lami):
        Hz = self.GPH.sample(Hzi)
        rhoz = self.GPrho.sample(rhozi)
        Lam0 = self.Lam - Lami
        Lam = np.abs(self.Lam + np.sqrt(1 - self.beta**2)*Lam0 + self.beta*self.sigmaLam*np.random.randn(1))
        #Flag if any of these less than zero
        if ((Hz < 0.0).any() or (rhoz <= 0.0).any() or (Lam<0.0)):
            print "Negative samples",(Hz < 0.0).any(),(rhoz <= 0.0).any(),(Lam<0.0)
            return Hzi, rhozi,Lami, 1
        else:
            return Hz, rhoz, Lam, 0
        
    def accept(self, Dz=None, dzdwz=None, D=None, S=None, Q=None, A=None, Z=None, rho=None, u=None, up=None, upp=None, udot=None,
               rhodot=None, rhop=None, Sp=None, Qp=None, Zp=None, LLTBCon=None, T1=None, T2=None, vmaxi=None,
               v=None, w0=None, NJ=None, NI=None):
        """
        Stores all values of interest (i.e the values returned by self.integrate)
        """
        self.Dz = Dz
        self.dzdwz = dzdwz
        self.D = D
        self.S = S
        self.Q = Q
        self.A = A
        self.Z = Z
        self.rho = rho
        self.u = u
        self.up = up
        self.upp = upp
        self.udot = udot
        self.rhodot = rhodot
        self.rhop = rhop
        self.Sp = Sp
        self.Qp = Qp
        self.Zp = Zp
        self.LLTBCon = LLTBCon
        self.T1 = T1
        self.T2 = T2
        self.vmaxi = vmaxi
        self.v = v
        self.w0 = w0
        self.NJ = NJ
        self.NI = NI
        #print " Actual Shape", self.T1.shape, " should be", NJ, NI
        return

    def get_age(self, Om0, Ok0, OL0, H0):
        """
        Here we return the current age of the Universe. quad seems to give the most reliable estimates
        TODO: figure out why the elliptic functions sometimes gives NaN
        """
        return quad(self.t0f, 0, 1, args=(Om0, Ok0, OL0, H0))[0]

    # def get_C_sol(self, Om0, Ok0, OL0, H0):
    #     """
    #     Since the Universe is FLRW along the central worldline (viz. C) we have analytic expressions for the input
    #     functions along C. These can be used as boundary data in the CIVP (not currently implemented)
    #     """
    #     #First get current age of universe
    #     amin = 0.5
    #     t0 = quad(self.t0f,0,1.0,args=(Om0,Ok0,OL0,H0))[0]
    #     #Get t(a) when a_0 = 1 (for some reason spline does not return correct result if we use a = np.linspace(1.0,0.2,1000)?)
    #     a = np.linspace(amin,1.0,5000)
    #     tmp = self.t0f(a,Om0,Ok0,OL0,H0)
    #     dto = uvs(a,tmp,k=3,s=0)
    #     to = dto.antiderivative()
    #     t = t0 + to(a)
    #     self.t = t
    #     #Invert to get a(t)
    #     aoto = uvs(t,a,k=3,s=0.0)
    #     aow = aoto(self.w0)
    #     #Now get rho(t)
    #     rho0 = Om0*3*H0**2/(kappa)
    #     rhoow = rho0*aow**(-3.0)
    #     #Get How (this is up_0)
    #     How = H0*np.sqrt(Om0*aow**(-3) + Ok0*aow**(-2) + OL0)
    #     upow = How
    #     #Now get dHz and hence uppow
    #     #First dimensionless densities
    #     Omow = kappa*rhoow/(3*How**2)
    #     OLow = self.Lam/(3*How**2)
    #     OKow = 1 - Omow - OLow
    #     dHzow = How*(3*Omow + 2*OKow)/2 #(because the terms in the np.sqrt adds up to 1)
    #     uppow = (dHzow + 2*upow)*upow
    #     return rhoow, upow, uppow
        
    def affine_grid(self, Hz, rhoz, Lam):
        """
        Get data on regular spatial grid
        """
        #First find dimensionless density params
        Om0 = kappa*rhoz[0]/(3*Hz[0]**2)
        OL0 = Lam/(3*Hz[0]**2)
        Ok0 = 1-Om0-OL0

        #Get t0
        t0 = self.get_age(Om0,Ok0,OL0,Hz[0])
        #print "t0 = ", t0, Om0, OL0, Ok0, rhoz[0]

        #Set affine parameter vals        
        dvo = uvs(self.z,1/(self.uz**2*Hz),k=3,s=0.0) #seems to bve the most accurate way to do the numerical integration
        vzo = dvo.antiderivative()
        vz = vzo(self.z)
        vz[0] = 0.0

        #Compute grid sizes that gives num error od err
        NJ = int(np.ceil(vz[-1]/np.sqrt(self.err) + 1))
        NI = int(np.ceil(3.0*(NJ - 1)*(t0 - self.tmin)/vz[-1] + 1))

        #Get functions on regular grid
        v = np.linspace(0,vz[-1],NJ)
        delv = (v[-1] - v[0])/(NJ-1)
        if delv > np.sqrt(self.err): #A sanity check
            print 'delv > sqrt(err)'
        Ho = uvs(vz,Hz,s=0.0,k=3)
        H = Ho(v)
        rhoo = uvs(vz,rhoz,s=0.0,k=3)
        rho = rhoo(v)
        uo = uvs(vz,self.uz,s=0.0,k=3)
        u = uo(v)
        u[0] = 1.0
        return v, vzo, H, rho, u, NJ, NI, delv, Om0, OL0, Ok0, t0
        
    def age_grid(self, NI, NJ, delv, t0):
        w0 = np.linspace(t0, self.tmin, NI)
        #self.w0 = w0
        delw = (w0[0] - w0[-1])/(NI-1)
        if delw/delv > 0.5:
            print "Warning CFL might be violated." 
        #Set u grid
        w = np.tile(w0, (NJ, 1)).T
        return w, delw

    def integrate(self,u,rho,Lam,v,delv,w,delw):
        """
        This is the routine that calls the compiled Fortran module to do the integration. We only return what we need
        from here but the Fortran code should return everything that could possibly be of interest.
        TODO: write the Fortran code to compute t(v) and r(v) and also find a current time slice t = tmin
        """
        NI, NJ = w.shape
        D,S,Q,A,Z,rho,rhod,rhop,u,ud,up,upp,vmax,vmaxi,r,t,X,dXdr,drdv,drdvp,Sp,Qp,Zp,LLTBCon,Dww,Aw,T1,T2 = CIVP.solve(v,delv,w,delw,u,rho,Lam,NI,NJ)
        #self.vmaxi = vmaxi
        return D,S,Q,A,Z,rho,u,up,upp,ud,rhod,rhop,Sp,Qp,Zp,LLTBCon,T1,T2,vmaxi

    # def get_tslice(self):
    #     #Here we get the constant time slice closest to t
    #     if (self.tfind >= self.w0[0] or self.tfind <= self.w0[-1]):
    #         #Check that time slice lies withing causal horizon
    #         print "Time slice beyond causal horizon"
    #         return 1
    #     else:
    #         I1 = np.argwhere(self.w0 >= self.tfind)[-1]
    #         I2 = np.argwhere(self.w0 < self.tfind)[0]
    #         #Choose whichever is closer
    #         if ( abs(self.w0[I1]-self.tfind) < abs(self.w0[I2] - self.tfind)):
    #             self.Istar = I1
    #         else:
    #             self.Istar = I2
    #         #get values on C
    #         self.tstar = self.w0[self.Istar]
    #         self.vstar = np.zeros(self.NI)
    #         self.rstar = np.zeros(self.NI)
    #         self.rhostar = np.zeros(self.NI)
    #         self.Dstar = np.zeros(self.NI)
    #         self.Xstar = np.zeros(self.NI)
    #         self.Hperpstar = np.zeros(self.NI)
    #         self.vstar[0] = 0.0
    #         self.rstar[0] = 0.0
    #         self.rhostar[0] = self.rho[self.Istar,0]
    #         self.Dstar[0] = 0.0
    #         self.Xstar[0] = self.X[self.Istar,0]
    #         self.Hperpstar[0] = self.Hperp[self.Istar,0]
    #         for i in range(1,self.Istar):
    #             n0 = self.Istar - i
    #             n = int(self.vmaxi[n0])
    #             I1 = np.argwhere(self.tv[n0,range(n)] > self.tstar)[-1]
    #             I2 = I1 + 1 #np.argwhere(self.tv[n0,range(n)] < self.tstar)[0]
    #             vi = np.squeeze(np.array([self.v[I1],self.v[I2]]))
    #             ti = np.squeeze(np.array([self.tv[n0,I1],self.tv[n0,I2]]))
    #             self.vstar[i] = np.interp(self.tstar,ti,vi)
    #             rhoi = np.squeeze(np.array([self.rho[n0,I1],self.rho[n0,I2]]))
    #             self.rhostar[i] = np.interp(self.vstar[i],vi,rhoi)
    #             rvi = np.squeeze(np.array([self.rv[n0,I1],self.rv[n0,I2]]))
    #             self.rstar[i] = np.interp(self.vstar[i],vi,rvi)
    #             Di = np.squeeze(np.array([self.D[n0,I1],self.D[n0,I2]]))
    #             self.Dstar[i] = np.interp(self.vstar[i],vi,Di)
    #             Xi = np.squeeze(np.array([self.X[n0,I1],self.X[n0,I2]]))
    #             self.Xstar[i] = np.interp(self.vstar[i],vi,Xi)
    #             Hperpi = np.squeeze(np.array([self.Hperp[n0,I1],self.Hperp[n0,I2]]))
    #             self.Hperpstar[i] = np.interp(self.vstar[i],vi,Hperpi)
    #         self.vmaxstar = self.vstar[self.Istar-1]
    #         return 0

    def get_dzdw(self, u=None, udot=None, up=None, A=None):
        return udot + up*(A - 1.0/u**2.0)/2.0
        
    def get_Chi2(self, Hz=None, D=None, rhoz=None, u=None, vzo=None, t0=None, NJ=None, udot=None, up=None, A = None):
        """
        The inputs H, D and u are functions of v. However vzo is the spline v(z) so we can interpolate directly
        """
        # Compute observables from CIVP soln
        current_F = self.get_PLC0_observables(vzo=vzo, D=D[:,0], A=A[:,0], Hz=Hz, u=u[:,0], udot=udot[:,0], up=up[:,0], rhoz=rhoz)
        chi2 = 0.0
        for s in self.my_F_data.keys(): # TODO: we might bail here if self.data is not set correctly
            chi2 += sum((self.my_F_data[s] - current_F[s])**2/(self.my_sF_data[s]**2))

        return chi2

    def get_PLC0_observables(self, vzo=None, D=None, A=None, Hz=None, u=None, udot=None, up=None, rhoz=None):
        """
        Here we compute observables on the PLC0

        :param vzo: interpolator for v(z) relation
        :param D: D(v)
        :param A: A(v)
        :param u: 1 + z(v)
        :param udot: dot{u}
        :param up: u'
        """
        #Get v(z) for interpolating
        z = u - 1.0
        vz = vzo(z)

        #Create empty dict to store current observables in
        obs_dict = {}
        if "dzdw" in self.data_lik:
            #Get dzdw(v)
            dzdw = self.get_dzdw(u = u, udot = udot, up = up, A = A)
            obs_dict["dzdw"] = uvs(vz, dzdw, k=3, s=0.0)(vzo(self.my_z_data["dzdw"]))
        if "D" in self.data_lik:
            obs_dict["D"] = uvs(vz, D, k=3, s=0.0)(vzo(self.my_z_data["D"]))
        if "H" in self.data_lik:
            obs_dict["H"] = uvs(self.z, Hz, k=3, s=0.0)(self.my_z_data["H"])
        if "rho" in self.data_lik:
            obs_dict["rho"] = uvs(self.z, rhoz, k=3, s=0.0)(self.my_z_data["rho"])
        # Add observables here
        return obs_dict

    def get_Dz_and_dzdwz(self,vzo=None, D=None, A=None, u=None, udot=None, up=None):
        z = u - 1.0
        vz = vzo(z)
        dzdw = self.get_dzdw(u=u, udot=udot, up=up, A=A)
        dzdwz = uvs(vz, dzdw, k=3, s=0.0)(vzo(self.z))
        Dz = uvs(vz, D, k=3, s=0.0)(vzo(self.z))
        return Dz, dzdwz

    def get_funcs(self):
        """
        Return quantities of interest
        """
        # Find index of w0 marking value closest to tfind
        I1 = np.argwhere(self.w0 >= self.tfind)[-1]
        I2 = np.argwhere(self.w0 < self.tfind)[0]
        #Choose whichever is closer
        if ( abs(self.w0[I1]-self.tfind) < abs(self.w0[I2] - self.tfind)):
            self.Istar = I1
        else:
            self.Istar = I2

        #Here we do the shear and curvature tests on two pncs
        umax = int(self.Istar)
        njf = int(self.vmaxi[umax]) #This is the max value of index on final pnc considered
        
        #All functions will be returned with the domain normalised between 0 and 1
        l = np.linspace(0, 1, self.Nret)
        #Curvetest

        #self.Kiraw = Ki
        try:
            T2io = uvs(self.v/self.v[-1], self.T2[:, 0], k=3, s=0.0)
            T2i = T2io(l)
        except:
            T2i = 0.0
            print "Failed at T2i"
        #T2f = self.curve_test(umax,self.NJ)
        #self.Kfraw = Kf
        try:
            T2fo = uvs(self.v[0:njf]/self.v[njf-1], self.T2[0:njf, umax], k=3, s=0.0)
            T2f = T2fo(l)
        except:
            T2f = 0.0
            print "Failed at T2f"
        #shear test
        #T1i = self.shear_test(0,self.NJ)
        try:
            T1io = uvs(self.v/self.v[-1], self.T1[:, 0], k=3, s=0.0)
            T1i = T1io(l)
        except:
            T1i = 0.0
            print "Failed at T1i"
        #T1f = self.shear_test(umax,self.NJ)
        try:
            T1fo = uvs(self.v[0:njf]/self.v[njf-1],self.T1[0:njf, umax],k=3,s=0.0)
            T1f = T1fo(l)
        except:
            T1f = 0.0
            print "Failed at T1f"
        #Get the LLTB consistency relation
        jmaxf = self.vmaxi[-1]
        try:
            LLTBConsi = uvs(self.v[0:jmaxf]/self.v[jmaxf-1],self.LLTBCon[0:jmaxf,0],k=3,s=0.0)(l)
        except:
            LLTBConsi = 0.0
            print "failed at LLTBConsi. jmaxf = ", jmaxf, " NI = ", self.NI
        try:
            LLTBConsf = uvs(self.v[0:jmaxf]/self.v[jmaxf-1],self.LLTBCon[0:jmaxf,-1],k=3,s=0.0)(l)
        except:
            LLTBConsf = 0.0
            print "failed at LLTBConsf. NI = ", self.NI
        try:
            Di = uvs(self.v/self.v[-1], self.D[:, 0], k=3, s=0.0)(l)
        except:
            Di = 0.0
            print "failed at Di"
        try:
            Df = uvs(self.v[0:jmaxf]/self.v[jmaxf-1], self.D[0:jmaxf,-1], k=3, s=0.0)(l)
        except:
            Df = 0.0
            print "failed at Df"
        try:
            Si = uvs(self.v/self.v[-1], self.S[:, 0], k=3, s=0.0)(l)
        except:
            Si = 0.0
            print "failed at Si"
        try:
            Sf = uvs(self.v[0:jmaxf]/self.v[jmaxf-1], self.S[0:jmaxf,-1], k=3, s=0.0)(l)
        except:
            Sf = 0.0
            print "failed at Sf"
        try:
            Qi = uvs(self.v/self.v[-1], self.Q[:, 0], k=3, s=0.0)(l)
        except:
            Qi = 0.0
            print "failed at Qi"
        try:
            Qf = uvs(self.v[0:jmaxf]/self.v[jmaxf-1], self.Q[0:jmaxf,-1], k=3, s=0.0)(l)
        except:
            Qf = 0.0
            print "failed at Qf"
        try:
            Ai = uvs(self.v/self.v[-1], self.A[:, 0], k=3, s=0.0)(l)
        except:
            Ai = 0.0
            print "failed at Ai"
        try:
            Af = uvs(self.v[0:jmaxf]/self.v[jmaxf-1], self.A[0:jmaxf,-1], k=3, s=0.0)(l)
        except:
            Af = 0.0
            print "failed at Af"
        try:
            Zi = uvs(self.v/self.v[-1], self.Z[:, 0], k=3, s=0.0)(l)
        except:
            Zi = 0.0
            print "failed at Zi"
        try:
            Zf = uvs(self.v[0:jmaxf]/self.v[jmaxf-1], self.Z[0:jmaxf,-1], k=3, s=0.0)(l)
        except:
            Zf = 0.0
            print "failed at Zf"
        try:
            Spi = uvs(self.v/self.v[-1], self.Sp[:, 0], k=3, s=0.0)(l)
        except:
            Spi = 0.0
            print "failed at Spi"
        try:
            Spf = uvs(self.v[0:jmaxf]/self.v[jmaxf-1], self.Sp[0:jmaxf,-1], k=3, s=0.0)(l)
        except:
            Spf = 0.0
            print "failed at Spf"
        try:
            Qpi = uvs(self.v/self.v[-1], self.Qp[:, 0], k=3, s=0.0)(l)
        except:
            Qpi = 0.0
            print "failed at Qpi"
        try:
            Qpf = uvs(self.v[0:jmaxf]/self.v[jmaxf-1], self.Qp[0:jmaxf,-1], k=3, s=0.0)(l)
        except:
            Qpf = 0.0
            print "failed at Qpf"
        try:
            Zpi = uvs(self.v/self.v[-1], self.Zp[:, 0], k=3, s=0.0)(l)
        except:
            Zpi = 0.0
            print "failed at Zpi"
        try:
            Zpf = uvs(self.v[0:jmaxf]/self.v[jmaxf-1], self.Zp[0:jmaxf,-1], k=3, s=0.0)(l)
        except:
            Zpf = 0.0
            print "failed at Zpf"
        try:
            ui = uvs(self.v/self.v[-1], self.u[:, 0], k=3, s=0.0)(l)
        except:
            ui = 0.0
            print "failed at ui"
        try:
            uf = uvs(self.v[0:jmaxf]/self.v[jmaxf-1], self.u[0:jmaxf,-1], k=3, s=0.0)(l)
        except:
            uf = 0.0
            print "failed at uf"
        try:
            upi = uvs(self.v/self.v[-1], self.up[:, 0], k=3, s=0.0)(l)
        except:
            upi = 0.0
            print "failed at upi"
        try:
            upf = uvs(self.v[0:jmaxf]/self.v[jmaxf-1], self.up[0:jmaxf,-1], k=3, s=0.0)(l)
        except:
            upf = 0.0
            print "failed at upf"
        try:
            uppi = uvs(self.v/self.v[-1], self.upp[:, 0], k=3, s=0.0)(l)
        except:
            uppi = 0.0
            print "failed at uppi"
        try:
            uppf = uvs(self.v[0:jmaxf]/self.v[jmaxf-1], self.upp[0:jmaxf,-1], k=3, s=0.0)(l)
        except:
            uppf = 0.0
            print "failed at uppf"
        try:
            udoti = uvs(self.v/self.v[-1], self.udot[:, 0], k=3, s=0.0)(l)
        except:
            udoti = 0.0
            print "failed at udoti"
        try:
            udotf = uvs(self.v[0:jmaxf]/self.v[jmaxf-1], self.udot[0:jmaxf,-1], k=3, s=0.0)(l)
        except:
            udotf = 0.0
            print "failed at udotf"
        try:
            rhoi = uvs(self.v/self.v[-1], self.rho[:, 0], k=3, s=0.0)(l)
        except:
            rhoi = 0.0
            print "failed at rhoi"
        try:
            rhof = uvs(self.v[0:jmaxf]/self.v[jmaxf-1], self.rho[0:jmaxf,-1], k=3, s=0.0)(l)
        except:
            rhof = 0.0
            print "failed at rhof"
        try:
            rhopi = uvs(self.v/self.v[-1], self.rhop[:, 0], k=3, s=0.0)(l)
        except:
            rhopi = 0.0
            print "failed at rhopi"
        try:
            rhopf = uvs(self.v[0:jmaxf]/self.v[jmaxf-1], self.rhop[0:jmaxf,-1], k=3, s=0.0)(l)
        except:
            rhopf = 0.0
            print "failed at rhopf"
        try:
            rhodoti = uvs(self.v/self.v[-1], self.rhodot[:, 0], k=3, s=0.0)(l)
        except:
            rhodoti = 0.0
            print "failed at rhodoti"
        try:
            rhodotf = uvs(self.v[0:jmaxf]/self.v[jmaxf-1], self.rhodot[0:jmaxf,-1], k=3, s=0.0)(l)
        except:
            rhodotf = 0.0
            print "failed at rhodotf"
        # #Get constant t slices
        # if F == 0:
        #     I = range(self.Istar)
        #     rmax = self.rstar[self.Istar-1]
        #     r = self.rstar[I]/rmax
        #     rhostar = np.interp(l,r,self.rhostar[I])
        #     Dstar = np.interp(l,r,self.Dstar[I])
        #     Dstar[0] = 0.0
        #     Xstar = np.interp(l,r,self.Xstar[I])
        #     Hperpstar = np.interp(l,r,self.Hperpstar[I])
        # else:
        #     rmax = self.rstar[0]
        #     rhostar = np.tile(self.rhostar[0],(self.Nret))
        #     Dstar = np.tile(self.Dstar[0],(self.Nret))
        #     Xstar = np.tile(self.Xstar[0],(self.Nret))
        #     Hperpstar = np.tile(self.Hperpstar[0],(self.Nret))
        return T1i, T1f,T2i,T2f,LLTBConsi,LLTBConsf, Di, Df, Si, Sf, Qi, Qf, Ai, Af, Zi, Zf, Spi, Spf, Qpi, Qpf, Zpi, \
               Zpf, ui, uf, upi, upf, uppi, uppf, udoti, udotf, rhoi, rhof, rhopi, rhopf, rhodoti, rhodotf, self.Dz, self.dzdwz
        
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
