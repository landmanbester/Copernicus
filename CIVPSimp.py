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
import sys
#sys.path.insert(0, '/home/bester/Algorithm') #On cluster
sys.path.insert(0, '/home/landman/Algorithm') #At home PC
#sys.path.insert(0, 'C:\Users\BMAX\Documents\Algorithm') #At home Laptop
import time
from numpy import inf,nan_to_num,log10, exp, size, dot, log, eye, diag, tile,ceil, zeros, linspace, sqrt, floor, argwhere, arange, pi, array, interp,squeeze, loadtxt, reshape
from numpy.linalg import cholesky, solve, inv, slogdet, LinAlgError, eigh
from numpy.random import multivariate_normal as mvn
from numpy.random import randn, random
from scipy.integrate import odeint,quad
from scipy.interpolate import UnivariateSpline as uvs
import matplotlib.pyplot as plt
import matplotlib as mpl
#from genFLRW import FLRW
import CIVP2
import FSupport as FS

class SSU(object):
    def __init__(self,zmax,tmin,np,err,nret):
        """
        This is the main untility class. 
        Input:  zmax = max redshift
                tmin = min time to integrate up to
                np = The number of redshift points to use for GPR
                err = the target error of the numerical integration scheme
                XH = The optimised hyperparameter values for GP_H
                Xrho = The optimised hyperparameter values for GP_rho
                sigmaLam = The variance of the prior over Lambda
        """
        #First set attributes that will remain fixed
        #Set number of spatial grid points
        self.np = np
        self.z = linspace(0,zmax,np)
        self.uz = 1 + self.z

        #Set function to get t0
        self.t0f = lambda x,a,b,c,d: sqrt(x)/(d*sqrt(a + b*x + c*x**3))

        #Set minimum time to integrate to
        self.tmin = tmin
        self.tfind = tmin + 0.1

        #Set number of spatial grid points at which to return quantities of interrest
        self.nret = nret

        #Set target error
        self.err = err #(fixed)

        #Max iterations to find vmaxi
        self.nitmax = 1000
        
        #Set initial conditions for hypersurface equations
        self.y0 = zeros(5)
        self.y0[1] = 1.0
        self.y0[3] = 1.0

    def doCIVP(self,Hz,rhoz,Lam):
        """
        This is the MCMC step for samples of rho and H. If redshift drift data are included it
        should also do the joint MCMC with samples of Lambda.
        """
        #Set up spatial grid
        v,vzo,H,rho,u,NJ,NI,delv,Om0,OL0,Ok0,t0 = self.affine_grid(Hz,rhoz,Lam)
        #Set temporal grid
        w, delw = self.age_grid(NI,NJ,delv,t0)
        #Do integration on initial PLC
        D, S, Q, A, Z, udot, rhodot, up, rhop, upp = self.evaluate(rho,u,v,NJ,Lam)
        #Accept sample
        self.accept(NJ,NI,delw,delv,v,w,D,u,rho,Lam,t0,Om0,OL0,Ok0,vzo)
        return

        
    def accept(self,NJ,NI,delw,delv,v,w,Di,ui,rhoi,Lam,t0,Om0,OL0,Ok0,vzo):
        self.NI = NI
        self.NJ = NJ
        self.vmax = zeros(NI)   #max radial extent on each PLC
        self.vmaxi = zeros(NI)  #index of max radial extent
        self.vmaxi[:] = int(NJ)
        self.vmax[0] = v[-1]
        #Do CIVP integration
        self.D,self.S,self.Q,self.A,self.Z,self.rho,self.u,self.up,self.upp,self.ud,self.rhod,self.rhop = self.integrate(NJ,NI,delw,delv,v,w,Di,ui,rhoi,Lam)
        #Get D(z),mu(z) and dzdw(z)
        self.Dz,self.muz,self.dzdw = self.get_PLC0_observables(vzo,self.D[0,:],self.A[0,:],self.u[0,:],self.ud[0,:],self.up[0,:])
        self.Hpar = self.up/self.u**2    
        self.Hperp = (self.u*self.Q - self.S/(2.0*self.u) + self.u*self.A*self.S/2.0)/self.D
        self.Hperp[:,0] = self.Hpar[:,0]
        self.t0 = t0
        self.H0 = self.Hpar[0,0]
        self.Om0 = Om0
        self.OL0 = OL0
        self.Ok0 = Ok0
        self.v = v
        self.Lam = Lam
        #Check LLTB consistency 
        self.check_LLTB(self.D,self.S,self.Q,self.A,self.Z,self.u,self.rho,delw,self.Lam,NI,NJ)
        self.transform()
        self.get_tslice()
        return

    def get_age(self,Om0,Ok0,OL0,H0):
        return quad(self.t0f,0,1,args=(Om0,Ok0,OL0,H0))[0]
        
    def affine_grid(self,Hz,rhoz,Lam):
        """
        Get data on regular spatial grid
        """
        #First find dimensionless density params
        Om0 = 8*pi*rhoz[0]/(3*Hz[0]**2)
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
        NJ = int(ceil(vz[-1]/sqrt(self.err) + 1))
        NI = int(ceil(3.0*(NJ - 1)*(t0 - self.tmin)/vz[-1] + 1))
        #Get functions on regular grid
        v = linspace(0,vz[-1],NJ)
        delv = (v[-1] - v[0])/(NJ-1)
        if delv > sqrt(self.err):
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
        w0 = linspace(t0,self.tmin,NI)
        self.w0 = w0
        delw = (w0[0] - w0[-1])/(NI-1)
        if delw/delv > 0.5:
            print "Warning CFL might be violated." 
        #Set u grid
        w = tile(w0,(NJ,1)).T
        return w, delw

    def hypeq(self,y,v,rhoo,uo,Lambda):
        dy = zeros(5)
        rho = rhoo(v)
        u = uo(v)
        dy[0] = y[1]
        dy[1] = -8.0*pi*u**2*rho*y[0]/2.0
        if y[0] == 0.0:
            dy[2] = 0.0
        else:
            dy[2] = (1.0 - y[0]*y[1]*y[4] - 2.0*y[2]*y[1] - y[3]*y[1]**2 + 4.0*pi*rho*y[0]**2*(y[3]*u**2 - 1.0) - Lambda*y[0]**2)/(2*y[0])
        dy[3] = y[4]
        if y[0] == 0.0:
            dy[4] = 8.0*pi*rho/3.0- 2.0*Lambda/3.0 
        else:
            dy[4] = 8.0*pi*rho + 4.0*y[2]*y[1]/y[0]**2 + 2.0*y[3]*y[1]**2/y[0]**2 - 2.0/y[0]**2
        return dy

    def get_App(self,rho,Lam,D,S,Q,A):
        App = 8.0*pi*rho + 4.0*Q*S/D**2 + 2.0*A*S**2/D**2 - 2.0/D**2
        App[:,0] = 8.0*pi*rho[:,0]/3.0- 2.0*Lam/3.0
        return App

    def dotu(self,u,A,up,Ap):
        return ((1.0/u**2 - A)*up - Ap*u)/2.0

    def dotrho(self,rho,u,rhop,up,D,Dp,dotD,A,vmaxi):
        rhodot = zeros(vmaxi)
        rhodot[0] = -3.0*rho[0]*up[0]
        rhodot[1::] = rho[1::]*(-up[1::]/u[1::]**3 - 2.0*dotD[1::]/D[1::] + Dp[1::]*(1.0/u[1::]**2 - A[1::])/D[1::]) + rhop[1::]*(1.0/u[1::]**2.0 - A[1::])/2.0
        return rhodot    

    def evaluate(self,rho,u,v,vmaxi,Lam):
        """
        This functions evaluates CIVP variables on a PLC from the values of rho and u on that PLC
        """
        #Interpolate u and rho
        uo = uvs(v,u,s=0.0,k=5)
        rhoo = uvs(v,rho,s=0.0,k=5)
        #Solve hyp equations
        y = odeint(self.hypeq,self.y0,v,args=(rhoo,uo,Lam),atol=0.01*self.err,rtol=0.01*self.err)
        #Get spatial derivatives of rho and u
        upo = uo.derivative()
        uppo = uo.derivative(2)
        up = upo(v)
        upp = uppo(v)
        rhopo = rhoo.derivative()
        rhop = rhopo(v)
        #Get udot and rhodot corresponding to these solutions
        ud = self.dotu(u,y[:,3],up,y[:,4])     
        rhod = self.dotrho(rho,u,rhop,up,y[:,0],y[:,1],y[:,2],y[:,3],vmaxi)
        return y[:,0], y[:,1], y[:,2], y[:,3], y[:,4], ud, rhod, up, rhop, upp

    def integrate(self,NJ,NI,delw,delv,v,w,D,u,rho,Lam):
        r,v1,rhof,W,v0,v1u,v1nu,v12nu,S,T,RR,rhod,rhop,self.vmaxi = CIVP2.solve(v,w,0.0025,0.005,D,-u,rho,Lam)
        A = zeros([NI,NJ])        
        A[:,1::] = 1.0 + W[:,1::]/r[:,1::]
        A[:,0] = 1.0
        Z = zeros([NI,NJ])
        Z[:,1::] = T[:,1::]/r[:,1::] - W[:,1::]*S[:,1::]/r[:,1::]**2
        Z[:,0] = 0.0
        self.v1 = v1
        self.v1nu=v1nu
        self.v1u = v1u
        self.delnu=delv
        self.delu = delw
        self.RR = RR
        return r,S,-RR,A,Z,rhof,-v1,-v1nu,-v12nu,v1u,-rhod,rhop

    def dprimecoef(self,u,A):
        return (A - 1/u**2)/2
    
    def dcoef(self,du,u,dA):
        return (du/u**3 + dA/2)    

    def F_d(self,n,j):
        return self.d[n,j]*(self.A[n,j] - 1.0/self.v1[n,j]**2)/2.0

    def transform(self):
        #get dt components
        self.dtdw = (self.A*self.v1**2 + 1)/(-2*self.v1)
        self.dtdv = self.v1
        self.dwdt = -self.v1
        self.dvdt = (self.A*self.v1**2 - 1)/(-2*self.v1)
        #set initial d data
        self.d = zeros([self.NI,self.NJ])
        self.c = zeros([self.NI,self.NJ])
        self.b = zeros([self.NI,self.NJ])
        self.a = zeros([self.NI,self.NJ])
        self.d[0,:] = -self.v1nu[0,:]*self.D[0,:] -self.v1[0,:]*self.S[0,:]
        self.c[0,:] = self.d[0,:]*(self.A[0,:]*self.v1[0,:]**2-1)/(2*self.v1[0,:]**2)
        self.a[0,:] = -self.v1[0,:]**2/self.d[0,:]
        self.b[0,:] = (1+self.A[0,:]*self.v1[0,:]**2)/(2*self.d[0,:])
        #Get dprime and ddot to evaluate dXdr below
        self.F = zeros([self.NI,self.NJ])
        self.ddot = zeros([self.NI,self.NJ])
        self.dprime = zeros([self.NI,self.NJ])
        self.F[0,:] = self.d[0,:]*(self.A[0,:] - 1.0/self.v1[0,:]**2)/2.0
        self.ddot[0,0] = (-self.F[0,2]/2.0 + 2.0*self.F[0,1] - 3.0*self.F[0,0]/2.0)/self.delnu
        self.ddot[0,1:self.NJ-1] = (self.F[0,2:self.NJ] - self.F[0,0:self.NJ-2])/(2*self.delnu)
        self.ddot[0,self.NJ-1] = (3.0*self.F[0,self.NJ-1]/2.0 - 2.0*self.F[0,self.NJ-2] + self.F[0,self.NJ-3]/2.0)/self.delnu        
        self.dprime[0,0] = (-self.d[0,2]/2.0 + 2.0*self.d[0,1] - 3.0*self.d[0,0]/2.0)/self.delnu
        self.dprime[0,1:self.NJ-1] = (self.d[0,2:self.NJ] - self.d[0,0:self.NJ-2])/(2*self.delnu)
        self.dprime[0,self.NJ-1] = (3.0*self.d[0,self.NJ-1]/2.0 - 2.0*self.d[0,self.NJ-2] + self.d[0,self.NJ-3]/2.0)/self.delnu
        self.X = zeros([self.NI,self.NJ])
        self.X[0,:] = -self.v1[0,:]/self.d[0,:]
        self.tv = zeros([self.NI,self.NJ])
        self.rv = zeros([self.NI,self.NJ])
        dto = uvs(self.v,self.v1[0,:],k=3,s=0.0)
        self.tv[0,:] = self.w0[0] + dto.antiderivative()(self.v)
        dro = uvs(self.v,self.d[0,:],k=3,s=0.0)
        self.rv[0,:] = dro.antiderivative()(self.v)
        for n in range(self.NI-1):
            jmax = int(self.vmaxi[n+1])
            #Use forward differences to get value at origin
            self.d[n+1,0] = self.d[n,0] + self.delu*(-self.F_d(n,2)/2.0 + 2.0*self.F_d(n,1) - 3.0*self.F_d(n,0)/2.0)/self.delnu
            arrup = arange(2,jmax)
            arrmid = arange(1,jmax-1)
            arrdown = arange(0,jmax-2)
            self.d[n+1,arrmid] = (self.d[n,arrup] + self.d[n,arrdown])/2.0 + self.delu*(self.F_d(n,arrup) - self.F_d(n,arrdown))/(2*self.delnu)
            #Use backward differences to get value at jmax
            self.d[n+1,jmax-1] = self.d[n,jmax-1] + self.delu*(3.0*self.F_d(n,jmax-1)/2.0 - 2*self.F_d(n,jmax-2) + self.F_d(n,jmax-3)/2.0)/self.delnu
            #get remaining transformation components
            self.c[n+1,0:jmax] = self.d[n+1,0:jmax]*(self.A[n+1,0:jmax]*self.v1[n+1,0:jmax]**2-1.0)/(2.0*self.v1[n+1,0:jmax]**2.0)
            self.a[n+1,0:jmax] = -self.v1[n+1,0:jmax]**2/self.d[n+1,0:jmax]
            self.b[n+1,0:jmax] = (1.0+self.A[n+1,0:jmax]*self.v1[n+1,0:jmax]**2.0)/(2.0*self.d[n+1,0:jmax])
            #Get dprime and ddot to evaluate dXdr below
            self.F[n+1,0:jmax] = self.d[n+1,0:jmax]*(self.A[n+1,0:jmax] - 1.0/self.v1[n+1,0:jmax]**2)/2.0
            self.ddot[n+1,0] = (-self.F[n+1,2]/2.0 + 2.0*self.F[n+1,1] - 3.0*self.F[n+1,0]/2.0)/self.delnu
            self.ddot[n+1,arrmid] = (self.F[n+1,arrup] - self.F[n+1,arrdown])/(2*self.delnu)
            self.ddot[n+1,jmax-1] = (3.0*self.F[n+1,jmax-1]/2.0 - 2.0*self.F[n+1,jmax-2] + self.F[n+1,jmax-3]/2.0)/self.delnu
            self.dprime[n+1,0] = (-self.d[n+1,2]/2.0 + 2.0*self.d[n+1,1] - 3.0*self.d[n+1,0]/2.0)/self.delnu
            self.dprime[n+1,arrmid] = (self.d[n+1,arrup] - self.d[n+1,arrdown])/(2*self.delnu)
            self.dprime[n+1,jmax-1] = (3.0*self.d[n+1,jmax-1]/2.0 - 2.0*self.d[n+1,jmax-2] + self.d[n+1,jmax-3]/2.0)/self.delnu
            self.X[n+1,0:jmax] = -self.v1[n+1,0:jmax]/self.d[n+1,0:jmax]
#            self.dXdr[n+1,0:jmax] = self.a[n+1,0:jmax]*(-self.v1u[n+1,0:jmax]/self.d[n+1,0:jmax] + self.v1[n+1,0:jmax]*self.ddot[n+1,0:jmax]/self.d[n+1,0:jmax]**2) + self.b[n+1,0:jmax]*(-self.v1nu[n+1,0:jmax]/self.d[n+1,0:jmax] + self.v1[n+1,0:jmax]*self.dprime[n+1,0:jmax]/self.d[n+1,0:jmax]**2)
#            self.Hr[n+1,0:jmax] = self.dXdr[n+1,0:jmax]/self.X[n+1,0:jmax]
#            self.dDdr[n+1,0:jmax] = self.a[n+1,0:jmax]*self.RR[n+1,0:jmax] + self.b[n+1,0:jmax]*self.S[n+1,0:jmax]
            #Get t(v) and r(v)
            dto = uvs(self.v[0:jmax],self.v1[n+1,0:jmax],k=3,s=0.0)
            self.tv[n+1,0:jmax] =  self.w0[n+1] + dto.antiderivative()(self.v[0:jmax])
            dro = uvs(self.v[0:jmax],self.d[n+1,0:jmax],k=3,s=0.0)
            self.rv[n+1,0:jmax] = dro.antiderivative()(self.v[0:jmax])
        return              

    def get_tslice(self):
        self.Istar = argwhere(self.w0 >= self.tmin)[-1]
        #get values on C
        self.tstar = self.w0[self.Istar]
        self.vstar = zeros(self.NI)
        self.rstar = zeros(self.NI)
        self.rhostar = zeros(self.NI)
        self.Dstar = zeros(self.NI)
        self.Xstar = zeros(self.NI)
        self.Hperpstar = zeros(self.NI)
        self.vstar[0] = 0.0
        self.rstar[0] = 0.0
        self.rhostar[0] = self.rho[self.Istar,0]
        self.Dstar[0] = 0.0
        self.Xstar[0] = self.X[self.Istar,0]
        self.Hperpstar[0] = self.Hperp[self.Istar,0]
        for i in range(1,self.Istar):
            n0 = self.Istar - i
            n = int(self.vmaxi[n0])
            I1 = argwhere(self.tv[n0,range(n)] > self.tstar)[-1]
            I2 = I1 + 1 #argwhere(self.tv[n0,range(n)] < self.tstar)[0]
            vi = squeeze(array([self.v[I1],self.v[I2]]))
            ti = squeeze(array([self.tv[n0,I1],self.tv[n0,I2]]))
            self.vstar[i] = interp(self.tstar,ti,vi)
            rhoi = squeeze(array([self.rho[n0,I1],self.rho[n0,I2]]))
            self.rhostar[i] = interp(self.vstar[i],vi,rhoi)
            rvi = squeeze(array([self.rv[n0,I1],self.rv[n0,I2]]))
            self.rstar[i] = interp(self.vstar[i],vi,rvi)
            Di = squeeze(array([self.D[n0,I1],self.D[n0,I2]]))
            self.Dstar[i] = interp(self.vstar[i],vi,Di)                
            Xi = squeeze(array([self.X[n0,I1],self.X[n0,I2]]))
            self.Xstar[i] = interp(self.vstar[i],vi,Xi)
            Hperpi = squeeze(array([self.Hperp[n0,I1],self.Hperp[n0,I2]]))
            self.Hperpstar[i] = interp(self.vstar[i],vi,Hperpi)                
        self.vmaxstar = self.vstar[self.Istar-1]                 
        return

    def get_vmaxi(self,A,i):
        #Initial guess (Note the dirty fix on the index of Ap)
        vp = self.vmax[i-1] - 0.5*A[self.vmaxi[i-1]-1]*self.delw
        #Place holder
        vprev = 0
        #Counter
        s = 0
        #Iterate to find vmax[i]
        while (abs(vp-vprev)>1e-5 and s < self.nitmax):
            vprev = vp
            jmax = int(floor(vp/self.delv + 1.0))             
            if (jmax < 2):
                #Flag if causal horizon reached
                print "Warning causal horizon reached"
                jmax = 2
            elif jmax > self.NJ:
                #Flag for unexpected behaviour
                print "Something went wrong, got jmax > NJ"
                jmax = self.NJ-1
            #Interpolate to find Ap (since vp is not necessarily on a grid point)
            Ap = A[jmax-2] + (vp-self.v[jmax-1])*(A[jmax-1] - A[jmax-2])/self.delv
            vp = self.vmax[i-1] - 0.5*Ap*self.delw
            s += 1
        if (s >= self.nitmax):
            print "Warning PNC cut-off did not converge"
        self.vmax[i] = vp
        self.vmaxi[i] = int(jmax)
        return

    def shear_test(self,i,NJ):
        n = int(self.vmaxi[i])
        tmp = zeros(NJ)
        tmp[0:n] = 1.0 - self.Hperp[i,0:n]/self.Hpar[i,0:n]
        return tmp

    def curve_test(self,i,NJ):
        """
        The curvature test on PNC i. Returns z and K on PNC i
        """
        n = int(self.vmaxi[i])
        #Set redshift
        u = self.u[i,0:n]
        up = self.up[i,0:n]
        upp = self.upp[i,0:n]
        #Get D, D' and D''
        D = self.D[i,0:n]
        Dp = self.S[i,0:n]
        Dpp = -8.0*pi*u**2.0*self.rho[i,0:n]*D/2.0
        #Get H and dHz
        H = self.Hpar[i,0:n]
        dH = (upp/u**2.0 - 2.0*up**2/u**3)/up
        #Get dDz
        dD = Dp/up
        #Get d2Dz
        d2D = (Dpp/up - Dp*upp/up**2)/up
        tmp = zeros(NJ)
        if i==0:
            tmp = 1.0 + H**2.0*(u**2.0*(D*d2D - dD**2.0) - D**2.0) + u*H*dH*D*(u*dD + D)
        else:
            tmp[0:n] = 1.0 + H**2.0*(u**2.0*(D*d2D - dD**2.0) - D**2.0) + u*H*dH*D*(u*dD + D)
        return tmp

    def get_dzdw(self,u,udot,up,A):
        return udot + up*(A - 1.0/u**2.0)/2.0

    def check_LLTB(self,D,S,Q,A,Z,u,rho,delw,Lam,NI,NJ):
        #Get App
        App = self.get_App(rho,Lam,D,S,Q,A)
        #To store w derivs
        Dww = zeros([NI,NJ])
        Aw = zeros([NI,NJ])
        #To store consistency rel
        self.LLTBCon = zeros([NI,NJ])
        jmax = self.vmaxi[-1]
        for i in range(jmax):
            if (i==0):
                Dww[:,i] = 0.0
                Aw[:,i] = 0.0
            else:
                #Get Dww
                Dww[:,i] = FS.dd5f1d(D[:,i],-delw,NI,NI)
                #Get Aw
                Aw[:,i] = FS.d5f1d(A[:,i],-delw,NI,NI)
            self.Dww = Dww
            self.Aw = Aw
            self.LLTBCon[:,i] = 0.5*A[:,i]*App[:,i]*D[:,i] - 2*Dww[:,i] + Z[:,i]*Q[:,i] - S[:,i]*Aw[:,i] + A[:,i]*Z[:,i]*S[:,i] - 0.25*8*pi*rho[:,i]*D[:,i]*(1/u[:,i]**2 + u[:,i]**2*A[:,i]**2) + Lam*A[:,i]*D[:,i]
        return

    def get_PLC0_observables(self,vzo,D,A,u,udot,up):
        #Get dzdw(v)
        dzdw = self.get_dzdw(u,udot,up,A)
        #Get mu(v)
        #mu = zeros(self.NJ)
        mu = zeros(self.NJ)
        mu[1::] = 5*log10(1e8*u[1::]**2*D[1::])
        mu[0] = -1e-15  #Should be close enough to -inf
        #Convert to functions of z
        z = u-1
        vz = vzo(z)
        Dz = uvs(vz,D,k=3,s=0.0)(vzo(self.z))
        muz=zeros(self.np)
        muz[0] = -inf
        muz[1::] = uvs(vz,mu,k=3,s=0.0)(vzo(self.z[1::]))
        dzdwz = uvs(vz,dzdw,k=3,s=0.0)(vzo(self.z))
        return Dz, muz, dzdwz

    def get_funcs(self):
        """
        Return quantities of interest
        """
        #Here we do the shear and curvature tests on two pncs
        umax = int(self.Istar)
        njf = int(self.vmaxi[umax]) #This is the max value of index on final pnc considered
        
        #All functions will be returned with the domain normalised between 0 and 1
        l = linspace(0,1,self.nret)
        #Curvetest
        T2i = self.curve_test(0,self.NJ)
        self.Kiraw = T2i
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
        I = range(self.Istar)
        rmax = self.rstar[self.Istar-1]
        r = self.rstar[I]/rmax
        rhostar = interp(l,r,self.rhostar[I])
        Dstar = interp(l,r,self.Dstar[I])
        Dstar[0] = 0.0
        Xstar = interp(l,r,self.Xstar[I])
        Hperpstar = interp(l,r,self.Hperpstar[I])
        return self.Dz,self.muz,self.dzdw,T1i, T1f,T2i,T2f,LLTBConsi,LLTBConsf,rhostar,Dstar,Xstar,Hperpstar,rmax,self.Om0,self.OL0,self.t0