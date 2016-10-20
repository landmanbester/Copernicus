# -*- coding: utf-8 -*-
"""
Created on Thu Aug 21 09:33:52 2014

@author: landman

This program simulates data for an LTB model

"""
import numpy as np
from scipy.integrate import quad
from scipy.interpolate import UnivariateSpline as uvs
import matplotlib.pyplot as plt
import matplotlib as mpl
from Copernicus.fortran_mods import CIVP


def affine_grid(Hz, rhoz, Lam, z, err, tmin):
    """
    Get data on regular spatial grid
    """
    # First find dimensionless density params
    Om0 = 8 * np.pi * rhoz[0] / (3 * Hz[0] ** 2)
    OL0 = Lam / (3 * Hz[0] ** 2)
    Ok0 = 1 - Om0 - OL0

    # Get t0
    t0f = lambda x, a, b, c, d: np.sqrt(x) / (d * np.sqrt(a + b * x + c * x ** 3))
    t0 = quad(t0f, 0, 1, args=(Om0, Ok0, OL0, Hz[0]))[0]

    # Set affine parameter vals
    dvo = uvs(z, 1 / ((1 + z) ** 2 * Hz), k=3, s=0.0)  # seems to be the most accurate way to do the numerical integration
    vzo = dvo.antiderivative()
    vz = vzo(z)
    vz[0] = 0.0

    # Compute grid sizes that gives num error od err
    NJ = int(np.ceil(vz[-1] / np.sqrt(err) + 1))
    NI = int(np.ceil(3.0 * (NJ - 1) * (t0 - tmin) / vz[-1] + 1))

    # Get functions on regular grid
    v = np.linspace(0, vz[-1], NJ)
    delv = (v[-1] - v[0]) / (NJ - 1)
    if delv > np.sqrt(err):  # A sanity check
        print 'delv > sqrt(err)'
    Ho = uvs(vz, Hz, s=0.0, k=3)
    H = Ho(v)
    rhoo = uvs(vz, rhoz, s=0.0, k=3)
    rho = rhoo(v)
    uo = uvs(vz, 1+z, s=0.0, k=3)
    u = uo(v)
    u[0] = 1.0
    return v, vzo, H, rho, u, NJ, NI, delv, Om0, OL0, Ok0, t0


def age_grid(NI, NJ, delv, t0, tmin):
    w0 = np.linspace(t0, tmin, NI)
    # self.w0 = w0
    delw = (w0[0] - w0[-1]) / (NI - 1)
    if delw / delv > 0.5:
        print "Warning CFL might be violated."
        # Set u grid
    w = np.tile(w0, (NJ, 1)).T
    return w, delw


def get_dzdw(u=None, udot=None, up=None, A=None):
    return udot + up * (A - 1.0 / u ** 2.0) / 2.0

if __name__=="__main__":
    # Load LTB z funcs
    z_funcs = np.load('/home/landman/Projects/CP_Dir/Processed_Data/LTB_z_funcs.npz')
    z = z_funcs['z']
    Hz = z_funcs['Hz']
    rhoz = z_funcs['rhoz']
    Dz1 = z_funcs['Dz']  #We can check to see that this is the same as CIVP version
    Lam = 0.0
    err = 1e-5
    tmin = 3.25
    zmax = z[-1]
    Np = zmax.size

    # Now get CIVP soln (this is mainly to get dzdw)
    # Set affine grid
    v, vzo, H, rho, u, NJ, NI, delv, Om0, OL0, Ok0, t0 = affine_grid(Hz, rhoz, Lam, z, err, tmin)

    # Set age grid
    w, delw = age_grid(NI, NJ, delv, t0, tmin)

    # Do CIVP integration
    D, S, Q, A, Z, rho, rhod, rhop, u, ud, up, upp, vmax, vmaxi, r, t, X, dXdr, drdv, drdvp, Sp, Qp, Zp, LLTBCon, Dww, Aw, T1, T2 = CIVP.solve(
        v, delv, w, delw, u, rho, Lam, NI, NJ)

    # Get dzdw and convert to function of redshift
    dzdw = get_dzdw(u=u, udot=ud, up=up, A=A)
    dzdwz = uvs(v,dzdw[:,0],k=3,s=0.0)(vzo(z))

    #Get D(z) for comparison
    Dz = uvs(v,D[:,0],k=3,s=0.0)(vzo(z))

    print "The two D(z) relations are identical: ", np.allclose(Dz, Dz1)

    # set z values for data points
    nD = 500
    nH = 50
    nrho = 60
    ndzdw = 10

    zD = np.sort(0.005 + np.random.ranf(nD) * (zmax - 0.005))
    zH = np.sort(np.random.ranf(nH) * zmax)
    zH[0] = 0
    zrho = np.sort(np.random.ranf(nrho)*zmax)
    zdzdw = 1.0 + np.sort(np.random.ranf(ndzdw) * zmax/2.0)

    # Set how the error grows
    alpha = 0.5
    errD = (1+zD)**alpha
    errH = (1+zH)**alpha
    errrho = (1 + zrho) ** alpha
    errdzdw = (1 + zdzdw) ** alpha

    # get functions values at data locations
    HzDat = uvs(z,Hz,k=3,s=0.0)(zH)
    DzDat = uvs(z, Dz, k=3, s=0.0)(zD)
    rhozDat = uvs(z, rhoz, k=3, s=0.0)(zrho)
    dzdwDat = uvs(v,dzdw[:,0],k=3,s=0.0)(vzo(zdzdw))

    # set number of trials (controls Gaussianity of distribution)
    N = 21
    delD = 0.05
    delH = 0.1
    delrho = 0.5
    deldzdw = 0.1

    # Do simulation
    SimD = np.zeros([N, nD])
    SimH = np.zeros([N, nH])
    Simrho = np.zeros([N, nrho])
    Simdzdw = np.zeros([N, ndzdw])
    zND = np.zeros(nD)
    zNH = np.zeros(nH)
    zNrho = np.zeros(nrho)
    zNdzdw = np.zeros(ndzdw)
    eyeND = np.eye(nD)
    eyeNH = np.eye(nH)
    eyeNrho = np.eye(nrho)
    eyeNdzdw = np.eye(ndzdw)

    for i in range(N):
        SimD[i,:] = DzDat + errD*delD*DzDat*np.random.multivariate_normal(zND,eyeND)
        SimH[i,:] = HzDat + errH*delH*HzDat*np.random.multivariate_normal(zNH,eyeNH)
        Simrho[i,:] = rhozDat + errrho*delrho*rhozDat*np.random.multivariate_normal(zNrho,eyeNrho)
        Simdzdw[i, :] = dzdwDat + errdzdw * deldzdw * dzdwDat * np.random.multivariate_normal(zNdzdw, eyeNdzdw)

    #sort columns in ascending order
    SimD.sort(axis=0)
    SimH.sort(axis=0)
    Simrho.sort(axis=0)
    Simdzdw.sort(axis=0)
    #get mean and 1-sigma error bars
    meanD = SimD[int(N/2),:]
    sigD = SimD[int(0.16*N),:]
    meandzdw = Simdzdw[int(N/2),:]
    sigdzdw = Simdzdw[int(0.16*N),:]
    meanH = SimH[int(N/2),:]
    sigH = SimH[int(0.16*N),:]
    meanrho = Simrho[int(N/2),:]
    sigrho = Simrho[int(0.16*N),:]

    ##save data
    savedzdw = np.column_stack((zdzdw,meandzdw,(meandzdw-sigdzdw)))
    dzdwf = open('/home/landman/Projects/CP_LTB/Data/Simdzdw.txt','w')
    np.savetxt(dzdwf,savedzdw,fmt='%s')
    dzdwf.close()
    saveH = np.column_stack((zH,meanH,(meanH-sigH)))
    Hf = open('/home/landman/Projects/CP_LTB/Data/SimH.txt','w')
    np.savetxt(Hf,saveH,fmt='%s')
    Hf.close()
    saverho = np.column_stack((zrho, meanrho, sigrho))
    rhof = open('/home/landman/Projects/CP_LTB/Data/Simrho.txt', 'w')
    np.savetxt(rhof, saverho, fmt='%s')
    rhof.close()
    saveD = np.column_stack((zD,meanD,(meanD-sigD)))
    Df = open('/home/landman/Projects/CP_LTB/Data/SimD.txt','w')
    np.savetxt(Df,saveD,fmt='%s')
    Df.close()

    mpl.rcParams.update({'font.size': 20, 'font.family': 'serif'})

    plt.figure('Dsim')
    plt.errorbar(zD,meanD,(meanD-sigD),fmt='xr',alpha=0.25)
    plt.plot(z,Dz,'k',linewidth=2,label=r'$D(z) from CIVP$')
    plt.plot(z, Dz1, 'b', linewidth=2, label=r'$D(z) from LTB$')
    plt.legend()
    plt.show()

    plt.figure('Hsim')
    plt.errorbar(zH,meanH,(meanH-sigH),fmt='xr')
    plt.plot(z,Hz,linewidth=2)
    plt.show()

    plt.figure('rhosim')
    plt.errorbar(zrho, meanrho, sigrho, fmt='xr')
    plt.plot(z, rhoz, linewidth=2)
    plt.show()

    plt.figure('dzdwsim')
    plt.errorbar(zdzdw,meandzdw,(meandzdw-sigdzdw),fmt='xr')
    plt.plot(z,dzdwz,linewidth=2)
    plt.show()
