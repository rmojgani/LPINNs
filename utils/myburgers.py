#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 09:27:31 2022

@author: rm99

"""
import numpy as np
from scipy.integrate import odeint

def myburgers(nu, C, N, M, tmax = 1.0):
    
    L = 2*np.pi
    dx = L/N
    x = np.arange(0,L,dx)
    
    kappa = 2 * np.pi*np.fft.fftfreq(N,d=dx)
    
    xmin, xmax = 0, 2*np.pi
    # u0 = 1/np.cosh(x)
    u0 = 1.0*np.sin(x*2*np.pi/(xmax-xmin)) + C
    
    # dt = 0.025
    
    dt = tmax/M
    t = np.arange(0,M*dt,dt)

    u = odeint(rhsBurgers, u0 , t , args=(kappa,nu)).T
    
    return u

def rhsBurgers(u,t,kappa,nu):
  uhat = np.fft.fft(u)
  d_uhat = (1j)*kappa*uhat
  dd_uhat = -np.power(kappa,2)*uhat
  d_u = np.fft.ifft(d_uhat)
  dd_u = np.fft.ifft(dd_uhat)
  du_dt = -u * d_u + nu * dd_u
  return du_dt.real