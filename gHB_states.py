#!/usr/bin/env python
# coding: utf-8

# In[31]:


# Construction of gHB states for joint-estimation of phase, loss, and diffusion

import numpy as np
import scipy as sc
import sympy as sp
import mpmath as mp
import cvxpy as cp
from numpy.linalg import matrix_rank
from numpy.linalg import inv
from numpy import pi
from sympy.physics.quantum import TensorProduct
from numpy.linalg import matrix_rank
from scipy.special import factorial
from mpmath import mp, hyp2f1
from functools import reduce

# weight matrix

def weightmatrix(y):
    
    m3=sp.Matrix([[2*y, 0], [0, 2*(1-y)]])
    return np.array(m3).astype(np.float64)

# Kravchuck coefficients

def krav(i, T, n):
    mp.dps = 100 
    mp.pretty = True
    return (-1)**n*np.sqrt(2**(-T)*factorial(T)/(factorial(n)*factorial(T-n))*factorial(T)/(factorial(i)*factorial(T-i)))*hyp2f1(-n, -i, -T, 2, maxterms=200000, maxprec=100000, zeroprec=1000)

# loss coefficients

def loss(i,T,k,l,eta_a,eta_b):
    return factorial(i)/(factorial(k)*factorial(i-k))*factorial(T-i)/(factorial(l)*factorial(T-i-l))*eta_a**(i-k)*(1-eta_a)**k*eta_b**(T-i-l)*(1-eta_b)**l


phi1,eta_a,eta_b,phi,delta,theta,n,avg,k,r,chi = sp.symbols('phi1 eta_a eta_b,phi,delta,theta,n,avg,k,r,chi',real=True,positive=True)


# full rank thermal state for regularization

def thermalstate(T,avg):

    dim=(T+1)**2
    mat3=sp.eye(dim)
    norm=sp.Sum(sp.Pow(avg,k)/sp.Pow(1+avg,k+1), (k, 0, dim-1)).doit()
    for i in range(dim):
        mat3[i,i]=(1/norm)*sp.Pow(avg,i)/sp.Pow(1+avg,i+1)
        
    return np.array(mat3).astype(np.complex128)


# regularized full rank gHB state after phase, loss, and phase diffusion operations

def phasediffetastatereg(T,n,eta_a,eta_b,phi,delta,avg,epsilon):
    
    phasediff1=np.zeros(((T+1)**2,(T+1)**2),dtype=np.complex128)
    
    for k in range(0,T+1):
        for l in range(0,(T-k)+1):
            for i in range(k, (T-l)+1):
                for j in range(k, (T-l)+1):
                    phasediff1[(i-k)*(T+1)+(T-i-l),(j-k)*(T+1)+(T-j-l)]+=krav(i,T,n)*krav(j,T,n)*np.exp(1j*phi*(i-j)-(delta**2*(i-j)**2)/2)*np.sqrt(loss(i,T,k,l,eta_a,eta_b))*np.sqrt(loss(j,T,k,l,eta_a,eta_b))
    
    return (1-epsilon)*phasediff1+epsilon*thermalstate(T,avg)

# loss coefficients in sympy

def losssp(i,T,k,l,eta_a,eta_b):
    return sp.factorial(i)/(sp.factorial(k)*sp.factorial(i-k))*sp.factorial(T-i)/(sp.factorial(l)*sp.factorial(T-i-l))*sp.Pow(eta_a,i-k)*sp.Pow(1-eta_a,k)*sp.Pow(eta_b,T-i-l)*sp.Pow(1-eta_b,l)

def dercoefficient(T,phi2,delta2,eta_a1,eta_b1,i,j,k,l):
    
    coeff=sp.exp(sp.I*phi*(i-j)-(delta**2*(i-j)**2)/2)*sp.sqrt(losssp(i,T,k,l,eta_a,eta_b))*sp.sqrt(losssp(j,T,k,l,eta_a,eta_b))
    derphi=sp.diff(coeff, phi)
    derphi_sub=derphi.subs([(phi, phi2), (delta, delta2), (eta_a, eta_a1), (eta_b, eta_b1)])
    dereta_a=sp.diff(coeff, eta_a)
    dereta_a_sub=dereta_a.subs([(phi, phi2), (delta, delta2), (eta_a, eta_a1), (eta_b, eta_b1)])
    derdelta=sp.diff(coeff, delta)
    derdelta_sub=derdelta.subs([(phi, phi2), (delta, delta2), (eta_a, eta_a1), (eta_b, eta_b1)])
    npderphi=sp.lambdify((phi,delta,eta_a,eta_b), derphi_sub, 'numpy')
    npdereta_a=sp.lambdify((phi,delta,eta_a,eta_b), dereta_a_sub, 'numpy')
    npderdelta=sp.lambdify((phi,delta,eta_a,eta_b), derdelta_sub, 'numpy')
    
    
    return [npderphi(phi2,delta2,eta_a1,eta_b1),npdereta_a(phi2,delta2,eta_a1,eta_b1),npderdelta(phi2,delta2,eta_a1,eta_b1)]

def derphasediffetastatereg(T,n,eta_a1,eta_b1,phi2,delta2,avg,epsilon):
    
    derphase=np.zeros(((T+1)**2,(T+1)**2),dtype=np.complex128)
    dereta_a=np.zeros(((T+1)**2,(T+1)**2),dtype=np.complex128)
    derdelta=np.zeros(((T+1)**2,(T+1)**2),dtype=np.complex128)
    
    for k in range(0,T+1):
        for l in range(0,(T-k)+1):
            for i in range(k, (T-l)+1):
                for j in range(k, (T-l)+1):
                    derphase[(i-k)*(T+1)+(T-i-l),(j-k)*(T+1)+(T-j-l)]+=krav(i,T,n)*krav(j,T,n)*dercoefficient(T,phi2,delta2,eta_a1,eta_b1,i,j,k,l)[0]
                    dereta_a[(i-k)*(T+1)+(T-i-l),(j-k)*(T+1)+(T-j-l)]+=krav(i,T,n)*krav(j,T,n)*dercoefficient(T,phi2,delta2,eta_a1,eta_b1,i,j,k,l)[1]
                    derdelta[(i-k)*(T+1)+(T-i-l),(j-k)*(T+1)+(T-j-l)]+=krav(i,T,n)*krav(j,T,n)*dercoefficient(T,phi2,delta2,eta_a1,eta_b1,i,j,k,l)[2]
    
    return [(1-epsilon)*derphase+epsilon*thermalstate(T,avg),(1-epsilon)*dereta_a+epsilon*thermalstate(T,avg),(1-epsilon)*derdelta+epsilon*thermalstate(T,avg)]


#-----------------------------------------------

# Construction of n qubit separable states for joint-estimation of phase and diffusion - no losses

def thermalstate_qb(n):

    dim=2**n
    mat3=sp.eye(dim)
    norm=sp.Sum(sp.Pow(n/2,k)/sp.Pow(1+(n/2),k+1), (k, 0, dim-1)).doit()
    for i in range(dim):
        mat3[i,i]=(1/norm)*sp.Pow(n/2,i)/sp.Pow(1+(n/2),i+1)
    return mat3

def gaussian(phi1,phi,delta):
    
    gauss=sp.exp(-((phi1-phi)**2)/(2*delta**2))/sp.sqrt(2*pi*delta**2)
    
    return gauss

# Building the n qubit phase diffused output state in sympy

def twoqubitphasediffstate(phi,delta,theta,epsilon,n):
    
    # input single qubit state
    p1=sp.Matrix([[sp.cos(theta/2)**2, sp.cos(theta/2)*sp.sin(theta/2)], [sp.cos(theta/2)*sp.sin(theta/2), sp.sin(theta/2)**2]])
    # defining the single qubit phase shift (phi) unitary e^{-i \phi \sigma_z/2}, where sigma_z is the Pauli z matrix
    phaseshift=sp.Matrix([[sp.exp(-sp.I*phi1/2), 0], [0, sp.exp(sp.I*phi1/2)]])
    # n qubit input state
    inputprobe=reduce(TensorProduct,[p1]*n)
    # n qubit unitary 
    phaseshift2=reduce(TensorProduct,[phaseshift]*n)
    # after phase shift operation, the phase diffusion (delta) noise channel is acted upon as follows. The output state is a mixed state with two unknown parameters (phi, delta) encoded
    outputprobe=(sp.Mul(phaseshift2,inputprobe,sp.conjugate(phaseshift2).T)*gaussian(phi1,phi,delta)).applyfunc(lambda x: sp.integrate(x, (phi1,-sp.oo,sp.oo)))
    # regularization: since the output state is NOT full rank, we make it full rank by adding an "epsilon" fraction of a thermal state of same dimension to it
    return (1-epsilon)*outputprobe+epsilon*thermalstate_qb(n)

# Converting the output state to numpy

def numpytwoqubitphasediffstate(phi,delta,theta,epsilon,n):

    return np.array(twoqubitphasediffstate(phi,delta,theta,epsilon,n)).astype(np.complex128)

# Taking the derivative of the output state with respect to phi in sympy

def derphitwoqubitphasediffstate(phi,delta,theta,epsilon,n):
    
    return twoqubitphasediffstate(phi,delta,theta,epsilon,n).applyfunc(lambda x: sp.diff(x,phi))

# Converting the phi derivative state to numpy

def numpyderphitwoqubitphasediffstate(phi2,delta,theta,epsilon,n):
    
    der1=derphitwoqubitphasediffstate(phi,delta,theta,epsilon,n).subs(phi,phi2)

    return np.array(der1).astype(np.complex128)

# Taking the derivative of the output state with respect to delta in sympy

def derdeltatwoqubitphasediffstate(phi,delta,theta,epsilon,n):
    
    return twoqubitphasediffstate(phi,delta,theta,epsilon,n).applyfunc(lambda x: sp.diff(x,delta))

# Converting the delta derivative state to numpy

def numpyderdeltatwoqubitphasediffstate(phi,delta1,theta,epsilon,n):
    
    der2=derdeltatwoqubitphasediffstate(phi,delta,theta,epsilon,n).subs(delta,delta1)
    
    return np.array(der2).astype(np.complex128)  


# In[25]:


# Note: Copy and paste the code from HCRB.py script here, and load the 'hcrb' function for evaluation

# Computing HCRB for gHB states

def hcrbphietaghb(T,n,eta_a1,eta_b1,phi2,delta2,y):
    
    derrhothnew=[derphasediffetastatereg(T,n,eta_a1,eta_b1,phi2,delta2,1,1e-7)[0],derphasediffetastatereg(T,n,eta_a1,eta_b1,phi2,delta2,1,1e-7)[1]]

    return hcrb(phasediffetastatereg(T,n,eta_a1,eta_b1,phi2,delta2,1,1e-7), derrhothnew, weightmatrix(y))

def hcrbphideltaghb(T,n,eta_a1,eta_b1,phi2,delta2,y):
    
    derrhothnew1=[derphasediffetastatereg(T,n,eta_a1,eta_b1,phi2,delta2,1,1e-7)[0],derphasediffetastatereg(T,n,eta_a1,eta_b1,phi2,delta2,1,1e-7)[2]]

    return hcrb(phasediffetastatereg(T,n,eta_a1,eta_b1,phi2,delta2,1,1e-7), derrhothnew1, weightmatrix(y))

def hcrbphideltaghb(T,n,eta_a1,eta_b1,phi2,delta2,y):
    
    derrhothnew1=[derphasediffetastatereg(T,n,eta_a1,eta_b1,phi2,delta2,1,1e-7)[0],derphasediffetastatereg(T,n,eta_a1,eta_b1,phi2,delta2,1,1e-7)[2]]

    return hcrb(phasediffetastatereg(T,n,eta_a1,eta_b1,phi2,delta2,1,1e-7), derrhothnew1, weightmatrix(y))

# Computing HCRB for qubits

def hcrbqubit(phi2,delta1,theta,n,y):

    derrhotq=[numpyderphitwoqubitphasediffstate(phi2,delta1,theta,1e-13,n),numpyderdeltatwoqubitphasediffstate(phi2,delta1,theta,1e-13,n)]
                    
    return hcrb(numpytwoqubitphasediffstate(phi2,delta1,theta,1e-13,n), derrhotq, weightmatrix(y))

