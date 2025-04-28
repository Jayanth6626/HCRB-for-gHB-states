# Construction of gHB states for joint-estimation of phase, loss, and diffusion

import numpy as np
import scipy as sc
import sympy as sp
import mpmath as mp
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

# derivatives of the state with respect to each parameter

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

#-----------------------------------------------------

# Note: Integrate the code from the HCRB.py script here, and load the 'hcrb' function to evaluate the following functions

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
