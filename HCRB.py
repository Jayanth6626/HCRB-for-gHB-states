import numpy as np
import scipy as sc
import cvxpy as cp

import mosek

def hcrb(state,derstate,W):

# 'state' is a density matrix, ρ.

# Note: If the state is not full rank, it can be regularized by adding a small fraction of a suitable state to
# avoid numerical instabilities. For example, one can use: (1 − ε)·ρ + ε·σ, where σ is a suitable full-rank
# state such as a thermal state.
 
# 'derstate' must be a python list of derivatives of the state with respect to each parameter.

# 'W' is a weight matrix
    
    d=len(state)
    num=d*d
    par=len(derstate)
    
# creating the basis of hermitian operators
    
    list2=[]
    list3=[]
    list4=[]
    list5=[]
    list6=[]

    
    for i in range(0,d):
        for j in range(i+1,d):
            
# create first list of sparse matrices with entries according to list indices
            m1=np.zeros((d,d),dtype=np.complex128)
            m1[i,j]=1j
            m1[j,i]=1j
            
            list2.append(m1/np.sqrt(2))
            
# create second list of sparse matrices with entries according to list indices
            m2=np.zeros((d,d),dtype=np.complex128)
            m2[i,j]=-1
            m2[j,i]=1
            
            list3.append(m2/np.sqrt(2))
            
    for i in range(0,d-1):
        
# create a list of vectors
        m3=np.zeros(d,dtype=np.complex128)
        m3[i]=1j
        m3[i+1]=-1j
        
        list4.append(m3)
        
# orgothonalize the vectors
        org=np.linalg.qr(np.array(list4).T)[0]
        l1=org[:,i]
        list5.append(l1)
        
# transform each vector to a diagonal matrix
        l2=np.diag(list5[i])
        list6.append(l2)
        
    b1=[np.identity(d)/np.sqrt(d)]+[mat*(-1j) for mat in (list2+list3+list6)]
    c1=np.array(b1).flatten().reshape(num,num)
    
# calculate Frobenius inner product
    
    frob=np.trace(np.dot(np.conjugate(np.transpose(b1[0])),b1[0]))
    
# calculate Hilbert-Schmidt inner product
    
    hs=np.trace(np.dot(b1[0],b1[0]))
    
# create the list of coefficients (s_theta)_i representating the state in the orthonormal basis

    vecrho=[]

    for i in range(0,num):
        #vecrho1=np.trace(np.dot(np.conjugate(np.transpose(state)),b1[i]))
        vecrho1=np.real(np.trace(np.dot(state,b1[i])))
        vecrho.append(vecrho1)
        
# checking basis transformation
    
    list7=[]

    for i in range(num):
        list8=b1[i]*vecrho[i]
        list7.append(list8)

# create the matrix containing derivatives of state with respect to parameters
    
    vecdrho=[]

    for i in range(0,par):
        for j in range(0,num):
            vecdrho1=np.real(np.trace(np.dot(derstate[i],b1[j])))
            #vecdrho1=np.real(np.trace(np.dot(np.conjugate(np.transpose(derstate[i])),b1[j])))
            vecdrho.append(vecdrho1)
            
    length = len(vecdrho)
            
    dersmatrix=np.array([vecdrho[i*length // par: (i+1)*length // par] for i in range(par)]).T
    
# construction the S and R matrix    

    l2=[]
    for j in range(num):
        l3=(b1[j] @ state).T.copy()
        l2.append(l3)
        
    c2=np.array(l2).flatten().reshape(num,num)
    S1=np.matmul(c1,np.transpose(c2))
    
    lu, d, perm = sc.linalg.ldl(S1, lower=0)

    R=np.conjugate(np.dot(lu,np.sqrt(d))).T

# optimization variables X and V 

    V = cp.Variable((par, par))
    X = cp.Variable((num, par))


# constraints

    constraints = [cp.bmat([[V, X.T @ R.conj().T], [R @ X, np.identity(num)]]) >> 0]
    constraints += [X.T @ dersmatrix == np.identity(par)]
    
    
# optimization
    
    prob = cp.Problem(cp.Minimize(cp.trace(W @ V)), constraints)
    prob.solve(solver=cp.MOSEK,verbose=True)
    
    return prob.value

