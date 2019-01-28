# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 12:04:57 2017

@author: Frederik
"""
from numpy import *
import numpy.random as npr
from scipy.linalg import *
from scipy.sparse import *
import datetime
import time
import os
import sys 
import logging as l
import numpy.fft as fft
import _pickle as pickle

from BasicFunctions import *
#==============================================================================
# 0: Simulation Parameters
#==============================================================================


treshlog=-12

### A: Time resolution parameters
# Resolution of driving field
Resolution_drive=300          # Resolution of driving field is  
                              # dt_drive= 1/Resolution_drive

# Resolution of time-evolution
Nres_out=0                    # Save a list of unitaries 
                              #
                              # {dU(t_n+dt,t_n),n=1..Resolution_drive}, 
                              #
                              # where dt=dt_drive*2^{-Nres_out}, 
                              # and t_n = n dt_drive.

# Resolution for trotterization
N_trotter=16                  # Effective evolution within a single step 
                              # is given by
                              #
                              # dU_step = [exp(-iH_1 dtau)...
                              # exp(-iH_3 dtau)]^(2^<N_trotter>), 
                              #
                              # where tau= dt/(2**<N_trotter>)

# Time interval between time evolution steps 
dt=1/(Resolution_drive*2**(Nres_out))

# Time interval for trotterization
dtau=dt/(2**N_trotter)

# Number of time evolution steps
TimeSteps=Resolution_drive*(2**Nres_out)


### B: Parameters for the  resolution of jump operators
FreqVecCutoff=100   # maximal harmonic to include in the jump operator 
#                     definition (i.e., the jump operators are time-dependent 
#                     in a periodic fashion, and we throw away the harmonics 
#                     with frequencies outside \pm  2pi FreqVecCutoff /T. Set to 100. 

Wmax=300
Wlist=concatenate((
arange(0,10,0.02),
arange(10,40,0.06),
arange(40,150,0.2),
arange(150,Wmax,0.6)))
#Wlist=arange(0,1)
nW= len(Wlist)

dWlist=concatenate((Wlist[1:]-Wlist[:nW-1],array([Wmax-Wlist[nW-1]])))


#BlockIntervals=



Resolution_jump=100    # Time resolution of decay opeator. Compute H_eff(t) for 
                    # <TimeStepsJump> evenly spaced times within the driving 
                    # period

Res_SSE=300     # Number of times per driving periods, jumps are attempted
                # Must be integer multiple of TimeStepsJump

### C: Output data time resolution
Dt_out=1/5   # Time resolution for output data. Must be a divisor of 
                # TimeStepJump


Ltresh=1e-9     #Treshold for discarding matrix elements of jump operator

N_fft=50000 #Number of matrix element to fourier transform at a time. 
#NBlocksW=int(NblocksTot/(NT_J)+0.1) 

#BlockIntervals=[int(k) for k in (arange(0,NBlocksW+1)*nW/NBlocksW)]



# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# 0.1 Check that input data are proper

#if not abs(Resolution_jump%Nsteps_out)<0.001:
#    raise ValueError("Nsteps_out must be a divisor of Resolution_jump")
  
if not abs(Res_SSE%Resolution_jump)<0.001:
    raise ValueError("dT must be an integer multple of Resolution_jump")  

if not abs(Resolution_drive % Res_SSE)<0.001:
    raise ValueError("Resolution_drive must be integer multiple of Res_SSE")
#==============================================================================
# 1: Basic functions
#==============================================================================


### A: Spectral density of energy Mat
def SD(Mat,T=0):
    K0=0.5*(Mat+abs(Mat))
    
    if T>0:
        K1=1/(exp(abs(Mat)/T)-1)
    else:
        K1=0        
    K=K1+K0
    
    return K 

def RangeFinder(W):
    fmin=min(freqvec)
    fmax=max(freqvec)
    
    # Find f such that \omega_2 f is closest to W.
    f=int(W/omega2+0.1)
    q=f-fmin
    
    # When constructing jump operator constrict Z and EnergyDenom
    # to the components q-dq... q+dq
    qmin=max(0,q-dq)
    qmax=min(len(freqvec)-1,q+dq+1)
    
    qmin=int(qmin+0.1)
    qmax=int(qmax+0.1)
    return[qmin,qmax]
    
    
def EnergyDenominator_gen(qmin,qmax):
    frequencies=freqvec[qmin:qmax]

    # Construct energy denominator
    
    EnergyDenom=ones((2*N,2*N,len(frequencies)),dtype=complex)
          
    for n in range(0,len(frequencies)):
        EnergyDenom[:,:,n]*=-omega2*frequencies[n]
        EnergyDenom[:,:,n]+=QEMat
        EnergyDenom[:,:,n]+=-(1j/2)*(GammaMat)+1j*10**(-12)

    return EnergyDenom
    
#def EnergyDenominator_bare_gen(qmin,qmax):
#    frequencies=freqvec[qmin:qmax]
#
#    # Construct energy denominator
#    
#    EnergyDenom=ones((2*N,2*N,len(frequencies)),dtype=complex)
#          
#    for n in range(0,len(frequencies)):
#        EnergyDenom[:,:,n]*=-omega2*frequencies[n]
#        EnergyDenom[:,:,n]+=QEMat
#        EnergyDenom[:,:,n]+=-(1j/2)*(gamma_c)+1j*10**(-12)
#
#    return EnergyDenom
#    
### B: Extracts jump operator form data
def JumpOperatorExtractor(Larray,t,Freqs):
    
    if not len(Larray)==len(Freqs):
        raise ValueError("Length of Larray must equal length of Freqs")
    # Frequency must be in physical units (i.e. driving frequency is 2pi)
    # Extracts unnormalized jump operator at frequency omega, at time t.
    # Computed in the way explained in the notes. 
    #
    # Required variables to be specified in calling script:
    #
    # J, 
    # Z,
    # omega2 (Driving frequency)

    # Here 
    #    
    # Z[a,b,n]=A^{k_n}_{ab} 
    #
    # J[a,b,n]=J(\varepsilon_b-\varepsilon_a +k_n\Omega)
    #  
    # Here k_n = freqvec[n]=n-NSF.FreqVecCutoff.
    # 


    # Time-evolution of blocks         
#    cvec=exp(-1j*omega2*freqvec*t)

        
    # Compute normalized jump operator (i.e. multiply with 2pi gamma_k afterwards)  M_{ab}(t) = \sum_k J(\vaerpsilon_b-\varepsilon_a+k \Omega )e^{-ik\Omega t} A_{ab}[k]
#    H=Z*sqrt(J)
#    L=einsum(H,[0,1,2],cvec,[2])
    
    L=sum([exp(-1j*Freqs[n]*t)*Larray[n] for n in range(len(Freqs))])
    L=L.toarray()

    
    return L
    
    
def BlockIntervalFinder(W):
    nb=0
    for b in BlockIntervals[1:]:

        if W>=b:
            nb+=1

    return nb
    
    
### C: Generates a gaussian wave packet(Coherent state) \otimes spin state. 

def GaussianPacketGenerator(N,n0,s,sigma,phase):

    # N         :   Number of photon states 
    # n0        :   Center of wave packet
    # s         :   Spin of wave packet (can be 1,-1 or 'x')
    # sigma     :   Width of wave packet
    # phase     :   Phase of wave packet

    # Defininglist with state
    Psi=zeros((N,2),dtype=complex) #Psi(n,s) gives the wave function on "site" n, at spin s

    # Filling out entries 
    for n in range(0,N):
        if s==-1:
            Psi[n,0]=exp(-(n-n0)**2/(2*sigma**2))*exp(1j*phase*n)
        
        elif s==1:
            Psi[n,1]=exp(-(n-n0)**2/(2*sigma**2))*exp(1j*phase*n)            

        elif s=="x":
            Psi[n,0]=exp(-(n-n0)**2/(2*sigma**2))*exp(1j*phase*n)
            Psi[n,1]=exp(-(n-n0)**2/(2*sigma**2))*exp(1j*phase*n)

        else:
            raise TypeError("s must be either 1, 0 or 'x'")

    # Normalizing 
    Psi=Psi/sqrt(sum(abs(Psi**2)))

    # Putting together vector as (2N,1) array
    PsiUp=array(Psi[:,0],ndmin=2).T
    PsiDown=array(Psi[:,1],ndmin=2).T
    Psi=concatenate((PsiUp,PsiDown))

    return Psi

### D: Return density in photon space
def Density(Psi):
    N=int(shape(Psi)[0]/2+0.1)
    return abs(Psi**2)[0:N,:]+abs(Psi**2)[N:2*N,:]
    
### E: Directory name associated with parameter choice    
def ParmDir_gen(N,ratio,eta,A_2,M,gamma_c,gamma_s):
    return "%d__%s__%s_%s_%s__%s_%s"%(N,FNS(ratio),FNS(eta/pi),FNS(A_2),FNS(M),FNS(gamma_c),FNS(gamma_s))
    
### F: Directory name associated with resolutino parameters
def ResDir_gen(Resolution_drive,Nres_out,nW,NT_j,CutoffRel,NBlocksW):
    return "%d_%d__%d_%d__%s_%d"%(Resolution_drive,Nres_out,nW,NT_j,FNS(CutoffRel),NBlocksW)

def FloquetDir_gen(N,ratio,eta,A_2,M,Resolution_drive,Nres_out):
    return "%d__%s__%s_%s_%s__%d_%d"%(N,FNS(ratio),FNS(eta/pi),FNS(A_2),FNS(M),Resolution_drive,Nres_out)




def AnnihilationOperator(N):
        # Identity operators on spin space, and photon space
    Iphoton=csr_matrix(eye(N))
    Ispin=eye(2)
        
    # Defining spin operators 
    Sx=array([[0,1],[1,0]])
    Sy=array([[0,1j],[-1j,0]])
    Sz=array([[1,0],[0,-1]])
    
    Sx=csr_matrix(Sx)
    Sy=csr_matrix(Sy)
    Sz=csr_matrix(Sz)
    
    # Defining photon annihilation operaor
    b=zeros((N,N))
    for n in range(1,N):
        b[n-1,n]=sqrt(n)
        
    # Creation operator 
    b_dag=b.T.conj()
    
    # Photon counting operator 
    N_cavity=dot(b_dag,b)
    
    # Photon annihilation operator (acting on combined Hilbert space)
    B=kron(Ispin,b)
    
    return B

def EVList(A,PsiList):
    
    APsi=A.dot(PsiList)
    
#    Psi1=PsiList.conj()

        
    return OVList(PsiList,APsi)
    
    
def OVList(Psi_L,Psi_R):
    Psi_L=Psi_L.conj()
    
    NY=shape(Psi_L)[1]
    
    
    return array([Psi_L[:,n].dot(Psi_R[:,n]) for n in range(0,NY)],dtype=complex)
    
def QuasiEnergy(Eigvals):
    return -imag(log(Eigvals))