# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 11:02:35 2016

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

import NoiseSimulationFunctions as NSF
from NoiseSimulationFunctions import *

import BasicFunctions as BF
from BasicFunctions import *

import scipy.sparse as sp
#import matplotlib.pyplot as plt
#from matplotlib.pyplot import *


# This script finds the normalized jump operator matrix elements \{L^{ab}_k[n]}} (i.e., they still need to be multiplied by \sqrt{2 pi gamma_k} to work in the master equation)
# For each operator (k), it saves the matrix elements as an array of sparse matrices [L_{-NSF.FrecVecCutoff+1}...L_{NSF.FreqVecCutoff}], where L_{n}[a,b]=L_k^{ab}[n]
# This script assumes the bath is at zero temperature, but this can easily be modified in the definition of the matrix J below. 
#-----------------------------------------------------------------------------#
#                                 1: Parameters                               #
#-----------------------------------------------------------------------------#

if not len(sys.argv)>1:    
    sys.argv[1:]=[400 ,1.618033988750,0.5*pi,10,10  ]#1.618033988750
    OutputToConsole=True

else:
    OutputToConsole=False
    
    # Log File
    timestring=datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    LogID=timestring
    LogFile="../Logs/JumpOperators/"+LogID+".log"
    l.basicConfig(filename=LogFile,filemode="w",format='%(message)s',level=l.DEBUG)
   
BF.OutputToConsole=OutputToConsole
BF.__init__()
   
   
# System paramters
N=int(sys.argv[1])              # size of lattice
ratio=float(sys.argv[2])        # 1.61803398875
eta=float(sys.argv[3])          # 5*pi     # coupling between spin and field
A_2=float(sys.argv[4]) #15      # Amplitude of driving field
M=float(sys.argv[5]) #7         # Magnetic field

omega2=2*pi;
omega1=omega2/ratio;



# Time resolution parameters
Nsteps_d=NSF.Resolution_drive      # Resolution of driving field is dt_drive= 1/Nsteps_d
Nres_out=NSF.Nres_out              # Save a list of {dU(t_n+dt_out,t_n),n=1..Nsteps_d}, where dt_out=dt_drive*2^{-Nres_out}, and t_n = n dt_drive. 
N_trotter=NSF.N_trotter            # Effective evolution within a single step is given by dU_step = [exp(-iH_1 dtau)...exp(-iH_3 dtau)]^(2^<N_trotter>), where tau= dt/(2**<N_trotter>)
Resultion_jump=NSF.Resolution_jump # Resolution of effective Hamiltonian / quantum jumps (does not need to be as good as the driving field's resolution)

dt=1/(Nsteps_d*2**(Nres_out))   # in practice, dt\sim 1/500 is good enough. 
dtau=dt/(2**N_trotter)

TimeSteps=Nsteps_d*(2**Nres_out)

treshlog= NSF.treshlog # Treshold for discarding entries of dU

# List with parameters 
parameters=[N,ratio,eta,A_2,M,Nsteps_d,Nres_out,N_trotter,treshlog]
explanation="parameters are: [N,ratio,eta,M,A_2,Nsteps_d,Nres_out,N_trotter,tresh (log10),"


N_fft=NSF.N_fft # Number of matrix elements to inclue in fourier transform at a time. 
#N_fft=50000
if N_fft<2*N:
    raise ValueError("Matrix dimension exceeds N_fft")
    
SliceWidth=min(2*N,int(N_fft/(2*N)))

SliceList=[x for x in range(0,2*N,SliceWidth)]
SliceList=SliceList+[2*N]
    
Nslices=len(SliceList)-1


#-----------------------------------------------------------------------------#
#                       2: Defining fixed operators                           #
#-----------------------------------------------------------------------------#

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# 2.1 Fundamenatal operators 
    
# Identity operators on spin space, and photon space
Iphoton=csr_matrix(eye(N))
Ispin=eye(2)

I=kron(Ispin,Iphoton)

# Defining spin operators 
sx=array([[0,1],[1,0]])
sy=array([[0,1j],[-1j,0]])
sz=array([[1,0],[0,-1]])

sx=csr_matrix(sx)
sy=csr_matrix(sy)
sz=csr_matrix(sz)

Sx=kron(sx,Iphoton)
Sy=kron(sy,Iphoton)
Sz=kron(sz,Iphoton)

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


#-----------------------------------------------------------------------------#
#                           4: Loading Prefabricated data                     #
#-----------------------------------------------------------------------------#
    
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# 4.1 Identify filename for input data

InputDataDir="../Data/FloquetSpectra/"
InputFileName=FloquetDir_gen(N,ratio,eta,A_2,M,Nsteps_d,Nres_out)
InputFileName=InputDataDir+InputFileName+".npz"

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# 4.2 Load data. Generate from new if they are not found
Log("Looking for data")
try:    
    InputData=load(InputFileName)
    Log("    Data Found")
    Log("")
except(FileNotFoundError):
    Log("Data Not Found -- generating them now")
    
    ArgList=[N,ratio,eta,A_2,M,Nsteps_d,Nres_out,N_trotter,treshlog]
    ArgString=argstr(ArgList)
    
    os.system("python3 FloquetOperatorGenerator.py 2>/dev/null "+ArgString)

    Log("    Done.")

    InputData=load(InputFileName)

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# 4.3 Load variables from data

V=InputData["Eigenvectors"]
EV=InputData["Eigenvalues"]
F=InputData["FloquetOperator"]
Ulist=InputData["Mlist"]

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# 4.4 Extract quasienergies
QE=-imag(log(EV))    # quasienergies are in the interval [-pi/T,pi/T]
  
dHeff=csr_matrix(diag(exp(1j*dt*QE)))


# Generating quasienergy difference matrix (QEMat[a,b]=\varepsilon_a-\varepsilon_b)
QEMat=array([[qe]*(2*N) for qe in QE])
QEMat=QEMat-QEMat.T




#-----------------------------------------------------------------------------#
#                      5: Compute fourier transform of operators              #
#-----------------------------------------------------------------------------#

Log("-------------------------------------------------------------------------")
Log("                         Finding Operator Spectrum")
Log("-------------------------------------------------------------------------")
Log("")
Log("Parameters: ")
Log("")
Log("    Number of photon states                   =  %d"%N)
Log("    omega2/omega1                             =  %.3f"%ratio)
Log("    eta                                       =  %.3f"%eta)
Log("    M                                         =  %.3f"%M)
Log("    A_2                                       =  %.3f"%A_2)
Log("")
Log("    Time resolution of drive                  =  %d" %Nsteps_d)
Log("    Time steps per drive-resolution step      =  2^%d" %Nres_out)
Log("    N_trotter                                 =  2^%d"%N_trotter)
Log("    Treshold for discarding matrix elements   =  10^%d"%treshlog)
Log("")
Log("")
Log("-------------------------------------------------------------------------")
Log("")

#==============================================================================
# We find the fourier transform of \tilde A(t), where \tilde A(t) = \tilde U^\dagger(t) A \tilde U(t),
# Where $\tilde U(t) = e^{iH_{\rm eff}t} U(t)  is strictly T-periodic. 
# Here the effective Hamiltonian is defined with branch cut at pi. 
# To make the computation easy, we work in the basis of Floquet eigenstates (i.e. eigenstates of U(T))
# 
# Fourier component of \tilde A, \{\tilde A^{z}_{nm}\}, we can find $A(\omega)$ by 
# 
# A(\omega)=\sum_{m,n,z}\delta(\omega z + \varepsilon_m-\varepsilon_n) \tilde A_{nm}^{z} 
# 
# This function in principle have shart peaks. 
# We smoothen it with a gaussian kernel of width \sigma_{\omega}
# Apart from isolated, sharp peaks (eg. at integer multiples of the driving frequency),
# The resulting A(\omega) is hopefully approximately independent of the smoothening factor $\sigma_\omega$, for small enough $\sigma_\omega$ 
#==============================================================================

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# 5.1 Initial lists and parameters

# Pamameters for monitoring progress 
StepsPerLogging=max(int(Nsteps_d/5),1)

# Operators to be time-evolved
Operators=[0.5*(B+B.T).toarray(),Sx.toarray(),Sy.toarray(),Sz.toarray()]
OperatorNames=["Cavity","Sx","Sy","Sz"]
nA0=-1

# List that contains all of the fourier transformed operators.
FourierTransformList=[]

# List with inverse lifetimes
GammaList=[]



#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# 5.2 Initial lists and parameters

DataDir="../Data/JumpOperators/"       #  JOS: JumpOperatorSpectrum. 

OutputDir=DataDir+FloquetDir_gen(N,ratio,eta,A_2,M,Nsteps_d,Nres_out)+"/"

try:
    os.mkdir(OutputDir)
except FileExistsError:
#    Log("Data directory already exists")    
    pass

Log("")

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# 5.2 Iterating over operators --

## NB!: can be modified to be less memory intensive if necesary by only fourier transforming slices of A at a time. 
#Nfreq=2*NSF.FreqVecCutoff 
#freqvec=arange(-NSF.FreqVecCutoff,NSF.FreqVecCutoff)

Nfreq=2*NSF.FreqVecCutoff

for A0 in Operators:
    Log("---------------------------------------------------------------------")
    Log("        Generating normalized jump operator from %s operator"%OperatorNames[nA0+1])
    Log("---------------------------------------------------------------------")

    Log("")
    
    ##- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    ### A: Initializing lists, counters and variables 
    
    nA0+=1                  # Counts which number operator we are at
    
    tnow=time.time()        # Current time   
    
    A=(V.conj().T).dot(A0).dot(V)   # Expressing Schrodinger picture operator in basis of Floquet eigenstates (matrices in Ulist denote time evolution in basis of Floquet eigenstates)

    Larray=[sp.csr_matrix((0,2*N),dtype=complex)]*Nfreq
    
    HeffList=zeros((2*N,2*N,NSF.Resolution_jump),dtype=complex) # Change into sparse later.
    
    for nslice in range(0,Nslices):
        
        s1=SliceList[nslice]
        s2=SliceList[nslice+1]
    
        print("At row %d:%d out of %d"%(s1,s2,2*N))


        # Initial time-evolution operator    
        U=eye(2*N,dtype=complex).toarray()
        
        # Incremental time-evolution from effective Hamiltonian
        dUeff=diag(exp(1j*(dt*QE)))
    
        # List with time-evolved operators to be filled
        Alist=zeros((s2-s1,2*N,TimeSteps),dtype=complex)
        
        ##- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        ### B: Time-evolving operator A
        
        Log("    Time evolving");tnow=time.time()
        
        for nstep_1 in range(0,Nsteps_d):
            
            # Time-evolution increment at this time
            dU=Ulist[nstep_1]
            
            # Evolve over the 2^Nres_out equally spaced time-steps from t_n to t_{n+1}
            for nstep_2 in range(0,2**Nres_out):
                
                # Find which slot in Alist data should go to. 
                nstep=nstep_1*(2**Nres_out)+nstep_2
                              
                # Save time-evolved operator to list Alist (in complex 64 format)
                Alist[:,:,nstep]=(U.conj().T).dot(A.dot(U))[s1:s2,:]
                
                # Generate time-evolution operator at this time
                U=dU.dot(U).dot(dUeff)
                

              
            # Display progress 
            if nstep_1%StepsPerLogging ==0:
                Log("        At time step %5d/%d.  Time spent: %.2f s"%(nstep_1+1,Nsteps_d,time.time()-tnow));tnow=time.time()
        Log("        Done.")
        Log("")
        
                
        ##- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
        ### C: Fourier transforming \tilde A(t)
                
        Log("    Fourier transforming micromotion part of evolution:");tnow=time.time()
        
        # Fourier transform of \tilde A(t). Z[a,b,n]= A_{ab}^n, where \tilde A(t) = \sum A{ab}^n e^{-in\Omega t}
        A_fourier=fft.fft(Alist,axis=2)*dt
        Zfull=A_fourier
        # Rearrange indices, and truncate to the 200 lowest frequency components, 
        # such that  Z[:,:,n] corresponds to frequency n-FreqCutoff
      
        Zfull=concatenate((Zfull[:,:,TimeSteps-NSF.FreqVecCutoff:],Zfull[:,:,:NSF.FreqVecCutoff]),axis=2)
        freqvec0=fft.fftfreq(TimeSteps)*TimeSteps
        freqvec=concatenate((freqvec0[TimeSteps-NSF.FreqVecCutoff:],freqvec0[:NSF.FreqVecCutoff]))
                
        Zfull=Zfull[:,:,::-1]    
        freqvec=freqvec[::-1]
        
        for nfreq in range(0,Nfreq):
            K=Zfull[:,:,nfreq]
            
            # Projecting out Floquet eigenstates with high weight. ### FN: This is where the photon number cutoff enters.
            
#            ExcludedCols=ExcludeElements
#            ExcludedRows=ExcludeElements[where((ExcludeElements>s1-1)*(ExcludeElements<s2))[0]]
#            
#            K[:,ExcludedCols]=0
#            K[ExcludedRows-s1,:]=0
#            
            
            
            # Computing Energy difference matrix: DeltaE[a,b] = \varepsilon_a-\varepsilon_b -\Omega k_nfreq
            DeltaE=+omega2*freqvec[nfreq]*ones((s2-s1,2*N),dtype=complex)+QEMat[s1:s2,:]
            
            # Computing spectral function  J[a,b,n]=(n(-DeltaE[a,b,n])+Theta(-DeltaE[a,b,n])*S(-DeltaE[a,b,n])) -                 
            # We work at temperature zero, and with S(E)=E, but the function can be easily modified
            
            # Bath spectral function
            J=sqrt(StepFunction(-DeltaE)*abs(DeltaE))   
    
            
            # Discarding irrelevant entries
 
        
            # Computing L^{ab}[k_{nfreq}] 
            L=sqrt(2*pi)*J*K
            
            L=L*(abs(L)>NSF.Ltresh)
            L=sp.csr_matrix(L,dtype=complex)
            
            Larray[nfreq]=sp.vstack((Larray[nfreq],L))

           

#            print("%d,%f"%(nfreq,norm(L.toarray())))
#            if norm(K)>0.1:
#                print("")
#                print(str(nfreq)+","+str(freqvec[nfreq]))
#                print(around(real(K),3))
#                print(real(around(DeltaE/omega1,3)))
                
#            if nfreq==109:
#                raise ValueError
                
    
    Log("        Done. Time spent: %.2f"%(time.time()-tnow))
    Log("")
    
    
    #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    ## F: Save fourier transformed operator
    
    FilePath=OutputDir+"JOS_"+OperatorNames[nA0]+".dat"
    save_pickle([k for k in Larray],FilePath)      

    ##- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    ### D: Finding contribution to effective Hamiltonian from operator. 
    
    tlist = arange(0,NSF.Resolution_jump)/NSF.Resolution_jump
    
    Log("    Finding contribution to effective Hamiltonian from operator")
    tnow=time.time()
    for nt in range(0,len(tlist)):
        t=tlist[nt]
        Lt=(sum([exp(-2*pi*1j*freqvec[n]*t)*Larray[n] for n in range(0,Nfreq)])).toarray()
        HeffList[:,:,nt]=Lt.conj().T.dot(Lt)

        if nt%StepsPerLogging ==0:
                Log("        At time step %5d/%d.  Time spent: %.2f s"%(nt+1,Resolution_jump,time.time()-tnow));tnow=time.time()
      
#    raise ValueError     
    
    FilePath=OutputDir+"Heff_"+OperatorNames[nA0]+".dat"
    save_pickle(HeffList,FilePath)        
#        
#    raise ValueError

#-----------------------------------------------------------------------------#
#                                6: Save Other Data                           #
#-----------------------------------------------------------------------------#

#LifeTimes=sum(array(GammaList),1)
FilePath=OutputDir+"Data.npz"

Eigenvectors=V 
savez(FilePath,Eigenvectors=Eigenvectors,freqvec=freqvec,QE=QE,GammaList=GammaList,parameters=parameters,explanation=explanation)
#save_pickle(FourierTransformList,OutputPath)
