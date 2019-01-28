# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 11:02:35 2016

@author: Frederik
"""

from numpy import *
import numpy.random as npr
from scipy.linalg import *
import numpy.linalg as npla
from scipy.sparse import *

import datetime
import time
import os
import sys 
import logging as l

# This script generates and diagonalizes the dissipation-less Floquet operator for the Frequency conversion setup
# Returns Floquet operator, Floquet eigenstates and quasienergies


import NoiseSimulationFunctions as NSF
from NoiseSimulationFunctions import *

import BasicFunctions as BF
from BasicFunctions import *

#-----------------------------------------------------------------------------#
#                                 1: Parameters                               #
#-----------------------------------------------------------------------------#

if not len(sys.argv)>1:    
    sys.argv[1:]=[800 ,1.618033988750,2*  pi, 15,    8 , 300,0,16,    -12]
    OutputToConsole=True 
    print(sys.argv[0])
else:
    OutputToConsole=False
    timestring=datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    LogID=timestring
    LogFile="../Logs/FloquetOperatorGenerator/"+LogID+".log"
    l.basicConfig(filename=LogFile,filemode="w",format='%(message)s',level=l.DEBUG)


BF.OutputToConsole=OutputToConsole
BF.__init__()

# System paramters
N=int(sys.argv[1])              # size of lattice
ratio=float(sys.argv[2])        # 1.61803398875zs
eta=float(sys.argv[3])          # 5*pi     # coupling between spin and field
A_2=float(sys.argv[4]) #15      # Amplitude of driving field
M=float(sys.argv[5]) #7         # Magnetic field

omega2=2*pi;
omega1=omega2/ratio;

# Time resolution parameters
Resolution_drive=int(sys.argv[6])      # Resolution of driving field is dt_drive= 1/Resolution_drive
Nres_out=int(sys.argv[7])      # Save a list of {dU(t_n+dt_out,t_n),n=1..Resolution_drive}, where dt_out=dt_drive*2^{-Nres_out}, and t_n = n dt_drive. 
N_trotter=int(sys.argv[8])       # Effective evolution within a single step is given by dU_step = [exp(-iH_1 dtau)...exp(-iH_3 dtau)]^(2^<N_trotter>), where tau= dt/(2**<N_trotter>)

dt=1/(Resolution_drive*2**(Nres_out))
dtau=dt/(2**N_trotter)

tresh=10**(float(sys.argv[9]))  # Treshold for discarding entries of dU

# List with parameters 
parameters=[N,ratio,eta,A_2,M,Resolution_drive,Nres_out,N_trotter,tresh]
explanation="parameters are: N,ratio,eta,M,A_2,Resolution_drive,Nres_out,N_trotter,log_10(tresh) "


#-----------------------------------------------------------------------------#
#                       2: Defining fixed operators                           #
#-----------------------------------------------------------------------------#

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# 2.1 Fundamenatal operators 
# Identity operators on spin space, and photon space
Iphoton=csr_matrix(eye(N))
Ispin=eye(2)
    
# Defining spin operators 
Sx=array([[0,1],[1,0]],dtype=complex)
Sy=array([[0,1j],[-1j,0]],dtype=complex)
Sz=array([[1,0],[0,-1]],dtype=complex)

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

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# 2.2 Fixed components of the time evolution
# 
# Defining the evolution operators from the field itself, from the external field, and from the quantized field 
# Evolutions are for a time interval dtau

# Effective evolution of cavity mode from its own energy
dU_mode=csr_matrix(diag(exp(-1j*dtau*omega1*arange(0,N))))    #Acts only on photon space

# Evolution of spin due to magnetic field
dU_Mag=csr_matrix(expm(csc_matrix(-1j*dtau*eta*M*Sz)))

# Evolution from cavity-spin interaction
H_spin=csr_matrix(eta*(0.5j*kron(Sx,b-b_dag)-0.5*kron(Sz,b+b_dag)),dtype=complex)
dU_spin=csr_matrix(expm(csc_matrix(-1j*dtau*H_spin)))

# Static part of unitary time evolution operator in the time interval dtau
dU_static=dU_spin.dot(kron(dU_Mag,dU_mode,format="csr"))


#-----------------------------------------------------------------------------#
#                               3: Functions                                  #
#-----------------------------------------------------------------------------#

#class StreamToLogger(object):
#   """
#   Fake file-like stream object that redirects writes to a logger instance.
#   """
#   def __init__(self, logger, log_level=l.INFO):
#      self.logger = logger
#      self.log_level = log_level
#      self.linebuf = ''
#
#   def write(self, buf):
#      for line in buf.rstrip().splitlines():
#         self.logger.log(self.log_level, line.rstrip())
#
#if OutputToConsole==False:
#    
#    stderr_logger = l.getLogger('STDERR')
#    sl = StreamToLogger(stderr_logger, l.ERROR)
#    sys.stderr = sl
#    
#def Log(string):
#    l.info(string)
#    if OutputToConsole:
#        print(string)
#
#def FNS(z,ndigits=5): #File-name friendly string from number (replaces "." with "," and keeps ndigits digits)
#    z1=int(z)
#    z2=int(round(10**(ndigits)*(z-z1))+0.1)
#    s2=str(z2)
#    
#    nzeros=ndigits-len(s2)
#    s2="0"*nzeros+s2
#
#    while s2[len(s2)-1]=="0":
#        s2=s2[:len(s2)-1]
#        
#        if len(s2)==0:
#            break
#    
#    string="%d,%s"%(z1,s2)
#    return string

    

#-----------------------------------------------------------------------------#
#                              4: Time-evolution                              #
#-----------------------------------------------------------------------------#

Log("-------------------------------------------------------------------------")
Log("                         Floquet operator generator")
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
Log("    Time resolution of drive                  =  %d" %Resolution_drive)
Log("    Time steps per drive-resolution step      =  2^%d" %Nres_out)
Log("    N_trotter                                 =  2^%d"%N_trotter)
Log("    Treshold for discarding matrix elements   =  10^%d"%round(log10(tresh)))
Log("")
Log("")
Log("-------------------------------------------------------------------------")
Log("")
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# 4.1 Initial lists and parameters


# Pamameters for monitoring progress 
Step_per_monitor=1     # 
tnow=time.time()    # current time

# Coefficients used for the time-dependent part of the evolution
c1=cos(eta*A_2*dtau)
s1=sin(eta*A_2*dtau)

# Lists 
Ulist=[]
tlist=[]

U=kron(Ispin,Iphoton,"csr")
FloquetOperator=kron(Ispin,Iphoton,"csr")
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# 4.2 Time-evolving

Log("Time evolving")
for nstep in range(0,Resolution_drive):
    t=nstep*1/Resolution_drive       
             
    ### A: Evolve with (time-dependent) effective Hamiltonian for time-step dt
    
    # Finding evolution over time-interval dtau
    dU_drive=kron(c1*Ispin-1j*s1*(Sy*sin(omega2*t)-Sz*cos(omega2*t)),Iphoton,"csr")
    dU=dU_drive.dot(dU_static)
    
    # Compute time-evolution from time t_n to time t_n+dt_out taking matrix powers
    for n in range(0,N_trotter):
        dU=dU.dot(dU)

    dU=dU.multiply(abs(dU)>tresh)
    
    Ulist.append(dU)

    # Compute time-evolution from time t_n to time t_{n+1} taking matrix powers
    for n in range(0,Nres_out):
        dU=dU.dot(dU)
 
    dU=dU.multiply(abs(dU)>tresh)

    # Evolve Floquet operator 
    FloquetOperator=dU.dot(FloquetOperator)
        
    tlist.append(t)
    
    ### C: Monitor progress 
    if nstep % Step_per_monitor==0:
        Log("        At time step "+str(nstep)+"/"+str(Resolution_drive)+". Time spent = "+str(round(time.time()-tnow,4))+"s ");tnow=time.time()

#-----------------------------------------------------------------------------#
#                              5: Diagonalization                             #
#-----------------------------------------------------------------------------#

Log("")
Log("Diagonalizing");tnow=time.time()
FloquetOperator=FloquetOperator.toarray()
[Eigenvalues,Eigenvectors]=eig(FloquetOperator)
Log("    Done. Time spent: %0.2f s" % (time.time()-tnow))

# Changing basis of incremental time-evolutions to the basis of Floquet eigenstates. 
Mlist=[]
for U in Ulist:
    Mat=(Eigenvectors.conj().T).dot(U.dot(Eigenvectors))
    Mlist.append(Mat)
    
# Mlist[n]: Time-evolution U(t_n+dt_out,t_n) in the basis of Floquet eigenstates. 
    

#-----------------------------------------------------------------------------#
#                          6: Saving to data file                             #
#-----------------------------------------------------------------------------#

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# 5.1 Generating data file name

DataDir="../Data/FloquetSpectra/"

FileName=FloquetDir_gen(N,ratio,eta,A_2,M,Resolution_drive,Nres_out)

FileName=DataDir+FileName+".npz"
    
savez(FileName,FloquetOperator=FloquetOperator,Eigenvalues=Eigenvalues,Eigenvectors=Eigenvectors,Mlist=Mlist,tlist=tlist,parameters=parameters,explanation=explanation)