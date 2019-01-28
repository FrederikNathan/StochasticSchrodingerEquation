# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 15:39:18 2017

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

from matplotlib.pyplot import *


#==============================================================================
# 1: Input data
#==============================================================================

FileName="12.283"
InputPath=RecursiveSearch(FileName,"../Data/InteractionPictureEvolutions")

InputData=load(InputPath)


[N,ratio,eta,A_2,MagField,gamma_c,gamma_s,
Resolution_drive,Nres_out,N_trotter,
treshlog,NT,Resolution_jump,Res_SSE] = InputData["parameters"]

N=int(N+0.1)
Resolution_drive=int(Resolution_drive+0.1)
Nres_out=int(Nres_out+0.1)
N_trotter=int(N_trotter+0.1)
treshlog=int(treshlog+0.1)
NT=int(NT+0.1)

Tmax=2000
Ymax=600
t0=Tmax/7
VMAX=0.03

Tmin=0

SaveFig=True
#Resolution
#[Wlist,BlockIntervals,Jump_Tlist]                       = InputData["ResLists"]
#
#[Cutoff,nW,Resolution_jump,dq]            = InputData["ResParms"]


Nsteps_out=InputData["Nsteps_out"]

tlist=arange(0,NT)


Eigenvectors=InputData["Eigenvectors"]
QE=InputData["QE"]
#Gamma=InputData["Gamma"]
InitializationNotes=InputData["InitializationNotes"][0]

#FloquetPath    ="../Data/FloquetSpectra/"+FloquetDir_gen(N,ratio,eta,A_2,MagField,Resolution_drive,Nres_out)+".npz"
#FloquetData=load(FloquetPath)

#Mlist=FloquetData["Mlist"]
#
# Number of data points per driving period 


print("-------------------------------------------------------------------------")
print("                         Data extractor ")
print("-------------------------------------------------------------------------")
print("")
print("Path: "+InputPath[37:])
print("")
print("Parameters: ")
print("")
print("    Number of photon states                   =  %d"%N)
print("")
print("    omega2/omega1                             =  %.3f"%ratio)
print("    eta * L / hbar                            =  %.3f"%eta)
print("    M                                         =  %.3f"%MagField)
print("    A_2                                       =  %.3f"%A_2)
print("")
print("    gamma_c                                   =  %.5f"%gamma_c)
print("    gamma_s                                   =  %.5f"%gamma_s)
print("")
print("    Resolution_jump                           =  %d"%Resolution_jump)
#print("    nW                                        =  %d"%nW)
#print("    Cutoff                                    =  %.3f"%Cutoff)
print("")
print("    Time resolution of drive                  =  %d" %Resolution_drive)
print("    Time steps per drive-resolution step      =  2^%d" %Nres_out)
print("    N_trotter                                 =  2^%d"%N_trotter)
print("    Treshold for discarding matrix elements   =  10^%d"%int(treshlog+0.1))
print("")
print("")
print("-------------------------------------------------------------------------")
print("")

#==============================================================================
# 2: Generating intitial lists etc
#==============================================================================

# Generating list with unitaries 
# MatList[n]=M[t_n], where M[t] is the micromotion operator, 
# and t_n = RecTimes[n]


#MatList=zeros((2*N,2*N,N_Rec),dtype=complex)
#TestMatList=zeros((2*N,2*N,N_Rec),dtype=complex)
#
#K=int(Resolution_drive/N_Rec+0.1)
#
#M=eye(2*N,dtype=complex).toarray()
#dt_Rec=1/N_Rec
#
#dUeff=diag(exp(1j*QE*dt_Rec))
#for nRec in range(0,N_Rec):
#    t=nRec/N_Rec
#
#    MatList[:,:,nRec]=M.dot(diag(exp(1j*QE*t)))
#    TestMatList[:,:,nRec]=M.dot(diag(exp(1j*QE*t)))
#    for k in range(0,K):
#        dM=Mlist[nRec*K+k]
#        
#        for nro in range(0,Nres_out):
#            dM=dM.dot(dM)
#        
#        M=dM.dot(M)#.dot(dUeff)
#        
#        
#M0=M
#NSF.N=N

#==============================================================================
# 3: Time evolution
#==============================================================================

PsiList=InputData["PsiList"]
DecayList=InputData["DecayList"]
N_Rec=int(shape(PsiList)[1]/NT+0.1)

PsiOut=PsiList[:,::N_Rec]
PsiOut=Eigenvectors.dot(PsiOut)


Rho=Density(PsiOut)

B=AnnihilationOperator(N)
N_cav=B.T.dot(B)
BEV=EVList(B,PsiOut)
NEV=abs(EVList(N_cav,PsiOut))

Coh=abs(BEV**2)/NEV

#InitializationNotes=InputData["InitializationNotes"][0]

#==============================================================================
# 4: Plotting
#==============================================================================


figure(1)
clf()
# Number of photons in the steady state
N_SS=1/(2*pi*gamma_c)


fig=gcf()
#fig.set_dpi(400)
fig.set_size_inches(6.5,10,forward=True)

#Rho=Rho.T

[tgrid,Ngrid]=meshgrid(tlist,arange(0,N))

tmin=amin(tgrid)
tmax=amax(tgrid)

Yres=20000
Skip=max(1,int(shape(tgrid)[1]/Yres))
Skip=1
tgrid=tgrid[:,::Skip]
Ngrid=Ngrid[:,::Skip]
Rho=Rho[:,::Skip]
Rho0=Rho[:,0]
RhoI=array([Rho0,Rho0]).T

[tg0,Ng0]=meshgrid(array([-t0,0]),arange(0,N))


pcolormesh(tgrid,Ngrid,Rho,vmin=-VMAX,vmax=VMAX,cmap="seismic")
pcolormesh(tg0,Ng0,RhoI,vmin=-VMAX,vmax=VMAX,cmap="seismic")

#pcolormesh(tgrid,Ngrid,Rho,vmax=0.05)
#pcolormesh(tg0,Ng0,RhoI,vmax=0.05)

N_min=abs((A_2-MagField)**2)
N_max=abs((A_2+MagField)**2)

#plot([tmin,tmax],[N_SS,N_SS],'w')
#plot([tmin,tmax],[N_min,N_min],'r')
#plot([tmin,tmax],[N_max,N_max],'r')
#plot([tmin,tmax],[tmin,tmax],'r')
plot([-t0,0],[N_SS,N_SS],'k')
plot([-t0,0],[N_min,N_min],'r')
plot([-t0,0],[N_max,N_max],'r')



xlabel("time")
ylabel("N")
xlim((Tmin,Tmax))
#xlim((1000,1400))
ylim((0,Ymax))
title("Wave function. "+InitializationNotes)

#if Tmax==NT:
    
ax=gca()
ax.set_aspect(0.88*Tmax/Ymax)

if SaveFig:
    FigFileName="/Users/frederik/Dropbox/Phd/PhotonConversion/Noise_simulation/Figures/%s.png"%FileName
    savefig(FigFileName,dpi=1000,format="png")
#%%

#xlim((75,100))
#==============================================================================
# 4: Decays
#==============================================================================

if gamma_c>0 or gamma_s > 0:
    if len(DecayList)>0:
        
            # Spin dcays
        Sdecay=where(array([k[0] for k in DecayList[:,1]])=="S")
#        plot(DecayList[Sdecay,0],100*ones(len(Sdecay)),'k.')
        DecayTimes=DecayList[:,0].astype(float)
    #    DecayFreqs=DecayList[:,1].astype(float)
        DecayTypes=DecayList[:,1]
        Ndec=len(DecayTimes)
        # Sort decays 
        Operators = ["Cavity","Sx","Sy","Sz"]
        DecTimeSort=[[],[],[],[]]
#        DecFreqSort=[[],[],[],[]]
        
        for n in range(0,Ndec):
            Type=DecayTypes[n]
            if Type=="Cavity":
                z=0
            if Type=="Sx":
                z=1
            if Type=="Sy":
                z=2
            if Type=="Sz":
                z=3
            
            
            DecTimeSort[z].append(DecayTimes[n])
#            DecFreqSort[z].append(DecayFreqs[n])
            
            
#        figure(3)
#        clf()
#        for z in range(0,4):
#            [H,B]=histogram(DecFreqSort[z],bins=30)
#            
#        #    Blist.append(B)
#        #    Hlist.append(H)
#        #for z in range(0,4):
#            color=colorlist[z]
#            plot(0.5*(B[1:]+B[:len(B)-1]),H,'-%s'%color)
        
#        xlabel("Frequency")
#        ylabel("Counts")
#        title("Photons counted. "+InitializationNotes)
#        legend(Operators)
#        
#    
#        figure(2)
#        clf()
#        colorlist=["k","b","g","r"]
#        title("Decay events. "+InitializationNotes)
#        for z in range(0,4):
#            color=colorlist[z]
#            plot(DecFreqSort[z],DecTimeSort[z],'o%s'%color)
#        
#        xlabel("Frequency")
#        ylabel("Time")
#        legend(Operators)
    
    

#        figure(5)
#        clf()
#        X=array(DecTimeSort[0])
#        Y=arange(0,len(DecTimeSort[0]))
#        plot(X,Y,'-')
#        t0=0
#        tmax=NT
#        plot([0,NT],[0,NT],'-')
#        xlabel("Time")
#        ylabel("Number of emitted cavity photons")
##        legend(["Simulation","Quantized Rate"])
#        title("Photon counting rate")
##        xlim((80,100))
#        ylim((0,80))
#
#figure(4)
#
#clf()
#plot(tlist,Coh)
#xlabel("Time")
#ylabel("Closeness to coherent state")
#ylim((0,1.1))

#%%
