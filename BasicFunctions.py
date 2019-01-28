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
import datetime

#==============================================================================
# 1: Basic functions
#==============================================================================

def __init__():
    
    # Decide wether output should be directed to log file or console. 
    class StreamToLogger(object):
       """
       Fake file-like stream object that redirects writes to a logger instance.
       """
       def __init__(self, logger, log_level=l.INFO):
          self.logger = logger
          self.log_level = log_level
          self.linebuf = ''
    
       def write(self, buf):
          for line in buf.rstrip().splitlines():
             self.logger.log(self.log_level, line.rstrip())
    
    if OutputToConsole==False:
        
        stderr_logger = l.getLogger('STDERR')
        sl = StreamToLogger(stderr_logger, l.ERROR)
        sys.stderr = sl


# A: Logging function
def Log(string):
    l.info(string)
    if OutputToConsole:
        print(string)
def RunID_gen():
    timestring=datetime.datetime.now().strftime("%y%m%d_%H%M-%S.%f")[:-3]
    
    return timestring 
    
# B: Converts number to file-name friendly string
def FNS(z,ndigits=5): #(replaces "." with "," and keeps ndigits digits)
    
    if z<0:
        sgnstr="-"
    else:
        sgnstr=""
        
    z1=int(z)
    z2=int(round(10**(ndigits)*(z-z1))+0.1)
    s2=str(z2)
    
    nzeros=ndigits-len(s2)
    s2="0"*nzeros+s2

    while s2[len(s2)-1]=="0":
        s2=s2[:len(s2)-1]
        
        if len(s2)==0:
            break
    

        
    if len(s2)==0:
        string="%d"%z1

    else: 
        if s2[0]=="-":
                s2=s2[1:]
        string="%d,%s"%(z1,s2)
    return string
    
# C: Converts list to argument string (to be passed to python by shell script)
def argstr(List):
    string=""
    for l in List:
        string+=str(l)+" "
    
    return string

# C: Pickle saving function
def save_pickle(matrix, filename):
    with open(filename, 'wb') as outfile:
        pickle.dump(matrix, outfile, 3)   
        outfile.close()
        
# D: Pickle loading function        
def load_pickle(filename):
    with open(filename, 'rb') as infile:
        matrix = pickle.load(infile)   
        infile.close()
    return matrix       
        
# E: Load Data from data file, or generate new if they are not found.
def DataSearch(InputPath,ScriptName,ArgList=None,CreateNewData=True):
    
    # Looking for data at Input Path. If data are not found, generate them, by executing script ScriptName, using the arguments in ArgList
    
    if InputPath[len(InputPath)-4:]==".npz":
        Type="npz"
    elif InputPath[len(InputPath)-4:]==".dat":
        Type="pickle"
    else:
        Log("Data type: %s"%InputPath[len(InputPath)-4:])
        raise TypeError("Data must be .dat or .npz")
        
#    Log("    Looking for data")
    
    try:   
        if Type=="npz":
            InputData=load(InputPath)
        elif Type=="pickle":
            InputData=load_pickle(InputPath)
            
        Log("   Data Found")
        Log("")
        
    except(FileNotFoundError):
        if (not ArgList==None) and CreateNewData==True :
            
            Log("    Data not Found -- generating them now")
            tnow=time.time()
            ArgString=argstr(ArgList)
            
            os.system("python3 "+ScriptName+" 2>/dev/null "+ArgString)
            
    
            InputData=load(InputPath)
            Log("       Done. Time spent: %.2f s"%(time.time()-tnow))
            Log("")
        else:
            raise FileNotFoundError("No data found at InputPath")

    return InputData        

# F: Expectation value 
def ExpVal(A,Psi):
    EV=Psi.conj().T.dot(A.dot(Psi))
    EV=EV[0,0]
    return EV
    
    
def RecursiveSearch(Pattern,Dir,nrec=0):
    FileList=os.listdir(Dir)
    PathList=[]
    PathFound=False
    
    for x in FileList:
        Path=Dir+"/"+x



        if os.path.isdir(Path):
            A=RecursiveSearch(Pattern,Path,nrec=nrec+1)
            PathList=PathList+A
                
            
        elif PatternMatch(Pattern,x):
            PathList.append(Path)
    
    
    if len(PathList)==0:
        if nrec>0:
            return []
        else: 
            raise FileNotFoundError("No files matched the search criterion \"%s\" "%Pattern)
    elif len(PathList)>1:
        raise FileNotFoundError("Multiple files match the pattern %s . Be more specific"%Pattern)
    
    else:
        if nrec>0:
            return PathList
        else:
            return PathList[0]
    
    
                
def StepFunction(X):
    return 0.5*(1+sign(X))
    
    
def FileNamePatternMatch(Pattern,Directory):
    FileList=os.listdir(Directory)
    
    nmatch=0
    OutFileList=[]
    for n in range(0,len( FileList)):
        file=FileList[n]
        K=PatternMatch(Pattern,file)
        

        
        if K:
            Kre=PatternMatch("_re.npz",file)
            if not Kre:
                nmatch+=1
                OutFileList.append(file)
            
    
    if nmatch==1:
        return OutFileList[0]
    elif nmatch==0:
        raise FileNotFoundError("No files found that matched the pattern %s"%Pattern)
    else:
        print(nmatch)
        print(OutFileList)
        raise FileNotFoundError("%d files match the pattern %s: %s. Be more specific"%(nmatch,Pattern,str(OutFileList)))
        
            

def PatternMatch(Pattern,Name):
    Lp=len(Pattern)
    Ln=len(Name)
    for offset in range(0,Ln-Lp+1):
        TestStr=Name[offset:offset+Lp]
        if TestStr==Pattern:
            return 1
            
    
    return 0    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    