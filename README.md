# StochasticSchrodingerEquation
Code for solving stochastic schrodinger equation in a dissipative, periodically driven cavity-spin system.


## Directory structure
Code generates auxilliary data on the disk, to avoid repeating expensive calculations. From the the directory where the code is saved, the following directories should therefore exist for the code to work:

Diectories for log files  
../Logs  
../Logs/SSE  
../Logs/JumpOperators  
../Logs/FloquetOperatorGenerator  
  
Directories for data  
../Data  
../Data/JumpOperators  
../Data/FloquetSpectra  
../Data/InteractionPictureEvolutions  
  
All output data is saved in ../Data/InteractionPictureEvolutions. This data can be analyzed using DataExtractorStrob.py.

## Code
The following elements of the code are  
  
BasicFunctions:  
basic helping functions, such as filename search, timing and string parsing  
  
NoiseSimulationFunctions:   
helping functions specific for the quantum problem and code, including creation of basic building block matrices, file-naming functions etc. Importantly, inert parameters (that have to do with the simulation's accuracy) are defined here

FloquetOperatorGenerator:  
Finds the Floquet operator for the unitary problem, and diagonalizes it. Saves data under ../Data/FloquetSpectra/. Writes logs to ../Logs/FloquetOperatorGenerator/  
  
JumpOperatorGeneratorNew:  
Constructs jump operators used in stochastic schrodinger equation. Relies on data from ../Data/FloquetSpectra. Saves data to ../Data/JumpOperators/. Writes logs to ../Logs/JumpOperators/  
  
SSE_Evolution:  
Runs stochastic Schrodinger Equation, using the jump operators. Uses data from ../Data/FloquetSpectra and ../Data/JumpOperators/.  Saves data in../Data/InteractionPictureEvolutions.  Writes logs to ../Logs/SSE/.  
  
DataExtractorStrob:  
Extracts data from ../Data/InteractionPictureEvolutions, and plots releveant data. Can be modified to plot other relevant data. Relies on data from ../Data/FloquetSpectra and ../Data/InteractionPictureEvolution/.  Saves data in../Data/InteractionPictureEvolutions.

## Running the code  
To run the code, make sure all directories above exist. Then Run SSE_Evolution.py with the desired parameters set. Auxilliary data will automatically be created using the other scripts, if it is not already generated. 

Data is saved to the disk every 30 minutes during the iterative solution. At any point, the already-generated data can be examined using DataExtractorStrob.py. 

Progress can be monitored continually by reading the log files. 
