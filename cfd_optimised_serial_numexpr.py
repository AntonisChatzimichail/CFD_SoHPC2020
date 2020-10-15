# Editor: Antonios-Kyrillos Chatzimichail, antonis.xatzimixail@gmail.com
# Project: CFD simulation of fluid flow in a cavity
# Arguments: -sf <scaleFactor> -i <iterations> -r <reynolds number> -t <tolerance> -p <plot image?>
# Example for scale factor 4, 5000 iterations, irrotational flow, no tolerance check and export output image should be:
# -sf 4 -i 5000 -r 0.0 -t 0.0 -p 1
# Version description: Optimised serial code using Numexpr module.

from time import time as getTime
import numpy as np
import argparse
# Import the local "util.py" methods
import util
# Import the local "plot_flow.py" methods
import plot_flow
import numexpr as ne

def boundaryPsi(psi, m, n, b, h, w):
    # BCs on bottom edge
    psi[b+1:b+w, 0] = [float(i) for i in range(1, w)]
    psi[b+w:m+1, 0] = float(w)
  
    # BCS on RHS
    psi[m+1, 1:h+1] = float(w)
    psi[m+1, h+1:h+w] = [float(w - i) for i in range(1, w)]

def boundaryZet(zet, psi, m, n, b, h, w):
    # Set top/bottom BCs
    zet[1:m+1, 0] = 2.0 * (psi[1:m+1, 1] - psi[1:m+1, 0])
    zet[1:m+1, n+1] = 2.0 * (psi[1:m+1, n] - psi[1:m+1, n+1])  
        
    # Set left BCs
    zet[0, 1:n+1] = 2.0 * (psi[1, 1:n+1] - psi[0, 1:n+1])
        
    # Set right BCs
    zet[m+1, 1:n+1] = 2.0 * (psi[m, 1:n+1] - psi[m+1, 1:n+1])

#calculations for jacobi steps are made faster with Numexpr (using ne.evaluate) that with Numpy
#to use ne.evaluate properly, we have to store the arrays in different variables (can't use psi[2:m+2, 1:n+1] inside the parameter string)
def jacobistep(psitmp, psi, m, n):
    # psitmp[1:m+1, 1:n+1] = 0.25 * (psi[2:m+2, 1:n+1] + psi[0:m, 1:n+1] + psi[1:m+1, 2:n+2] + psi[1:m+1, 0:n])
    d = psi[2:m+2, 1:n+1]
    u = psi[0:m, 1:n + 1]
    r = psi[1:m+1, 2:n+2]
    l = psi[1:m+1, 0:n]
    psitmp[1:m + 1, 1:n + 1] = ne.evaluate("0.25 * (d + u + r + l)")
    
def jacobistepvort(zettmp, psitmp, zet, psi, m, n, re):
    # psitmp[1:m+1, 1:n+1] = 0.25 * (psi[2:m+2, 1:n+1] + psi[0:m, 1:n+1] + psi[1:m+1, 2:n+2] + psi[1:m+1, 0:n] - zet[1:m+1, 1:n+1])
    # zettmp[1:m+1, 1:n+1] = 0.25 * (zet[2:m+2, 1:n+1] + zet[0:m, 1:n+1] + zet[1:m+1, 2:n+2] + zet[1:m+1, 0:n]) - re / 16.0 * ((psi[1:m+1, 0:n] - psi[1:m+1, 2:n+2]) * (zet[0:m, 1:n+1] - zet[2:m+2, 1:n+1]) - (zet[1:m+1, 0:n] - zet[1:m+1, 2:n+2]) * (psi[0:m, 1:n+1] - psi[2:m+2, 1:n+1]))
    pd = psi[2:m + 2, 1:n + 1]
    pu = psi[0:m, 1:n + 1]
    pr = psi[1:m + 1, 2:n + 2]
    pl = psi[1:m + 1, 0:n]
    zc = zet[1:m+1, 1:n+1]
    zd = zet[2:m+2, 1:n+1]
    zu = zet[0:m, 1:n+1]
    zr = zet[1:m+1, 2:n+2]
    zl = zet[1:m+1, 0:n]
    psitmp[1:m + 1, 1:n + 1] = ne.evaluate("0.25 * (pd + pu + pr + pl - zc)")
    zettmp[1:m + 1, 1:n + 1] = ne.evaluate("0.25 * (zd + zu + zr + zl) - re / 16.0 * ((pl - pr) * (zu - zd) - (zl - zr) * (pu - pd))")

def deltaSquaredError(newarr, oldarr, m, n):
    return np.power(newarr[1: m+1, 1:n+1] - oldarr[1: m+1, 1:n+1], 2).sum()
#    delta = newarr[1: m+1, 1:n+1] - oldarr[1: m+1, 1:n+1]
#    return (delta*delta).sum()

parser = argparse.ArgumentParser()
 
parser.add_argument("-sf")
parser.add_argument("-i")
parser.add_argument("-r")
parser.add_argument("-t")
parser.add_argument("-p")   # plot or not

argument = parser.parse_args()

# Arguments
scaleFactor = int(argument.sf)  # Scale factor of simulation sizes
iteration = int(argument.i) # Number of iterations
printFrequence = 1000  # Print frequence through iterations
reynold = float(argument.r)  # Reynold's number in Jacobi - must be less than 3.7
tolerance = float(argument.t)  # Tolerance for convergence
re = 0 #need reynold number for print so generated a new variable
checkIter = iteration
# Initalize irrotationality
irrotational = True

# Check errors if tolerance is provided
checkError = tolerance > 0
#print("withNumpyFast code Start")
#if not checkError:
#    print("Scale Factor = {} | Iterations = {} | Checkerror off | Reynold Number = {}  ".format(scaleFactor, iteration, re))
#else:
#    print("Scale Factor = {} | Iterations = {} | Tolerance = {} |Reynold Number = {}".format(scaleFactor, iteration, tolerance, re))
# Reynold's number check
if reynold == 0:
    re = -1.0
else:
    re = float(reynold)
    irrotational = False
  
#if irrotational:
#   print("Irrotational flow")
#else:
#    print("Reynolds number = {}".format(re))

# Scale boundary values
b = 10 * scaleFactor
h = 15 * scaleFactor
w = 5 * scaleFactor

# Scale grid sizes
m = 32 * scaleFactor
n = 32 * scaleFactor

# Scale Reynold's number
re = re / (scaleFactor)

# Initialize zero array
psi = np.empty((m+2,n+2), dtype = float)
zet = np.empty((m+2,n+2), dtype = float)

#print("Running CFD on {} x {} grid in serial".format(m, n))
    
#construct psi
psi[:m+2, :n+2] = 0.0
boundaryPsi(psi, m, n, b, h, w)

#initial psitmp
psitmp = np.empty((m+2, n+2), dtype = float)
        
#update psitmp    
psitmp[:m+2, :n+2] = psi[:m+2, :n+2]   

#construct zet
zet[:m+2, :n+2] = 0.0
    
#initial zettmp
zettmp = np.empty((m+2, n+2), dtype = float)

# Find bnorm
bnorm = np.power(psi, 2).sum()


if not irrotational:
    
    boundaryZet(zet, psi, m, n, b, h, w)
    
    #update zettmp
    zettmp[:m+2, :n+2] = zet[:m+2, :n+2]       
    bnorm += np.power(zet, 2).sum()
            
bnorm = np.sqrt(bnorm)

# Begin iterative Jacobi loop
#print("Starting main loop...\n")

tStart = getTime()
zettmp = np.asarray(zettmp)
psitmp = np.asarray(psitmp)

for iter in range(1, iteration+1):
    # Calculate psi for next iteration
    if irrotational:
        jacobistep(psitmp, psi, m, n)
    else:
        jacobistepvort(zettmp, psitmp, zet, psi, m, n, re)
    
    # Calculate current error if required
    if checkError or iter == iteration:
        error = deltaSquaredError(psitmp, psi, m, n)
        if not irrotational:
            error += deltaSquaredError(zettmp, zet ,m, n)
        error = (np.sqrt(error)) / bnorm   
    
    # quit early if we have reached required tolerance
    if checkError:
        if error < tolerance:
            checkIter = iter
            #print("Converged on iteration : {} \n ".format(iter))
            #print(" iteration {} , the error is {} ".format(iter, error))
            break

    #Copy psitmp back
    psi[1:m+1, 1:n+1] = psitmp[1:m+1, 1:n+1]
    
    # Copy zettmp back
    if not irrotational:
        zet[1:m+1, 1:n+1] = zettmp[1:m+1, 1:n+1]
        # update zeta BCs that depend on psi
        boundaryZet(zet, psi, m, n, b, h, w)    

tStop = getTime()

tTotal = tStop - tStart
tIter = tTotal / checkIter

# filePsi = open("psiNUM.dat", "w")
# filePsi.write(str(psi))
# filePsi.close()

# Print out some stats

print("After {} iterations, the error is {}".format(checkIter, error))
print("Time was {} seconds".format(tTotal))
print("Each iteration took {} seconds in average".format(tIter))

# creating dat files
if argument.p==str(1):
    print("Calling plot")
    util.write_data(m, n, scaleFactor, psi, "velocityNUM.dat", "colourmapNUM.dat")
    args = ["velocityNUM.dat", "colourmapNUM.dat", "outNUM_sf" + str(scaleFactor) + "_r" + argument.r + ".png"]
    plot_flow.main(args)