# Editor: Antonios-Kyrillos Chatzimichail, antonis.xatzimixail@gmail.com
# Project: CFD simulation of fluid flow in a cavity
# Arguments: -sf <scaleFactor> -i <iterations> -r <reynolds number> -t <tolerance> -p <plot image?>
# Example for scale factor 4, 5000 iterations, irrotational flow, no tolerance check and export output image should be:
# -sf 4 -i 5000 -r 0.0 -t 0.0 -p 1
# Version description: Numba CUDA code for GPU. CPU launches GPU kernels for the calculations.

from time import time as getTime
import numpy as np
import argparse
# Import the local "util.py" methods
import util
# Import the local "plot_flow.py" methods
import plot_flow
from numba import cuda
import math

def boundaryPsi(psi, m, n, b, h, w):
    # BCs on bottom edge
    psi[b+1:b+w, 0] = [float(i) for i in range(1, w)]
    psi[b+w:m+1, 0] = float(w)
  
    # BCS on RHS
    psi[m+1, 1:h+1] = float(w)
    psi[m+1, h+1:h+w] = [float(w - i) for i in range(1, w)]

# a function is turned into a GPU kernel by adding the "@cuda.jit" decorator
# when the kernel is launched, the function code is executed by every single GPU thread
# row and col are the coordinates of a thread inside the grid
@cuda.jit
def boundaryZet(zet, psi, m, n, b, h, w):
    row, col = cuda.grid(2)
    # Set top/bottom BCs
    if col==0 and (row >= 1 and row < m+1):
        zet[row, col] = 2.0 * (psi[row, col+1] - psi[row, col])
    if col== n+1 and (row >= 1 and row < m+1):
        zet[row, col] = 2.0 * (psi[row, col-1] - psi[row, col])
        
    # Set left BCs
    if row==0 and (col >= 1 and col < n+1):
        zet[row, col] = 2.0 * (psi[row+1, col] - psi[row, col])
        
    # Set right BCs
    if row== m+1 and (col >= 1 and col < n+1):
        zet[row, col] = 2.0 * (psi[row-1, col] - psi[row, col])

# each thread is responsible for only one point in the matrix, the one with the corresponding coordinates
@cuda.jit
def jacobistep(psitmp, psi, m, n):
    row, col = cuda.grid(2)
    if (row >= 1 and row < m+1) and (col >= 1 and col < n+1):
        psitmp[row, col] = 0.25 * (psi[row+1, col] + psi[row-1, col] + psi[row, col+1] + psi[row, col-1])
    # psitmp[1:m+1, 1:n+1] = 0.25 * (psi[2:m+2, 1:n+1] + psi[0:m, 1:n+1] + psi[1:m+1, 2:n+2] + psi[1:m+1, 0:n])

# kernel that copies one array to another element-wise
@cuda.jit
def copy_array_d2d(src, dst, m, n):
    row, col = cuda.grid(2)
    if (row >= 1 and row < m + 1) and (col >= 1 and col < n + 1):
        dst[row, col] = src[row, col]

# reduction kernel, helps to add the elements of an 1-D array using the GPU
@cuda.reduce
def sum_reduce(a, b):
    return a + b
    
@cuda.jit
def jacobistepvort(zettmp, psitmp, zet, psi, m, n, re):
    row, col = cuda.grid(2)
    if (row >= 1 and row < m + 1) and (col >= 1 and col < n + 1):
        psitmp[row, col] = 0.25 * (psi[row+1, col] + psi[row-1, col] + psi[row, col+1] + psi[row, col-1] - zet[row, col])
        zettmp[row, col] = 0.25 * (zet[row+1, col] + zet[row-1, col] + zet[row, col+1] + zet[row, col-1]) - re / 16.0 * ((psi[row, col-1] - psi[row, col+1]) * (zet[row-1, col] - zet[row+1, col]) - (zet[row, col-1] - zet[row, col+1]) * (psi[row-1, col] - psi[row+1, col]))

# this kernel now is calculating the squared differences and stores them to an array
# the delta squared error is the sum of the elements of this array (the sum is done separately)
@cuda.jit
def deltaSquaredError(storarr, newarr, oldarr, m, n):
    row, col = cuda.grid(2)
    if (row >= 1 and row < m + 1) and (col >= 1 and col < n + 1):
        storarr[row-1, col-1] = (newarr[row, col] - oldarr[row, col])**2

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

#construct zet
zet[:m+2, :n+2] = 0.0
    
#initial zettmp
# zettmp = np.empty((m+2, n+2), dtype = float)

# Find bnorm
bnorm = np.power(psi, 2).sum()

# allocate memory on GPU and copy the array there
d_psi = cuda.to_device(psi)
# allocate memory on GPU
d_psitmp = cuda.device_array((m+2, n+2))
d_zet = cuda.to_device(zet)
d_zettmp = cuda.device_array((m+2, n+2))
d_array_dse = cuda.device_array((m, n))
# specify the number of threads per block for the kernel launch
threadsperblock = (16, 16)
# specify the number of blocks per grid for the kernel launch,
# make sure that there are enough blocks for the problem size, having more is ok, having less will produce wrong result
blockspergrid_x = int(math.ceil(m+2 / threadsperblock[0]))
blockspergrid_y = int(math.ceil(n+2 / threadsperblock[1]))
blockspergrid = (blockspergrid_x, blockspergrid_y)

# update psitmp
# psitmp[:m+2, :n+2] = psi[:m+2, :n+2]

# copy the array by launching a kernel
copy_array_d2d[blockspergrid, threadsperblock](d_psi, d_psitmp, m, n)

if not irrotational:
    
    boundaryZet[blockspergrid, threadsperblock](d_zet, d_psi, m, n, b, h, w)
    # copy zet back to host (CPU)
    zet = d_zet.copy_to_host()
    #update zettmp
    # zettmp[:m+2, :n+2] = zet[:m+2, :n+2]
    copy_array_d2d[blockspergrid, threadsperblock](d_zet, d_zettmp, m, n)
    bnorm += np.power(zet, 2).sum()
            
bnorm = np.sqrt(bnorm)

# Begin iterative Jacobi loop
#print("Starting main loop...\n")

tStart = getTime()
# zettmp = np.asarray(zettmp)
# psitmp = np.asarray(psitmp)

for iter in range(1, iteration+1):
    # Calculate psi for next iteration
    if irrotational:
        jacobistep[blockspergrid, threadsperblock](d_psitmp, d_psi, m, n)
    else:
        jacobistepvort[blockspergrid, threadsperblock](d_zettmp, d_psitmp, d_zet, d_psi, m, n, re)
    
    # Calculate current error if required
    if checkError or iter == iteration:
        deltaSquaredError[blockspergrid, threadsperblock](d_array_dse, d_psitmp, d_psi, m, n)
        error = 0.0
        # the sum of d_array_dse has to be calculated row by row because sum_reduce supports only 1-D arrays
        for dse_row in range(m):
            error += sum_reduce(d_array_dse[dse_row, :n])
        if not irrotational:
            deltaSquaredError[blockspergrid, threadsperblock](d_array_dse, d_zettmp, d_zet, m, n)
            for dse_row in range(m):
                error += sum_reduce(d_array_dse[dse_row, :n])
        error = (np.sqrt(error)) / bnorm   
    
    # quit early if we have reached required tolerance
    if checkError:
        if error < tolerance:
            checkIter = iter
            #print("Converged on iteration : {} \n ".format(iter))
            #print(" iteration {} , the error is {} ".format(iter, error))
            break

    #Copy psitmp back
    # psi[1:m+1, 1:n+1] = psitmp[1:m+1, 1:n+1]
    copy_array_d2d[blockspergrid, threadsperblock](d_psitmp, d_psi, m, n)
    
    # Copy zettmp back
    if not irrotational:
        # zet[1:m+1, 1:n+1] = zettmp[1:m+1, 1:n+1]
        copy_array_d2d[blockspergrid, threadsperblock](d_zettmp, d_zet, m, n)
        # update zeta BCs that depend on psi
        boundaryZet[blockspergrid, threadsperblock](d_zet, d_psi, m, n, b, h, w)

tStop = getTime()

tTotal = tStop - tStart
tIter = tTotal / checkIter

# retrieve psi matrix back from the GPU
psi = d_psi.copy_to_host()

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
