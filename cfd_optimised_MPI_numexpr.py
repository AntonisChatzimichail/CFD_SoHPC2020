# Editor: Antonios-Kyrillos Chatzimichail, antonis.xatzimixail@gmail.com
# Project: CFD simulation of fluid flow in a cavity
# Arguments: -sf <scaleFactor> -i <iterations> -r <reynolds number> -t <tolerance> -p <plot image?>
# Example for scale factor 4, 5000 iterations, irrotational flow, no tolerance check and export output image should be:
# -sf 4 -i 5000 -r 0.0 -t 0.0 -p 1
# Version description: Optimised MPI code using Numexpr module.

import numpy as np
from mpi4py import MPI
import argparse
# Import the local "util.py" methods
import util
# Import the local "plot_flow.py" methods
import plot_flow
import numexpr as ne

def boundaryPsi(psi, m, n, b, h, w, comm):
    rank = comm.Get_rank()
    size = comm.Get_size()
    istart = m*rank + 1
    istop = istart + m -1
    # BCs on bottom edge
    for i in range(b + 1, b + w):
        if (i >= istart and i <= istop):
            psi[i-istart+1][0] = float(i-b)
    for i in range(b + w, m*size + 1):
        if (i >= istart and i <= istop):
            psi[i-istart+1][0] = float(w)
    # BCS on RHS
    if rank == size-1:
      for j in range(1, h + 1):
          psi[m+1][j] = float(w)
      for j in range(h + 1, h + w):
          psi[m+1][j]=float(w-j+h)

def boundaryZet(zet, psi, m, n, b, h, w, comm):
    rank = comm.Get_rank()
    size = comm.Get_size()
    # Set top/bottom BCs
    zet[1:m+1, 0] = 2.0 * (psi[1:m+1, 1] - psi[1:m+1, 0])
    zet[1:m+1, n+1] = 2.0 * (psi[1:m+1, n] - psi[1:m+1, n+1])
    
    # set left BC:
    if rank == 0:
        zet[0, 1:n+1] = 2.0 * (psi[1, 1:n+1] - psi[0, 1:n+1])
    # set right BCs
    if rank == size-1:
        zet[m+1, 1:n+1] = 2.0 * (psi[m, 1:n+1] - psi[m+1, 1:n+1]) 
    
#calculations for jacobi steps are made faster with Numexpr (using ne.evaluate) that with Numpy
#to use ne.evaluate properly, we have to store the arrays in different variables (can't use psi[2:m+2, 1:n+1] inside the parameter string)
def jacobistep(psitmp, psi, m, n):
    # psitmp[1:m+1, 1:n+1] = 0.25 * (psi[2:m+2, 1:n+1] + psi[0:m, 1:n+1] + psi[1:m+1, 2:n+2] + psi[1:m+1, 0:n])
    d = psi[2:m + 2, 1:n + 1]
    u = psi[0:m, 1:n + 1]
    r = psi[1:m + 1, 2:n + 2]
    l = psi[1:m + 1, 0:n]
    psitmp[1:m + 1, 1:n + 1] = ne.evaluate("0.25 * (d + u + r + l)")
    
def jacobistepvort(zettmp, psitmp, zet, psi, m, n, re):
    # psitmp[1:m+1, 1:n+1] = 0.25 * (psi[2:m+2, 1:n+1] + psi[0:m, 1:n+1] + psi[1:m+1, 2:n+2] + psi[1:m+1, 0:n] - zet[1:m+1, 1:n+1])
    # zettmp[1:m+1, 1:n+1] = 0.25 * (zet[2:m+2, 1:n+1] + zet[0:m, 1:n+1] + zet[1:m+1, 2:n+2] + zet[1:m+1, 0:n]) - re / 16.0 * ((psi[1:m+1, 0:n] - psi[1:m+1, 2:n+2]) * (zet[0:m, 1:n+1] - zet[2:m+2, 1:n+1]) - (zet[1:m+1, 0:n] - zet[1:m+1, 2:n+2]) * (psi[0:m, 1:n+1] - psi[2:m+2, 1:n+1]))
    pd = psi[2:m + 2, 1:n + 1]
    pu = psi[0:m, 1:n + 1]
    pr = psi[1:m + 1, 2:n + 2]
    pl = psi[1:m + 1, 0:n]
    zc = zet[1:m + 1, 1:n + 1]
    zd = zet[2:m + 2, 1:n + 1]
    zu = zet[0:m, 1:n + 1]
    zr = zet[1:m + 1, 2:n + 2]
    zl = zet[1:m + 1, 0:n]
    psitmp[1:m + 1, 1:n + 1] = ne.evaluate("0.25 * (pd + pu + pr + pl - zc)")
    zettmp[1:m + 1, 1:n + 1] = ne.evaluate("0.25 * (zd + zu + zr + zl) - re / 16.0 * ((pl - pr) * (zu - zd) - (zl - zr) * (pu - pd))")

def deltaSquaredError(newarr, oldarr, m, n):
    return np.power(newarr[1: m+1, 1:n+1] - oldarr[1: m+1, 1:n+1], 2).sum()

def haloswap(x,lm,n,comm): 
    tag = 1
    status = MPI.Status()
    rank = comm.Get_rank()
    size = comm.Get_size()

    # no need to halo swap if serial:
    if size > 1:
        # send right boundaries and receive left ones
        if rank == 0:
            comm.Send(x[lm][1:n+1], rank+1, tag)
        elif rank == size-1:
            comm.Recv(x[0][1:n+1], rank-1, tag, status)
        else:
            comm.Sendrecv(x[lm][1:n+1], rank+1, tag, x[0][1:n+1], rank-1, tag, status)
        # send left boundary and receive right
        if rank == 0:
            comm.Recv(x[lm+1][1:n+1], rank+1, tag, status)
        elif rank == size-1:
            comm.Send(x[1][1:n+1], rank-1, tag)
        else:
            comm.Sendrecv(x[1][1:n+1], rank-1, tag, x[lm+1][1:n+1], rank+1, tag, status)

# Arguments

parser = argparse.ArgumentParser()

parser.add_argument("-sf")
parser.add_argument("-i")
parser.add_argument("-r")
parser.add_argument("-t")
parser.add_argument("-p")   # plot or not

argument = parser.parse_args()

scaleFactor = int(argument.sf)  # Scale factor of simulation sizes
iteration = int(argument.i) # Number of iterations
printFrequence = 100000  # Print frequence through iterations
re = float(argument.r)  # Reynold's number in Jacobi - must be less than 3.7
tolerance = float(argument.t)  # Tolerance for convergence

# Initalize irrotationality
irrotational = True

# Parallelisation parameters
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


# Check errors if tolerance is provided
checkError = tolerance > 0

# Reynold's number check
if re == 0:
    re = -1.0
else:
    re = float(re)
    irrotational = False

if not checkError:
    if rank == 0:
        print("Scale Factor = {} | Iterations = {}".format(scaleFactor, iteration))
else:
    if rank == 0:
        print("Scale Factor = {} | Iterations = {} | Tolerance = {}".format(scaleFactor, iteration, tolerance))
  
if irrotational:
    if rank == 0:
        print("Irrotational flow")
else:
    if rank == 0:
        print("Reynolds number = {}".format(re))

# broadcast runtime parameters to other processors
parameters = np.asarray((scaleFactor, iteration, re, irrotational))
comm.Bcast(parameters, root=0)
scaleFactor, iteration, re, irrotational = parameters
scaleFactor = int(scaleFactor)
iteration = int(iteration)
irrotational = irrotational == 1.0


# Scale boundary values
b = 10 * scaleFactor
h = 15 * scaleFactor
w = 5 * scaleFactor

# Scale grid sizes
m = 32 * scaleFactor
n = 32 * scaleFactor

# Scale Reynold's number
re = re / (scaleFactor)

# calculate local size
lm = m // size

# consistency check
if (size*lm != m):
    if rank == 0:
        print("ERROR: m= {}  does not divide onto {} processes".format(m, size))
    MPI.Finalize()
if rank == 0:
    print("Running CFD on {} x {} grid using {} process(es) ".format(m,n,size))

# Initialize zero array
psi = np.empty((lm+2,n+2), dtype = float)
zet = np.empty((lm+2,n+2), dtype = float)
zettmp = np.empty((lm+2,n+2), dtype = float)
    
#construct psi
psi[:lm+2, :n+2] = 0.0

#initial psitmp
psitmp = np.empty((lm+2, n+2), dtype = float)
        
#update psitmp    
psitmp[:lm+2, :n+2] = psi[:lm+2, :n+2]   

if not irrotational:
    #construct zet
    zet[:lm+2, :n+2] = 0.0

boundaryPsi(psi, lm, n, b, h, w, comm)

# Find bnorm
localbnorm = np.array([np.power(psi, 2).sum()])
bnorm = np.array([0.0])
haloswap(psi,lm,n,comm)
#boundary swap of psi

if not irrotational:
    boundaryZet(zet, psi, lm, n, b, h, w, comm)
    localbnorm += np.power(zet, 2).sum()
    #boundary swap zeta
    haloswap(zet,lm,n,comm)
     
comm.Allreduce(sendbuf=localbnorm, recvbuf=bnorm, op=MPI.SUM)     
   
#get global bnorm            
bnorm = np.sqrt(bnorm[0])

    
# barrier for accurate timing - not needed for correctness
comm.Barrier()

tStart = MPI.Wtime()

for iter in range(1, iteration+1):
    # Calculate psi for next iteration
    if irrotational:
        jacobistep(psitmp, psi, lm, n)
    else:
        jacobistepvort(zettmp, psitmp, zet, psi, lm, n, re)
    
    # Calculate current error if required
    if checkError or iter == iteration:
        localerror = np.array([deltaSquaredError(psitmp, psi, lm, n)])
        error = np.array([0.0])
        if not irrotational:
            localerror += deltaSquaredError(zettmp, zet ,lm, n)
        comm.Allreduce(localerror, error, op=MPI.SUM)
        error = (np.sqrt(error)) / bnorm    

    # quit early if we have reached required tolerance
    if checkError:
        if error < tolerance:
            if rank == 0:
                #print("Converged on iteration : {} \n ".format(iter))
                print(" iteration {} , the error is {} ".format(iter, error))
            break

    #Copy psitmp back
    psi[1:lm+1, 1:n+1] = psitmp[1:lm+1, 1:n+1]
    
    # Copy zettmp back
    if not irrotational:
        zet[1:lm+1, 1:n+1] = zettmp[1:lm+1, 1:n+1]
  
    # do a boundary swap
    haloswap(psi,lm,n,comm);

    if not irrotational:
        haloswap(zet,lm,n,comm);
        # update zeta BCs that depend on psi
        boundaryZet(zet, psi, lm, n, b, h, w, comm)
   
comm.Barrier()
tStop = MPI.Wtime()

tTotal = tStop - tStart
tIter = tTotal / iteration

# Print out some stats
if rank==0:
    print("After {} iterations, the error is {}".format(iteration, error))
    print("Time was {} seconds".format(tTotal))
    print("Each iteration took {} seconds in average".format(tIter))

gatheredPsi = comm.gather(psi, 0)

# drop dat files
if rank == 0 and argument.p==str(1):
    finalPsi = np.empty((m + 2, n + 2), dtype=float)
    finalPsi[:m + 2, :n + 2] = 0.0
    # copy first (boundary) line
    finalPsi[0, :n+2] = gatheredPsi[0][0, :n+2]
    # copy last (boundary) line
    finalPsi[m+1, :n + 2] = gatheredPsi[len(gatheredPsi)-1][lm+1, :n + 2]
    # copy the rest of the lines
    for i in range(len(gatheredPsi)):
        finalPsi[i*lm+1:(i+1)*lm+1, 0:n+2] = gatheredPsi[i][1:lm+1, 0:n+2]      #i=0, 1:lm+1 | i=1, lm+1:2*lm+1 | ... | i=3, 3*lm+1:4*lm+1
    print("Calling plot")
    util.write_data(m, n, scaleFactor, finalPsi, "velocityMPI.dat", "colourmapMPI.dat")
    args = ["velocityMPI.dat", "colourmapMPI.dat", "outMPI_sf"+str(scaleFactor)+"_r"+argument.r+".png"]
    plot_flow.main(args)
