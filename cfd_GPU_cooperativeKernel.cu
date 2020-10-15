/* Editor: Antonios-Kyrillos Chatzimichail, antonis.xatzimixail@gmail.com
 Project: CFD simulation of fluid flow in a cavity
 Version description: CUDA C cooperative kernel code. Uses cudaLaunchCooperativeKernel() instead of the standard <<<>>> syntax, which makes grid sync possible.
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <cooperative_groups.h>

using namespace cooperative_groups;
namespace cg = cooperative_groups;

//Threads Per Block (in one dimension)
#define TPB 16

/* wall-clock time */
double gettime(void)
{
  struct timeval tp;
  gettimeofday (&tp, NULL);
  return tp.tv_sec + tp.tv_usec/(double)1.0e6;
}

double** arraymalloc2d_device(int nx, int ny){
	double **array2d;
	cudaMallocManaged(&array2d, nx*sizeof(double*));
	for(int i=0; i<nx; i++)
		cudaMallocManaged(&array2d[i], ny*sizeof(double));
	return array2d;
}

void boundarypsi(double **psi, int m, int n, int b, int h, int w)
{

  int i,j;

  //BCs on bottom edge

  for (i=b+1;i<=b+w-1;i++)
    {
      psi[i][0] = (double)(i-b);
    }

  for (i=b+w;i<=m;i++)
    {
      psi[i][0] = (double)(w);
    }

  //BCS on RHS

  for (j=1; j <= h; j++)
    {
      psi[m+1][j] = (double) w;
    }

  for (j=h+1;j<=h+w-1; j++)
    {
      psi[m+1][j]=(double)(w-j+h);
    }
}

void boundaryzet(double **zet, double **psi, int m, int n)
{
  int i,j;

  //set top/bottom BCs:

  for (i=1;i<m+1;i++)
    {
      zet[i][0]   = 2.0*(psi[i][1]-psi[i][0]);
      zet[i][n+1] = 2.0*(psi[i][n]-psi[i][n+1]);
    }

  //set left BCs:

  for (j=1;j<n+1;j++)
    {
      zet[0][j] = 2.0*(psi[1][j]-psi[0][j]);
    }

  //set right BCs

  for (j=1;j<n+1;j++)
    {
      zet[m+1][j] = 2.0*(psi[m][j]-psi[m+1][j]);
    }
}

//one kernel does the whole job, all the iterations
//returns in retarr the error and the number of iterations made
//one GPU thread is responsible for multiple pixels, which are located every stride_step pixels (horizontally and vertically)  
__global__ void jacobikernel(double **psitmp, double **psi, int m, int n, double **dsearr, int checkerr, int numiter, int tolerance, double bnorm, int stride_step, double **retarr){
	int si, sj;	//starting i, j
	int i, j;
	si = blockIdx.x * blockDim.x + threadIdx.x;
	sj = blockIdx.y * blockDim.y + threadIdx.y;
	cg::grid_group grid = cg::this_grid();
	
	double error;
	
	for(int iter=1;iter<=numiter;iter++){
		for(i=si + 1; i<m+1; i+=stride_step){
			for(j=sj + 1; j<n+1; j+=stride_step){
				//jacobistep
				psitmp[i][j]=0.25*(psi[i-1][j]+psi[i+1][j]+psi[i][j-1]+psi[i][j+1]);
			}
		}
		
		if (checkerr || iter == numiter){
			for(i=si + 1; i<m+1; i+=stride_step){
				for(j=sj + 1; j<n+1; j+=stride_step){				
					//deltasq array
					dsearr[i-1][j-1] = psitmp[i][j]-psi[i][j];
					dsearr[i-1][j-1] *= dsearr[i-1][j-1];
				}
			}
		}
		grid.sync(); //sync after jacobistep and deltasq array
		
		if (checkerr || iter == numiter){
			error = 0.0;
			for(int dse_row=0; dse_row<m; dse_row++){
				for(int dse_col=0; dse_col<n; dse_col++)
					error += dsearr[dse_row][dse_col];
			}
			error = sqrt(error) / bnorm;
		}
		if (checkerr)
		{
			if (error < tolerance)
			{
				if(si==0 && sj==0){
					printf("Converged on iteration %d\n",iter);
					retarr[0][0] = error;
					retarr[0][1] = iter;
				}
				break;
			}
		}
		
		for(i=si + 1; i<m+1; i+=stride_step){
			for(j=sj + 1; j<n+1; j+=stride_step){
				//copy psi back			
				psi[i][j] = psitmp[i][j];
			}
		}
		grid.sync(); //sync after copy psi back
	}
	
	if(si==0 && sj==0){
		retarr[0][0] = error;
		retarr[0][1] = numiter;
	}
}

__global__ void jacobistep(double **psinew, double **psi, int m, int n)
{
  int i, j;
  i = blockIdx.x * blockDim.x + threadIdx.x;
  j = blockIdx.y * blockDim.y + threadIdx.y;
  if(i>=1 && i<m+1 && j>=1 && j<n+1)
	psinew[i][j]=0.25*(psi[i-1][j]+psi[i+1][j]+psi[i][j-1]+psi[i][j+1]);
}

__global__ void jacobivortkernel(double **zettmp, double **psitmp, double **zet, double **psi, int m, int n, double re, double **dsearr, int checkerr, int numiter, int tolerance, double bnorm, int stride_step, double **retarr){
	int si, sj;	//starting i, j
	int i,j;
	si = blockIdx.x * blockDim.x + threadIdx.x;
	sj = blockIdx.y * blockDim.y + threadIdx.y;
	cg::grid_group grid = cg::this_grid();
	
	double error;
	
	for(int iter=1;iter<=numiter;iter++){
		for(i=si + 1; i<m+1; i+=stride_step){
			for(j=sj + 1; j<n+1; j+=stride_step){
				//jacobistepvort
				psitmp[i][j]=0.25*(  psi[i-1][j]+psi[i+1][j]+psi[i][j-1]+psi[i][j+1]
					 - zet[i][j] );
				zettmp[i][j]=0.25*(zet[i-1][j]+zet[i+1][j]+zet[i][j-1]+zet[i][j+1])
					- re/16.0*(
				   (  psi[i][j+1]-psi[i][j-1])*(zet[i+1][j]-zet[i-1][j])
				   - (psi[i+1][j]-psi[i-1][j])*(zet[i][j+1]-zet[i][j-1])
				   );
				//end jacobistepvort
			}
		}
		
		
		if (checkerr || iter == numiter){
			for(i=si + 1; i<m+1; i+=stride_step){
				for(j=sj + 1; j<n+1; j+=stride_step){
					//deltasq array
					dsearr[i-1][j-1] = psitmp[i][j]-psi[i][j];
					dsearr[i-1][j-1] *= dsearr[i-1][j-1];
					dsearr[i-1][j-1] += (zettmp[i][j]-zet[i][j])*(zettmp[i][j]-zet[i][j]);
				}
			}
		}
		grid.sync(); //sync after jacobistepvort and deltasq array
		if (checkerr || iter == numiter){
			error = 0.0;
			for(int dse_row=0; dse_row<m; dse_row++){
				for(int dse_col=0; dse_col<n; dse_col++)
					error += dsearr[dse_row][dse_col];
			}
			error = sqrt(error) / bnorm;
		}
		
		//quit early if needed
		if (checkerr)
		{
			if (error < tolerance)
			{
				if(si==0 && sj==0){
					printf("Converged on iteration %d\n",iter);
					retarr[0][0] = error;
					retarr[0][1] = iter;
				}
				break;
			}
		}
		//copy psi and zet back
		for(i=si + 1; i<m+1; i+=stride_step){
			for(j=sj + 1; j<n+1; j+=stride_step){
				psi[i][j] = psitmp[i][j];
				zet[i][j] = zettmp[i][j];
			}
		}
		grid.sync(); //sync after copy psi and zet back
		for(i=si; i<m+2; i+=stride_step){
			for(j=sj; j<n+2; j+=stride_step){
				//boundaryzet
				if(i>=1 && i<m+1){
					if(j==0)
						zet[i][j]   = 2.0*(psi[i][j+1]-psi[i][j]);
					if(j==n+1)
						zet[i][j] = 2.0*(psi[i][j-1]-psi[i][j]);
				}
				if(j>=1 && j<n+1){
					if(i==0)
						zet[i][j] = 2.0*(psi[i+1][j]-psi[i][j]);
					if(i==m+1)
						zet[i][j] = 2.0*(psi[i-1][j]-psi[i][j]);
				}
			}
		}
		grid.sync(); //sync after boundaryzet
	}
	
	if(si==0 && sj==0){
		retarr[0][0] = error;
		retarr[0][1] = numiter;
	}
}

void jacobistepvort(double **zetnew, double **psinew,
		    double **zet, double **psi,
		    int m, int n, double re)
{
  int i, j;

  for(i=1;i<=m;i++)
    {
      for(j=1;j<=n;j++)
	{
	  psinew[i][j]=0.25*(  psi[i-1][j]+psi[i+1][j]+psi[i][j-1]+psi[i][j+1]
			     - zet[i][j] );
	}
    }

  for(i=1;i<=m;i++)
    {
      for(j=1;j<=n;j++)
	{
	  zetnew[i][j]=0.25*(zet[i-1][j]+zet[i+1][j]+zet[i][j-1]+zet[i][j+1])
	    - re/16.0*(
		       (  psi[i][j+1]-psi[i][j-1])*(zet[i+1][j]-zet[i-1][j])
		       - (psi[i+1][j]-psi[i-1][j])*(zet[i][j+1]-zet[i][j-1])
		       );
	}
    }
}

double deltasq(double **newarr, double **oldarr, int m, int n)
{
  int i, j;

  double dsq=0.0;
  double tmp;

  for(i=1;i<=m;i++)
    {
      for(j=1;j<=n;j++)
	{
	  tmp = newarr[i][j]-oldarr[i][j];
	  dsq += tmp*tmp;
        }
    }

  return dsq;
}

void print_array(double **array, int nx, int ny){
	for(int i=0; i<nx; i++){
		for(int j=0; j<ny; j++)
			printf("%.6lf ", array[i][j]);
		printf("\n");
	}
}

int main(int argc, char **argv)
{
  //int printfreq=1000; //output frequency
  double error, bnorm;
  double tolerance=0.0; //tolerance for convergence. <=0 means do not check

  //main arrays
  double **psi, **zet;
  //temporary versions of main arrays
  double **psitmp, **zettmp;
  //deltasqerror array and return array
  double **dsearr, **retarr;

  //command line arguments
  int scalefactor, numiter;

  double re; // Reynold's number - must be less than 3.7

  //simulation sizes
  int bbase=10;
  int hbase=15;
  int wbase=5;
  int mbase=32;
  int nbase=32;

  int irrotational = 1, checkerr = 0;

  int m,n,b,h,w;
  int iter;
  int i,j;

  double tstart, tstop, ttot, titer;
  
  dim3 tpb, nBlocks;

  //do we stop because of tolerance?
  if (tolerance > 0) {checkerr=1;}

  //check command line parameters and parse them

  if (argc <3|| argc >4)
    {
      printf("Usage: cfd <scale> <numiter> [reynolds]\n");
      return 0;
    }

  scalefactor=atoi(argv[1]);
  numiter=atoi(argv[2]);

  if (argc == 4)
    {
      re=atof(argv[3]);
      irrotational=0;
    }
  else
    {
      re=-1.0;
    }

  if(!checkerr)
    {
      printf("Scale Factor = %i, iterations = %i\n",scalefactor, numiter);
    }
  else
    {
      printf("Scale Factor = %i, iterations = %i, tolerance= %g\n",scalefactor,numiter,tolerance);
    }

  if (irrotational)
    {
      printf("Irrotational flow\n");
    }
  else
    {
      printf("Reynolds number = %f\n",re);
    }

  //Calculate b, h & w and m & n
  b = bbase*scalefactor;
  h = hbase*scalefactor;
  w = wbase*scalefactor;
  m = mbase*scalefactor;
  n = nbase*scalefactor;

  re = re / (double)scalefactor;

  printf("Running CFD on %d x %d grid in serial\n",m,n);

  //allocate arrays

  /*psi    = (double **) arraymalloc2d(m+2,n+2,sizeof(double));
  psitmp = (double **) arraymalloc2d(m+2,n+2,sizeof(double));*/
  psi = arraymalloc2d_device(m+2, n+2);
  psitmp = arraymalloc2d_device(m+2, n+2);
  
  dsearr = arraymalloc2d_device(m, n);
  retarr = arraymalloc2d_device(1, 2);

  //zero the psi array
  for (i=0;i<m+2;i++)
    {
      for(j=0;j<n+2;j++)
	{
	  psi[i][j]=0.0;
	}
    }

  if (!irrotational)
    {
      //allocate arrays

      zet =   arraymalloc2d_device(m+2, n+2);
      zettmp =arraymalloc2d_device(m+2, n+2);

      //zero the zeta array

      for (i=0;i<m+2;i++)
	{
	  for(j=0;j<n+2;j++)
	    {
	      zet[i][j]=0.0;
	    }
	}
    }
  
  //set the psi boundary conditions

  boundarypsi(psi,m,n,b,h,w);

  //compute normalisation factor for error

  bnorm=0.0;

  for (i=0;i<m+2;i++)
    {
      for (j=0;j<n+2;j++)
	{
	  bnorm += psi[i][j]*psi[i][j];
	}
    }

  if (!irrotational)
    {
      //update zeta BCs that depend on psi
      boundaryzet(zet,psi,m,n);

      //update normalisation

      for (i=0;i<m+2;i++)
	{
	  for (j=0;j<n+2;j++)
	    {
	      bnorm += zet[i][j]*zet[i][j];
	    }
	}
    }

  bnorm=sqrt(bnorm);
  
  tpb.x = TPB;
  tpb.y = TPB;
  //ensure that the number of the blocks is enough for the problem size
  nBlocks.x = (m+2)/ tpb.x +1;
  nBlocks.y = (n+2)/ tpb.y +1;

	int numBlocksPerSm = 0;
	int dev = 0;
	// Number of threads the kernel will be launched with
	int numThreads = tpb.x * tpb.y;
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, dev);
	dim3 dimBlock(tpb.x, tpb.y, 1);
	dim3 dimGrid(nBlocks.x, nBlocks.y, 1);
	cudaError_t cuError;	
	int needed_blocks = nBlocks.x * nBlocks.y;
	int stride_step = m+2;	//default value, one thread for every element 
  
  //begin Jacobi kernel

  printf("\nStarting main loop...\n\n");
  
  tstart=gettime();
	if (irrotational){
		//find out what is the max number of Blocks Per Sm (Streaming Multiprocessor)
		cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, jacobikernel, numThreads, 0);
		//calculate the max number of blocks that can be launched
		int max_blocks = deviceProp.multiProcessorCount*numBlocksPerSm;
		if(needed_blocks > max_blocks){
			//printf("\tWarning: Too many Blocks for this device!\n\tMax: %d, asked for %d\n", max_blocks, needed_blocks);
			int b1d = (int)(sqrt(max_blocks)); //blocks in 1 dimension, it will create (b1d x b1d) blocks in the grid, it is always (b1d x b1d) <= max_blocks
			stride_step = b1d * tpb.x;	//the step for each thread
			dimGrid.x = b1d;
			dimGrid.y = b1d;
		}
		void *kernelArgs[] = {&psitmp, &psi, &m, &n, &dsearr, &checkerr, &numiter, &tolerance, &bnorm, &stride_step, &retarr};
		//launch kernel with the API, not standard <<<>>> syntax
		cuError = cudaLaunchCooperativeKernel((void*)jacobikernel, dimGrid, dimBlock, kernelArgs);
		if(cuError != cudaSuccess)
			printf("\tKernel launch unsuccessful!\n");
		else
			printf("\tKernel launched with %dx%d blocks and stride_step = %d\n", dimGrid.x, dimGrid.y, stride_step);
	}
	else{
		cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, jacobivortkernel, numThreads, 0);
		int max_blocks = deviceProp.multiProcessorCount*numBlocksPerSm;
		if(needed_blocks > max_blocks){
			//printf("\tWarning: Too many Blocks for this device!\n\tMax: %d, asked for %d\n", max_blocks, needed_blocks);
			int b1d = (int)(sqrt(max_blocks)); //blocks in 1 dimension
			stride_step = b1d * tpb.x;
			dimGrid.x = b1d;
			dimGrid.y = b1d;
		}
		void *kernelArgs[] = {&zettmp, &psitmp, &zet, &psi, &m, &n, &re, &dsearr, &checkerr, &numiter, &tolerance, &bnorm, &stride_step, &retarr};
		cuError = cudaLaunchCooperativeKernel((void*)jacobivortkernel, dimGrid, dimBlock, kernelArgs);
		if(cuError != cudaSuccess)
			printf("\tKernel launch unsuccessful!\n");
		else
			printf("\tKernel launched with %dx%d blocks and stride_step = %d\n", dimGrid.x, dimGrid.y, stride_step);
	}
	//wait the GPU work to end before continuing
	cudaDeviceSynchronize();
	//cudaDeviceReset();
  /*for(iter=1;iter<=numiter;iter++)
    {
      //calculate psi for next iteration

      if (irrotational)
	{
	  //jacobistep(psitmp,psi,m,n);
	  jacobistep<<<nBlocks, tpb>>>(psitmp,psi,m,n);
	  cudaDeviceSynchronize();
	}
      else
	{
	  jacobistepvort(zettmp,psitmp,zet,psi,m,n,re);
	}

      //calculate current error if required

      if (checkerr || iter == numiter)
	{
	  error = deltasq(psitmp,psi,m,n);

	  if(!irrotational)
	    {
	      error += deltasq(zettmp,zet,m,n);
	    }

	  error=sqrt(error);
	  error=error/bnorm;
	}

      //quit early if we have reached required tolerance

      if (checkerr)
	{
	  if (error < tolerance)
	    {
	      printf("Converged on iteration %d\n",iter);
	      break;
	    }
	}

      //copy back

      for(i=1;i<=m;i++)
	{
	  for(j=1;j<=n;j++)
	    {
	      psi[i][j]=psitmp[i][j];
	    }
	}

      if (!irrotational)
	{
	  for(i=1;i<=m;i++)
	    {
	      for(j=1;j<=n;j++)
		{
		  zet[i][j]=zettmp[i][j];
		}
	    }
	}

      if (!irrotational)
	{
	  //update zeta BCs that depend on psi
	  boundaryzet(zet,psi,m,n);
	}

      //print loop information

      if(iter%printfreq == 0)
	{
	  if (!checkerr)
	    {
	      printf("Completed iteration %d\n",iter);
	    }
	  else
	    {
	      printf("Completed iteration %d, error = %g\n",iter,error);
	    }
	}
    }

  if (iter > numiter) iter=numiter;
*/
	error = retarr[0][0];
	iter  = retarr[0][1];

  tstop=gettime();

  ttot=tstop-tstart;
  titer=ttot/(double)iter;


  //print out some stats

  printf("\n... finished\n");
  printf("After %d iterations, the error is %g\n",iter,error);
  printf("Time for %d iterations was %g seconds\n",iter,ttot);
  printf("Each iteration took %g seconds\n",titer);

  //output results
/*
  writedatafiles(psi,m,n, scalefactor);

  writeplotfile(m,n,scalefactor);
*/
  //free un-needed arrays
  /*free(psi);
  free(psitmp);*/
  cudaFree(psi);
  cudaFree(psitmp);

  if (!irrotational)
    {
      /*free(zet);
      free(zettmp);*/
	  cudaFree(zet);
	  cudaFree(zettmp);
    }

  printf("... finished\n");

  return 0;
}
