/* Editor: Antonios-Kyrillos Chatzimichail, antonis.xatzimixail@gmail.com
 Project: CFD simulation of fluid flow in a cavity
 Version description: CUDA C standard kernels code. Uses the standard <<<>>> syntax to launch kernels on GPU. Kernels do small jobs, because there is no grid sync among threads.
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
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

__global__ void boundaryzet(double **zet, double **psi, int m, int n)
{
	int i,j;
	i = blockIdx.x * blockDim.x + threadIdx.x;
	j = blockIdx.y * blockDim.y + threadIdx.y;
  //set top/bottom BCs:
	if(i>=1 && i<m+1)
    {
      if(j==0)
		zet[i][j] = 2.0*(psi[i][j+1]-psi[i][j]);
      if(j==n+1)
		zet[i][j] = 2.0*(psi[i][j-1]-psi[i][j]);
    }

  //set left/right BCs:
	if(j>=1 && j<n+1)
    {
      if(i==0)
		zet[i][j] = 2.0*(psi[i+1][j]-psi[i][j]);
	  if(i==m+1)
		zet[i][j] = 2.0*(psi[i-1][j]-psi[i][j]);
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

__global__ void jacobistepvort(double **zetnew, double **psinew,
		    double **zet, double **psi,
		    int m, int n, double re)
{
  int i, j;
  i = blockIdx.x * blockDim.x + threadIdx.x;
  j = blockIdx.y * blockDim.y + threadIdx.y;
  if(i>=1 && i<m+1 && j>=1 && j<n+1){
	  psinew[i][j]=0.25*(  psi[i-1][j]+psi[i+1][j]+psi[i][j-1]+psi[i][j+1]
			     - zet[i][j] );
	  zetnew[i][j]=0.25*(zet[i-1][j]+zet[i+1][j]+zet[i][j-1]+zet[i][j+1])
	    - re/16.0*(
		       (  psi[i][j+1]-psi[i][j-1])*(zet[i+1][j]-zet[i-1][j])
		       - (psi[i+1][j]-psi[i-1][j])*(zet[i][j+1]-zet[i][j-1])
		       );
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

__global__ void copy_array_d2d(double **src, double **dst, int m, int n){
	int i, j;
	i = blockIdx.x * blockDim.x + threadIdx.x;
	j = blockIdx.y * blockDim.y + threadIdx.y;
	if(i>=1 && i<m+1 && j>=1 && j<n+1)
		dst[i][j] = src[i][j];
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
  
  tpb.x = TPB;
  tpb.y = TPB;
  //ensure that the number of the blocks is enough for the problem size
  nBlocks.x = (m+2)/ tpb.x +1;
  nBlocks.y = (n+2)/ tpb.y +1;

  printf("Running CFD on %d x %d grid in serial\n",m,n);

  //allocate arrays

  /*psi    = (double **) arraymalloc2d(m+2,n+2,sizeof(double));
  psitmp = (double **) arraymalloc2d(m+2,n+2,sizeof(double));*/
  psi = arraymalloc2d_device(m+2, n+2);
  psitmp = arraymalloc2d_device(m+2, n+2);

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

      /*zet =   (double **) arraymalloc2d(m+2,n+2,sizeof(double));
      zettmp =(double **) arraymalloc2d(m+2,n+2,sizeof(double));*/
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
      boundaryzet<<<nBlocks, tpb>>>(zet,psi,m,n);
	  cudaDeviceSynchronize();

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
  
  //begin iterative Jacobi loop

  printf("\nStarting main loop...\n\n");
  
  tstart=gettime();

  for(iter=1;iter<=numiter;iter++)
    {
      //calculate psi for next iteration

      if (irrotational)
	{
	  jacobistep<<<nBlocks, tpb>>>(psitmp,psi,m,n);
	}
      else
	{
	  jacobistepvort<<<nBlocks, tpb>>>(zettmp,psitmp,zet,psi,m,n,re);
	}
	//wait the GPU work to end before continuing (the alternative of synchronising threads)
	cudaDeviceSynchronize();
	
      //calculate current error if required

      //deltasq is done on the CPU because, typically, the tolerance parameter is 0, so this is done only once in the last iteration 
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

      /*for(i=1;i<=m;i++)
	{
	  for(j=1;j<=n;j++)
	    {
	      psi[i][j]=psitmp[i][j];
	    }
	}*/
	
	//copy psi back row by row
	/*for(i=1;i<=m;i++)
		memcpy(&psi[i][1], &psitmp[i][1], n*sizeof(double));*/
	
	copy_array_d2d<<<nBlocks, tpb>>>(psitmp, psi, m, n);

      if (!irrotational)
	{
	  /*for(i=1;i<=m;i++)
	    {
	      for(j=1;j<=n;j++)
		{
		  zet[i][j]=zettmp[i][j];
		}
	    }*/
		copy_array_d2d<<<nBlocks, tpb>>>(zettmp, zet, m, n);
	}
	cudaDeviceSynchronize();
      if (!irrotational)
	{
	  //update zeta BCs that depend on psi
	  boundaryzet<<<nBlocks, tpb>>>(zet,psi,m,n);
	}
	cudaDeviceSynchronize();
    }

  cudaDeviceSynchronize();
  if (iter > numiter) iter=numiter;

  tstop=gettime();

  ttot=tstop-tstart;
  titer=ttot/(double)iter;


  //print out some stats

  printf("\n... finished\n");
  printf("After %d iterations, the error is %g\n",iter,error);
  printf("Time for %d iterations was %g seconds\n",iter,ttot);
  printf("Each iteration took %g seconds\n",titer);

  //output results

  //writedatafiles(psi,m,n, scalefactor);

  //writeplotfile(m,n,scalefactor);

  //free un-needed arrays
  cudaFree(psi);
  cudaFree(psitmp);

  if (!irrotational)
    {
      cudaFree(zet);
      cudaFree(zettmp);
    }

  printf("... finished\n");

  return 0;
}
