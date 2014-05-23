#include<stdio.h>
#include<stdlib.h>
#include<assert.h>
#include<math.h>
#include<string.h>
#include "mpi.h"


int main(int argc, char **argv)
{
	int i, j, k, x, y;
	int nprocs, mype;
	double starttime, endtime;
	int N = 400; /* Matrix Dimensions*/
	int NT = 20000; /* Number of timesteps*/
	double L = 1.0; /* Physical Domain Length*/
	double T = 1.0e6; /* Total Physical Timespan*/
	double u = 5.0e-7; /* X velocity Scalar*/
	double v1 = 0.57; /* Y velocity Scalar*/
	double x0 = .5, y0 = .5;
	double sd = 0.25;
	/*create file pointer, bring in file with inits via argv - future
	 *      */
	FILE *fp;
	fp  = fopen(argv[1], "r");
	fscanf(fp, "%d %d %lf %lf %lf %lf", &N, &NT, &L, &T, &u, &v1);
	fclose(fp);

	/*derived inits*/
	double v = u*v1; // Y velocity Scalar
	double delta = L/N; /*derived inits*/
	double delT = T/NT; /*derived inits*/
	double d = 0.5*delT/delta;

	assert(delT <= (delta/(sqrt(2)*sqrt(u*u+v*v))));

	MPI_Status stat;
	MPI_Request req, req2;
	MPI_Comm comm2d;

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
	MPI_Comm_rank(MPI_COMM_WORLD, &mype);
        assert((int)sqrt(nprocs) * (int)sqrt(nprocs) == nprocs);
	int mysize = (N/sqrt(nprocs))+2;
	/* contiguous memory bitches */
	double **matrix = (double **) malloc (sizeof(double) * mysize);
	double *data = (double *) malloc (sizeof(double) * mysize * mysize);
	double **matrix1 = (double **) malloc (sizeof(double) * mysize);
	double *data1 = (double *) malloc (sizeof(double) * mysize * mysize);

	for(i=0;i<mysize;i++)
	{
		*(matrix+i) = data + mysize*i;
		*(matrix1+i) = data1 + mysize*i;
	}

	int rows = mysize;
	int cols = mysize;
	int mystart, myend;
	int dims[2];
	int periods[2];
	int coords[2];
	periods[0] = 1;
	periods[1] = 1;
	dims[0] = nprocs/sqrt(nprocs);
	dims[1] = nprocs/sqrt(nprocs);
	int source, dest;
	int up, down;

	/* ghost cell init */
	MPI_Datatype ghost_type;
	MPI_Type_vector( mysize-2 , 1 , mysize , MPI_DOUBLE , &ghost_type );
	MPI_Type_commit( &ghost_type );

	/* who am i and where do i start */
	MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &comm2d);
	MPI_Cart_coords(comm2d, mype, 2, coords);
	mystart = coords[0]*mysize*sqrt(nprocs)+coords[1]*mysize;
	myend = mystart + mysize;

	/************** BEGIN ***********/
	/************** INIT Gaussian ***********/
	for(i=1;i<mysize-1;i++){
		for(j=1;j<mysize-1;j++){
			*(*(matrix+i)+j) =  exp(-((((i+mystart)*delta-x0)*((i+mystart)*delta-x0))/(2*sd*sd) + (((j+mystart)*delta-y0)*((j+mystart)*delta-y0))/(2*sd*sd)));
			*(*(matrix1+i)+j) =  exp(-((((i+mystart)*delta-x0)*((i+mystart)*delta-x0))/(2*sd*sd) + (((j+mystart)*delta-y0)*((j+mystart)*delta-y0))/(2*sd*sd)));
		}
	}
	/************** INIT Gaussian ***********/
	/************** END ***********/

	/************** BEGIN INITIAL FILL****************/
	/************Cart Shift ----  Sendrecv row and ghost**********/
	MPI_Cart_shift(comm2d, 0, 1, &source, &dest);
	MPI_Isend(*(matrix+1),mysize, MPI_DOUBLE, source, 0, comm2d, &req);
	MPI_Irecv(*(matrix+(mysize-1)), mysize, MPI_DOUBLE, dest, MPI_ANY_TAG, comm2d, &req2);			   
	MPI_Wait(&req, &stat);
	MPI_Wait(&req2, &stat);
	MPI_Isend(*(matrix+(mysize-2)),mysize, MPI_DOUBLE, dest, 0, comm2d, &req);
	MPI_Irecv(*matrix, mysize, MPI_DOUBLE, source, MPI_ANY_TAG, comm2d, &req2);			   
	MPI_Wait(&req, &stat);
	MPI_Wait(&req2, &stat);
	MPI_Cart_shift(comm2d, 1, 1, &source, &dest);
	MPI_Isend(*(matrix+1)+1,1, ghost_type, source, 0, comm2d, &req);
	MPI_Irecv(*(matrix+1)+(mysize-1), 1, ghost_type, dest, MPI_ANY_TAG, comm2d, &req2);			   
	MPI_Wait(&req, &stat);
	MPI_Wait(&req2, &stat);
	MPI_Isend(*(matrix+1)+mysize-2, 1, ghost_type, dest, 0, comm2d, &req);
	MPI_Irecv(*(matrix+1), 1, ghost_type, source, MPI_ANY_TAG, comm2d, &req2);			   
	MPI_Wait(&req, &stat);
	MPI_Wait(&req2, &stat);
	/************Cart Shift ----  Sendrecv row and ghost**********/
	/****************** END INITIAL FILL**************/

	if(mype ==0){
		starttime = MPI_Wtime();
	}
	/********** outer timestep loop *************/
	for(k = 0; k < NT; k++)
	{

		/************** BEGIN ***********/
		/************** check print for matrix and matrix1 ***********/
		/*
		   if(k == 15000)
		   {
		   for(x=0;x<nprocs;x++){
		   for(y=0;y<nprocs;y++){
		   if(coords[0] == x && coords[1] == y){
		   for(i=0;i<mysize;i++)
		   {
		   for(j=0;j<mysize;j++)
		   {
		   printf("%f, ", *(*(matrix + i) + j));
		   }
		   printf("\n");
		   }
		   printf("\n");
		   }
		   }
		   }
		   }
		   */
		/************** END ***********/
		/************** check print for matrix and matrix1 ***********/


		/*********** exchange n-1 with n data ***********/
		for(x=0; x<mysize;x++)
		{
			for(y = 0;y<mysize;y++)
			{
				*(*(matrix1+x)+y) = *(*(matrix + x)+y);
			}

		}
		MPI_Barrier(MPI_COMM_WORLD);

		/************Cart Shift ----  Sendrecv row and ghost**********/
		MPI_Cart_shift(comm2d, 0, 1, &source, &dest);
		MPI_Isend(*(matrix+1),mysize, MPI_DOUBLE, source, 0, comm2d, &req);
		MPI_Irecv(*(matrix+(mysize-1)), mysize, MPI_DOUBLE, dest, MPI_ANY_TAG, comm2d, &req2);			   
		MPI_Wait(&req, &stat);
		MPI_Wait(&req2, &stat);
		MPI_Isend(*(matrix+(mysize-2)),mysize, MPI_DOUBLE, dest, 0, comm2d, &req);
		MPI_Irecv(*matrix, mysize, MPI_DOUBLE, source, MPI_ANY_TAG, comm2d, &req2);			   
		MPI_Wait(&req, &stat);
		MPI_Wait(&req2, &stat);
		MPI_Cart_shift(comm2d, 1, 1, &source, &dest);
		MPI_Isend(*(matrix+1)+1,1, ghost_type, source, 0, comm2d, &req);
		MPI_Irecv(*(matrix+1)+(mysize-1), 1, ghost_type, dest, MPI_ANY_TAG, comm2d, &req2);			   
		MPI_Wait(&req, &stat);
		MPI_Wait(&req2, &stat);
		MPI_Isend(*(matrix+1)+mysize-2, 1, ghost_type, dest, 0, comm2d, &req);
		MPI_Irecv(*(matrix+1), 1, ghost_type, source, MPI_ANY_TAG, comm2d, &req2);			   
		MPI_Wait(&req, &stat);
		MPI_Wait(&req2, &stat);
		/************Cart Shift ----  Sendrecv row and ghost**********/

		/***********  double for loop to generate next time step values *********/
		for(i = 1; i< mysize-1; i++)
		{
			for(j =1; j<mysize-1; j++)
			{
				*(*(matrix+i)+j) = .25*(matrix1[i+1][j]+matrix1[i-1][j]+matrix1[i][j+1]+matrix1[i][j-1])-\
						   d*(u*(matrix1[i+1][j]-matrix1[i-1][j])+v*(matrix1[i][j+1]-matrix1[i][j-1]));
			}

		}
		/*********** end for loop for time step values *************/
	}



	if(mype ==0){
		endtime = MPI_Wtime();
		printf("%f\n",endtime-starttime);
	}
	MPI_Type_free(&ghost_type);
	MPI_Finalize();

	return 0;

}
