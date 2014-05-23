#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<omp.h>
#include<assert.h>

int main(int argc, char **argv){

	/*first initialize the conditions C to test*/
	int N = 10000; // Matrix Dimensions
	int NT = 2000; // Number of timesteps
	double L = 1.0; // Physical Domain Length
	double T = 1.0e6; // Total Physical Timespan
	double u = 5.0e-7; // X velocity Scalar
	double v1 = 0.57; // Y velocity Scalar


	/*create file pointer, bring in file with inits via argv - future
	 *
	 */
	FILE *fp;
	fp  = fopen(argv[1], "r");
	fscanf(fp, "%d %d %lf %lf %lf %lf", &N, &NT, &L, &T, &u, &v1);
	fclose(fp);	
//	printf("%d %d %f %f %f %f", N, NT, L, T, u, v1);

	/*derived inits*/
	double v = u*v1;
	double delta = L/N;
	double delT = T/NT;
	/*space i need*/
	int i, j, k, s,t;
	double x0 = .5, y0 = .5; 
	
/* contiguous memory bitches */
        double **matrixn = (double **) malloc (sizeof(double) * N);
        double *data = (double *) malloc (sizeof(double) * N * N);
        double **matrixn1 = (double **) malloc (sizeof(double) * N);
        double *data1 = (double *) malloc (sizeof(double) * N * N);

        for(i=0;i<N;i++)
        {
                *(matrixn+i) = data + N*i;
                *(matrixn1+i) = data1 + N*i;
        }




double sd = 0.25;
	double d = 0.5*delT/delta;
	assert(delT <= delta/(sqrt(2)*sqrt(u*u+v*v)));
	double start, end;
	int up, down, left, right;
	
	/*now initialize the n and n-1 matrix to the gaussian by x -> y*/
	for(i=0;i<N;i++){
		for(j=0;j<N;j++){
			matrixn[i][j] =  exp(-(((i*delta-x0)*(i*delta-x0))/(2*sd*sd) + ((j*delta-y0)*(j*delta-y0))/(2*sd*sd)));
			matrixn1[i][j] =  exp(-(((i*delta-x0)*(i*delta-x0))/(2*sd*sd) + ((j*delta-y0)*(j*delta-y0))/(2*sd*sd)));
		}
	}
	
	/*now build the loops for the advection function*/
	start = omp_get_wtime();
	for(k=0;k<NT;k++){
		/*this will be area for ghost cell exchange*/
		/*exchange n-1 timestep matrix with nth timestep matrix*/
		for(s=0;s<N;s++){
			for(t=0;t<N;t++){
				matrixn1[s][t] = matrixn[s][t];
			}
		}
#pragma omp parallel for private(i, j, up, down, left, right)
		for(i=0;i<N;i++){
			for(j=0;j<N;j++){
				/*handle the various cases for up, down, left, right*/
				up = i-1;
				down = i+1;
				left = j-1;
				right = j+1;
				if(i == N-1 && j != N-1){
					down = 0;
				}
				else if(j == N-1 && i != N-1 && i!=0){
					right =0;
				}
				else if(i == 0 && j != 0 && j !=N-1){
					up = N-1;
				}
				else if(j == 0 && i !=0){
					left = N-1;
				}
				else if(j==0 && i==0){
					up = N-1;
					left = N-1;
				}
				else if(i ==N-1 && j==N-1){
					down = 0;
					right = 0;
				}
				else if(i==N-1 && j==0){
					left = N-1;
					down = 0;
				}
				else if(i==0 && j ==N-1){
					right = 0;
					up = N-1;
				}
				else if(i==0 && j == 0){
                                        up = N-1;
					left = N-1;
                                }
				matrixn[i][j] = .25*(matrixn1[down][j]+matrixn1[up][j]+matrixn1[i][right]+matrixn1[i][left])-\
						d*(u*(matrixn1[down][j]-matrixn1[up][j])+v*(matrixn1[i][right]-matrixn1[i][left]));	
			}
		}
	} /*this brace is end of parallel for*/
	end = omp_get_wtime();
	printf("%f\n", end-start);



	return 0;
}

