////////////// ME766 PROJECT////////////////////////////////

////////////// Parallelization of PageRank Algorithm using CUDA and OpenMp  /////////////////////////

//// Instructor:
//// Shivasubramanian Gopalakrishnan 

//// Submitted by :
//// Arpit Tiwari
//// Ansh Thamke 
//// Raj Ingole 
//// Sumit Bhong

#include <iostream>
#include <fstream>
#include <string>
#include<stdlib.h>
#include<bits/stdc++.h>
#include <stdio.h>
#include<time.h>
#include <sys/time.h>
using namespace std;

#define TILE_WIDTH 32

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }						////////// Function to check error in device functions///////
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__global__ void matrixmult(float *a, float *b, float *c,int N,int d) 					/////////// Matrix Multiplication  a*v1=v2 ////////
{
	int row = blockIdx.y * blockDim.y +threadIdx.y;
	int col = blockIdx.x * blockDim.x +threadIdx.x;
	
	if(row < N && col < d)
	{	
		float temp=0;

		for(int p=0; p<N ; p++)
		{
			temp=temp + a[row * N + p] * b[p * d + col];

		}

		c[row*d + col] = temp ;
	}
	
}

float *length(float *a,float *len,int n)
{

for(int j=0;j<n;j++)
{
int sum=0;
	for(int i=0;i<n;i++)
	{
		if(a[i*n+j]>0){sum++;}
	}

len[j]=sum;
}
return len;
}

float *stochastic(float *a,float *len,float t,int n)		///////////// Formation of Stochastic Matrix/////////////
{
for(int i=0;i<n;i++)
{
float g=len[i];
	if(g>0){
				for(int j=0;j<n;j++)
				{
					a[j*n+i]=(a[j*n+i]*t)/(g);
				}
			}
	else {
for(int j=0;j<n;j++)
{ a[j*n+i]=(1.0*t)/n;  }
}

}
return a;
}

float *transmatrix(float *a,float t , int n)			////////////// Formation of Transformation matrix considering damping factor/////////
{
float b[n][n];
	for(int i=0;i<n;i++)
	{
		for(int j=0;j<n;j++)
		{
			b[i][j]=((1-t)*1.0)/n;
		}
	}
	for(int i=0;i<n;i++)
	{
		for(int j=0;j<n;j++)
		{
			a[i*n+j]=a[i*n+j]+b[i][j];
		}
	}
return a;
}

bool error(float *b, float *c,float tol,float sum,int n,int f,int d)	////////// Function to check Convergence /////////
{
	for(int i=0;i<n;i++)
	{
		sum =sum + (c[d*i+f]-b[d*i+f])*(c[d*i+f]-b[d*i+f]);
	}
if(sum<tol)
{
return false;
}
else return true;
}

int main(){														////// Main function starts //////
	struct timeval t1, t2;
	gettimeofday(&t1, 0);
  
  
  int n=100;                        /////// NUMBER OF NODES	///////////
  int d=15;
  
  float tol=0.00000000005;           //////////// TOLERANCE VALUE	////////
  float *len;
  
  float b=0.85;                       ////////// Damping Factor	//////////
  float *v1,*v2,*matri;
  
  size_t bytes = n*n*sizeof(float);
  
	  matri = (float*)malloc( bytes );                       ////// DYNAMIC MEMORY ALLOCATION /////////
		v1 = (float*)malloc( n*d*sizeof(float) );
	  v2 = (float*)malloc(  n*d*sizeof(float) );
		len = (float*)malloc( n*sizeof(float));
	
	gpuErrchk( cudaMallocManaged(&matri,bytes));				//////// Device Memory Allocation///////
	gpuErrchk(cudaMallocManaged(&v1,d*n*sizeof(float)));
	gpuErrchk(cudaMallocManaged(&v2,d*n*sizeof(float)));
	

for(int i=0;i<d*n;i++){
v1[i]=1.0/n;
v2[i]=1.0;}


///////
int i=0;
float *first,*second;
int t=291;
first = (float*)malloc( t*sizeof(float));
second = (float*)malloc( t*sizeof(float));
string line;
ifstream myfile("Barbasi.txt");
while(std::getline(myfile,line))
{
std::stringstream linestream(line);

float val1;
float val2;

while(linestream>>val1>>val2){

first[i]=val1;
second[i]=val2;
}
i++;
}
myfile.close();




int f1,f2;
for(int i=0;i<n;i++)
{
    for(int j=0;j<n;j++)
    {
        matri[i*n+j]=0;
     }
 }


for(int i=0;i<t;i++){


f1=first[i];
f2=second[i];
matri[f2*n+f1]=1;
}

length(matri,len,n);
stochastic(matri,len,b,n);
transmatrix(matri,b,n);

int blocks = (n+TILE_WIDTH-1)/TILE_WIDTH;				////////// Number of blocks to be used///////
	
	dim3 dim_block(TILE_WIDTH, TILE_WIDTH);				////////// Number of threads per block ///////
	dim3 dim_grid( blocks, blocks);

matrixmult<<<dim_grid, dim_block>>>(matri, v1, v2, n,d);		///////// Cuda function call /////
cudaDeviceSynchronize();

int f=0;

  
while(error(v1,v2,tol,0,n,f,d))									/////// Iterative Multiplication /////
{
    for(int i=0;i<n;i++){v1[i*d+f+1]=v2[i*d+f];}

	matrixmult<<<dim_grid, dim_block>>>(matri, v1, v2, n,d);
	cudaDeviceSynchronize();
	
f++;
}
double coutnnn=0;										
for(int i=0;i<n;i++){
//cout<<v2[i*d+f]<<endl;
coutnnn=coutnnn+v2[i*d+f];
}

cout<<coutnnn<<endl;
cout<<"Number of iterations to converge = "<<f<<endl;

cout<<endl<<endl;
gettimeofday(&t2, 0);

double timee = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000.0;		////////// Time taken to to calculate final PageRank//////
cout<<timee/1000<<endl;

}
