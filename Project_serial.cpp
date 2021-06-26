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
#include<omp.h>


using namespace std;

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

float *matmul(float *a,float *v1,float *v2,int n,int d)					/////////// Matrix Multiplication  a*v1=v2 ////////
{	
   float sub=0;
   

for(int i=0;i<n;i++)
{
	for(int j=0;j<d;j++)
	{
    	 sub=0;
    	for(int p=0;p<n;p++)
    	{
    		sub =sub + a[i*n+p]*v1[p*d+j];
    	}
    	v2[i*d+j]=sub;
	}
}
return v2;
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

void sortArr(float arr[], int n)									////////// Function to sort pages according to final rank ///////
{

	// Vector to store element
	// with respective present index
	vector<pair<float, float> > vp;

	// Inserting element in pair vector
	// to keep track of previous indexes
	for (int i = 0; i < n; ++i) {
		vp.push_back(make_pair(arr[i], i));
	}

	// Sorting pair vector
	sort(vp.begin(), vp.end());

	// Displaying sorted element
	// with previous indexes
	// corresponding to each element
	cout << "Probability  \t"
		<< "Page   \t" <<"Pagerank   \t"<< endl;
	for (int i =  (vp.size()-1); i>= 0; i--) {
		cout << vp[i].first <<"   \t"
			<< vp[i].second << "   \t"<<(n-i)<< endl;
	}
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

for(int i=0;i<n*d;i++)
{
v1[i]=1.0/n;
v2[i]=1.0;
}

                                                    ////////// IMPORTING DATASET //////////
int i=0;
float *first,*second;
int t=291;											////////// Number of links in graph ( to be changed for different values of n) ////////
first = (float*)malloc( t*sizeof(float));
second = (float*)malloc( t*sizeof(float));
string line;
ifstream myfile("barbasi.txt");					////////// Text file containing all the information of graph ///////////
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

/////////

f1=first[i];
f2=second[i];
matri[f2*n+f1]=1;
}

length(matri,len,n);
stochastic(matri,len,b,n);
transmatrix(matri,b,n);
matmul(matri,v1,v2,n,d);

int f=0;                                    ///////// NUMBER OF ITERATIONS ////////////

while(error(v1,v2,tol,0,n,f,d))
{
for(int i=0;i<n;i++){v1[i*d+f+1]=v2[i*d+f];}
															/////////// Iterative Matrix multiplication till convergence ////////
	matmul(matri,v1,v2,n,d);
	
f++;
}
for(int i=0;i<n;i++){v1[i*d+f]=v2[i*d+f];}
double coutnnn=0;

/*for(int z=0;z<f;z++)
{
for(int i=0;i<n;i++)
{
cout<<v1[i*d+z]<<endl;
//coutnnn=coutnnn+v2[i*d+f];
}
cout<<endl;
}
*/
 /*for(int p=0 ; p< n; p++)
	  {
		for(int i=0 ; i<(f+1); i++)
		{
			cout<<v1[d*p+i]<<"      ";
		}
		cout<<endl;	
	  }  
    cout<<endl;

float sort[n];
*/
for(int i=0;i<n;i++)
{
 // sort[i]=v2[i*d+f];
//cout<<v2[i*d+f]<<endl;
coutnnn=coutnnn+v2[i*d+f];
}
cout<<endl;

//int p = sizeof(sort) / sizeof(sort[0]);
//	sortArr(sort, p);

//cout<<v2[1]<<endl;
//cout<<v2[n-1]<<endl;
cout<<coutnnn<<endl;
cout<<"Number of iterations to converge = "<<f<<endl;



cout<<endl<<endl;
gettimeofday(&t2, 0);

double timee = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000.0;		////////// Time taken to to calculate final PageRank//////
cout<<timee/1000<<endl;

}
