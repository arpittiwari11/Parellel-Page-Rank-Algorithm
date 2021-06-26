# Parellelised Google Page rank algorithm using OpenMP, Cuda

PageRank (PR) is an algorithm used by Google Search to rank web pages in their search engine results. PageRank is a way of measuring the importance of website pages. According to Google:
PageRank works by counting the number and quality of links to a page to determine a rough estimate of how important the website is. The underlying assumption is that more important websites are likely to receive more links from other websites.

I parellised the serial code of page rank algorithm using OpenMp (which is an implementation of multithreading, a method of parallelizing whereby a primary thread (a series of instructions executed consecutively) forks a specified number of sub-threads and the system divides a task among them ) and CUDA (an acronym for Compute Unified Device Architecture) which is a parallel computing platform and application programming interface (API) model created by Nvidia.


I implemented the Page rank algorithm on the Barabasi Graphs from the Networkx library in python.

Here is the sample Barabasi Graph for N=10 nodes:

<img src="https://user-images.githubusercontent.com/82596857/123515741-5564e680-d6b6-11eb-86de-9d183f4110e4.png" width="400" >



There is a significant speed up in the algorithm after parellisation which could be seen by the following time study analysis.

Shown below is the comparative time study graphs by varying the size of the network(Nodes) of various parellisation techniques such as OpenMP,CUDA.

<img src="https://user-images.githubusercontent.com/82596857/123517976-85fe4d80-d6c1-11eb-87c2-930079688a0b.png" width="500" height="320"/> <img src="https://user-images.githubusercontent.com/82596857/123517995-9dd5d180-d6c1-11eb-964e-180951545c94.png" width="450" height="320"/> 
