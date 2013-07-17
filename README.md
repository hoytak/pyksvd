pyksvd
======

A highly optimized, parallel implementation of the Batch-OMP version of the KSVD learning algorithm. It implements the algorithm in the paper

Efficient Implementation of the K-SVD Algorithm and the Batch-OMP Method by Ron Rubinstein, Michael Zibulevsky and Michael Elad, 2009.

available from

http://www.cs.technion.ac.il/users/wwwb/cgi-bin/tr-get.cgi/2008/CS/CS-2008-08.pdf

The computation is done in highly optimized C++ code with OpenMP
implementations for multicore archetectures.

It currently requires Eigen and g++ >= 4.6 to compile.  Setup is done using the standard 

python setup.py install 

method. License is BSD. 

