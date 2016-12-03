Efficient Continuous Relaxations for Dense CRFs - Code
=====================================================

This code implments the continuous relaxation based algorihms proposed in the following papers

["Efficient Continuous Relaxations for Dense CRF"](https://arxiv.org/abs/1608.06192).
Alban Desmaison, Rudy Bunel, Pushmeet Kohli, Philip H. S. Torr, and M. Pawan Kumar.
ECCV, Springer, 2016.

["Efficient Linear Programming for Dense CRFs"](https://arxiv.org/abs/1611.09718).
Thalaiyasingam Ajanthan, Alban Desmaison, Rudy Bunel, Mathieu Salzmann, Philip H. S. Torr, and M. Pawan Kumar.

If you're using this code in a publication, please cite our papers.

This code is for research purposes only.
If you want to use it for commercial purpose please contact us.

Contact: alban at robots.ox.ac.uk, rudy at robots.ox.ac.uk or thalaiyasingam.ajanthan at anu.edu.au

Our code is built on top of the software provided by Philipp Krähenbühl, which is downloaded from
http://graphics.stanford.edu/projects/drf/.

How to compile the code
-----------------------

Dependencies:
 * [cmake](http://www.cmake.org/)
 * [OpenCV](http://opencv.org/)
 * Eigen (included)
 * liblbfgs (included)

Linux, Mac OS X and Windows (cygwin,wsl):
 * mkdir build
 * cd build
 * cmake -D CMAKE_BUILD_TYPE=Release ..
 * make
 * cd ..

How to run the example
----------------------

An example on how to use the code can be found in _examples/inference.cpp_. 
A sample image from MSRC dataset is given in data/ folder.

Example usage:
 * ./examples/inference /path/to/data/2_14_s.bmp /path/to/data/2_14_s.c_unary dc-neg /path/to/results/
 
 This runs the dc-neg mehod (with the default energy parameters) on the sample image and write the results to the results folder.

Cross validated parameters:
 * The cross-validated energy paramters used in the paper are given in data/cv-results.txt

