/*
    Copyright (c) 2013, Philipp Krähenbühl
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:
        * Redistributions of source code must retain the above copyright
        notice, this list of conditions and the following disclaimer.
        * Redistributions in binary form must reproduce the above copyright
        notice, this list of conditions and the following disclaimer in the
        documentation and/or other materials provided with the distribution.
        * Neither the name of the Stanford University nor the
        names of its contributors may be used to endorse or promote products
        derived from this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY Philipp Krähenbühl ''AS IS'' AND ANY
    EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL Philipp Krähenbühl BE LIABLE FOR ANY
    DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
    LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
    ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

    -----------------------------------------------------------------------------
    The following functions are copyright (c) 2016 of Alban Desmaison, Rudy Bunel
    and Thalaiyasingam Ajanthan.
    -----------------------------------------------------------------------------
    seqCompute_upper_minus_lower_dc
    seqCompute_upper_minus_lower_ord
    seqCompute_upper_minus_lower_ord_restricted
    -----------------------------------------------------------------------------
*/
#pragma once
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <iostream>
#include <vector>
#include <Eigen/Core>
using namespace Eigen;

// Data structure for the splitted arrays 
#define RESOLUTION 10   // no of discrete levels H
typedef struct {
    float data[RESOLUTION];
} split_array;

/************************************************/
/***          Permutohedral Lattice           ***/
/************************************************/

class Permutohedral
{
protected:
    struct Neighbors{
        int n1, n2;
        Neighbors( int n1=0, int n2=0 ):n1(n1),n2(n2){
        }
    };
    // allocated once in "seqCompute_upper_minus_lower_ord", dealocate in destructor
    int L_;             
    split_array * values_;       
    split_array * new_values_;   
    // allocated once in "seqCompute_upper_minus_lower_ord_restricted", dealocate in destructor
    split_array * stored_values_l_; 
    split_array * stored_values_u_; 
    std::vector<int> stored_active_list_; 
    //

    std::vector<int> offset_, rank_;
    std::vector<float> barycentric_;
    std::vector<Neighbors> blur_neighbors_;
    // Number of elements, size of sparse discretized space, dimension of features
    int N_, M_, d_;
    void sseCompute ( float* out, const float* in, int value_size, bool reverse=false ) const;
    void seqCompute ( float* out, const float* in, int value_size, bool reverse=false ) const;
    void seqCompute_upper_minus_lower_dc ( float* out, int low, int middle_low, int middle_high, 
            int high ) const;
    void seqCompute_upper_minus_lower_ord ( float* out, const float* in, int value_size ); 
    void seqCompute_upper_minus_lower_ord_restricted ( float* out, const float* in, int value_size, 
        const std::vector<int> & pI, const float* extIn, const bool store );
public:
    Permutohedral();
    ~Permutohedral();
    void init ( const MatrixXf & features );
    // full-matrix
    MatrixXf compute ( const MatrixXf & v, bool reverse=false ) const;
    void compute ( MatrixXf & out, const MatrixXf & in, bool reverse=false ) const;
    // divide-and-conquer
    void compute_upper_minus_lower_dc ( MatrixXf & out, int low, int middle_low, int middle_high, 
            int high ) const;
    // discretize version - linear time
    void compute_upper_minus_lower_ord ( MatrixXf & out, const MatrixXf & Q );
    // works on subset of pixels
    void compute_upper_minus_lower_ord_restricted ( MatrixXf & rout, const MatrixXf & rQ,  
        const std::vector<int> & pI, const MatrixXf & Q, const bool store );
    // Compute the gradient of a^T K b
    void gradient ( float* df, const float * a, const float* b, int value_size ) const;
};
