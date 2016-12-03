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
*/
#pragma once
#include "labelcompatibility.h"
#include "permutohedral.h"

#define SMALLEST_BLOCK 30

// The filter in the dense CRF can be normalized in a few different ways
enum NormalizationType {
    NO_NORMALIZATION,    // No normalization whatsoever (will lead to a substantial approximation error)
    NORMALIZE_BEFORE,    // Normalize before filtering (Not used, just there for completeness)
    NORMALIZE_AFTER,     // Normalize after filtering (original normalization in NIPS 11 work)
    NORMALIZE_SYMMETRIC, // Normalize before and after (ICML 2013, low approximation error and preserves the symmetry of CRF)
};
enum KernelType {
    CONST_KERNEL,   // Constant kernel, no parameters
    DIAG_KERNEL,    // Diagonal kernel (scaling features)
    FULL_KERNEL,    // Full kernel matrix (arbitrary squared matrix)
};

class Kernel {
public:
    virtual ~Kernel();
    virtual void apply( MatrixXf & out, const MatrixXf & Q ) const = 0;
    virtual void applyTranspose( MatrixXf & out, const MatrixXf & Q ) const = 0;
    virtual void apply_upper_minus_lower_dc( MatrixXf & out, int low, int middle_low, 
            int middle_high, int high) const = 0;
    virtual void apply_upper_minus_lower_ord( MatrixXf & out, const MatrixXf & Q) = 0;
    virtual void apply_upper_minus_lower_ord_restricted(MatrixXf & rout, const MatrixXf & rQ, 
        const std::vector<int> & pI, const MatrixXf & Q, const bool store) = 0;
    virtual VectorXf parameters() const = 0;
    virtual void setParameters( const VectorXf & p ) = 0;
    virtual VectorXf gradient( const MatrixXf & b, const MatrixXf & Q ) const = 0;
    virtual MatrixXf const & features() const = 0;
    virtual KernelType ktype() const = 0;
    virtual NormalizationType ntype() const = 0;
};

class PairwisePotential{
protected:
    LabelCompatibility * compatibility_;
    Kernel * kernel_;
    PairwisePotential( const PairwisePotential &o ){}
    void filter( MatrixXf & out, const MatrixXf & in, bool transpose=false ) const;
public:
    virtual ~PairwisePotential();
    PairwisePotential(const MatrixXf & features, LabelCompatibility * compatibility, 
            KernelType ktype=CONST_KERNEL, NormalizationType ntype=NORMALIZE_SYMMETRIC);
    // full-matrix 
    void apply(MatrixXf & out, const MatrixXf & Q) const;
    void applyTranspose(MatrixXf & out, const MatrixXf & Q) const;
    // upper-traingular minus lower-triangluar: divide-and-conquer (ECCV-16) 
    void apply_upper_minus_lower_dc(MatrixXf & out, const MatrixXi & ind) const;
    void apply_upper_minus_lower_sorted_slice(MatrixXf & out, int min, int max) const;
    // upper-traingular minus lower-triangluar: new implementation (arxiv-17)
    void apply_upper_minus_lower_ord(MatrixXf & out, const MatrixXf & Q);
    void apply_upper_minus_lower_ord_restricted(MatrixXf & rout, const MatrixXf & rQ,  
        const std::vector<int> & pI, const MatrixXf & Q, const bool store); // works on subset of pixels
    
    // Get the parameters
    virtual VectorXf parameters() const;
    virtual VectorXf kernelParameters() const;
    virtual MatrixXf features() const;
    virtual KernelType ktype() const;
    virtual NormalizationType ntype() const;
    virtual Kernel* getKernel() const;
    virtual MatrixXf compatibility_matrix(int nb_labels) const;
    virtual void setParameters( const VectorXf & v );
    virtual void setKernelParameters( const VectorXf & v );
    virtual VectorXf gradient( const MatrixXf & b, const MatrixXf & Q ) const;
    virtual VectorXf kernelGradient( const MatrixXf & b, const MatrixXf & Q ) const;
};
