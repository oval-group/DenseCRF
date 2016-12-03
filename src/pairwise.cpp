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
#include <iostream>

#include "pairwise.h"

Kernel::~Kernel() {
}
class DenseKernel: public Kernel {
protected:
  NormalizationType ntype_;
  KernelType ktype_;
  Permutohedral lattice_;
  VectorXf norm_;
  MatrixXf f_;
  MatrixXf parameters_;
  void initLattice( const MatrixXf & f ) {
    const int N = f.cols();
    lattice_.init( f );
    
    
    if ( ntype_ != NO_NORMALIZATION ) {
      norm_ = lattice_.compute( VectorXf::Ones( N ).transpose() ).transpose();

      if ( ntype_ == NORMALIZE_SYMMETRIC ) {
      for ( int i=0; i<N; i++ )
          norm_[i] = 1.0 / sqrt(norm_[i]+1e-20);
      }
      else {
        for ( int i=0; i<N; i++ )
          norm_[i] = 1.0 / (norm_[i]+1e-20);
      }
    }
  }
  void filter( MatrixXf & out, const MatrixXf & in, bool transpose ) const {
    // Read in the values
    if( ntype_ == NORMALIZE_SYMMETRIC || (ntype_ == NORMALIZE_BEFORE && !transpose) 
                || (ntype_ == NORMALIZE_AFTER && transpose))
      out = in*norm_.asDiagonal();
    else
      out = in;
  
    // Filter
    if( transpose )
      lattice_.compute( out, out, true );
    else
      lattice_.compute( out, out );
    //lattice_.compute( out.data(), out.data(), out.rows() );
  
    // Normalize again
    if( ntype_ == NORMALIZE_SYMMETRIC || (ntype_ == NORMALIZE_BEFORE && transpose) 
                || (ntype_ == NORMALIZE_AFTER && !transpose))
      out = out*norm_.asDiagonal();
  }
  void filter_upper_minus_lower_dc( MatrixXf & out, int low, int middle_low, int middle_high, 
            int high ) const {
    // Normalization makes no sense here since this would always return 1
  
    // Filter
    lattice_.compute_upper_minus_lower_dc( out, low, middle_low, middle_high, high );
  }
  void filter_upper_minus_lower_ord( MatrixXf & out, const MatrixXf & Q) {
    lattice_.compute_upper_minus_lower_ord( out, Q );
  }
  void filter_upper_minus_lower_ord_restricted(MatrixXf & rout, const MatrixXf & rQ, 
        const std::vector<int> & pI, const MatrixXf & Q, const bool store) {
    lattice_.compute_upper_minus_lower_ord_restricted(rout, rQ, pI, Q, store);
  }
  // Compute d/df a^T*K*b
  MatrixXf kernelGradient( const MatrixXf & a, const MatrixXf & b ) const {
    MatrixXf g = 0*f_;
    lattice_.gradient( g.data(), a.data(), b.data(), a.rows() );
    return g;
  }
  MatrixXf featureGradient( const MatrixXf & a, const MatrixXf & b ) const {
    if (ntype_ == NO_NORMALIZATION )
      return kernelGradient( a, b );
    else if (ntype_ == NORMALIZE_SYMMETRIC ) {
      MatrixXf fa = lattice_.compute( a*norm_.asDiagonal(), true );
      MatrixXf fb = lattice_.compute( b*norm_.asDiagonal() );
      MatrixXf ones = MatrixXf::Ones( a.rows(), a.cols() );
      VectorXf norm3 = norm_.array()*norm_.array()*norm_.array();
      MatrixXf r = kernelGradient( 0.5*( a.array()*fb.array() + fa.array()*b.array()).matrix()
                    * norm3.asDiagonal(), ones );
      return - r + kernelGradient( a*norm_.asDiagonal(), b*norm_.asDiagonal() );
    }
    else if (ntype_ == NORMALIZE_AFTER ) {
      MatrixXf fb = lattice_.compute( b );
      
      MatrixXf ones = MatrixXf::Ones( a.rows(), a.cols() );
      VectorXf norm2 = norm_.array()*norm_.array();
      MatrixXf r = kernelGradient( ( a.array()*fb.array() ).matrix()*norm2.asDiagonal(), ones );
      return - r + kernelGradient( a*norm_.asDiagonal(), b );
    }
    else /*if (ntype_ == NORMALIZE_BEFORE )*/ {
      MatrixXf fa = lattice_.compute( a, true );
      
      MatrixXf ones = MatrixXf::Ones( a.rows(), a.cols() );
      VectorXf norm2 = norm_.array()*norm_.array();
      MatrixXf r = kernelGradient( ( fa.array()*b.array() ).matrix()*norm2.asDiagonal(), ones );
      return -r+kernelGradient( a, b*norm_.asDiagonal() );
    }
  }
public:
  DenseKernel(const MatrixXf & f, KernelType ktype, NormalizationType ntype):f_(f), ktype_(ktype), 
            ntype_(ntype) {
    if (ktype_ == DIAG_KERNEL)
      parameters_ = VectorXf::Ones( f.rows() );
    else if( ktype == FULL_KERNEL )
      parameters_ = MatrixXf::Identity( f.rows(), f.rows() );
    initLattice( f );
  }
  virtual void apply_upper_minus_lower_dc( MatrixXf & out, int low, int middle_low, int middle_high, 
            int high) const {
    filter_upper_minus_lower_dc(out, low, middle_low, middle_high, high);
  }
  virtual void apply_upper_minus_lower_ord( MatrixXf & out, const MatrixXf & Q) {
    filter_upper_minus_lower_ord(out, Q);
  }
    virtual void apply_upper_minus_lower_ord_restricted(MatrixXf & rout, const MatrixXf & rQ, 
        const std::vector<int> & pI, const MatrixXf & Q, const bool store) {
    filter_upper_minus_lower_ord_restricted(rout, rQ, pI, Q, store);
    }
  virtual void apply( MatrixXf & out, const MatrixXf & Q ) const {
    filter( out, Q, false );
  }
  virtual void applyTranspose( MatrixXf & out, const MatrixXf & Q ) const {
    filter( out, Q, true );
  }
  virtual VectorXf parameters() const {
    if (ktype_ == CONST_KERNEL)
      return VectorXf();
    else if (ktype_ == DIAG_KERNEL)
      return parameters_;
    else {
      MatrixXf p = parameters_;
      p.resize( p.cols()*p.rows(), 1 );
      return p;
    }
  }
  virtual void setParameters( const VectorXf & p ) {
    if (ktype_ == DIAG_KERNEL) {
      parameters_ = p;
      initLattice( p.asDiagonal() * f_ );
    }
    else if (ktype_ == FULL_KERNEL) {
      MatrixXf tmp = p;
      tmp.resize( parameters_.rows(), parameters_.cols() );
      parameters_ = tmp;
      initLattice( tmp * f_ );
    }
  }
  virtual VectorXf gradient( const MatrixXf & a, const MatrixXf & b ) const {
    if (ktype_ == CONST_KERNEL)
      return VectorXf();
    MatrixXf fg = featureGradient( a, b );
    if (ktype_ == DIAG_KERNEL)
      return (f_.array()*fg.array()).rowwise().sum();
    else {
      MatrixXf p = fg*f_.transpose();
      p.resize( p.cols()*p.rows(), 1 );
      return p;
    }
  }

  virtual MatrixXf const & features() const {
    return f_;
  }

  virtual KernelType ktype() const {
    return ktype_;
  }

  virtual NormalizationType ntype() const {
    return ntype_;
  }
};

PairwisePotential::~PairwisePotential(){
  delete compatibility_;
  delete kernel_;
}
PairwisePotential::PairwisePotential(const MatrixXf & features, LabelCompatibility * compatibility, 
        KernelType ktype, NormalizationType ntype) : compatibility_(compatibility) {
  kernel_ = new DenseKernel( features, ktype, ntype );
}
void PairwisePotential::apply(MatrixXf & out, const MatrixXf & Q) const {
  kernel_->apply( out, Q );
  
  // Apply the compatibility
  compatibility_->apply( out, out );
}
void PairwisePotential::applyTranspose(MatrixXf & out, const MatrixXf & Q) const {
  kernel_->applyTranspose( out, Q );
  // Apply the compatibility
  compatibility_->applyTranspose( out, out );
}
void PairwisePotential::apply_upper_minus_lower_ord(MatrixXf & out, const MatrixXf & Q) {
    // pass rescaled_Q to reduce the discretization error! ("rescale" function is in densecrf_utils.cpp)
    // when no of labels are high, each probabilty is small and all fall into one or two buckets!
  assert(Q.maxCoeff() <= 1);
  assert(Q.minCoeff() >= 0);  // values truncated to be [0,1], doesn't need to sum to 1
    out.fill(0);
  kernel_->apply_upper_minus_lower_ord(out, Q);

  // Apply the compatibility
  compatibility_->apply( out, out );
} 
void PairwisePotential::apply_upper_minus_lower_ord_restricted(MatrixXf & rout, const MatrixXf & rQ, 
    const std::vector<int> & pI, const MatrixXf & Q, const bool store) {
    // pass rescaled_Q to reduce the discretization error! ("rescale" function is in densecrf_utils.cpp)
    // when no of labels are high, each probabilty is small and all fall into one or two buckets!
    assert(rQ.maxCoeff() <= 1);
  assert(rQ.minCoeff() >= 0);  // values truncated to be [0,1], doesn't need to sum to 1
    rout.fill(0);
  kernel_->apply_upper_minus_lower_ord_restricted(rout, rQ, pI, Q, store);
  
  // Apply the compatibility
  compatibility_->apply( rout, rout );
}    
void PairwisePotential::apply_upper_minus_lower_dc(MatrixXf & out, const MatrixXi & ind) const {
  MatrixXf const & features = kernel_->features();
  MatrixXf sorted_features = features;
  MatrixXf single_label_out(1, features.cols());

  for(int label=0; label<ind.rows(); ++label) {
    // Sort the features with the scores for this label
    for(int j=0; j<features.cols(); ++j) {
      sorted_features.col(j) = features.col(ind(label, j));
    }

    PairwisePotential pairwise(sorted_features, new PottsCompatibility(1));

    single_label_out.fill(0);
    pairwise.apply_upper_minus_lower_sorted_slice(single_label_out, 0, sorted_features.cols());
    out.row(label) = single_label_out;
  }
  compatibility_->apply(out, out);
}
void PairwisePotential::apply_upper_minus_lower_sorted_slice(MatrixXf & out, int min, int max) const {
  int size = max-min;
  if(size <= 0) {
    // This should never happen
    assert(false);
  } else if(size<=SMALLEST_BLOCK) {
    // Alpha is a magic scaling constant (write Rudy if you really wanna understand this)
    MatrixXf const & features = kernel_->features();
    double alpha = 1.0 / 0.6;
    // Lower
    for(int c=min; c<max; ++c) {
      for(int b=min; b<c; ++b) {
        VectorXf featDiff = (features.col(c) - features.col(b));
        out(0, c) -= exp(-featDiff.squaredNorm()) * alpha;
      }
    }
        // Upper
    for(int c=min; c<max; ++c) {
      for(int b=c+1; b<max; ++b) {
        VectorXf featDiff = (features.col(c) - features.col(b));
        out(0, c) += exp(-featDiff.squaredNorm()) * alpha;
      }
    }
  } else {
    int middle_low, middle_high;
    if(size%2==0) {
      middle_low = min + size/2;
      middle_high = min + size/2;
    } else if(size%2==1) {
      middle_low = floor(min + size/2.0);
      middle_high = floor(min + size/2.0) + 1;
    }

    // Upper left
    apply_upper_minus_lower_sorted_slice(out, min, middle_high);
    // Lower right
    apply_upper_minus_lower_sorted_slice(out, middle_low, max);
    // Lower left
    kernel_->apply_upper_minus_lower_dc(out, min, middle_low, middle_high, max);
  }
}
VectorXf PairwisePotential::parameters() const {
  return compatibility_->parameters();
}
void PairwisePotential::setParameters( const VectorXf & v ) {
  compatibility_->setParameters( v );
}
VectorXf PairwisePotential::gradient( const MatrixXf & b, const MatrixXf & Q ) const {
  MatrixXf filtered_Q = 0*Q;
  // You could reuse the filtered_b from applyTranspose
  kernel_->apply( filtered_Q, Q );
  return compatibility_->gradient(b,filtered_Q);
}
VectorXf PairwisePotential::kernelParameters() const {
  return kernel_->parameters();
}
void PairwisePotential::setKernelParameters( const VectorXf & v ) {
  kernel_->setParameters( v );
}
VectorXf PairwisePotential::kernelGradient( const MatrixXf & b, const MatrixXf & Q ) const {
  MatrixXf lbl_Q = 0*Q;
  // You could reuse the filtered_b from applyTranspose
  compatibility_->apply( lbl_Q, Q );
  return kernel_->gradient(b,lbl_Q);
}
MatrixXf PairwisePotential::features() const {
  return kernel_->features();
}
KernelType PairwisePotential::ktype() const {
  return kernel_->ktype();
}
NormalizationType PairwisePotential::ntype() const {
  return kernel_->ntype();
}
Kernel* PairwisePotential::getKernel() const {
  return kernel_;
}
MatrixXf PairwisePotential::compatibility_matrix(int nb_labels) const {
  return compatibility_->matrixForm(nb_labels);
}
