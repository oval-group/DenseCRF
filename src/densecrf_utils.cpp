#include <fstream>
#include <iostream>

#include "densecrf_utils.h"
#include <Eigen/Eigenvalues>

/////////////////////////////
/////  eigen-utils      /////
/////////////////////////////
bool all_close_to_zero(const VectorXf & vec, float ref){
    for (int i = 0; i<vec.size() ; i++) {
        if(vec(i)> ref or vec(i)<-ref){
            return false;
        }
    }
    return true;
}

bool all_positive(const VectorXf & vec){
    for (int i=0; i < vec.size(); i++) {
        if(vec(i)< 0){
            return false;
        }
    }
    return true;
}

bool all_strict_positive(const VectorXf & vec){
    for (int i=0; i < vec.size(); i++) {
        if(vec(i)<= 0){
            return false;
        }
    }
    return true;
}

void clamp_and_normalize(VectorXf & prob){
    for (int row=0; row < prob.size(); row++) {
        if (prob(row) < 0) {
            prob(row) = 0;
        }
    }
    prob = prob / prob.sum();
}

bool valid_probability(const MatrixXf & proba){
    for (int i=0; i<proba.cols(); i++) {
        if (not all_positive(proba.col(i))) {
            return false;
        }
        if (fabs(proba.col(i).sum()-1)>1e-5) {
            return false;
        }
    }
    return true;
}

bool valid_probability_debug(const MatrixXf & proba){
    for (int i=0; i<proba.cols(); i++) {
        if (not all_positive(proba.col(i))) {
            std::cout << "Col " << i << " has negative values"<< '\n';
            std::cout << proba.col(i).transpose() << '\n';
        }
        if (fabs(proba.col(i).sum()-1)>1e-6) {
            std::cout << "Col " << i << " doesn't sum to 1, sum to " << proba.col(i).sum()<< '\n';
        }
    }
    return true;
}

typeP dotProduct(const MatrixXf & M1, const MatrixXf & M2, MatrixP & temp){
    // tmp is an already allocated and well dimensioned temporary
    // matrix so that we don't need to allocate a new one. This may
    // very well be premature optimisation.
    temp = (M1.cast<typeP>()).cwiseProduct(M2.cast<typeP>());
    return temp.sum();
}

void sortRows(const MatrixXf & M, MatrixXi & ind) {
    int nbrRows = M.rows();
    int nbrCols = M.cols();
    // Initialize the indices matrix
    for(int i=0; i<ind.cols(); ++i) {
        ind.col(i).fill(i);
    }

    // we need indices for the row to be contiguous in memory
    ind.transposeInPlace();

    for(int i=0; i<nbrRows; ++i) {
        std::sort(ind.data()+i*nbrCols,
                  ind.data()+(i+1)*nbrCols,
                  [&M, i](int a, int b){return M(i, a)>M(i, b);});
    }

    ind.transposeInPlace();
}

void sortCols(const MatrixXf & M, MatrixXi & ind) {
    // Sort each row of M independantly and store indices in ind
    int nbrRows = M.rows();
    int nbrCols = M.cols();
    for(int i=0; i<ind.rows(); ++i) {
        ind.row(i).fill(i);
    }

    // indices for the cols are already contiguous in memory
    for(int i=0; i<nbrCols; ++i) {
        std::sort(ind.data()+i*nbrRows,
                  ind.data()+(i+1)*nbrRows,
                  [&M, i](int a, int b){
                    return M(a, i)>M(b, i);
                  });
    }

}

float infiniteNorm(const MatrixXf & M) {
    return M.cwiseAbs().maxCoeff();
}

/////////////////////////////
/////  qp-utils         /////
/////////////////////////////
void descent_direction(MatrixXf & out, const MatrixXf & grad){
    out.resize(grad.rows(), grad.cols());
    out.fill(0);
    int N =  grad.cols();

    int m;
    for (int i=0; i < N; i++) {
        grad.col(i).minCoeff(&m);
        out(m,i) = 1;
    }
}

float pick_lambda_eig_to_convex(const MatrixXf & lbl_compatibility){
    // We assume that the label compatibility matrix is symmetric in
    // order to use eigens eigenvalue code.
    VectorXf eigs = lbl_compatibility.selfadjointView<Eigen::Upper>().eigenvalues();
    float lambda_eig = eigs.minCoeff();
    lambda_eig = lambda_eig < 0 ? lambda_eig: 0;
    return lambda_eig;
}

/////////////////////////////
///// newton-cccp-utils /////
/////////////////////////////
void newton_cccp(VectorXf & state, const VectorXf & cste, float lambda_eig){
    int M_ = cste.size();
    VectorXf kkts(M_ + 1);
    kkts.head(M_) = 2 * lambda_eig * state.head(M_);
    kkts.head(M_) += state.head(M_).array().log().matrix();
    kkts.head(M_) += cste + state(M_) * VectorXf::Ones(M_);
    kkts(M_) = state.head(M_).sum() - 1;
    while (not all_close_to_zero(kkts, 0.001)) {
        // Compute J-1
        VectorXf inv_proba(M_);
        float z_norm = 0;
        for (int l=0; l < M_; l++) {
            inv_proba(l) = state(l) / (1+ 2 * lambda_eig * state(l));
            z_norm += inv_proba(l);
        }
        MatrixXf J1(M_+1, M_+1);
        for (int l=0; l < M_; l++) {
            J1(l,l) = (1-inv_proba(l)/z_norm) * inv_proba(l);
        }
        for (int l1=0; l1 < M_; l1++) {
            for (int l2=l1+1; l2< M_; l2++) {
                float temp = (- inv_proba(l2)/z_norm) * inv_proba(l1);
                J1(l1,l2) = temp;
                J1(l2,l1) = temp;
            }
        }
        for (int l=0; l<M_; l++) {
            J1(M_, l) = inv_proba(l)/z_norm;
            J1(l, M_) = inv_proba(l)/z_norm;
        }
        J1(M_, M_) = -1 / z_norm;

        // Apply J-1 to kkts
        VectorXf step = J1 * (-kkts);

        // Step progressively
        VectorXf new_state = state + step;
        float step_size = 1;
        while(true){
            if(all_strict_positive(new_state.head(M_))){
                break;
            } else {
                step_size = step_size /2;
                new_state = new_state - step_size * step;
            }
        }
        state = new_state;
        kkts.head(M_) = 2 * lambda_eig * state.head(M_);
        kkts.head(M_) += state.head(M_).array().log().matrix();
        kkts.head(M_) += cste + state(M_) * VectorXf::Ones(M_);
        kkts(M_) = state.head(M_).sum() - 1;
    }

}

float pick_lambda_eig_to_concave(MatrixXf const & lbl_compatibility){
    // We make the assumption that the label compatibility is symmetric,
    // which other places in the code do.
    VectorXf eigs = lbl_compatibility.selfadjointView<Upper>().eigenvalues();
    float lambda_eig = eigs.maxCoeff();
    lambda_eig = lambda_eig > 0? lambda_eig : 0;
    return lambda_eig;
}

/////////////////////////////
/////  mf-utils         /////
/////////////////////////////
void expAndNormalize ( MatrixXf & out, const MatrixXf & in ) {
    out.resize( in.rows(), in.cols() );
    for( int i=0; i<out.cols(); i++ ){
        VectorXf b = in.col(i);
        b.array() -= b.maxCoeff();
        b = b.array().exp();
        out.col(i) = b / b.array().sum();
    }
}

/////////////////////////////
/////  dc-neg-utils     /////
/////////////////////////////
void kkt_solver(const VectorXf & lin_part, const MatrixXf & inv_KKT, VectorXf & out) {
    int M = lin_part.size();
    VectorXf state(M + 1);
    VectorXf target(M + 1);
    target.head(M) = -lin_part;
    target(M) = 1;
    state = inv_KKT * target;
    out = state.head(M);
}


/////////////////////////////
/////  lp-utils         /////
/////////////////////////////
// enure Q is valid probability distribution
void renormalize(MatrixXf & Q) {
    double sum;
    double uniform = 1.0/Q.rows();
    for(int i=0; i<Q.cols(); i++) {
        sum = Q.col(i).sum();
        if(sum == 0 || sum != sum) {
            Q.col(i).fill(uniform);
        } else {
            Q.col(i) /= sum;
        }
    }
}

// Project current estimates on valid space
void feasible_Q(MatrixXf & tmp, MatrixXi & ind, VectorXd & sum, VectorXi & K, const MatrixXf & Q) {
    sortCols(Q, ind);
    for(int i=0; i<Q.cols(); ++i) {
        sum(i) = Q.col(i).sum()-1;
        K(i) = -1;
    }
    for(int i=0; i<Q.cols(); ++i) {
        for(int k=Q.rows(); k>0; --k) {
            double uk = Q(ind(k-1, i), i);
            if(sum(i)/k < uk) {
                K(i) = k;
                break;
            }
            sum(i) -= uk;
        }
    }
    tmp.fill(0);
    for(int i=0; i<Q.cols(); ++i) {
        for(int k=0; k<Q.rows(); ++k) {
            tmp(k, i) = std::min(std::max(Q(k, i) - sum(i)/K(i), 0.0), 1.0);
        }
    }
}

// get relevant subset of labels given Q
void get_limited_indices(MatrixXf const & Q, std::vector<int> & indices) {
    VectorXd accum = Q.cast<double>().rowwise().sum();
    indices.clear();

    double represented = 0;
    int max_ind;
    while(represented < 0.99 * Q.cols() && indices.size() < Q.rows()) {
        int max_val = accum.maxCoeff(&max_ind);
        indices.push_back(max_ind);
        accum[max_ind] = -1e9;
        represented += max_val;
    }
}

// return the rows given in indices
MatrixXf get_restricted_matrix(MatrixXf const & in, std::vector<int> const & indices) {
    MatrixXf out(indices.size(), in.cols());
    out.fill(0);

    for(int i=0; i<indices.size(); i++) {
        out.row(i) = in.row(indices[i]);
    }

    return out;
}

// get the full matrix, new rows are all zero
MatrixXf get_extended_matrix(MatrixXf const & in, std::vector<int> const & indices, int max_rows) {
    MatrixXf out(max_rows, in.cols());
    out.fill(0);

    for(int i=0; i<indices.size(); i++) {
        out.row(indices[i]) = in.row(i);
    }

    return out;
}

// update the relavant rows of the full matrix
VectorXs get_original_label(VectorXs const & restricted_labels, std::vector<int> const & indices) {
    VectorXs extended_labels(restricted_labels.rows());
    for(int i=0; i<restricted_labels.rows(); i++) {
        extended_labels[i] = indices[restricted_labels[i]];
    }
    return extended_labels;
}

/////////////////////////////
/////  prox-lp-utils    /////
/////////////////////////////
LP_inf_params::LP_inf_params(float prox_reg_const, float dual_gap_tol, float prox_energy_tol, 
        int prox_max_iter, int fw_max_iter, 
        int qp_max_iter, float qp_tol, float qp_const, 
        bool best_int, bool accel_prox, 
        float less_confident_percent, float confidence_tol):
        prox_reg_const(prox_reg_const), dual_gap_tol(dual_gap_tol), prox_energy_tol(prox_energy_tol), 
        prox_max_iter(prox_max_iter), fw_max_iter(fw_max_iter), 
        qp_max_iter(qp_max_iter), qp_tol(qp_tol), qp_const(qp_const), 
        best_int(best_int), accel_prox(accel_prox),
        less_confident_percent(less_confident_percent), confidence_tol(confidence_tol) {}

LP_inf_params::LP_inf_params() {
    prox_reg_const = 0.1;   
    dual_gap_tol = 1e3;     
    prox_energy_tol = 1e3;      
    prox_max_iter = 10;     
    fw_max_iter = 5;        
    qp_max_iter = 1000;     
    qp_tol = 1000;          
    qp_const = 1e-16;           
    best_int = true;
    accel_prox = true;
    less_confident_percent = 0;  // don't check for less confident pixels
    confidence_tol = 0.95;
}

LP_inf_params::LP_inf_params(const LP_inf_params& params) {
    prox_reg_const = params.prox_reg_const;
    dual_gap_tol = params.dual_gap_tol;     
    prox_energy_tol = params.prox_energy_tol;       
    prox_max_iter = params.prox_max_iter;       
    fw_max_iter = params.fw_max_iter;       
    qp_max_iter = params.qp_max_iter;       
    qp_tol = params.qp_tol;         
    qp_const = params.qp_const;         
    best_int = params.best_int;
    accel_prox = params.accel_prox;
    less_confident_percent = params.less_confident_percent;
    confidence_tol = params.confidence_tol;
}

// rescale Q to be within [0,1] -- order of Q values preserved!
void rescale(MatrixXf & out, const MatrixXf & Q) {
    out = Q;
    float minval, maxval;
    // don't do label-wise rescaling -> introduces different error in each label!
    // but one rescaling for teh entire matrix
    minval = out.minCoeff();
    out = out.array() - minval;
    maxval = out.maxCoeff();
    assert(maxval >= 0);
    if (maxval > 0) out /= maxval;
}

// make a step of qp_gamma: -- \cite{NNQP solver Xiao and Chen 2014} - O(n) implementation!!
void qp_gamma_step(VectorXf & v_gamma, const VectorXf & v_pos_h, const VectorXf & v_neg_h, 
        const float qp_delta, const int M, const float lambda, VectorXf & v_step, VectorXf & v_tmp, 
        VectorXf & v_tmp2) {
    // C: lambda*I - (lambda/m)*ones
    // neg_C: max(-C, 0)- elementwise
    // abs_C: abs(C)- elementwise
    float sum = v_gamma.sum();
    v_tmp2.fill(sum);
    
    v_tmp = v_tmp2 - v_gamma;
    v_tmp *= (2*lambda/float(M));  // 2 * neg_C * v_gamma
    v_step = v_tmp + v_pos_h;
    v_step = v_step.array() + qp_delta;

    v_tmp = (1.0/float(M)) * v_tmp2 + (1-2.0/float(M)) * v_gamma;
    v_tmp *= lambda;    // abs_C * v_gamma
    v_tmp = v_tmp + v_neg_h;
    v_tmp = v_tmp.array() + qp_delta;
    v_step = v_step.cwiseQuotient(v_tmp);
    v_gamma = v_gamma.cwiseProduct(v_step);
}

// multiply by C in linear time!
void qp_gamma_multiplyC(VectorXf & v_out, const VectorXf & v_in, const int M, const float lambda) {
    // C: lambda*I - (lambda/m)*ones
    float sum = v_in.sum();
    v_out.fill(sum);
    v_out = (-1.0/float(M)) * v_out + v_in;
    v_out *= lambda;
}

// return the indices of pixels for which maximum probability fo taking a label is less than tol
void less_confident_pixels(std::vector<int> & indices, const MatrixXf & Q, float tol) {
    indices.clear();
    for (int i = 0; i < Q.cols(); ++i) {
        float max_prob = Q.col(i).maxCoeff();
        if (max_prob <= tol) indices.push_back(i);  // indices are in ascending order! (used later!!)
    }
}

// update the restricted matrix with the corresponding values from the full matrix
void update_restricted_matrix(MatrixXf & out, const MatrixXf & in, const std::vector<int> & pindices) {
    assert(out.cols() == pindices.size());
    out.fill(0);

    for(int i=0; i<pindices.size(); i++) {
        out.col(i) = in.col(pindices[i]);
    }
}

// update the relavant rows of the full matrix that is in the restricted matrix
void update_extended_matrix(MatrixXf & out, const MatrixXf & in, const std::vector<int> & pindices) {
    assert(in.cols() == pindices.size());

    for(int i=0; i<pindices.size(); i++) {
        out.col(pindices[i]) = in.col(i);
    }
}

