#pragma once
#include <Eigen/Core>
#include <vector>

using namespace Eigen;

typedef MatrixXd MatrixP;
typedef VectorXd VectorP;
typedef double typeP;
typedef Matrix<short,Dynamic,1> VectorXs;

// eigen-utils //
bool all_close_to_zero(const VectorXf & vect, float ref);
bool all_positive(const VectorXf & vect);
bool all_strict_positive(const VectorXf & vect);
bool valid_probability(const MatrixXf & proba);
bool valid_probability_debug(const MatrixXf & proba);
void clamp_and_normalize(VectorXf & prob);

typeP dotProduct(const MatrixXf & M1, const MatrixXf & M2, MatrixP & temp);
void sortRows(const MatrixXf & M, MatrixXi & ind);
void sortCols(const MatrixXf & M, MatrixXi & ind);
float infiniteNorm(const MatrixXf & M);

// qp-utils //
void descent_direction(Eigen::MatrixXf & out, const Eigen::MatrixXf & grad );
float pick_lambda_eig_to_convex(const MatrixXf & lbl_compatibility);

// newton-cccp-utils //
void newton_cccp(VectorXf & state, const VectorXf & cste, float lamda_eig);
float pick_lambda_eig_to_concave(const MatrixXf & lbl_compatibility);

// mf-utils //
void expAndNormalize( MatrixXf & out, const MatrixXf & in);

// dc-neg-utils //
void kkt_solver(const VectorXf & lin_part, const MatrixXf & inv_KKT, VectorXf & out);

// lp-utils //
void renormalize(MatrixXf & Q);
void feasible_Q(MatrixXf & tmp, MatrixXi & ind, VectorXd & sum, VectorXi & K, const MatrixXf & Q);

void get_limited_indices(MatrixXf const & Q, std::vector<int> & indices);
MatrixXf get_restricted_matrix(MatrixXf const & in, std::vector<int> const & indices);
MatrixXf get_extended_matrix(MatrixXf const & in, std::vector<int> const & indices, int max_rows);
VectorXs get_original_label(VectorXs const & restricted_labels, std::vector<int> const & indices);

// prox-lp-utils //
class LP_inf_params {
public: 
    float prox_reg_const;   // proximal regularization constant
    float dual_gap_tol;     // dual gap tolerance
    float prox_energy_tol;  // proximal energy tolerance
    int prox_max_iter;      // maximum proximal iterations
    int fw_max_iter;        // maximum FW iterations
    int qp_max_iter;        // maximum qp-gamma iterations
    float qp_tol;           // qp-gamma tolerance
    float qp_const;         // const used in qp-gamma
    bool best_int;          // return the Q that yields the best integer energy
    bool accel_prox;        // accelerated proximal method
    // less-confident switch
    float less_confident_percent;   // percentage of less confident pixels to break
    float confidence_tol;           // tolerance to decide a pixel to be less-confident
    LP_inf_params(float prox_reg_const, float dual_gap_tol, float prox_energy_tol, 
        int prox_max_iter, int fw_max_iter, 
        int qp_max_iter, float qp_tol, float qp_const, 
        bool best_int, bool accel_prox, 
        float less_confident_percent, float confident_tol);
    LP_inf_params();    // default values
    LP_inf_params(const LP_inf_params& params); // copy constructor
};

void rescale(MatrixXf & out, const MatrixXf & Q);

void qp_gamma_step(VectorXf & v_gamma, const VectorXf & v_pos_h, const VectorXf & v_neg_h, 
        const float qp_delta, const int M, const float lambda, VectorXf & v_step, VectorXf & v_tmp, 
        VectorXf & v_tmp2);
void qp_gamma_multiplyC(VectorXf & v_out, const VectorXf & v_in, const int M, const float lambda);

void less_confident_pixels(std::vector<int> & indices, const MatrixXf & Q, float tol = 0.95);
void update_restricted_matrix(MatrixXf & out, const MatrixXf & in, const std::vector<int> & pindices);
void update_extended_matrix(MatrixXf & out, const MatrixXf & in, const std::vector<int> & pindices);

