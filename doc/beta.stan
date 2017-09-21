
//    This file is part of rstanarm.
//    Copyright (C) 2015, 2016 Trustees of Columbia University
    

/*
    rstanarm is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    rstanarm is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with rstanarm.  If not, see <http://www.gnu.org/licenses/>.
*/

// GLM for a Gaussian, Gamma, inverse Gaussian, or Beta outcome
functions {

  /* for multiple .stan files */
  
  /** 
   * Create group-specific block-diagonal Cholesky factor, see section 2 of
   * https://cran.r-project.org/web/packages/lme4/vignettes/lmer.pdf
   * @param len_theta_L An integer indicating the length of returned vector, 
   *   which lme4 denotes as m
   * @param p An integer array with the number variables on the LHS of each |
   * @param dispersion Scalar standard deviation of the errors, calles sigma by lme4
   * @param tau Vector of scale parameters whose squares are proportional to the 
   *   traces of the relative covariance matrices of the group-specific terms
   * @param scale Vector of prior scales that are multiplied by elements of tau
   * @param zeta Vector of positive parameters that are normalized into simplexes
   *   and multiplied by the trace of the covariance matrix to produce variances
   * @param rho Vector of radii in the onion method for creating Cholesky factors
   * @param z_T Vector used in the onion method for creating Cholesky factors
   * @return A vector that corresponds to theta in lme4
   */
  vector make_theta_L(int len_theta_L, int[] p, real dispersion,
                      vector tau, vector scale, vector zeta,
                      vector rho, vector z_T) {
    vector[len_theta_L] theta_L;
    int zeta_mark = 1;
    int rho_mark = 1;
    int z_T_mark = 1;
    int theta_L_mark = 1;

    // each of these is a diagonal block of the implicit Cholesky factor
    for (i in 1:size(p)) { 
      int nc = p[i];
      if (nc == 1) { // "block" is just a standard deviation
        theta_L[theta_L_mark] = tau[i] * scale[i] * dispersion;
        // unlike lme4, theta[theta_L_mark] includes the dispersion term in it
        theta_L_mark = theta_L_mark + 1;
      }
      else { // block is lower-triangular               
        matrix[nc,nc] T_i; 
        real std_dev;
        real T21;
        real trace_T_i = square(tau[i] * scale[i] * dispersion) * nc;
        vector[nc] pi = segment(zeta, zeta_mark, nc); // gamma(zeta | shape, 1)
        pi = pi / sum(pi);                            // thus dirichlet(pi | shape)
        
        // unlike lme4, T_i includes the dispersion term in it
        zeta_mark = zeta_mark + nc;
        std_dev = sqrt(pi[1] * trace_T_i);
        T_i[1,1] = std_dev;
        
        // Put a correlation into T_i[2,1] and scale by std_dev
        std_dev = sqrt(pi[2] * trace_T_i);
        T21 = 2.0 * rho[rho_mark] - 1.0;
        rho_mark = rho_mark + 1;
        T_i[2,2] = std_dev * sqrt(1.0 - square(T21));
        T_i[2,1] = std_dev * T21;
        
        for (r in 2:(nc - 1)) { // scaled onion method to fill T_i
          int rp1 = r + 1;
          vector[r] T_row = segment(z_T, z_T_mark, r);
          real scale_factor = sqrt(rho[rho_mark] / dot_self(T_row)) * std_dev;
          z_T_mark = z_T_mark + r;
          std_dev = sqrt(pi[rp1] * trace_T_i);
          for(c in 1:r) T_i[rp1,c] = T_row[c] * scale_factor;
          T_i[rp1,rp1] = sqrt(1.0 - rho[rho_mark]) * std_dev;
          rho_mark = rho_mark + 1;
        }
        
        // now vech T_i
        for (c in 1:nc) for (r in c:nc) {
          theta_L[theta_L_mark] = T_i[r,c];
          theta_L_mark = theta_L_mark + 1;
        }
      }
    }
    return theta_L;
  }
  
  /** 
  * Create group-specific coefficients, see section 2 of
  * https://cran.r-project.org/web/packages/lme4/vignettes/lmer.pdf
  *
  * @param z_b Vector whose elements are iid normal(0,sigma) a priori
  * @param theta Vector with covariance parameters as defined in lme4
  * @param p An integer array with the number variables on the LHS of each |
  * @param l An integer array with the number of levels for the factor(s) on 
  *   the RHS of each |
  * @return A vector of group-specific coefficients
  */
  vector make_b(vector z_b, vector theta_L, int[] p, int[] l) {
    vector[rows(z_b)] b;
    int b_mark = 1;
    int theta_L_mark = 1;
    for (i in 1:size(p)) {
      int nc = p[i];
      if (nc == 1) {
        real theta_L_start = theta_L[theta_L_mark];
        for (s in b_mark:(b_mark + l[i] - 1)) 
          b[s] = theta_L_start * z_b[s];
        b_mark = b_mark + l[i];
        theta_L_mark = theta_L_mark + 1;
      }
      else {
        matrix[nc,nc] T_i = rep_matrix(0, nc, nc);
        for (c in 1:nc) {
          T_i[c,c] = theta_L[theta_L_mark];
          theta_L_mark = theta_L_mark + 1;
          for(r in (c+1):nc) {
            T_i[r,c] = theta_L[theta_L_mark];
            theta_L_mark = theta_L_mark + 1;
          }
        }
        for (j in 1:l[i]) {
          vector[nc] temp = T_i * segment(z_b, b_mark, nc);
          b_mark = b_mark - 1;
          for (s in 1:nc) b[b_mark + s] = temp[s];
          b_mark = b_mark + nc + 1;
        }
      }
    }
    return b;
  }

  /** 
   * Prior on group-specific parameters
   *
   * @param z_b A vector of primitive coefficients
   * @param z_T A vector of primitives for the unit vectors in the onion method
   * @param rho A vector radii for the onion method
   * @param zeta A vector of primitives for the simplexes
   * @param tau A vector of scale parameters
   * @param regularization A real array of LKJ hyperparameters
   * @param delta A real array of concentration paramters
   * @param shape A vector of shape parameters
   * @param t An integer indicating the number of group-specific terms
   * @param p An integer array with the number variables on the LHS of each |
   * @return nothing
   */
  void decov_lp(vector z_b, vector z_T, vector rho, vector zeta, vector tau,
                real[] regularization, real[] delta, vector shape,
                int t, int[] p) {
    int pos_reg = 1;
    int pos_rho = 1;
    target += normal_lpdf(z_b | 0, 1);
    target += normal_lpdf(z_T | 0, 1);
    for (i in 1:t) if (p[i] > 1) {
      vector[p[i] - 1] shape1;
      vector[p[i] - 1] shape2;
      real nu = regularization[pos_reg] + 0.5 * (p[i] - 2);
      pos_reg = pos_reg + 1;
      shape1[1] = nu;
      shape2[1] = nu;
      for (j in 2:(p[i]-1)) {
        nu = nu - 0.5;
        shape1[j] = 0.5 * j;
        shape2[j] = nu;
      }
      target += beta_lpdf(rho[pos_rho:(pos_rho + p[i] - 2)] | shape1, shape2);
      pos_rho = pos_rho + p[i] - 1;
    }
    target += gamma_lpdf(zeta | delta, 1);
    target += gamma_lpdf(tau  | shape, 1);
  }
  
  /**
   * Hierarchical shrinkage parameterization
   *
   * @param z_beta A vector of primitive coefficients
   * @param global A real array of positive numbers
   * @param local A vector array of positive numbers
   * @param global_prior_scale A positive real number
   * @param error_scale 1 or sigma in the Gaussian case
   * @return A vector of coefficientes
   */
  vector hs_prior(vector z_beta, real[] global, vector[] local, 
                  real global_prior_scale, real error_scale) {
    vector[rows(z_beta)] lambda;
    int K;
    K = rows(z_beta);
    for (k in 1:K) lambda[k] = local[1][k] * sqrt(local[2][k]);
    return z_beta .* lambda * global[1] * sqrt(global[2]) * 
           global_prior_scale * error_scale;
  }

  /** 
   * Hierarchical shrinkage plus parameterization
   *
   * @param z_beta A vector of primitive coefficients
   * @param global A real array of positive numbers
   * @param local A vector array of positive numbers
   * @param global_prior_scale A positive real number
   * @param error_scale 1 or sigma in the Gaussian case
   * @return A vector of coefficientes
   */
  vector hsplus_prior(vector z_beta, real[] global, vector[] local, 
                      real global_prior_scale, real error_scale) {
    return z_beta .* (local[1] .* sqrt(local[2])) .* 
           (local[3] .* sqrt(local[4])) * global[1] * sqrt(global[2]) * 
           global_prior_scale * error_scale;
  }
  
  /** 
   * Divide a scalar by a vector
   *
   * @param x The scalar in the numerator
   * @param y The vector in the denominator
   * @return An elementwise vector
   */
  vector divide_real_by_vector(real x, vector y) {
    int K = rows(y); 
    vector[K] ret;
    for (k in 1:K) ret[k] = x / y[k];
    return ret;
  }

  /** 
   * Cornish-Fisher expansion for standard normal to Student t
   *
   * See result 26.7.5 of
   * http://people.math.sfu.ca/~cbm/aands/page_949.htm
   *
   * @param z A scalar distributed standard normal
   * @param df A scalar degrees of freedom
   * @return An (approximate) Student t variate with df degrees of freedom
   */
  real CFt(real z, real df) {
    real z2 = square(z);
    real z3 = z2 * z;
    real z5 = z2 * z3;
    real z7 = z2 * z5;
    real z9 = z2 * z7;
    real df2 = square(df);
    real df3 = df2 * df;
    real df4 = df2 * df2;
    return z + (z3 + z) / (4 * df) + (5 * z5 + 16 * z3 + 3 * z) / (96 * df2)
           + (3 * z7 + 19 * z5 + 17 * z3 - 15 * z) / (384 * df3)
           + (79 * z9 + 776 * z7 + 1482 * z5 - 1920 * z3 - 945 * z) / (92160 * df4);
  }

  /** 
   * Return two-dimensional array of group membership
   *
   * @param N An integer indicating the number of observations
   * @param t An integer indicating the number of grouping variables
   * @return An two-dimensional integer array of group membership
   */
  int[,] make_V(int N, int t, int[] v) {
    int V[t,N];
    int pos = 1;
    if (t > 0) for (j in 1:N) for (i in 1:t) {
      V[i,j] = v[pos];
      pos = pos + 1;
    }
    return V;
  }


  /** 
   * Apply inverse link function to linear predictor
   *
   * @param eta Linear predictor vector
   * @param link An integer indicating the link function
   * @return A vector, i.e. inverse-link(eta)
   */
  vector linkinv_gauss(vector eta, int link) {
    if (link == 1)      return eta;
    else if (link == 2) return exp(eta); 
    else if (link == 3) return inv(eta);
    else reject("Invalid link");
    return eta; // never reached
  }

  /** 
  * Pointwise (pw) log-likelihood vector
  *
  * @param y The integer array corresponding to the outcome variable.
  * @param link An integer indicating the link function
  * @return A vector
  */
  vector pw_gauss(vector y, vector eta, real sigma, int link) {
    return -0.5 * log(6.283185307179586232 * sigma) - 
            0.5 * square((y - linkinv_gauss(eta, link)) / sigma);
  }

  /** 
  * Apply inverse link function to linear predictor
  *
  * @param eta Linear predictor vector
  * @param link An integer indicating the link function
  * @return A vector, i.e. inverse-link(eta)
  */
  vector linkinv_gamma(vector eta, int link) {
    if (link == 1)      return eta;
    else if (link == 2) return exp(eta);
    else if (link == 3) return inv(eta);
    else reject("Invalid link");
    return eta; // never reached
  }

  /** 
  * Pointwise (pw) log-likelihood vector
  *
  * @param y A vector corresponding to the outcome variable.
  * @param eta A vector of linear predictors
  * @param shape A real number for the shape parameter
  * @param link An integer indicating the link function
  * @param sum_log_y A scalar equal to the sum of log(y)
  * @return A scalar log-likelihood
  */
  real GammaReg(vector y, vector eta, real shape, 
                int link, real sum_log_y) {
    real ret;
    if (link < 1 || link > 3) reject("Invalid link");
    ret = rows(y) * (shape * log(shape) - lgamma(shape)) +
      (shape - 1) * sum_log_y;
    if (link == 2)      // link is log
      ret = ret - shape * sum(eta) - shape * sum(y ./ exp(eta));
    else if (link == 1) // link is identity
      ret = ret - shape * sum(log(eta)) - shape * sum(y ./ eta);
    else                // link is inverse
      ret = ret + shape * sum(log(eta)) - shape * dot_product(eta, y);
    return ret;
  }
  
  /** 
  * Pointwise (pw) log-likelihood vector
  *
  * @param y A vector corresponding to the outcome variable.
  * @param shape A real number for the shape parameter
  * @param link An integer indicating the link function
  * @return A vector
  */
  vector pw_gamma(vector y, vector eta, real shape, int link) {
    int N = rows(eta);
    vector[N] ll;
    if (link == 3) { // link = inverse
      for (n in 1:N) {
        ll[n] = gamma_lpdf(y[n] | shape, shape * eta[n]);
      }
    }
    else if (link == 2) { // link = log
      for (n in 1:N) {
        ll[n] = gamma_lpdf(y[n] | shape, shape / exp(eta[n]));
      }
    }
    else if (link == 1) { // link = identity
      for (n in 1:N) {
        ll[n] = gamma_lpdf(y[n] | shape, shape / eta[n]);
      }
    }
    else reject("Invalid link");
    return ll;
  }

  /** 
  * Apply inverse link function to linear predictor
  *
  * @param eta Linear predictor vector
  * @param link An integer indicating the link function
  * @return A vector, i.e. inverse-link(eta)
  */
  vector linkinv_inv_gaussian(vector eta, int link) {
    if (link == 1)      return eta;
    else if (link == 2) return exp(eta);
    else if (link == 3) return inv(eta);
    else if (link == 4) return inv_sqrt(eta);
    else reject("Invalid link");
    return eta; // never reached
  }

  /** 
  * inverse Gaussian log-PDF (for data only, excludes constants)
  *
  * @param y The vector of outcomes
  * @param mu The vector of conditional means
  * @param lambda A positive scalar dispersion parameter
  * @param sum_log_y A scalar equal to the sum of log(y)
  * @param sqrt_y A vector equal to sqrt(y)
  * @return A scalar
  */
  real inv_gaussian(vector y, vector mu, real lambda, 
                    real sum_log_y, vector sqrt_y) {
    return 0.5 * rows(y) * log(lambda / 6.283185307179586232) - 
      1.5 * sum_log_y - 
      0.5 * lambda * dot_self( (y - mu) ./ (mu .* sqrt_y) );
  }
  
  /** 
  * Pointwise (pw) log-likelihood vector
  *
  * @param y The integer array corresponding to the outcome variable.
  * @param eta The linear predictors
  * @param lamba A positive scalar dispersion parameter
  * @param link An integer indicating the link function
  * @param log_y A precalculated vector of the log of y
  * @param sqrt_y A precalculated vector of the square root of y
  * @return A vector of log-likelihoods
  */
  vector pw_inv_gaussian(vector y, vector eta, real lambda, 
                         int link, vector log_y, vector sqrt_y) {
    vector[rows(y)] mu;
    mu = linkinv_inv_gaussian(eta, link); // link checked
    return -0.5 * lambda * square( (y - mu) ./ (mu .* sqrt_y) ) +
            0.5 * log(lambda / 6.283185307179586232) - 1.5 * log_y;
  }
  
  /** 
  * PRNG for the inverse Gaussian distribution
  *
  * Algorithm from wikipedia 
  *
  * @param mu The expectation
  * @param lambda The dispersion
  * @return A draw from the inverse Gaussian distribution
  */
  real inv_gaussian_rng(real mu, real lambda) {
    real mu2 = square(mu);
    // compound declare & define does not work with _rng
    real z;
    real y;
    real x;
    z = uniform_rng(0,1);
    y = square(normal_rng(0,1));
    x = mu + ( mu2 * y - mu * sqrt(4 * mu * lambda * y + mu2 * square(y)) )
      / (2 * lambda);
    if (z <= (mu / (mu + x))) return x;
    else return mu2 / x;
  }
  
  /** 
  * Apply inverse link function to linear predictor for beta models
  *
  * @param eta Linear predictor vector
  * @param link An integer indicating the link function
  * @return A vector, i.e. inverse-link(eta)
  */
  vector linkinv_beta(vector eta, int link) {
    vector[rows(eta)] mu;
    if (link < 1 || link > 6) reject("Invalid link");
    if (link == 1)  // logit
      for(n in 1:rows(eta)) mu[n] = inv_logit(eta[n]);
    else if (link == 2)  // probit
      for(n in 1:rows(eta)) mu[n] = Phi(eta[n]);
    else if (link == 3)  // cloglog
      for(n in 1:rows(eta)) mu[n] = inv_cloglog(eta[n]);
    else if (link == 4) // cauchy
      for(n in 1:rows(eta)) mu[n] = cauchy_cdf(eta[n], 0.0, 1.0);
    else if (link == 5)  // log 
      for(n in 1:rows(eta)) {
          mu[n] = exp(eta[n]);
          if (mu[n] < 0 || mu[n] > 1)
            reject("mu needs to be between 0 and 1");
      }
    else if (link == 6) // loglog
      for(n in 1:rows(eta)) mu[n] = 1-inv_cloglog(-eta[n]); 
      
    return mu;
  }
  
  /** 
  * Apply inverse link function to linear predictor for dispersion for beta models
  *
  * @param eta Linear predictor vector
  * @param link An integer indicating the link function
  * @return A vector, i.e. inverse-link(eta)
  */
  vector linkinv_beta_z(vector eta, int link) {
    vector[rows(eta)] mu;
    if (link < 1 || link > 3) reject("Invalid link");
    if (link == 1)        // log
      for(n in 1:rows(eta)) mu[n] = exp(eta[n]);
    else if (link == 2)   // identity
      return eta;
    else if (link == 3)   // sqrt
      for(n in 1:rows(eta)) mu[n] = square(eta[n]);
    return mu;
  }
  
  /** 
  * Pointwise (pw) log-likelihood vector for beta models
  *
  * @param y The vector of outcomes
  * @param eta The linear predictors
  * @param dispersion Positive dispersion parameter
  * @param link An integer indicating the link function
  * @return A vector of log-likelihoods
  */
  vector pw_beta(vector y, vector eta, real dispersion, int link) {
    vector[rows(y)] ll;
    vector[rows(y)] mu;
    vector[rows(y)] shape1;
    vector[rows(y)] shape2;
    if (link < 1 || link > 6) reject("Invalid link");
    mu = linkinv_beta(eta, link);
    shape1 = mu * dispersion;
    shape2 = (1 - mu) * dispersion;
    for (n in 1:rows(y)) {
      ll[n] = beta_lpdf(y[n] | shape1[n], shape2[n]);
    }
    return ll;
  }

  /** 
  * Pointwise (pw) log-likelihood vector for beta models with z variables
  *
  * @param y The vector of outcomes
  * @param eta The linear predictors (for y)
  * @param eta_z The linear predictors (for dispersion)
  * @param link An integer indicating the link function passed to linkinv_beta
  * @param link_phi An integer indicating the link function passed to linkinv_beta_z
  * @return A vector of log-likelihoods
  */
  vector pw_beta_z(vector y, vector eta, vector eta_z, int link, int link_phi) {
    vector[rows(y)] ll;
    vector[rows(y)] mu;
    vector[rows(y)] mu_z;
    if (link < 1 || link > 6) reject("Invalid link");
    if (link_phi < 1 || link_phi > 3) reject("Invalid link");
    mu = linkinv_beta(eta, link);
    mu_z = linkinv_beta_z(eta_z, link_phi);
    for (n in 1:rows(y)) {
      ll[n] = beta_lpdf(y[n] | mu[n] * mu_z[n], (1-mu[n]) * mu_z[n]);
    }
    return ll;
  }
  
  /** 
  * test function for csr_matrix_times_vector
  *
  * @param m Integer number of rows
  * @param n Integer number of columns
  * @param w Vector (see reference manual)
  * @param v Integer array (see reference manual)
  * @param u Integer array (see reference manual)
  * @param b Vector that is multiplied from the left by the CSR matrix
  * @return A vector that is the product of the CSR matrix and b
  */
  vector test_csr_matrix_times_vector(int m, int n, vector w, 
                                      int[] v, int[] u, vector b) {
    return csr_matrix_times_vector(m, n, w, v, u, b); 
  }
  
}
data {

  // dimensions
  int<lower=0> N;  // number of observations
  int<lower=0> K;  // number of predictors
  
  // data
  vector[K] xbar;               // predictor means
  int<lower=0,upper=1> dense_X; // flag for dense vs. sparse
  matrix[N,K] X[dense_X];       // centered predictor matrix in the dense case

  // stuff for the sparse case
  int<lower=0> nnz_X;                    // number of non-zero elements in the implicit X matrix
  vector[nnz_X] w_X;                     // non-zero elements in the implicit X matrix
  int<lower=0> v_X[nnz_X];               // column indices for w_X
  int<lower=0> u_X[dense_X ? 0 : N + 1]; // where the non-zeros start in each row of X
  real lb_y; // lower bound on y
  real<lower=lb_y> ub_y; // upper bound on y
  vector<lower=lb_y, upper=ub_y>[N] y; // continuous outcome

  // flag indicating whether to draw from the prior
  int<lower=0,upper=1> prior_PD;  // 1 = yes
  
  // intercept
  int<lower=0,upper=1> has_intercept;  // 1 = yes
  
  // family (interpretation varies by .stan file)
  int<lower=1> family;
  
  // link function from location to linear predictor 
  int<lower=1> link;  // interpretation varies by .stan file
  
  // prior family: 0 = none, 1 = normal, 2 = student_t, 3 = hs, 4 = hs_plus, 
  //   5 = laplace, 6 = lasso, 7 = product_normal
  int<lower=0,upper=7> prior_dist;
  int<lower=0,upper=2> prior_dist_for_intercept;
  
  // prior family: 0 = none, 1 = normal, 2 = student_t, 3 = exponential
  int<lower=0,upper=3> prior_dist_for_aux;
  

  // weights
  int<lower=0,upper=1> has_weights;  // 0 = No, 1 = Yes
  vector[has_weights ? N : 0] weights;
  
  // offset
  int<lower=0,upper=1> has_offset;  // 0 = No, 1 = Yes
  vector[has_offset ? N : 0] offset;
  // declares prior_{mean, scale, df}, prior_{mean, scale, df}_for_intercept, prior_{mean, scale, df}_for_aux

  // hyperparameter values are set to 0 if there is no prior
  vector<lower=0>[K] prior_scale;
  real<lower=0> prior_scale_for_intercept;
  real<lower=0> prior_scale_for_aux;
  vector[K] prior_mean;
  real prior_mean_for_intercept;
  real<lower=0> prior_mean_for_aux;
  vector<lower=0>[K] prior_df;
  real<lower=0> prior_df_for_intercept;
  real<lower=0> prior_df_for_aux;  
  real<lower=0> global_prior_df;    // for hs priors only
  real<lower=0> global_prior_scale; // for hs priors only
  int<lower=2> num_normals[prior_dist == 7 ? K : 0];
  // declares t, p[t], l[t], q, len_theta_L, shape, scale, {len_}concentration, {len_}regularization

  // glmer stuff, see table 3 of
  // https://cran.r-project.org/web/packages/lme4/vignettes/lmer.pdf
  int<lower=0> t;               // num. terms (maybe 0) with a | in the glmer formula
  int<lower=1> p[t];            // num. variables on the LHS of each |
  int<lower=1> l[t];            // num. levels for the factor(s) on the RHS of each |
  int<lower=0> q;               // conceptually equals \sum_{i=1}^t p_i \times l_i
  int<lower=0> len_theta_L;     // length of the theta_L vector

  // hyperparameters for glmer stuff; if t > 0 priors are mandatory
  vector<lower=0>[t] shape; 
  vector<lower=0>[t] scale;
  int<lower=0> len_concentration;
  real<lower=0> concentration[len_concentration];
  int<lower=0> len_regularization;
  real<lower=0> regularization[len_regularization];

  int<lower=0> num_non_zero;         // number of non-zero elements in the Z matrix
  vector[num_non_zero] w;            // non-zero elements in the implicit Z matrix
  int<lower=0> v[num_non_zero];      // column indices for w
  int<lower=0> u[t > 0 ? N + 1 : 0]; // where the non-zeros start in each row
  int<lower=0,upper=1> special_case; // is the only term (1|group)

  // betareg data
  int<lower=0, upper=1> has_intercept_z;  // presence of z intercept
  int<lower=0> link_phi;                  // link transformation for eta_z (0 => no z in model)
  int<lower=0> z_dim;                     // dimensions of z vars
  matrix[N, z_dim] betareg_z;             // matrix of z vars
  row_vector[z_dim] zbar;                 // mean of predictors
  // betareg hyperparameters
  int<lower=0,upper=7> prior_dist_z;
  int<lower=0,upper=2> prior_dist_for_intercept_z;
  vector<lower=0>[z_dim] prior_scale_z;
  real<lower=0> prior_scale_for_intercept_z;
  vector[z_dim] prior_mean_z;
  real prior_mean_for_intercept_z;
  vector<lower=0>[z_dim] prior_df_z;
  real<lower=0> prior_df_for_intercept_z;
  real<lower=0> global_prior_scale_z;
  int<lower=2> num_normals_z[prior_dist_z == 7 ? z_dim : 0];
}
transformed data {
  vector[family == 3 ? N : 0] sqrt_y;
  vector[family == 3 ? N : 0] log_y;
  real sum_log_y = family == 1 ? not_a_number() : sum(log(y));
  int<lower=1> V[t, N] = make_V(N, t, v);
  int<lower=0> hs_z;                  // for tdata_betareg.stan

  int<lower=0> len_z_T = 0;
  int<lower=0> len_var_group = sum(p) * (t > 0);
  int<lower=0> len_rho = sum(p) - t;
  int<lower=0, upper=1> is_continuous = 0; // changed in continuous.stan
  int<lower=1> pos = 1;
  real<lower=0> delta[len_concentration];
  int<lower=0> hs;
  if (prior_dist <= 2) hs = 0;
  else if (prior_dist == 3) hs = 2;
  else if (prior_dist == 4) hs = 4;
  else hs = 0;
  len_z_T = 0;
  len_var_group = sum(p) * (t > 0);
  len_rho = sum(p) - t;
  pos = 1;
  for (i in 1:t) {
    if (p[i] > 1) {
      for (j in 1:p[i]) {
        delta[pos] = concentration[j];
        pos = pos + 1;
      }
    }
    for (j in 3:p[i]) len_z_T = len_z_T + p[i] - 1;
  }

  if (prior_dist_z <= 2) hs_z = 0;
  else if (prior_dist_z == 3) hs_z = 2;
  else if (prior_dist_z == 4) hs_z = 4;
  else hs_z = 0;
  is_continuous = 1;

  if (family == 3) {
    sqrt_y = sqrt(y);
    log_y = log(y);
  }
}
parameters {
  real<lower=((family == 1 || link == 2) || (family == 4 && link == 5) ? negative_infinity() : 0.0), 
       upper=((family == 4 && link == 5) ? 0.0 : positive_infinity())> gamma[has_intercept];

  vector[prior_dist == 7 ? sum(num_normals) : K] z_beta;
  real<lower=0> global[hs];
  vector<lower=0>[K] local[hs];
  vector<lower=0>[K] S[prior_dist == 5 || prior_dist == 6];
  real<lower=0> one_over_lambda[prior_dist == 6];
  vector[q] z_b;
  vector[len_z_T] z_T;
  vector<lower=0,upper=1>[len_rho] rho;
  vector<lower=0>[len_concentration] zeta;
  vector<lower=0>[t] tau;
  real<lower=0> aux_unscaled; # interpretation depends on family!

  vector[prior_dist_z == 7 ? sum(num_normals_z) : z_dim] z_omega; // betareg z variable coefficients
  real gamma_z[has_intercept_z];  // betareg intercept
  real<lower=0> global_z[hs_z];
  vector<lower=0>[z_dim] local_z[hs_z];
  vector<lower=0>[z_dim] S_z[prior_dist_z == 5 || prior_dist_z == 6];
  real<lower=0> one_over_lambda_z[prior_dist_z == 6];
}
transformed parameters {
  // aux has to be defined first in the hs case
  real aux = prior_dist_for_aux == 0 ? aux_unscaled : (prior_dist_for_aux <= 2 ? 
             prior_scale_for_aux * aux_unscaled + prior_mean_for_aux :
             prior_scale_for_aux * aux_unscaled);
  vector[z_dim] omega; // used in tparameters_betareg.stan             

  vector[K] beta;
  vector[q] b;
  vector[len_theta_L] theta_L;
  if      (prior_dist == 0) beta = z_beta;
  else if (prior_dist == 1) beta = z_beta .* prior_scale + prior_mean;
  else if (prior_dist == 2) for (k in 1:K) {
    beta[k] = CFt(z_beta[k], prior_df[k]) * prior_scale[k] + prior_mean[k];
  }
  else if (prior_dist == 3) {
    if (is_continuous == 1 && family == 1)
      beta = hs_prior(z_beta, global, local, global_prior_scale, aux);
    else beta = hs_prior(z_beta, global, local, global_prior_scale, 1);
  }
  else if (prior_dist == 4) {
    if (is_continuous == 1 && family == 1)
      beta = hsplus_prior(z_beta, global, local, global_prior_scale, aux);
    else beta = hsplus_prior(z_beta, global, local, global_prior_scale, 1);
  }
  else if (prior_dist == 5) // laplace
    beta = prior_mean + prior_scale .* sqrt(2 * S[1]) .* z_beta;
  else if (prior_dist == 6) // lasso
    beta = prior_mean + one_over_lambda[1] * prior_scale .* sqrt(2 * S[1]) .* z_beta;
  else if (prior_dist == 7) { // product_normal
    int z_pos = 1;
    for (k in 1:K) {
      beta[k] = z_beta[z_pos];
      z_pos = z_pos + 1;
      for (n in 2:num_normals[k]) {
        beta[k] = beta[k] * z_beta[z_pos];
        z_pos = z_pos + 1;
      }
      beta[k] = beta[k] * prior_scale[k] ^ num_normals[k] + prior_mean[k];
    }
  }

  if (prior_dist_z == 0) omega = z_omega;
  else if (prior_dist_z == 1) omega = z_omega .* prior_scale_z + prior_mean_z;
  else if (prior_dist_z == 2) for (k in 1:z_dim) {
    omega[k] = CFt(omega[k], prior_df_z[k]) * prior_scale_z[k] + prior_mean_z[k];
  }
  else if (prior_dist_z == 3) 
    omega = hs_prior(z_omega, global_z, local_z, global_prior_scale, 1);
  else if (prior_dist_z == 4) 
    omega = hsplus_prior(z_omega, global_z, local_z, global_prior_scale, 1);
  else if (prior_dist_z == 5)
    omega = prior_mean_z + prior_scale_z .* sqrt(2 * S_z[1]) .* z_omega;
  else if (prior_dist_z == 6)
    omega = prior_mean_z + one_over_lambda_z[1] * prior_scale_z .* sqrt(2 * S_z[1]) .* z_omega;
  else if (prior_dist_z == 7) {
    int z_pos = 1;
    for (k in 1:z_dim) {
      omega[k] = z_omega[z_pos];
      z_pos = z_pos + 1;
      for (n in 2:num_normals_z[k]) {
        omega[k] = omega[k] * z_omega[z_pos];
        z_pos = z_pos + 1;
      }
      omega[k] = omega[k] * prior_scale_z[k] ^ num_normals_z[k] + prior_mean_z[k];
    }
  }
    
  
  if (prior_dist_for_aux == 0) // none
    aux = aux_unscaled;
  else {
    aux = prior_scale_for_aux * aux_unscaled;
    if (prior_dist_for_aux <= 2) // normal or student_t
      aux = aux + prior_mean_for_aux;
  }

  if (t > 0) {
    if (special_case == 1) {
      int start = 1;
      theta_L = tau * aux;
      if (t == 1) b = theta_L[1] * z_b;
      else for (i in 1:t) {
        int end = start + l[i] - 1;
        b[start:end] = theta_L[i] * z_b[start:end];
        start = end + 1;
      }
    }
    else {
      theta_L = make_theta_L(len_theta_L, p, 
                             aux, tau, scale, zeta, rho, z_T);
      b = make_b(z_b, theta_L, p, l);
    }
  }
}
model {
  vector[N] eta_z; // beta regression linear predictor for phi

  vector[N] eta;  // linear predictor
  if (K > 0) {
    if (dense_X) eta = X[1] * beta;
    else eta = csr_matrix_times_vector(N, K, w_X, v_X, u_X, beta);
  }
  else eta = rep_vector(0.0, N);
  if (has_offset == 1) eta = eta + offset;
  if (t > 0) {

    if (special_case) for (i in 1:t) eta = eta + b[V[i]];
    else eta = eta + csr_matrix_times_vector(N, q, w, v, u, b);
  }
  if (has_intercept == 1) {
    if ((family == 1 || link == 2) || (family == 4 && link != 5)) eta = eta + gamma[1];
    else if (family == 4 && link == 5) eta = eta - max(eta) + gamma[1];
    else eta = eta - min(eta) + gamma[1];
  }
  else {

  // correction to eta if model has no intercept (because X is centered)
  eta = eta + dot_product(xbar, beta); 
  }
  

 if (family == 4 && z_dim > 0 && link_phi > 0) {
    eta_z = betareg_z * omega;
  }
  else if (family == 4 && z_dim == 0 && has_intercept_z == 1){
    eta_z = rep_vector(0.0, N); 
  }
  // adjust eta_z according to links
  if (has_intercept_z == 1) {
    if (link_phi > 1) {
      eta_z = eta_z - min(eta_z) + gamma_z[1];
    }
    else {
      eta_z = eta_z + gamma_z[1];
    }
  }
  else { // has_intercept_z == 0

  if (link_phi > 1) {
    eta_z = eta_z - min(eta_z) + dot_product(zbar, omega);
  }
  else {
    eta_z = eta_z + dot_product(zbar, omega);
  }
  }

  // Log-likelihood 
  if (has_weights == 0 && prior_PD == 0) { // unweighted log-likelihoods
    if (family == 1) {
      if (link == 1) 
        target += normal_lpdf(y | eta, aux);
      else if (link == 2) 
        target += normal_lpdf(y | exp(eta), aux);
      else 
        target += normal_lpdf(y | divide_real_by_vector(1, eta), aux);
      // divide_real_by_vector() is defined in common_functions.stan
    }
    else if (family == 2) {
      target += GammaReg(y, eta, aux, link, sum_log_y);
    }
    else if (family == 3) {
      target += inv_gaussian(y, linkinv_inv_gaussian(eta, link), 
                             aux, sum_log_y, sqrt_y);
    }
    else if (family == 4 && link_phi == 0) {
      vector[N] mu;
      mu = linkinv_beta(eta, link);
      target += beta_lpdf(y | mu * aux, (1 - mu) * aux);
    }
    else if (family == 4 && link_phi > 0) {
      vector[N] mu;
      vector[N] mu_z;
      mu = linkinv_beta(eta, link);
      mu_z = linkinv_beta_z(eta_z, link_phi);
      target += beta_lpdf(y | rows_dot_product(mu, mu_z), 
                          rows_dot_product((1 - mu) , mu_z));
    }
  }
  else if (prior_PD == 0) { // weighted log-likelihoods
    vector[N] summands;
    if (family == 1) summands = pw_gauss(y, eta, aux, link);
    else if (family == 2) summands = pw_gamma(y, eta, aux, link);
    else if (family == 3) summands = pw_inv_gaussian(y, eta, aux, link, log_y, sqrt_y);
    else if (family == 4 && link_phi == 0) summands = pw_beta(y, eta, aux, link);
    else if (family == 4 && link_phi > 0) summands = pw_beta_z(y, eta, eta_z, link, link_phi);
    target += dot_product(weights, summands);
  }

  // Log-priors
  if (prior_dist_for_aux > 0 && prior_scale_for_aux > 0) {
    if (prior_dist_for_aux == 1)
      target += normal_lpdf(aux_unscaled | 0, 1);
    else if (prior_dist_for_aux == 2)
      target += student_t_lpdf(aux_unscaled | prior_df_for_aux, 0, 1);
    else 
     target += exponential_lpdf(aux_unscaled | 1);
  }
    

  // Log-priors for coefficients
       if (prior_dist == 1) target += normal_lpdf(z_beta | 0, 1);
  else if (prior_dist == 2) target += normal_lpdf(z_beta | 0, 1); // Student t
  else if (prior_dist == 3) { // hs
    target += normal_lpdf(z_beta | 0, 1);
    target += normal_lpdf(local[1] | 0, 1);
    target += inv_gamma_lpdf(local[2] | 0.5 * prior_df, 0.5 * prior_df);
    target += normal_lpdf(global[1] | 0, 1);
    target += inv_gamma_lpdf(global[2] | 0.5 * global_prior_df, 0.5 * global_prior_df);
  }
  else if (prior_dist == 4) { // hs+
    target += normal_lpdf(z_beta | 0, 1);
    target += normal_lpdf(local[1] | 0, 1);
    target += inv_gamma_lpdf(local[2] | 0.5 * prior_df, 0.5 * prior_df);
    target += normal_lpdf(local[3] | 0, 1);
    // unorthodox useage of prior_scale as another df hyperparameter
    target += inv_gamma_lpdf(local[4] | 0.5 * prior_scale, 0.5 * prior_scale);
    target += normal_lpdf(global[1] | 0, 1);
    target += inv_gamma_lpdf(global[2] | 0.5 * global_prior_df, 0.5 * global_prior_df);
  }
  else if (prior_dist == 5) { // laplace
    target += normal_lpdf(z_beta | 0, 1);
    target += exponential_lpdf(S[1] | 1);
  }
  else if (prior_dist == 6) { // lasso
    target += normal_lpdf(z_beta | 0, 1);
    target += exponential_lpdf(S[1] | 1);
    target += chi_square_lpdf(one_over_lambda[1] | prior_df[1]);
  }
  else if (prior_dist == 7) { // product_normal
    target += normal_lpdf(z_beta | 0, 1);
  }
  /* else prior_dist is 0 and nothing is added */
  
  // Log-prior for intercept  
  if (has_intercept == 1) {
    if (prior_dist_for_intercept == 1)  // normal
      target += normal_lpdf(gamma | prior_mean_for_intercept, prior_scale_for_intercept);
    else if (prior_dist_for_intercept == 2)  // student_t
      target += student_t_lpdf(gamma | prior_df_for_intercept, prior_mean_for_intercept, 
                               prior_scale_for_intercept);
    /* else prior_dist is 0 and nothing is added */
  }

  // Log-priors for coefficients
  if (prior_dist_z == 1)  target += normal_lpdf(z_omega | 0, 1);
  else if (prior_dist_z == 2) target += normal_lpdf(z_omega | 0, 1);
  else if (prior_dist_z == 3) { // hs
    target += normal_lpdf(z_omega | 0, 1);
    target += normal_lpdf(local_z[1] | 0, 1);
    target += inv_gamma_lpdf(local_z[2] | 0.5 * prior_df_z, 0.5 * prior_df_z);
    target += normal_lpdf(global_z[1] | 0, 1);
    target += inv_gamma_lpdf(global_z[2] | 0.5, 0.5);
  }
  else if (prior_dist_z == 4) { // hs+
    target += normal_lpdf(z_omega | 0, 1);
    target += normal_lpdf(local_z[1] | 0, 1);
    target += inv_gamma_lpdf(local_z[2] | 0.5 * prior_df_z, 0.5 * prior_df_z);
    target += normal_lpdf(local_z[3] | 0, 1);
    // unorthodox useage of prior_scale as another df hyperparameter
    target += inv_gamma_lpdf(local_z[4] | 0.5 * prior_scale_z, 0.5 * prior_scale_z);
    target += normal_lpdf(global_z[1] | 0, 1);
    target += inv_gamma_lpdf(global_z[2] | 0.5, 0.5);
  }
  else if (prior_dist_z == 5) { // laplace
    target += normal_lpdf(z_omega | 0, 1);
    target += exponential_lpdf(S_z[1] | 1);
  }
  else if (prior_dist_z == 6) { // lasso
    target += normal_lpdf(z_omega | 0, 1);
    target += exponential_lpdf(S_z[1] | 1);
    target += chi_square_lpdf(one_over_lambda_z[1] | prior_df_z[1]);
  }
  else if (prior_dist_z == 7) { // product_normal
    target += normal_lpdf(z_omega | 0, 1);
  }
  /* else prior_dist is 0 and nothing is added */
  
  // Log-prior for intercept  
  if (has_intercept_z == 1) {
    if (prior_dist_for_intercept_z == 1)  // normal
      target += normal_lpdf(gamma_z | prior_mean_for_intercept_z, prior_scale_for_intercept_z);
    else if (prior_dist_for_intercept_z == 2)  // student_t
      target += student_t_lpdf(gamma_z | prior_df_for_intercept_z, prior_mean_for_intercept_z, 
                               prior_scale_for_intercept_z);
    /* else prior_dist is 0 and nothing is added */
  }
  if (t > 0) decov_lp(z_b, z_T, rho, zeta, tau, 
                      regularization, delta, shape, t, p);
}
generated quantities {
  real alpha[has_intercept];
  real omega_int[has_intercept_z];
  real mean_PPD = 0;
  if (has_intercept == 1) {
    if (dense_X) alpha[1] = gamma[1] - dot_product(xbar, beta);
    else alpha[1] = gamma[1];
  }
  if (has_intercept_z == 1) { 
    omega_int[1] = gamma_z[1] - dot_product(zbar, omega);  // adjust betareg intercept 
  }
  {
    real nan_count; // for the beta_rng underflow issue
    real yrep; // pick up value to test for the beta_rng underflow issue
    vector[N] eta_z;

  vector[N] eta;  // linear predictor
  if (K > 0) {
    if (dense_X) eta = X[1] * beta;
    else eta = csr_matrix_times_vector(N, K, w_X, v_X, u_X, beta);
  }
  else eta = rep_vector(0.0, N);
  if (has_offset == 1) eta = eta + offset;
    nan_count = 0;
    if (t > 0) {

    if (special_case) for (i in 1:t) eta = eta + b[V[i]];
    else eta = eta + csr_matrix_times_vector(N, q, w, v, u, b);
    }
    if (has_intercept == 1) {
      if ((family == 1 || link == 2) || (family == 4 && link != 5)) eta = eta + gamma[1];
      else if (family == 4 && link == 5) {
        real max_eta;
        max_eta = max(eta);
        alpha[1] = alpha[1] - max_eta;
        eta = eta - max_eta + gamma[1];
      }
      else {
        real min_eta = min(eta);
        alpha[1] = alpha[1] - min_eta;
        eta = eta - min_eta + gamma[1];
      }
    }
    else {

  // correction to eta if model has no intercept (because X is centered)
  eta = eta + dot_product(xbar, beta); 
    }
    

 if (family == 4 && z_dim > 0 && link_phi > 0) {
    eta_z = betareg_z * omega;
  }
  else if (family == 4 && z_dim == 0 && has_intercept_z == 1){
    eta_z = rep_vector(0.0, N); 
  }
    // adjust eta_z according to links
    if (has_intercept_z == 1) {
      if (link_phi > 1) {
        omega_int[1] = omega_int[1] - min(eta_z);
        eta_z = eta_z - min(eta_z) + gamma_z[1];
      }
      else {
        eta_z = eta_z + gamma_z[1];
      }
    }
    else { // has_intercept_z == 0

  if (link_phi > 1) {
    eta_z = eta_z - min(eta_z) + dot_product(zbar, omega);
  }
  else {
    eta_z = eta_z + dot_product(zbar, omega);
  }
    }
    
    if (family == 1) {
      if (link > 1) eta = linkinv_gauss(eta, link);
      for (n in 1:N) mean_PPD = mean_PPD + normal_rng(eta[n], aux);
    }
    else if (family == 2) {
      if (link > 1) eta = linkinv_gamma(eta, link);
      for (n in 1:N) mean_PPD = mean_PPD + gamma_rng(aux, aux / eta[n]);
    }
    else if (family == 3) {
      if (link > 1) eta = linkinv_inv_gaussian(eta, link);
      for (n in 1:N) mean_PPD = mean_PPD + inv_gaussian_rng(eta[n], aux);
    }
    else if (family == 4 && link_phi == 0) { 
      eta = linkinv_beta(eta, link);
      for (n in 1:N) 
        mean_PPD = mean_PPD + beta_rng(eta[n] * aux, (1 - eta[n]) * aux);
    }
    else if (family == 4 && link_phi > 0) {
      eta = linkinv_beta(eta, link);
      eta_z = linkinv_beta_z(eta_z, link_phi);
      for (n in 1:N) {
        yrep = beta_rng(eta[n] * eta_z[n], (1 - eta[n]) * eta_z[n]);
          if (is_nan(yrep) == 1) {
            mean_PPD = mean_PPD;
            nan_count = nan_count + 1;
          }
          else if (is_nan(yrep) == 0) {
            mean_PPD = mean_PPD + yrep; 
          }
      }
    }
    if (family == 4 && link_phi > 0) {
      mean_PPD = mean_PPD / (N - nan_count);
    }
    else {
      mean_PPD = mean_PPD / N;
    }
  }
}
