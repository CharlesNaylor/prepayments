data {
    int N; #Number of records
    int K; #number of betas
    int month[N]; 
    matrix[N,K] exogs;
    int endo[N];
}
parameters {
    row_vector[K] beta;
    real intercept;
    real month_intercept[12]; #seasonality
    real<lower=0, upper=1> theta;
}
transformed parameters {
    vector<lower=0>[N] lambda;
    vector[N] mu;
    real log_lik[N];
    for(n in 1:N) {
      mu[n] = intercept +  month_intercept[month[n]] + beta * exogs[n]';
      lambda[n] = exp(mu[n]);
      if(endo[n] == 0) 
        log_lik[n] = log(theta);
      else
        log_lik[n] = log1m(theta) + poisson_lpmf(endo[n] | lambda[n]) 
                      - log1m_exp(-lambda[n]);
    }
}
model {
  intercept ~ normal(0, 0.1);
  to_vector(month_intercept) ~ normal(0, 0.1);
  beta[1] ~ normal(1,5); #incentive + upfront_mip
  beta[2] ~ normal(0,5); #cato
  beta[3] ~ normal(1,1); #hpa
  beta[4] ~ normal(-1,1); #sato
  theta ~ pareto(0.1, 1.5); #as per Gelman, 2013, ch.5
  for(n in 1:N) {
    target += log_lik[n];
  }
}
generated quantities {
  real endo_hat[N];
  for(n in 1:N) {
      endo_hat[n] = (1-theta) * poisson_rng(lambda[n]);
  }
}

