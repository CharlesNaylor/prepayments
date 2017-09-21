# stan code for models.PoolModel
# I'm going to declare the exogs separately b/c I expect different priors
# For now, however, I think I'll do them all the same.

functions {
    real cpr_pred_rng(real[] season, real[] beta, real sigma, real alpha,
                    int month, real cato, real sato, real hpa, real lockin,
                    real burnout, real incentive, real mip) {
        real cpr_pred;
        cpr_pred = normal_rng(season[month] + 
                beta[1] * cato + 
                beta[2] * sato + 
                beta[3] * hpa + 
                beta[4] * lockin + 
                beta[5] * burnout + 
                beta[6] * incentive + 
                beta[7] * mip,
            sigma);
        return fmax(0.0, cpr_pred);
        }
}
data {
    int N; #number of records

    real<lower=0> cpr[N]; # next month's CPR (endo)

    real cato[N]; #Curve at origination
    real sato[N]; #spread at origination
    real hpa[N]; # home price appreciation
    real<lower=0> lockin[N]; #Lock-in rate
    real<lower=0> burnout[N]; #burnout
    real<lower=0> incentive[N]; #purchase rate spread over mkt mortgage rate
    real<lower=0> mip[N]; #mortgage insurance rate
    int<lower=1,upper=12> month[N]; #month of year (seasonality)
}
parameters {
    real season[12]; #seasonality constant
    real beta[7]; #factor betas
    real<lower=0> sigma; #cpr error term
    real alpha; #base alpha (so seasonality doesn't take it)
}
model {
    to_vector(season) ~ normal(0, 1);
    alpha ~ normal(0,1);
    to_vector(beta) ~ normal(0, 1);
    sigma ~ cauchy(0, 5);
    for(n in 1:N) {
        cpr[n] ~ normal(alpha + season[month[n]] + 
                                beta[1] * cato[n] + 
                                beta[2] * sato[n] + 
                                beta[3] * hpa[n] + 
                                beta[4] * lockin[n] + 
                                beta[5] * burnout[n] + 
                                beta[6] * incentive[n] + 
                                beta[7] * mip[n],
                        sigma);
    }
}
generated quantities {
    vector[N] cpr_pred;
    vector[N] log_lik;
    for(n in 1:N) {
        log_lik[n] = normal_lpdf(cpr[n] | alpha + season[month[n]] + 
                                            beta[1] * cato[n] + 
                                            beta[2] * sato[n] + 
                                            beta[3] * hpa[n] + 
                                            beta[4] * lockin[n] + 
                                            beta[5] * burnout[n] + 
                                            beta[6] * incentive[n] + 
                                            beta[7] * mip[n],
                        sigma);

        cpr_pred[n] = cpr_pred_rng(season, beta, sigma, alpha,
            month[n], cato[n], sato[n], hpa[n], lockin[n],
            burnout[n], incentive[n], mip[n]);
    }
}
