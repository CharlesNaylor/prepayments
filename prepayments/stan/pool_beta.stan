# stan code for models.PoolModel
# Now I'm going to fit a proportions model, which should be more appropriate
# for the data.
#Confusingly, Phi is used to denote the mean in mc-stan reference, but to
# denote precision in the Ferrari paper.
#c.f. https://www.ime.usp.br/~sferrari/beta.pdf
#     https://cran.r-project.org/web/packages/betareg/vignettes/betareg.pdf
#     mc-stan reference 2.14, p.267


functions {
    real cpr_pred_abbrev_rng(real lambda, real phi) {
        real cpr_pred;
        cpr_pred = beta_rng(lambda * phi, lambda * (1-phi));
        return cpr_pred;
    }
    real cpr_pred_rng(real[] season, real[] beta, real lambda, real intercept,
                    int month, real cato, real sato, real hpa, real lockin,
                    real burnout, real incentive, real mip) {
        real cpr_pred;
        real phi; #mu
        phi = inv_logit(intercept + season[month] +
                    beta[1] * cato +
                    beta[2] * sato +
                    beta[3] * hpa +
                    beta[4] * lockin +
                    beta[5] * burnout +
                    beta[6] * incentive +
                    beta[7] * mip);
        return cpr_pred_abbrev_rng(lambda, phi);
        }
}
data {
    int N; #number of records

    real<lower=0> cpr[N]; # next month's CPR (endo)

    real cato[N]; #Curve at origination
    real sato[N]; #spread at origination
    real hpa[N]; # home price appreciation
    real<lower=0> lockin[N]; #Lock-in rate
    real burnout[N]; #burnout
    real incentive[N]; #purchase rate spread over mkt mortgage rate
    real<lower=0> mip[N]; #mortgage insurance rate
    int<lower=1,upper=12> month[N]; #month of year (seasonality)
}
parameters {
    real season[12]; #seasonality constant
    real beta[7]; #factor betas
    real intercept; #base alpha
    real<lower=0.1> lambda; #dispersion
}
transformed parameters {
    vector[12] shrunk_season;
    vector[N] phi; #mu
    for(i in 1:12) {
        shrunk_season[i] = intercept + season[i];
    }
    for(n in 1:N) {
        phi[n] = inv_logit(shrunk_season[month[n]] +
                                beta[1] * cato[n] +
                                beta[2] * sato[n] +
                                beta[3] * hpa[n] +
                                beta[4] * lockin[n] +
                                beta[5] * burnout[n] +
                                beta[6] * incentive[n] +
                                beta[7] * mip[n]);
    }
}
model {
    to_vector(season) ~ normal(0, 0.1);
    intercept ~ normal(0,0.1);
    to_vector(beta) ~ normal(0, 10);
    lambda ~ pareto(0.1, 1.5); #As per Gelman, 2013, ch. 5
    cpr ~ beta(lambda * phi, lambda*(1-phi));
}
generated quantities {
   vector[N] log_lik;
   vector[N] cpr_pred;

   for(n in 1:N) {
        log_lik[n] = beta_lpdf(cpr[n] | lambda*phi[n],lambda*(1-phi[n]));
        cpr_pred[n] = cpr_pred_abbrev_rng(lambda, phi[n]);
   }
}

