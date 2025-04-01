data {
  int<lower=1> M;
  int<lower=1> N0;
  array[M] int<lower=1> N;
  int<lower=1> N2;
  array[N2, M] int deaths;
  array[N2, M] real f;
  array[M] int EpidemicStart;
  array[M] real pop;
  int W;
  array[M, N2] int week_index;
  array[N2] real SI;
  real AR_SD_MEAN;
}

parameters {
  array[M] real<lower=0> y;
  real<lower=0> phi;
  real<lower=0> tau;
  matrix[W+1, M] weekly_effect;
  real<lower=0, upper=1> weekly_rho;
  real<lower=0, upper=1> weekly_rho1;
  real<lower=0> weekly_sd;
}

transformed parameters {
  matrix[N2, M] prediction = rep_matrix(0, N2, M);
  matrix[N2, M] E_deaths = rep_matrix(0, N2, M);
  matrix[N2, M] Rt = rep_matrix(0, N2, M);
  matrix[N2, M] Rt_adj = Rt;

  {
    matrix[N2, M] cumm_sum = rep_matrix(0, N2, M);
    for (m in 1:M) {
      for (i in 2:N0) {
        cumm_sum[i, m] = cumm_sum[i-1, m] + y[m];
      }
      prediction[1:N0, m] = rep_vector(y[m], N0);
      Rt[, m] = 3.3 * 2 * inv_logit(-weekly_effect[week_index[m], m]);
      Rt_adj[1:N0, m] = Rt[1:N0, m];
      for (i in (N0+1):N2) {
        real convolution = 0;
        for (j in 1:(i-1)) {
          convolution += prediction[j, m] * SI[i-j];
        }
        cumm_sum[i, m] = cumm_sum[i-1, m] + prediction[i-1, m];
        Rt_adj[i, m] = ((pop[m] - cumm_sum[i, m]) / pop[m]) * Rt[i, m];
        prediction[i, m] = Rt_adj[i, m] * convolution;
      }

      E_deaths[1, m] = 1e-15 * prediction[1, m];
      for (i in 2:N2) {
        for (j in 1:(i-1)) {
          E_deaths[i, m] += prediction[j, m] * f[i-j, m];
        }
      }
    }
  }
}

model {
  tau ~ exponential(0.03);
  weekly_sd ~ normal(0, AR_SD_MEAN);
  weekly_rho ~ normal(0.8, 0.05);
  weekly_rho1 ~ normal(0.1, 0.05);
  for (m in 1:M) {
    y[m] ~ exponential(1/tau);
    weekly_effect[3:(W+1), m] ~ normal(
      weekly_effect[2:W, m] * weekly_rho + weekly_effect[1:(W-1), m] * weekly_rho1,
      weekly_sd * sqrt(1 - pow(weekly_rho, 2) - pow(weekly_rho1, 2) - 2 * pow(weekly_rho, 2) * weekly_rho1 / (1 - weekly_rho1))
    );
  }
  weekly_effect[2, ] ~ normal(0, weekly_sd * sqrt(1 - pow(weekly_rho, 2) - pow(weekly_rho1, 2) - 2 * pow(weekly_rho, 2) * weekly_rho1 / (1 - weekly_rho1)));
  weekly_effect[1, ] ~ normal(0, 0.01);
  phi ~ normal(0, 5);
  for (m in 1:M) {
    deaths[EpidemicStart[m]:(N[m]), m] ~ neg_binomial_2(E_deaths[EpidemicStart[m]:(N[m]), m], phi);
  }
}
