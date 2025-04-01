functions {
  // Function for reduce_sum to compute log probability for a slice of groups
  real partial_log_prob(array[] int group_slice, int start, int end,
                        array[] int EpidemicStart, array[] int N, array[,] int deaths,
                        matrix E_deaths, real phi) {
    real lp = 0;
    for (i in 1:size(group_slice)) {
      int m = group_slice[i];               // Current group index
      int start_idx = EpidemicStart[m];     // Start time for this group
      int end_idx = N[m];                   // End time for this group
      for (t in start_idx:end_idx) {
        lp += neg_binomial_2_lpmf(deaths[t, m] | E_deaths[t, m], phi);
      }
    }
    return lp;
  }
}

data {
  int<lower=1> M;               // Number of groups
  int<lower=1> N0;              // Initial period length
  array[M] int N;               // Number of observations per group
  int<lower=1> N2;              // Total time points
  array[N2, M] int deaths;      // Observed deaths
  array[N2, M] real f;          // Factor (e.g., infection fatality rate)
  array[M] int EpidemicStart;   // Start of epidemic per group
  array[M] real pop;            // Population per group
  int W;                        // Number of weeks
  array[M, N2] int week_index;  // Week index for each group and time
  array[N2] real SI;            // Serial interval distribution
  real AR_SD_MEAN;              // Mean for weekly_sd prior
}

transformed data {
  // Array of group indices from 1 to M for reduce_sum slicing
  array[M] int group_indices;
  for (m in 1:M)
    group_indices[m] = m;
}

parameters {
  array[M] real<lower=0> y;     // Initial infection rate per group
  real<lower=0> phi;            // Precision parameter for negative binomial
  real<lower=0> tau;            // Rate parameter for y
  matrix[W+1, M] weekly_effect; // Weekly effects per group
  real<lower=0, upper=1> weekly_rho;   // AR(1) coefficient
  real<lower=0, upper=1> weekly_rho1;  // AR(2) coefficient
  real<lower=0> weekly_sd;      // Standard deviation for weekly effects
}

transformed parameters {
  matrix[N2, M] prediction = rep_matrix(0, N2, M);  // Predicted infections
  matrix[N2, M] E_deaths = rep_matrix(0, N2, M);    // Expected deaths
  matrix[N2, M] Rt = rep_matrix(0, N2, M);         // Reproduction number
  matrix[N2, M] Rt_adj = Rt;                        // Adjusted Rt

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
  // Priors
  tau ~ exponential(0.03);
  weekly_sd ~ normal(0, AR_SD_MEAN);
  weekly_rho ~ normal(0.8, 0.05);
  weekly_rho1 ~ normal(0.1, 0.05);
  
  for (m in 1:M) {
    y[m] ~ exponential(1/tau);
    weekly_effect[3:(W+1), m] ~ normal(
      weekly_effect[2:W, m] * weekly_rho + weekly_effect[1:(W-1), m] * weekly_rho1,
      weekly_sd * sqrt(1 - pow(weekly_rho, 2) - pow(weekly_rho1, 2)
                        - 2 * pow(weekly_rho, 2) * weekly_rho1 / (1 - weekly_rho1))
    );
  }
  weekly_effect[2, ] ~ normal(0, 
    weekly_sd * sqrt(1 - pow(weekly_rho, 2) - pow(weekly_rho1, 2)
                        - 2 * pow(weekly_rho, 2) * weekly_rho1 / (1 - weekly_rho1)));
  weekly_effect[1, ] ~ normal(0, 0.01);
  phi ~ normal(0, 5);
  
  // Parallelized likelihood computation
  target += reduce_sum(partial_log_prob, group_indices, 1, EpidemicStart, N, deaths, E_deaths, phi);
}