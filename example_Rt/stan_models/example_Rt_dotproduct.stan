data {
  int<lower=1> M;                     // Number of regions
  int<lower=1> N0;                    // Initial number of days
  array[M] int<lower=1> N;            // Number of observations per region
  int<lower=1> N2;                    // Total number of days
  array[N2, M] int deaths;            // Death counts
  array[N2, M] real f;                // Fatality rates
  array[M] int EpidemicStart;         // Start day of the epidemic per region
  array[M] real pop;                  // Population per region
  int W;                              // Number of weeks
  array[M, N2] int week_index;        // Week indices
  array[N2] real SI;                  // Serial interval
  real AR_SD_MEAN;                    // Mean of the autoregressive standard deviation
}

transformed data {
  array[N2] real SI_rev;              // Reversed serial interval
  matrix[N2, M] f_rev;                // Reversed fatality rates as matrix

  for (i in 1:N2) {
    SI_rev[i] = SI[N2 - i + 1];
    for (m in 1:M) {
      f_rev[i, m] = f[N2 - i + 1, m];
    }
  }
}

parameters {
  array[M] real<lower=0> y;           // Initial infections
  real<lower=0> phi;                  // Overdispersion parameter
  real<lower=0> tau;                  // Scale parameter for initial infections
  matrix[W + 1, M] weekly_effect;     // Weekly effects
  real<lower=0, upper=1> weekly_rho;  // Autoregressive coefficient
  real<lower=0, upper=1> weekly_rho1; // Second autoregressive coefficient
  real<lower=0> weekly_sd;            // Standard deviation of weekly effects
}

transformed parameters {
  matrix[N2, M] prediction = rep_matrix(0.0, N2, M);
  matrix[N2, M] E_deaths = rep_matrix(0.0, N2, M);
  matrix[N2, M] Rt = rep_matrix(0.0, N2, M);
  matrix[N2, M] Rt_adj = rep_matrix(0.0, N2, M);
  matrix[N2, M] cumm_sum = rep_matrix(0.0, N2, M);

  for (m in 1:M) {
    // Initialize cumulative sum and predictions for the first N0 days
    cumm_sum[1, m] = y[m];
    for (i in 2:N0) {
      cumm_sum[i, m] = cumm_sum[i - 1, m] + y[m];
    }
    prediction[1:N0, m] = rep_vector(y[m], N0);

    // Compute Rt and adjusted Rt for all days
    Rt[:, m] = 3.3 * 2 * inv_logit(-weekly_effect[week_index[m], m]);
    Rt_adj[1:N0, m] = Rt[1:N0, m];

    // Compute predictions for days beyond N0
    for (i in (N0 + 1):N2) {
      real convolution = dot_product(to_vector(prediction[1:(i - 1), m]), to_vector(SI_rev[(N2 - i + 2):N2]));
      cumm_sum[i, m] = cumm_sum[i - 1, m] + prediction[i - 1, m];
      Rt_adj[i, m] = ((pop[m] - cumm_sum[i, m]) / pop[m]) * Rt[i, m];
      prediction[i, m] = Rt_adj[i, m] * convolution;
    }

    // Compute expected deaths
    E_deaths[1, m] = 1e-15 * prediction[1, m];
    for (i in 2:N2) {
      E_deaths[i, m] = dot_product(prediction[1:(i - 1), m], f_rev[(N2 - i + 2):N2, m]);
    }
  }
}

model {
  // Priors
  tau ~ exponential(0.03);
  weekly_sd ~ normal(0, AR_SD_MEAN);
  weekly_rho ~ normal(0.8, 0.05);
  weekly_rho1 ~ normal(0.1, 0.05);
  phi ~ normal(0, 5);

  for (m in 1:M) {
    y[m] ~ exponential(1 / tau);
    weekly_effect[3:(W + 1), m] ~ normal(
      weekly_effect[2:W, m] * weekly_rho + weekly_effect[1:(W - 1), m] * weekly_rho1,
      weekly_sd * sqrt(1 - pow(weekly_rho, 2) - pow(weekly_rho1, 2) - 2 * pow(weekly_rho, 2) * weekly_rho1 / (1 - weekly_rho1))
    );
  }

  weekly_effect[2, ] ~ normal(0, weekly_sd * sqrt(1 - pow(weekly_rho, 2) - pow(weekly_rho1, 2) - 2 * pow(weekly_rho, 2) * weekly_rho1 / (1 - weekly_rho1)));
  weekly_effect[1, ] ~ normal(0, 0.01);

  // Likelihood
  for (m in 1:M) {
    deaths[EpidemicStart[m]:N[m], m] ~ neg_binomial_2(E_deaths[EpidemicStart[m]:N[m], m], phi);
  }
}