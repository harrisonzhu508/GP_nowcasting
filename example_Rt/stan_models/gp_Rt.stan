data {
  int<lower=1> M;               // Number of regions
  int<lower=1> N0;              // Initial number of days
  array[M] int<lower=1> N;       // Number of observations per region (daily)
  int<lower=1> N2;              // Total number of days (daily)
  int<lower=1> W;               // Number of weeks
  array[N2, M] int deaths;      // Death counts (daily)
  array[N2, M] real f;          // Fatality rates (daily)
  array[M] int EpidemicStart;   // Start day per region (daily)
  array[M] real pop;            // Population per region
  array[N2] real SI;            // Serial interval (daily)
  vector[W] x_week;             // Weekly time index, e.g., x_week[i]=i
  // Optionally, you can still pass daily x if needed for other computations.
}

transformed data {
  array[N2] real SI_rev;
  matrix[N2, M] f_rev;
  for (i in 1:N2) {
    SI_rev[i] = SI[N2 - i + 1];
    for (m in 1:M) {
      f_rev[i, m] = f[N2 - i + 1, m];
    }
  }
}

parameters {
  array[M] real<lower=0> y;        // Initial infections (daily)
  real<lower=0> phi;               // Overdispersion parameter
  real<lower=0> tau;               // Scale parameter for initial infections
  
  // GP hyperparameters for weekly effect
  vector<lower=0>[M] alpha_gp_week;  // Weekly GP amplitude
  vector<lower=0>[M] rho_gp_week;    // Weekly GP lengthscale
  
  // Latent weekly GP draws for each region: dimension [W x M]
  matrix[W, M] gp_raw_week;          
}

transformed parameters {
  // Compute weekly GP values for each region
  matrix[W, M] gp_week;
  {
    for (m in 1:M) {
      matrix[W, W] K_week;
      // Use cov_exp_quad for efficiency if available:
      K_week = cov_exp_quad(x_week, alpha_gp_week[m], rho_gp_week[m]);
      for (i in 1:W)
        K_week[i, i] = K_week[i, i] + 1e-6;
      matrix[W, W] L_K_week = cholesky_decompose(K_week);
      gp_week[, m] = L_K_week * gp_raw_week[, m];
    }
  }
  
  // Upsample weekly GP to daily effect
  // Here, we assume piecewise constant: each day in week i gets the GP value for week i.
  vector[N2] gp_daily[M];
  {
    for (m in 1:M) {
      for (i in 1:N2) {
        // Map day i to week index: week = ceil(i/7)
        int week_i = min( (i + 6) / 7, W); // integer division rounding up
        gp_daily[m][i] = gp_week[week_i, m];
      }
    }
  }
  
  // Now use gp_daily to compute Rt
  matrix[N2, M] Rt;
  for (m in 1:M) {
    for (i in 1:N2) {
      Rt[i, m] = 3.3 * 2 * inv_logit(gp_daily[m][i]);
    }
  }
  
  // The rest of the model stays similar:
  matrix[N2, M] prediction = rep_matrix(0.0, N2, M);
  matrix[N2, M] E_deaths = rep_matrix(0.0, N2, M);
  matrix[N2, M] Rt_adj = rep_matrix(0.0, N2, M);
  matrix[N2, M] cumm_sum = rep_matrix(0.0, N2, M);

  for (m in 1:M) {
    cumm_sum[1, m] = y[m];
    for (i in 2:N0) {
      cumm_sum[i, m] = cumm_sum[i - 1, m] + y[m];
    }
    for (i in 1:N0) {
      prediction[i, m] = y[m];
      Rt_adj[i, m] = Rt[i, m];
    }
    for (i in (N0 + 1):N2) {
      real convolution = 0.0;
      for (j in 1:(i - 1)) {
        convolution += prediction[j, m] * SI_rev[i - j];
      }
      cumm_sum[i, m] = cumm_sum[i - 1, m] + prediction[i - 1, m];
      Rt_adj[i, m] = ((pop[m] - cumm_sum[i, m]) / pop[m]) * Rt[i, m];
      prediction[i, m] = Rt_adj[i, m] * convolution;
    }
    E_deaths[1, m] = 1e-15 * prediction[1, m];
    for (i in 2:N2) {
      real sum_ = 0.0;
      for (j in 1:(i - 1)) {
        sum_ += prediction[j, m] * f_rev[i - j, m];
      }
      E_deaths[i, m] = sum_;
    }
  }
}

model {
  tau ~ exponential(0.03);
  phi ~ normal(0, 5);
  
  // GP priors for weekly effect
  alpha_gp_week ~ normal(0, 2);
  rho_gp_week ~ normal(0, 30);
  to_vector(gp_raw_week) ~ normal(0, 1);
  
  for (m in 1:M) {
    y[m] ~ exponential(1 / tau);
  }
  
  for (m in 1:M) {
    deaths[EpidemicStart[m]:N[m], m] ~ neg_binomial_2(E_deaths[EpidemicStart[m]:N[m], m], phi);
  }
}
