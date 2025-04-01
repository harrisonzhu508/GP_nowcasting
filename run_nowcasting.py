import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pystan
import datetime
import arviz
import seaborn as sns
import os
sns.set()
from nowcasting_functions import *
cwd = os.getcwd()
states = ['AC', 'SP', 'PA', 'DF', 'RO', 'RR', 'CE', 'PE', 'RJ', 'SC', 'AM', 'Brazil']

# Load the data
data_new = pd.read_csv('data/df_SIVEP_nowcast_allStates_08-02-2021.csv')
print("First date:", data_new.Date.values[0])
print("Last date:", data_new.Date.values[-1])

model = pystan.StanModel(file=cwd+'/stan_models/4comp_longshort_SE.stan')
print('Model compiled')

model_nobbs = pystan.StanModel(file=cwd+'/stan_models/nobbs.stan')
print('Model compiled')

def get_state_data(df, state):
    df_state = df.copy()
    if state == 'Brazil':
        # need to merge!!!!!!
        df_state.drop(columns=['State'], inplace=True)
        columns = list(df_state.columns)
        columns.remove('Deaths')
        df_state = df_state.groupby(columns, as_index=False)['Deaths'].sum()
        return df_state
    return df_state[df_state['State'] == state]

def do_nowcast_daily_for_state(df, state, model):
    data = get_state_data(df, state)
    fit, results, delays_data, triangle = fit_model_daily(data, precompiled=False, modelname=model, date_nowcast=None,
              maxD=10, iters=1000, warmup=400, chains=4, adapt_delta=0.9,
              max_treedepth=12, seed=9876,
              pickle_run=False, save=False, savepath='')
    return fit, results, delays_data, triangle


fits_nobbs = {}
results_nobbs = {}
delays_data_nobbs = {}
triangle_nobbs = {}

for s in list(states):
    print('Nowcasting for ' + s)
    fit, result, delays_data, triangle = do_nowcast_daily_for_state(data_new, s, model_nobbs)
    fits_nobbs.update({s: fit})
    results_nobbs.update({s: result})
    delays_data_nobbs.update({s: delays_data})
    triangle_nobbs.update({s: triangle})
    
print('All nowcasting completed')

state = list(states)[1]
data_new[data_new['State'] == state]["Date"].nunique()
sum_n_predict = fits_nobbs[state].extract()['sum_n_predict']
print(sum_n_predict.shape)

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 1. Filter AC state data
df_ac = data_new[data_new['State'] == state].copy()

# 2. Get the model input dates from the triangle (this ensures alignment with T)
# Sort triangle by date (just to be sure)
triangle_sorted = triangle.sort_index()
dates_model = pd.to_datetime(triangle_sorted.index)

# Sort predictions to match
sorted_indices = triangle.index.argsort()
sum_n_predict_sorted = sum_n_predict[:, sorted_indices]

# Get credible intervals
mean_pred = sum_n_predict_sorted.mean(axis=0)
q025 = np.percentile(sum_n_predict_sorted, 2.5, axis=0)
q975 = np.percentile(sum_n_predict_sorted, 97.5, axis=0)


# 4. Plot
plt.figure(figsize=(12, 6))
plt.scatter(pd.to_datetime(df_ac['Date']), df_ac['Deaths'], label="Reported Deaths", alpha=0.6)
plt.plot(dates_model, mean_pred, color='orange', label="Nowcast (mean prediction)", linewidth=2)
# plot weekly reported deaths
plt.fill_between(dates_model, q025, q975, color='orange', alpha=0.3, label="95% CrI")

plt.title(f"Nowcasting Daily Deaths in {state}")
plt.xlabel("Date")
plt.ylabel("Deaths")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
plt.savefig(f"nowcast_daily_{state}.png")

# save the dates_model with mean_pred to CSV for each state 
# df_ac['Date'] = pd.to_datetime(df_ac['Date'])
# df_ac['mean_pred'] = mean_pred

for state in states:
    print(f"Processing: {state}")
    
    # 1. Get model data
    triangle = triangle_nobbs[state]  # This is the daily triangle used as input
    triangle_sorted = triangle.sort_index()
    dates_model = pd.to_datetime(triangle_sorted.index)

    # 2. Extract and sort predictions
    sum_n_predict = fits_nobbs[state].extract()['sum_n_predict']  # shape: (n_draws, T)
    sorted_indices = triangle.index.argsort()
    sum_n_predict_sorted = sum_n_predict[:, sorted_indices]
    mean_pred = sum_n_predict_sorted.mean(axis=0)

    # 3. Combine into a DataFrame
    df_out = pd.DataFrame({
        'Date': dates_model,
        'mean_pred': mean_pred
    })

    # 4. Export to CSV
    df_out.to_csv(f'nowcast_daily_{state}.csv', index=False)
