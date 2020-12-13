import pandas as pd
import numpy as np
import time

from scipy import stats as sps
from statistics import mean

import argparse

parser = argparse.ArgumentParser(description='Calcuate Rt for Covid-19 - iNDIA')
parser.add_argument('-s', '--state', help='State name')
args = parser.parse_args()
states = args.state

start = time.time()

url = 'https://api.covid19india.org/csv/latest/state_wise_daily.csv'
GAMMA = 1/7
R_T_MAX =12

# Calling the covid19india.org's API and parsing CSV file to a dataframe
df = pd.read_csv(url, index_col=["Date"], parse_dates=["Date"])
# dropping "Deceased" and "Recovered" rows as that data is not required to calculate Rt
df = df[df["Status"]=="Confirmed"]
# Dropping status column as it is no longer required.
df = df.drop(["Status", "Date_YMD"], axis=1)
#Rename "TT" to "IN"
df = df.rename(columns={"TT": "IN"})

def prepare_cases(cases, cutoff=10):
    smoothed = cases.rolling(7,win_type='gaussian', min_periods=1, center=True).mean(std=2).round()
    idx_start = np.searchsorted(smoothed, cutoff)
    smoothed = smoothed.iloc[idx_start:]
    original = cases.loc[smoothed.index]
    return original, smoothed

r_t_range = np.linspace(0, R_T_MAX, R_T_MAX*100+1)
def get_posteriors(sr, sigma=0.15):
    # Calculate Lambda
    lam = sr[:-1].values * np.exp(GAMMA * (r_t_range[:, None] - 1))
    # Calculate each day's likelihood
    likelihoods = pd.DataFrame(data = sps.poisson.pmf(sr[1:].values, lam), index = r_t_range, columns = sr.index[1:])
    # Create the Gaussian Matrix
    process_matrix = sps.norm(loc=r_t_range, scale=sigma ).pdf(r_t_range[:, None]) 
    # Normalize all rows to sum to 1
    process_matrix /= process_matrix.sum(axis=0)
    # Calculate the initial prior
    prior0 = sps.gamma(a=4).pdf(r_t_range)
    prior0 /= prior0.sum()
    # Create a DataFrame that will hold outher posteriors for each day
    # Insert the prior as the first posterior.
    posteriors = pd.DataFrame(index=r_t_range, columns=sr.index, data={sr.index[0]: prior0})
    # log of probability of the data for maximum likelihood calculation
    log_likelihood = 0.0
    # Iteratively apply Bayes' rule
    for previous_day, current_day in zip(sr.index[:-1], sr.index[1:]):
        # Calculate the new prior
        current_prior = process_matrix @ posteriors[previous_day]
        # Calculate the numerator of Bayes' Rule: P(k|R_t)P(R_t)
        numerator = likelihoods[current_day] * current_prior
        # Calcluate the denominator of Bayes' Rule P(k)
        denominator = np.sum(numerator)
        # Execute full Bayes' Rule
        posteriors[current_day] = numerator/denominator
        # Add to the running sum of log likelihoods
        log_likelihood += np.log(denominator)
    return posteriors, log_likelihood

def highest_density_interval(pmf, p=.9):
    try:
        # If a DataFrame is passed, just call this recursively on the columns
        if(isinstance(pmf, pd.DataFrame)):
            return pd.DataFrame([highest_density_interval(pmf[col], p=p) for col in pmf], index=pmf.columns)
        cumsum = np.cumsum(pmf.values)
        # N x N matrix of total probability mass for each low, high
        total_p = cumsum - cumsum[:, None]
        # Return all indices with total_p > p
        lows, highs = (total_p > p).nonzero()
        # Find the smallest range (highest density)
        best = (highs - lows).argmin()
        low = pmf.index[lows[best]]
        high = pmf.index[highs[best]]
        return pd.Series([low, high], index=[f'Low_{p*100:.0f}', f'High_{p*100:.0f}'])
    except:
        return pd.Series([0.0,0.0], index=['Low_90', 'High_90'])


sigmas = np.linspace(1/20, 1, 20)
if not states :
    states = df.columns.values.tolist()
else:
    states = [states]

results = {}

print("\nSmoothing the cases...")
for state_name in states:
    cases = df[state_name]
    print(state_name)
    if sum(cases) == 0:
        print("0 cases reported for " + state_name + " :)")
        continue
    elif sum(cases)<100:
        print("Cases < 100 for " + state_name + " - too low!!!")
        continue
    #Taking mean of first 25 non-zeroes values from daily cases
    cases_cutoff = cases.to_numpy()[cases.to_numpy()!=0][:25].mean()
    if cases_cutoff >= 1:
        print("Cutoff (Mean of first 25 non-zero case): " + str(cases_cutoff))
        new, smoothed = prepare_cases(cases, cutoff=cases_cutoff)
    else:
        new, smoothed = prepare_cases(cases, cutoff=1)
    #print(smoothed)
    result = {}
    # Holds all posteriors with every given value of sigma
    result['posteriors'] = []
    # Holds the log likelihood across all k for each value of sigma
    result['log_likelihoods'] = []
    for sigma in sigmas:
        try:
            posteriors, log_likelihood = get_posteriors(smoothed, sigma=sigma)
        except:
            print("Too low cases for " + state_name + "-> sigma: " + str(round(sigma, 2)))
        result['posteriors'].append(posteriors)
        result['log_likelihoods'].append(log_likelihood)
    # Store all results keyed off of state name
    results[state_name] = result
print("Done!!!")

# Each index of this array holds the total of the log likelihoods for the corresponding index of the sigmas array.
total_log_likelihoods = np.zeros_like(sigmas)

# Loop through each state's results and add the log likelihoods to the running total.
for state_name, result in results.items():
    total_log_likelihoods += result['log_likelihoods']

# Select the index with the largest log likelihood total
max_likelihood_index = total_log_likelihoods.argmax()

final_results = None

print("\nComputing final results...")
for state_name, result in results.items():
    print(state_name)
    posteriors = result['posteriors'][max_likelihood_index]
    hdis_90 = highest_density_interval(posteriors, p=.9)
    most_likely = posteriors.idxmax().rename('ML')
    result = pd.concat([most_likely, hdis_90], axis=1)
    result["State"] = state_name
    if final_results is None:
        final_results = result
    else:
        final_results = pd.concat([final_results, result])

# Since we now use a uniform prior, the first datapoint is pretty bogus, so just truncating it here
final_results = final_results.groupby('State').apply(lambda x: x.iloc[1:])
# Groupby leaves a duplicate "State" column but as it is already in index so, removing it here
final_results = final_results.drop(columns="State")

# Creating a dataframe to merge no. of cases per day in original dataframe
temp = pd.DataFrame()
for state_name in states:
    df_temp = pd.DataFrame(df[state_name])
    df_temp["State"] = state_name
    temp = temp.append(df_temp.rename(columns={state_name:"Cases"}))
temp = temp.reset_index().set_index(["State", "Date"])

# Merging
final_results = final_results.merge(temp, on= ["State", "Date"])

#Renaming a few columns
final_results = final_results.rename(columns={"ML": "Rt", "Low_90": "Low", "High_90": "High"})

print("Done!!!")
print("\nIt took " + str(round((time.time()-start)/60, 2)) + " min(s).!")

final_results.to_csv("rt.csv")
print("\nRt data exported to CSV -> rt.csv")