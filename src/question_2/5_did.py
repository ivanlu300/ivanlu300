import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path
from sspipe import p

import statsmodels.formula.api as smf
import linearmodels as plm
from psmpy import PsmPy
from scipy.stats import ttest_1samp
from sklearn.neighbors import NearestNeighbors

# Set up paths
main_path = Path().resolve().parents[2] / 'Data' / 'elsa_10' / 'tab'
derived_path = Path().resolve().parents[2] / 'Data' / 'derived_10'
figure_path = Path().resolve().parents[1] / 'output' / 'figure'
table_path = Path().resolve().parents[1] / 'output' / 'table'

# Read data
sample = pd.read_csv(derived_path / 'wave_910_pca.csv')

# ########## Further sample selection: only respondents who did not change their digital literacy between Wave 9 and 10
# pd.crosstab(sample['PC1_b_9'], sample['PC1_b_10'])
# sample_910 = sample.loc[sample['PC1_b_9'] == sample['PC1_b_10'], :]  # N = 2881
# sample_910['group'] = sample_910['PC1_b_9']
# sample_910['group'].value_counts(dropna=False)  # 1: 565, 0: 658

sample_did = sample[['idauniq', 'PC1_b_9',
                     'srh_9', 'srh_10',
                     'high_bp_9', 'high_bp_10', 'high_chol_9', 'high_chol_10', 'diabetes_9', 'diabetes_10',
                     'asthma_9', 'asthma_10', 'arthritis_9', 'arthritis_10', 'cancer_9', 'cancer_10',
                     'cesd_9', 'cesd_10', 'anxiety_9', 'anxiety_10', 'mood_9', 'mood_10',
                     'employ_status_9', 'total_income_bu_d_9',
                     'age_9', 'sex_9', 'ethnicity_9', 'marital_status_9',
                     'memory_9', 'numeracy_9', 'edu_age_9', 'edu_qual_9', 'n_deprived_9']].dropna()

sample_did['PC1_b_9'].value_counts(dropna=False)  # 1: 555, 0: 281

####################
# Logit model
logit_ps = smf.logit(
    'PC1_b_9 ~ total_income_bu_d_9 + employ_status_9 + age_9 + sex_9 + ethnicity_9 + C(marital_status_9) + memory_9 + numeracy_9 + edu_age_9 + C(edu_qual_9) + n_deprived_9',
    data=sample_did).fit()

sample_did['ps'] = logit_ps.predict(sample_did)

# Perform matching
# split the low and high PC1_b_9
sample_did_low = sample_did.loc[sample_did['PC1_b_9'] == 0, :]
sample_did_high = sample_did.loc[sample_did['PC1_b_9'] == 1, :]

# Nearest Neighbour Matching
nn = NearestNeighbors(n_neighbors=1)
nn.fit(sample_did_high[['ps']])

# Find nearest neighbors for each control observation
distances, indices = nn.kneighbors(sample_did_low[['ps']])

# Extract matched ids
matched_ids = sample_did_high.iloc[indices.flatten()]['idauniq'].values

# Create DataFrame of matches
matches = pd.DataFrame({
    'low_ID': sample_did_low['idauniq'].values,
    'high_ID': matched_ids
})

# Loop to get results
did_result = pd.DataFrame({'Outcome': ['Self-rated health', '', 'High blood pressure', '', 'High cholesterol', '',
                                       'Diabetes', '', 'Asthma', '', 'Arthritis', '', 'Cancer', '', 'Depression score',
                                       '',
                                       'Anxiety disorder', '', 'Mood swings', ''],
                           'DiD': np.nan,
                           'p_value': np.nan})

outcome_list = ['srh', 'high_bp', 'high_chol', 'diabetes', 'asthma', 'arthritis', 'cancer', 'cesd', 'anxiety', 'mood']

# Write a loop to estimate DiD for each outcome
for j in range(len(outcome_list)):
    did = []
    for i in range(len(matches)):
        did.append((sample_did.loc[sample_did['idauniq'] == matches['high_ID'][i], f'{outcome_list[j]}_10'].values[0] -
                    sample_did.loc[sample_did['idauniq'] == matches['high_ID'][i], f'{outcome_list[j]}_9'].values[0])
                   -
                   (sample_did.loc[sample_did['idauniq'] == matches['low_ID'][i], f'{outcome_list[j]}_10'].values[0] -
                    sample_did.loc[sample_did['idauniq'] == matches['low_ID'][i], f'{outcome_list[j]}_9'].values[0]))

        # mean
        did_result.iloc[j * 2, 1] = np.mean(did)

        # t-test
        t_stat, p_value = ttest_1samp(did, 0)
        did_result.iloc[j * 2, 2] = p_value
        did_result.iloc[j * 2 + 1, 1] = t_stat

##### Format
# Round to 3 decimal places
did_result['DiD'] = did_result['DiD'] | p(round, 3)

# Add * to indicate statistical significance
for i in np.arange(0, did_result.shape[0], 2):
    if abs(did_result.iloc[i, 2]) <= 0.001:
        did_result.iloc[i, 1] = str(did_result.iloc[i, 1]) + '***'
    elif (abs(did_result.iloc[i, 2]) <= 0.01) & (abs(did_result.iloc[i, 2]) > 0.001):
        did_result.iloc[i, 1] = str(did_result.iloc[i, 1]) + '**'
    elif (abs(did_result.iloc[i, 2]) <= 0.05) & (abs(did_result.iloc[i, 2]) > 0.01):
        did_result.iloc[i, 1] = str(did_result.iloc[i, 1]) + '*'

# Add square brackets to t-values
did_result['DiD'] = did_result['DiD'].astype(str)
did_result.loc[did_result.index % 2 == 1, 'DiD'] = '[' + did_result['DiD'] + ']'

# Add a description column
did_result['Description'] = [''] * did_result.shape[0]

########## LaTeX
did_result[['Outcome', 'Description', 'DiD']].to_latex(index=False, float_format="%.3f") | p(print)

########## APPENDIX ##########
# 5 Nearest Neighbours Matching
logit_ps = smf.logit(
    'PC1_b_9 ~ total_income_bu_d_9 + employ_status_9 + age_9 + sex_9 + ethnicity_9 + C(marital_status_9) + memory_9 + numeracy_9 + edu_age_9 + C(edu_qual_9) + n_deprived_9',
    data=sample_did).fit()

sample_did['ps'] = logit_ps.predict(sample_did)

# Perform matching
# split the low and high PC1_b_9
sample_did_low = sample_did.loc[sample_did['PC1_b_9'] == 0, :]
sample_did_high = sample_did.loc[sample_did['PC1_b_9'] == 1, :]

# Nearest Neighbour Matching
nn = NearestNeighbors(n_neighbors=5)
nn.fit(sample_did_high[['ps']])

# Find nearest neighbors for each control observation
distances, indices = nn.kneighbors(sample_did_low[['ps']])

# Create DataFrame of matches
matches = pd.DataFrame({
    'low_ID': np.repeat(sample_did_low['idauniq'].values, 5),
    'high_ID': sample_did_high.iloc[indices.flatten(), 0].values
})

# Loop to get results
did_result = pd.DataFrame({'Outcome': ['Self-rated health', '', 'High blood pressure', '', 'High cholesterol', '',
                                       'Diabetes', '', 'Asthma', '', 'Arthritis', '', 'Cancer', '',
                                       'Depressive symptoms', '',
                                       'Anxiety disorder', '', 'Mood swings', ''],
                           'DiD': np.nan,
                           'p_value': np.nan})

outcome_list = ['srh', 'high_bp', 'high_chol', 'diabetes', 'asthma', 'arthritis', 'cancer', 'cesd', 'anxiety', 'mood']

def average_in_chunks(lst, chunk_size=5):
    # Using list comprehension to create chunks and calculate their averages
    return [sum(lst[i:i + chunk_size]) / chunk_size for i in range(0, len(lst), chunk_size)]

# Write a loop to estimate DiD for each outcome
for j in range(len(outcome_list)):
    did = []
    for i in range(len(matches)):
        did.append((sample_did.loc[sample_did['idauniq'] == matches['high_ID'][i], f'{outcome_list[j]}_10'].values[0] -
                    sample_did.loc[sample_did['idauniq'] == matches['high_ID'][i], f'{outcome_list[j]}_9'].values[0])
                   -
                   (sample_did.loc[sample_did['idauniq'] == matches['low_ID'][i], f'{outcome_list[j]}_10'].values[0] -
                    sample_did.loc[sample_did['idauniq'] == matches['low_ID'][i], f'{outcome_list[j]}_9'].values[0]))

    did = average_in_chunks(did)
        
    # mean
    did_result.iloc[j * 2, 1] = np.mean(did)

    # t-test
    t_stat, p_value = ttest_1samp(did, 0)
    did_result.iloc[j * 2, 2] = p_value
    did_result.iloc[j * 2 + 1, 1] = t_stat

##### Format
# Round to 3 decimal places
did_result['DiD'] = did_result['DiD'] | p(round, 3)

# Add * to indicate statistical significance
for i in np.arange(0, did_result.shape[0], 2):
    if abs(did_result.iloc[i, 2]) <= 0.001:
        did_result.iloc[i, 1] = str(did_result.iloc[i, 1]) + '***'
    elif (abs(did_result.iloc[i, 2]) <= 0.01) & (abs(did_result.iloc[i, 2]) > 0.001):
        did_result.iloc[i, 1] = str(did_result.iloc[i, 1]) + '**'
    elif (abs(did_result.iloc[i, 2]) <= 0.05) & (abs(did_result.iloc[i, 2]) > 0.01):
        did_result.iloc[i, 1] = str(did_result.iloc[i, 1]) + '*'

# Add square brackets to t-values
did_result['DiD'] = did_result['DiD'].astype(str)
did_result.loc[did_result.index % 2 == 1, 'DiD'] = '[' + did_result['DiD'] + ']'

# Add a description column
did_result['Description'] = [''] * did_result.shape[0]

########## LaTeX
did_result[['Outcome', 'Description', 'DiD']].to_latex(index=False, float_format="%.3f") | p(print)


########## APPENDIX ##########
# 3 Nearest Neighbours Matching
logit_ps = smf.logit(
    'PC1_b_9 ~ total_income_bu_d_9 + employ_status_9 + age_9 + sex_9 + ethnicity_9 + C(marital_status_9) + memory_9 + numeracy_9 + edu_age_9 + C(edu_qual_9) + n_deprived_9',
    data=sample_did).fit()

sample_did['ps'] = logit_ps.predict(sample_did)

# Perform matching
# split the low and high PC1_b_9
sample_did_low = sample_did.loc[sample_did['PC1_b_9'] == 0, :]
sample_did_high = sample_did.loc[sample_did['PC1_b_9'] == 1, :]

# Nearest Neighbour Matching
nn = NearestNeighbors(n_neighbors=3)
nn.fit(sample_did_high[['ps']])

# Find nearest neighbors for each control observation
distances, indices = nn.kneighbors(sample_did_low[['ps']])

# Create DataFrame of matches
matches = pd.DataFrame({
    'low_ID': np.repeat(sample_did_low['idauniq'].values, 3),
    'high_ID': sample_did_high.iloc[indices.flatten(), 0].values
})

# Loop to get results
did_result = pd.DataFrame({'Outcome': ['Self-rated health', '', 'High blood pressure', '', 'High cholesterol', '',
                                       'Diabetes', '', 'Asthma', '', 'Arthritis', '', 'Cancer', '',
                                       'Depressive symptoms', '',
                                       'Anxiety disorder', '', 'Mood swings', ''],
                           'DiD': np.nan,
                           'p_value': np.nan})

outcome_list = ['srh', 'high_bp', 'high_chol', 'diabetes', 'asthma', 'arthritis', 'cancer', 'cesd', 'anxiety', 'mood']

def average_in_chunks(lst, chunk_size=3):
    # Using list comprehension to create chunks and calculate their averages
    return [sum(lst[i:i + chunk_size]) / chunk_size for i in range(0, len(lst), chunk_size)]

# Write a loop to estimate DiD for each outcome
for j in range(len(outcome_list)):
    did = []
    for i in range(len(matches)):
        did.append((sample_did.loc[sample_did['idauniq'] == matches['high_ID'][i], f'{outcome_list[j]}_10'].values[0] -
                    sample_did.loc[sample_did['idauniq'] == matches['high_ID'][i], f'{outcome_list[j]}_9'].values[0])
                   -
                   (sample_did.loc[sample_did['idauniq'] == matches['low_ID'][i], f'{outcome_list[j]}_10'].values[0] -
                    sample_did.loc[sample_did['idauniq'] == matches['low_ID'][i], f'{outcome_list[j]}_9'].values[0]))

    did = average_in_chunks(did)
        
    # mean
    did_result.iloc[j * 2, 1] = np.mean(did)

    # t-test
    t_stat, p_value = ttest_1samp(did, 0)
    did_result.iloc[j * 2, 2] = p_value
    did_result.iloc[j * 2 + 1, 1] = t_stat

##### Format
# Round to 3 decimal places
did_result['DiD'] = did_result['DiD'] | p(round, 3)

# Add * to indicate statistical significance
for i in np.arange(0, did_result.shape[0], 2):
    if abs(did_result.iloc[i, 2]) <= 0.001:
        did_result.iloc[i, 1] = str(did_result.iloc[i, 1]) + '***'
    elif (abs(did_result.iloc[i, 2]) <= 0.01) & (abs(did_result.iloc[i, 2]) > 0.001):
        did_result.iloc[i, 1] = str(did_result.iloc[i, 1]) + '**'
    elif (abs(did_result.iloc[i, 2]) <= 0.05) & (abs(did_result.iloc[i, 2]) > 0.01):
        did_result.iloc[i, 1] = str(did_result.iloc[i, 1]) + '*'

# Add square brackets to t-values
did_result['DiD'] = did_result['DiD'].astype(str)
did_result.loc[did_result.index % 2 == 1, 'DiD'] = '[' + did_result['DiD'] + ']'

# Add a description column
did_result['Description'] = [''] * did_result.shape[0]

########## LaTeX
did_result[['Outcome', 'Description', 'DiD']].to_latex(index=False, float_format="%.3f") | p(print)





########## APPENDIX ##########
# Probit model
probit_ps = smf.probit(
    'PC1_b_9 ~ total_income_bu_d_9 + employ_status_9 + age_9 + sex_9 + ethnicity_9 + C(marital_status_9) + memory_9 + numeracy_9 + edu_age_9 + C(edu_qual_9) + n_deprived_9',
    data=sample_did).fit()

sample_did['ps'] = probit_ps.predict(sample_did)

# Perform matching
# split the low and high PC1_b_9
sample_did_low = sample_did.loc[sample_did['PC1_b_9'] == 0, :]
sample_did_high = sample_did.loc[sample_did['PC1_b_9'] == 1, :]

# Nearest Neighbour Matching
nn = NearestNeighbors(n_neighbors=1)
nn.fit(sample_did_high[['ps']])

# Find nearest neighbors for each control observation
distances, indices = nn.kneighbors(sample_did_low[['ps']])

# Extract matched ids
matched_ids = sample_did_high.iloc[indices.flatten()]['idauniq'].values

# Create DataFrame of matches
matches = pd.DataFrame({
    'low_ID': sample_did_low['idauniq'].values,
    'high_ID': matched_ids
})

# Loop to get results
did_result = pd.DataFrame({'Outcome': ['Self-rated health', '', 'High blood pressure', '', 'High cholesterol', '',
                                       'Diabetes', '', 'Asthma', '', 'Arthritis', '', 'Cancer', '', 'Depression score',
                                       '',
                                       'Anxiety disorder', '', 'Mood swings', ''],
                           'DiD': np.nan,
                           'p_value': np.nan})

outcome_list = ['srh', 'high_bp', 'high_chol', 'diabetes', 'asthma', 'arthritis', 'cancer', 'cesd', 'anxiety', 'mood']

# Write a loop to estimate DiD for each outcome
for j in range(len(outcome_list)):
    did = []
    for i in range(len(matches)):
        did.append((sample_did.loc[sample_did['idauniq'] == matches['high_ID'][i], f'{outcome_list[j]}_10'].values[0] -
                    sample_did.loc[sample_did['idauniq'] == matches['high_ID'][i], f'{outcome_list[j]}_9'].values[0])
                   -
                   (sample_did.loc[sample_did['idauniq'] == matches['low_ID'][i], f'{outcome_list[j]}_10'].values[0] -
                    sample_did.loc[sample_did['idauniq'] == matches['low_ID'][i], f'{outcome_list[j]}_9'].values[0]))

        # mean
        did_result.iloc[j * 2, 1] = np.mean(did)

        # t-test
        t_stat, p_value = ttest_1samp(did, 0)
        did_result.iloc[j * 2, 2] = p_value
        did_result.iloc[j * 2 + 1, 1] = t_stat

##### Format
# Round to 3 decimal places
did_result['DiD'] = did_result['DiD'] | p(round, 3)

# Add * to indicate statistical significance
for i in np.arange(0, did_result.shape[0], 2):
    if abs(did_result.iloc[i, 2]) <= 0.001:
        did_result.iloc[i, 1] = str(did_result.iloc[i, 1]) + '***'
    elif (abs(did_result.iloc[i, 2]) <= 0.01) & (abs(did_result.iloc[i, 2]) > 0.001):
        did_result.iloc[i, 1] = str(did_result.iloc[i, 1]) + '**'
    elif (abs(did_result.iloc[i, 2]) <= 0.05) & (abs(did_result.iloc[i, 2]) > 0.01):
        did_result.iloc[i, 1] = str(did_result.iloc[i, 1]) + '*'

# Add square brackets to t-values
did_result['DiD'] = did_result['DiD'].astype(str)
did_result.loc[did_result.index % 2 == 1, 'DiD'] = '[' + did_result['DiD'] + ']'

# Add a description column
did_result['Description'] = [''] * did_result.shape[0]
