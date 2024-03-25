import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import plotnine as pn
from pathlib import Path
from sspipe import p, px

# Set up paths
main_path = Path('1_sample_selection.py').resolve().parents[3] / 'Data' / 'elsa_10' / 'tab'
derived_path = Path('1_sample_selection.py').resolve().parents[3] / 'Data' / 'derived_10'
figure_path = Path('1_sample_selection.py').resolve().parents[2] / 'output' / 'figure'
table_path = Path('1_sample_selection.py').resolve().parents[2] / 'output' / 'table'

# Read data
main_10 = pd.read_csv(derived_path / 'wave_10_pca.csv')

# Remove reverse causality
main_10['SCINNO05'].value_counts(dropna=False)  # NAs due to refused
main_10['SCINNO06'].value_counts(dropna=False)

sample = main_10.loc[(main_10['SCINNO05'] != 1) & (main_10['SCINNO06'] != 1), :]

########## Treatment: digital literacy
# Drop NAs
sample_literacy = sample.dropna(subset=['PC1_b', 'total_income_bu_d', 'age', 'sex', 'ethnicity', 'edu_age', 'edu_qual', 'n_deprived', 'employ_status', 'marital_status', 'memory', 'numeracy', 'comprehension'])
sample_literacy['PC1_b'].value_counts(dropna=False)  # 1: 2113, 0: 1447

# logit model
logit_literacy = smf.logit('PC1_b ~ total_income_bu_d + age + sex + ethnicity + edu_age + C(edu_qual) + n_deprived + employ_status + C(marital_status) + memory + numeracy + comprehension',
                           data=sample_literacy).fit()
logit_literacy.summary()

sample_literacy['ps_literacy'] = logit_literacy.predict(sample_literacy)

# IPTW
sample_literacy['iptw_literacy'] = np.select(condlist=[sample_literacy['PC1_b'] == 1, sample_literacy['PC1_b'] == 0],
                                             choicelist=[1 / sample_literacy['ps_literacy'],
                                                         1 / (1 - sample_literacy['ps_literacy'])],
                                             default=np.nan)

# stabilised IPTW
sample_literacy['iptw_literacy_s'] = np.select(condlist=[sample_literacy['PC1_b'] == 1, sample_literacy['PC1_b'] == 0],
                                               choicelist=[np.nanmean(sample_literacy['PC1_b']) / sample_literacy[
                                                   'ps_literacy'],
                                                           (1 - np.nanmean(sample_literacy['PC1_b'])) / (
                                                                   1 - sample_literacy['ps_literacy'])],
                                               default=np.nan)

########## Write a for loop to save data
outcome_list = ['srh', 'high_bp', 'high_chol', 'diabetes', 'asthma', 'arthritis', 'cancer', 'cesd', 'cesd_b']
table_ate = pd.DataFrame({'Outcome': ['Self-reported health', '', 'High blood pressure', '', 'High cholesterol', '',
                                      'Diabetes', '', 'Asthma', '', 'Arthritis', '', 'Cancer', '', 'CES-D items', '',
                                      'CES-D diagnosis', ''],
                          'Digital literacy': np.nan,
                          'p_literacy': np.nan})

for i in range(len(outcome_list)):
    outcome = outcome_list[i]

    model_literacy = smf.wls(f'{outcome} ~ PC1_b',
                             data=sample_literacy,
                             weights=sample_literacy['iptw_literacy_s']).fit()
    table_ate.loc[i * 2, 'Digital literacy'] = model_literacy.params['PC1_b']
    table_ate.loc[i * 2 + 1, 'Digital literacy'] = model_literacy.tvalues['PC1_b']
    table_ate.loc[i * 2, 'p_literacy'] = model_literacy.pvalues['PC1_b']

# looks like the digital literacy one is much better
literacy_ate = table_ate.loc[:, ['Outcome', 'Digital literacy', 'p_literacy']]
literacy_ate['Digital literacy'] = literacy_ate['Digital literacy'] | p(round, 3)

# Add * to indicate statistical significance
for i in np.arange(0, literacy_ate.shape[0], 2):
    if abs(literacy_ate.loc[i, 'p_literacy']) <= 0.001:
        literacy_ate.loc[i, 'Digital literacy'] = str(literacy_ate.loc[i, 'Digital literacy']) + '***'
    elif (abs(literacy_ate.loc[i, 'p_literacy']) <= 0.01) & (abs(literacy_ate.loc[i, 'p_literacy']) > 0.001):
        literacy_ate.loc[i, 'Digital literacy'] = str(literacy_ate.loc[i, 'Digital literacy']) + '**'
    elif (abs(literacy_ate.loc[i, 'p_literacy']) <= 0.05) & (abs(literacy_ate.loc[i, 'p_literacy']) > 0.01):
        literacy_ate.loc[i, 'Digital literacy'] = str(literacy_ate.loc[i, 'Digital literacy']) + '*'

# Add square brackets to t-values
literacy_ate['Digital literacy'] = literacy_ate['Digital literacy'].astype(str)
literacy_ate.loc[literacy_ate.index % 2 == 1, 'Digital literacy'] = '[' + literacy_ate['Digital literacy'] + ']'

# Add a description column
literacy_ate['Description'] = [''] * literacy_ate.shape[0]

literacy_ate[['Outcome', 'Description', 'Digital literacy']].to_latex(index=False, float_format="%.3f") | p(print)

########## Inspection
sample_literacy['cesd'].value_counts(dropna=False)
sample_literacy['cesd_b'].value_counts(dropna=False)
