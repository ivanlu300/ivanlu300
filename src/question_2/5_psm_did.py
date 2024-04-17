import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path
from sspipe import p

import statsmodels.formula.api as smf
from causalinference import CausalModel

# Set up paths
main_path = Path().resolve().parents[2] / 'Data' / 'elsa_10' / 'tab'
derived_path = Path().resolve().parents[2] / 'Data' / 'derived_10'
figure_path = Path().resolve().parents[1] / 'output' / 'figure'
table_path = Path().resolve().parents[1] / 'output' / 'table'

# Read data
sample = pd.read_csv(derived_path / 'wave_910_pca.csv')

########## Further sample selection: only respondents who did not change their digital literacy between Wave 9 and 10
pd.crosstab(sample['PC1_b_9'], sample['PC1_b_10'])
sample_910 = sample.loc[sample['PC1_b_9'] == sample['PC1_b_10'], :]  # N = 2881
sample_910['group'] = sample_910['PC1_b_9']
sample_910['group'].value_counts(dropna=False)  # 1: 1449, 0: 1432

########## Treatment: digital literacy
# Drop NAs
sample_literacy = sample_910.dropna(subset=['group', 'total_income_bu_d_9', 'age_9', 'sex_9', 'ethnicity_9', 'edu_age_9', 'edu_qual_9', 'n_deprived_9', 'employ_status_9', 'marital_status_9', 'memory_9', 'numeracy_9'])
sample_literacy['group'].value_counts(dropna=False)  # 1: 1380, 0: 1366

# Calculate propensity score
logit_literacy = smf.logit('group ~ total_income_bu_d_9 + age_9 + sex_9 + ethnicity_9 + edu_age_9 + C(edu_qual_9) + n_deprived_9 + employ_status_9 + C(marital_status_9) + memory_9 + numeracy_9',
                           data=sample_literacy).fit()
logit_literacy.summary()

sample_literacy['ps_literacy'] = logit_literacy.predict(sample_literacy)

# plot common support
fig = px.histogram(sample_literacy, x='ps_literacy', color='group',
                   barmode='overlay', 
                   labels={'ps_literacy': 'Propensity score', 'group': 'Treatment'})

fig.update_yaxes(title_text='Count')
fig.for_each_trace(lambda trace: trace.update(name = 'High digital literacy' if trace.name == '1.0' else 'Low digital literacy'))

fig.write_image(figure_path / 'ps_common_support_q2.png')

fig.show()

########## Get the difference in outcome between Wave 9 and Wave 10
# self-reported health
sample_literacy['srh_diff'] = sample_literacy['srh_10'] - sample_literacy['srh_9']
sample_literacy['srh_diff'].value_counts(dropna=False)

# cardiovascular diseases
sample_literacy['high_bp_diff'] = sample_literacy['high_bp_10'] - sample_literacy['high_bp_9']
sample_literacy['high_chol_diff'] = sample_literacy['high_chol_10'] - sample_literacy['high_chol_9']
sample_literacy['diabetes_diff'] = sample_literacy['diabetes_10'] - sample_literacy['diabetes_9']

# non-cardiovascular diseases
sample_literacy['asthma_diff'] = sample_literacy['asthma_10'] - sample_literacy['asthma_9']
sample_literacy['arthritis_diff'] = sample_literacy['arthritis_10'] - sample_literacy['arthritis_9']
sample_literacy['cancer_diff'] = sample_literacy['cancer_10'] - sample_literacy['cancer_9']

# mental health
sample_literacy['cesd_diff'] = sample_literacy['cesd_10'] - sample_literacy['cesd_9']
sample_literacy['cesd_b_diff'] = sample_literacy['cesd_b_10'] - sample_literacy['cesd_b_9']

########## Matching estimation
# self-reported health
sample_srh = sample_literacy.dropna(subset=['srh_diff', 'group', 'ps_literacy'])
cm_srh = CausalModel(Y=sample_srh['srh_diff'].values,
                     D=sample_srh['group'].values,
                     X=sample_srh[['ps_literacy']].values)
cm_srh.est_via_matching(matches=5)
print(cm_srh.estimates)

# cardiovascular diseases
sample_high_bp = sample_literacy.dropna(subset=['high_bp_diff', 'group', 'ps_literacy'])
cm_high_bp = CausalModel(Y=sample_high_bp['high_bp_diff'].values,
                         D=sample_high_bp['group'].values,
                         X=sample_high_bp[['ps_literacy']].values)
cm_high_bp.est_via_matching(matches=5)
print(cm_high_bp.estimates)

sample_high_chol = sample_literacy.dropna(subset=['high_chol_diff', 'group', 'ps_literacy'])
cm_high_chol = CausalModel(Y=sample_high_chol['high_chol_diff'].values,
                           D=sample_high_chol['group'].values,
                           X=sample_high_chol[['ps_literacy']].values)
cm_high_chol.est_via_matching(matches=5)
print(cm_high_chol.estimates)

sample_high_diabetes = sample_literacy.dropna(subset=['diabetes_diff', 'group', 'ps_literacy'])
cm_high_diabetes = CausalModel(Y=sample_high_diabetes['diabetes_diff'].values,
                               D=sample_high_diabetes['group'].values,
                               X=sample_high_diabetes[['ps_literacy']].values) 
cm_high_diabetes.est_via_matching(matches=5)
print(cm_high_diabetes.estimates)

# noncardiovascular diseases
sample_asthma = sample_literacy.dropna(subset=['asthma_diff', 'group', 'ps_literacy'])
cm_asthma = CausalModel(Y=sample_asthma['asthma_diff'].values,
                        D=sample_asthma['group'].values,
                        X=sample_asthma[['ps_literacy']].values)
cm_asthma.est_via_matching(matches=5)
print(cm_asthma.estimates)

sample_arthritis = sample_literacy.dropna(subset=['arthritis_diff', 'group', 'ps_literacy'])
cm_arthritis = CausalModel(Y=sample_arthritis['arthritis_diff'].values,
                           D=sample_arthritis['group'].values,
                           X=sample_arthritis[['ps_literacy']].values)
cm_arthritis.est_via_matching(matches=5)
print(cm_arthritis.estimates)

sample_cancer = sample_literacy.dropna(subset=['cancer_diff', 'group', 'ps_literacy'])
cm_cancer = CausalModel(Y=sample_cancer['cancer_diff'].values,
                        D=sample_cancer['group'].values,
                        X=sample_cancer[['ps_literacy']].values)
cm_cancer.est_via_matching(matches=5)
print(cm_cancer.estimates)
pd.crosstab(sample_cancer['group'], sample_cancer['cancer_diff'])  # probably because there are too few cases

# mental health
sample_cesd = sample_literacy.dropna(subset=['cesd_diff', 'group', 'ps_literacy'])
cm_cesd = CausalModel(Y=sample_cesd['cesd_diff'].values,
                      D=sample_cesd['group'].values,
                      X=sample_cesd[['ps_literacy']].values)
cm_cesd.est_via_matching(matches=5)
print(cm_cesd.estimates)

sample_cesd_b = sample_literacy.dropna(subset=['cesd_b_diff', 'group', 'ps_literacy'])
cm_cesd_b = CausalModel(Y=sample_cesd_b['cesd_b_diff'].values,
                        D=sample_cesd_b['group'].values,
                        X=sample_cesd_b[['ps_literacy']].values)
cm_cesd_b.est_via_matching(matches=5)
print(cm_cesd_b.estimates)
pd.crosstab(sample_cesd_b['group'], sample_cesd_b['cesd_b_diff'])

