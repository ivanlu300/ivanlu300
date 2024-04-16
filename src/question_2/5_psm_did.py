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

########## Treatment: digital literacy
# Drop NAs
sample_literacy = sample.dropna(subset=['PC1_b', 'total_income_bu_d_9', 'age_9', 'sex_9', 'ethnicity_9', 'edu_age_9', 'edu_qual_9', 'n_deprived_9', 'employ_status_9', 'marital_status_9', 'memory_9', 'numeracy_9'])
sample_literacy['PC1_b'].value_counts(dropna=False)  # 1: 2074, 0: 1316

# Calculate propensity score
logit_literacy = smf.logit('PC1_b ~ total_income_bu_d_9 + age_9 + sex_9 + ethnicity_9 + edu_age_9 + C(edu_qual_9) + n_deprived_9 + employ_status_9 + C(marital_status_9) + memory_9 + numeracy_9',
                           data=sample_literacy).fit()
logit_literacy.summary()

sample_literacy['ps_literacy'] = logit_literacy.predict(sample_literacy)

# plot common support
fig = px.histogram(sample_literacy, x='ps_literacy', color='PC1_b', 
                   barmode='overlay', 
                   labels={'ps_literacy': 'Propensity score', 'PC1_b': 'Treatment'})

fig.update_yaxes(title_text='Count')
fig.for_each_trace(lambda trace: trace.update(name = 'High digital literacy' if trace.name == '1.0' else 'Low digital literacy'))

fig.write_image(figure_path / 'ps_common_support_q2.png')

fig.show()

########## Get the difference in outcome between Wave 9 and Wave 10
# self-reported health
sample_literacy['srh_diff'] = sample_literacy['srh_10'] - sample_literacy['srh_9']
pd.crosstab(sample_literacy['srh_9'], sample_literacy['srh_10'])

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
sample_srh = sample_literacy.dropna(subset=['srh_diff', 'PC1_b', 'ps_literacy'])
cm_srh = CausalModel(Y=sample_srh['srh_diff'].values,
                     D=sample_srh['PC1_b'].values,
                     X=sample_srh[['ps_literacy']].values)
cm_srh.est_via_matching(matches=5)
print(cm_srh.estimates)

# cardiovascular diseases
sample_high_bp = sample_literacy.dropna(subset=['high_bp_diff', 'PC1_b', 'ps_literacy'])
cm_high_bp = CausalModel(Y=sample_high_bp['high_bp_diff'].values,
                         D=sample_high_bp['PC1_b'].values,
                         X=sample_high_bp[['ps_literacy']].values)
cm_high_bp.est_via_matching(matches=5)
print(cm_high_bp.estimates)

sample_high_chol = sample_literacy.dropna(subset=['high_chol_diff', 'PC1_b', 'ps_literacy'])
cm_high_chol = CausalModel(Y=sample_high_chol['high_chol_diff'].values,
                           D=sample_high_chol['PC1_b'].values,
                           X=sample_high_chol[['ps_literacy']].values)
cm_high_chol.est_via_matching(matches=5)
print(cm_high_chol.estimates)

sample_high_diabetes = sample_literacy.dropna(subset=['diabetes_diff', 'PC1_b', 'ps_literacy'])
cm_high_diabetes = CausalModel(Y=sample_high_diabetes['diabetes_diff'].values,
                               D=sample_high_diabetes['PC1_b'].values,
                               X=sample_high_diabetes[['ps_literacy']].values) 
cm_high_diabetes.est_via_matching(matches=5)
print(cm_high_diabetes.estimates)

# noncardiovascular diseases
sample_asthma = sample_literacy.dropna(subset=['asthma_diff', 'PC1_b', 'ps_literacy'])
cm_asthma = CausalModel(Y=sample_asthma['asthma_diff'].values,
                        D=sample_asthma['PC1_b'].values,
                        X=sample_asthma[['ps_literacy']].values)
cm_asthma.est_via_matching(matches=5)
print(cm_asthma.estimates)

sample_arthritis = sample_literacy.dropna(subset=['arthritis_diff', 'PC1_b', 'ps_literacy'])
cm_arthritis = CausalModel(Y=sample_arthritis['arthritis_diff'].values,
                           D=sample_arthritis['PC1_b'].values,
                           X=sample_arthritis[['ps_literacy']].values)
cm_arthritis.est_via_matching(matches=5)
print(cm_arthritis.estimates)

sample_cancer = sample_literacy.dropna(subset=['cancer_diff', 'PC1_b', 'ps_literacy'])
cm_cancer = CausalModel(Y=sample_cancer['cancer_diff'].values,
                        D=sample_cancer['PC1_b'].values,
                        X=sample_cancer[['ps_literacy']].values)
cm_cancer.est_via_matching(matches=5)
print(cm_cancer.estimates)
pd.crosstab(sample_cancer['PC1_b'], sample_cancer['cancer_diff'])  # probably because there are too few cases

# mental health
sample_cesd = sample_literacy.dropna(subset=['cesd_diff', 'PC1_b', 'ps_literacy'])
cm_cesd = CausalModel(Y=sample_cesd['cesd_diff'].values,
                      D=sample_cesd['PC1_b'].values,
                      X=sample_cesd[['ps_literacy']].values)
cm_cesd.est_via_matching(matches=5)
print(cm_cesd.estimates)

sample_cesd_b = sample_literacy.dropna(subset=['cesd_b_diff', 'PC1_b', 'ps_literacy'])
cm_cesd_b = CausalModel(Y=sample_cesd_b['cesd_b_diff'].values,
                        D=sample_cesd_b['PC1_b'].values,
                        X=sample_cesd_b[['ps_literacy']].values)
cm_cesd_b.est_via_matching(matches=5)
print(cm_cesd_b.estimates)

