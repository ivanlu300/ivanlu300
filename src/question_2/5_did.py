import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path
from sspipe import p

import statsmodels.formula.api as smf
import linearmodels as plm

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

sample_did = sample_910[['idauniq', 'group',
                         'srh_9', 'srh_10',
                         'high_bp_9', 'high_bp_10', 'high_chol_9', 'high_chol_10', 'diabetes_9', 'diabetes_10',
                         'asthma_9', 'asthma_10', 'arthritis_9', 'arthritis_10', 'cancer_9', 'cancer_10',
                         'cesd_9', 'cesd_10', 'cesd_b_9', 'cesd_b_10',
                         'employ_status_9', 'employ_status_10', 'total_income_bu_d_9', 'total_income_bu_d_10',
                         'age_9', 'age_10', 'sex_9', 'ethnicity_9', 'marital_status_9', 'marital_status_10',
                         'memory_9', 'memory_10', 'numeracy_9', 'numeracy_10', 'edu_age_9', 'edu_qual_9',
                         'n_deprived_9', 'n_deprived_10']]

########## Transform the data into long format
sample_did_long = pd.wide_to_long(sample_did,
                                  i='idauniq',
                                  j='wave',
                                  stubnames=['srh', 'high_bp', 'high_chol', 'diabetes', 'asthma', 'arthritis', 'cancer',
                                             'cesd', 'cesd_b', 'employ_status', 'total_income_bu_d', 'age',
                                             'marital_status', 'memory', 'numeracy', 'n_deprived'],
                                  sep='_',
                                  suffix='\\d+'
                                  ).reset_index()

did_srh = smf.ols('srh ~ group * C(wave)',
                  data=sample_did_long).fit()
did_srh.summary()

did_srh = smf.ols(
    'srh ~ group * C(wave) + employ_status + total_income_bu_d + C(marital_status) + memory + numeracy + n_deprived + age + sex_9 + ethnicity_9 + edu_age_9 + C(edu_qual_9)',
    data=sample_did_long).fit()
did_srh.summary()

did_high_bp = smf.ols(
    'high_bp ~ group * C(wave) + employ_status + total_income_bu_d + C(marital_status) + memory + numeracy + n_deprived + age + sex_9 + ethnicity_9 + edu_age_9 + C(edu_qual_9)',
    data=sample_did_long).fit()
did_high_bp.summary()

did_high_chol = smf.ols(
    'high_chol ~ group * C(wave) + employ_status + total_income_bu_d + C(marital_status) + memory + numeracy + n_deprived + age + sex_9 + ethnicity_9 + edu_age_9 + C(edu_qual_9)',
    data=sample_did_long).fit()
did_high_chol.summary()  # positive

did_diabetes = smf.ols(
    'diabetes ~ group * C(wave) + employ_status + total_income_bu_d + C(marital_status) + memory + numeracy + n_deprived + age + sex_9 + ethnicity_9 + edu_age_9 + C(edu_qual_9)',
    data=sample_did_long).fit()
did_diabetes.summary()

did_asthma = smf.ols(
    'asthma ~ group * C(wave) + employ_status + total_income_bu_d + C(marital_status) + memory + numeracy + n_deprived + age + sex_9 + ethnicity_9 + edu_age_9 + C(edu_qual_9)',
    data=sample_did_long).fit()
did_asthma.summary()

did_arthritis = smf.ols(
    'arthritis ~ group * C(wave) + employ_status + total_income_bu_d + C(marital_status) + memory + numeracy + n_deprived + age + sex_9 + ethnicity_9 + edu_age_9 + C(edu_qual_9)',
    data=sample_did_long).fit()
did_arthritis.summary()  # positive

did_cancer = smf.ols(
    'cancer ~ group * C(wave) + employ_status + total_income_bu_d + C(marital_status) + memory + numeracy + n_deprived + age + sex_9 + ethnicity_9 + edu_age_9 + C(edu_qual_9)',
    data=sample_did_long).fit()
did_cancer.summary()  # only cancer is statistically significant

did_cesd = smf.ols(
    'cesd ~ group * C(wave) + employ_status + total_income_bu_d + C(marital_status) + memory + numeracy + n_deprived + age + sex_9 + ethnicity_9 + edu_age_9 + C(edu_qual_9)',
    data=sample_did_long).fit()
did_cesd.summary()

did_cesd_b = smf.ols(
    'cesd_b ~ group * C(wave) + employ_status + total_income_bu_d + C(marital_status) + memory + numeracy + n_deprived + age + sex_9 + ethnicity_9 + edu_age_9 + C(edu_qual_9)',
    data=sample_did_long).fit()
did_cesd_b.summary()

########## Using a for loop to save results in one table
models = [did_srh, did_high_bp, did_high_chol, did_diabetes, did_asthma, did_arthritis, did_cancer, did_cesd,
          did_cesd_b]
