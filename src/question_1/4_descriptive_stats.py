import pandas as pd
import numpy as np
from pathlib import Path
from sspipe import p, px

# Set up paths
main_path = Path('1_sample_selection.py').resolve().parents[3] / 'Data' / 'elsa_10' / 'tab'
derived_path = Path('1_sample_selection.py').resolve().parents[3] / 'Data' / 'derived_10'
figure_path = Path('1_sample_selection.py').resolve().parents[2] / 'output' / 'figure'
table_path = Path('1_sample_selection.py').resolve().parents[2] / 'output' / 'table'

# Read data
main_10 = pd.read_csv(derived_path / 'wave_10_pca.csv')

# Treatment
main_10['PC1_b'].value_counts(dropna=False)  # 1: 2088, 0: 1454, NaN: 54

# Descriptive variables
desc_vars = ['srh', 'high_bp', 'high_chol', 'diabetes', 'asthma', 'arthritis', 'cancer', 'cesd', 'cesd_b',
             'total_income_bu_d', 'age', 'sex', 'ethnicity', 'edu_age', 'edu_qual', 'n_deprived']

# descriptive statistics by treatment status
desc_df = main_10.groupby('PC1_b')[desc_vars].mean().round(3).T

# rename columns
desc_df.rename(columns={1: 'High', 0: 'Low'}, inplace=True)

# rename rows
desc_df.rename(index={'srh': 'Self-reported health',
                      'high_bp': 'High blood pressure',
                      'high_chol': 'High cholesterol',
                      'diabetes': 'Diabetes',
                      'asthma': 'Asthma',
                      'arthritis': 'Arthritis',
                      'cancer': 'Cancer',
                      'cesd': 'CES-D items',
                      'cesd_b': 'CES-D diagnosis',
                      'total_income_bu_d': 'Decile of total income',
                      'age': 'Age',
                      'sex': 'Sex',
                      'ethnicity': 'Ethnicity',
                      'edu_age': 'Age left full-time education',
                      'edu_qual': 'Highest educational qualification',
                      'n_deprived': 'Deprivation index'},
               inplace=True)

# Add description
desc_df['Description'] = [''] * len(desc_df)

# Rearrange columns
desc_df = desc_df[['Description', 'Low', 'High']]

# LaTeX
desc_df.to_latex(index=True, float_format="%.3f") | p(print)

########## Inspection
main_10['total_income_bu_d'].value_counts(dropna=False)
main_10['n_deprived'].value_counts(dropna=False)
main_10['edu_age'].value_counts(dropna=False)
pd.crosstab(main_10['fqendm'], main_10['edu_age'], dropna=False)
pd.crosstab(main_10['edend'], main_10['edu_age'], dropna=False)
pd.crosstab(main_10['edend'], main_10['fqendm'], dropna=False)
