import pandas as pd
import numpy as np
from pathlib import Path
from sspipe import p, px

# Set up paths
main_path = Path().resolve().parents[2] / 'Data' / 'elsa_10' / 'tab'
derived_path = Path().resolve().parents[2] / 'Data' / 'derived_10'
figure_path = Path().resolve().parents[1] / 'output' / 'figure'
table_path = Path().resolve().parents[1] / 'output' / 'table'

# Read data
main_10 = pd.read_csv(derived_path / 'wave_10_pca.csv')

# Treatment
main_10['PC1_b'].value_counts(dropna=False)  # 1: 994, 0: 994, NaN: 2029

# Descriptive statistics
desc_vars = ['srh', 'high_bp', 'high_chol', 'diabetes', 'asthma', 'arthritis', 'cancer', 'cesd', 'anxiety', 'mood',
             'age', 'sex', 'ethnicity', 'edu_age', 'employ_status', 'total_income_bu_d', 'n_deprived', 'memory', 'numeracy', 'comprehension']

# descriptive statistics by treatment status for non-categorical variables
desc_df = main_10.groupby('PC1_b')[desc_vars].mean().round(3).T

# descriptive statistics by treatment status for categorical variables
marital_status_df = main_10.groupby('PC1_b')['marital_status'].value_counts(normalize=True).unstack().T
edu_qual_df = main_10.groupby('PC1_b')['edu_qual'].value_counts(normalize=True).unstack().T

# concatenate all dataframes
desc_df = pd.concat([desc_df, marital_status_df, edu_qual_df])

# rename columns
desc_df.rename(columns={1: 'High', 0: 'Low'}, inplace=True)

# rename rows
desc_df.index = desc_df.index.astype(str)
desc_df.rename(index={'srh': 'Self-rated health',
                      'high_bp': 'High blood pressure',
                      'high_chol': 'High cholesterol',
                      'diabetes': 'Diabetes',
                      'asthma': 'Asthma',
                      'arthritis': 'Arthritis',
                      'cancer': 'Cancer',
                      'cesd': 'Depression score',
                      'anxiety': 'Anxiety disorder',
                      'mood': 'Mood swings',
                      'age': 'Age',
                      'sex': 'Gender',
                      'ethnicity': 'Ethnicity',
                      '1': 'Single',
                      '2': 'Married',
                      '3': 'Remarried',
                      '4': 'Separated',
                      '5': 'Divorced',
                      '6': 'Widowed',
                      'edu_age': 'Age left full-time education',
                      '1.0': 'Degree or equivalent',
                      '2.0': 'HE below degree',
                      '3.0': 'A-level or equivalent',
                      '4.0': 'O-level or equivalent',
                      '5.0': 'CSE or equivalent',
                      '6.0': 'Foreign/other qualification',
                      '7.0': 'No qualification',
                      'employ_status': 'Employment status',
                      'total_income_bu_d': 'Household income (decile)',
                      'n_deprived': 'Deprivation index',
                      'memory': 'Memory',
                      'numeracy': 'Numeracy index',
                      'comprehension': 'Comprehension index'},
               inplace=True)

# Add description
desc_df['Description'] = [''] * len(desc_df)

# Rearrange columns
desc_df = desc_df[['Description', 'Low', 'High']]

# LaTeX
desc_df.to_latex(index=True, float_format="%.3f") | p(print)

########## Inspection
