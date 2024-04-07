import pandas as pd
import numpy as np
from pathlib import Path
from sspipe import p, px

# Set up paths
main_path = Path('../question_1/1_sample_selection.py').resolve().parents[3] / 'Data' / 'elsa_10' / 'tab'
derived_path = Path('../question_1/1_sample_selection.py').resolve().parents[3] / 'Data' / 'derived_10'

# Read data
main_10_var = ['idauniq', 'W10scout',
               'CaHHa', 'askinst',  # sample selection
               'Hehelf',  # self-reported health
               'WpDes', 'DhWork',  # employment status
               'PScedA', 'PScedB', 'PScedC', 'PScedD', 'PScedE', 'PScedF', 'PScedG', 'PScedH',  # ces-d
               'HOBB', 'SCINT'] + \
              ['SCIND'] + [f'SCIND0{number}' for number in range(1, 6)] + \
              ['SCINA'] + [f'SCINA0{number}' for number in range(1, 10)] + [f'SCINA{number}' for number in
                                                                            range(10, 22)] + \
              ['SCINNO'] + [f'SCINNO0{number}' for number in range(1, 10)] + \
              ['HEHaveBP', 'HEHaveHC', 'HEEverDI',  # (blood pressure, cholesterol still has, diabetes ever diagnosed)
               'HEHaveAS', 'HEHaveAR', 'HEHaveCA'] + \
              ['CaFam'] + [f'CaFam{number}' for number in range(2, 26)] + \
              ['EXRela', 'exrelefo', 'exreleme', 'exreleou', 'exrelede', 'exreleel', 'exrelefa', 'exrelepr', 'exreleho',
               'exreletr', 'EXRele96'] + \
              ['indager', 'indsex', 'fqethnmr', 'dimarr', 'FqMqua', 'fqqumnv5', 'fqqumnv4', 'fqqumnv3', 'fqqumnv2',
               'fqqumnv1', 'fqendm'] + \
              ['IaSInc', 'IaAIm', 'IaAIp', 'IaPPmo', 'IaPPei', 'IasPa', 'IaPAM', 'IaSPp', 'IaPPAm'] + \
              ['IaP'] + [f'IaP{number}' for number in range(2, 43)] + \
              ['IaA'] + [f'IaA{number}' for number in range(2, 43)] + \
              ['IaSint', 'IaNSi', 'IaNPBP', 'IaIsaD', 'IaSSSi', 'Iauiti', 'Iabgi', 'Iaira', 'IafBA', 'IaSiOi'] + \
              ['CfMetM'] + \
              ['CfLitB', 'CfLitC', 'CfLitD', 'CfLitE'] + \
              ['CfSvA', 'CfSvB', 'CfSvC', 'CfSvD', 'CfSvE']

main_9_var = ['idauniq', 'w9scout',
              'cahha', 'askinst',  # sample selection
              'hehelf',  # self-reported health
              'wpdes', 'dhwork',  # employment status
              'psceda', 'pscedb', 'pscedc', 'pscedd', 'pscede', 'pscedf', 'pscedg', 'pscedh',  # ces-d
              'hobb', 'scint'] + \
             ['scinddt', 'scindlt', 'scindtb', 'scindph', 'scind95', 'scind96'] + \
             ['scinaem', 'scinacl', 'scinaed', 'scinahe', 'scinabk', 'scinash', 'scinasl', 'scinasn', 'scinact', 'scinanw', 'scinast', 'scinagm', 'scinajb', 'scinaps', 'scina95', 'scina96'] + \
             ['hedasbp', 'hedasch', 'hedawdi',  # (blood pressure, cholesterol still has, diabetes ever diagnosed)
              'hedbsas', 'hedbsar', 'hedbsca'] + \
             ['cafam', 'cafam2', 'cafam3', 'cafam4', 'cafam5', 'cafam6', 'cafam7', 'cafam8', 'cafam9', 'cafam10', 'cafam11', 'cafam12', 'cafam13', 'cafam14', 'cafam15', 'cafam16', 'cafam17', 'cafam18', 'cafam19', 'cafam20', 'cafam21', 'cafam22', 'cafam23', 'cafam24', 'cafam25'] + \
             ['exrela', 'exrelefo', 'exreleme', 'exreleou', 'exrelede', 'exreleel', 'exrelefa', 'exrelepr', 'exreleho', 'exreletr', 'exrele96'] + \
             ['cfmetm'] + \
             ['cflitb', 'cflitc', 'cflitd', 'cflite'] + \
             ['cfsva', 'cfsvb', 'cfsvc', 'cfsvd', 'cfsve']

ifs_9_var = ['idauniq', 'age', 'sex', 'nonwhite', 'marstat', 'edend', 'edqual', 'ndepriv']

financial_9_var = ['idauniq', 'eqtotinc_bu_s', 'yq10_bu_s']  # BU equivalised total income, and its decile

main_10 = pd.read_table(main_path / 'wave_10_elsa_data_eul_v1.tab', usecols=main_10_var)
main_9 = pd.read_table(main_path / 'wave_9_elsa_data_eul_v1.tab', usecols=main_9_var)
ifs_9 = pd.read_table(main_path / 'wave_9_ifs_derived_variables.tab', usecols=ifs_9_var)
financial_9 = pd.read_table(main_path / 'wave_9_financial_derived_variables.tab', usecols=financial_9_var)

# Merge data
full_9 = main_9.merge(ifs_9, on='idauniq', how='left').merge(financial_9, on='idauniq', how='left')
sample = full_9.merge(main_10, on='idauniq', how='inner', suffixes=('_9', '_10'))  # only conflicting variables have suffixes

########## Sample selection
sample['w9scout'].value_counts(dropna=False)  # remove respondents who did not receive or were not eligible for self-completion questionnaire
sample['W10scout'].value_counts(dropna=False)
sample = sample.loc[(sample['w9scout'] == 1) & (sample['W10scout'] == 1), :]  # N = 4482

sample['askinst_9'].value_counts(dropna=False)  # remove respondents in institution
sample['askinst_10'].value_counts(dropna=False)
sample = sample.loc[(sample['askinst_9'] == 0) & (sample['askinst_10'] == 0), :]  # N = 4477

sample['cahha'].value_counts(dropna=False)  # remove respondents living with formal care (e.g. home care worker)
sample['CaHHa'].value_counts(dropna=False)
sample = sample.loc[(sample['cahha'] == -1) & (sample['CaHHa'] == -1), :]  # N = 4419

sample['cafam'].value_counts(dropna=False)  # remove respondents living with informal care (e.g. son)
help_list_9 = ['cafam'] + [f'cafam{number}' for number in range(2, 26)]
help_list_10 = ['CaFam'] + [f'CaFam{number}' for number in range(2, 26)]
sample = sample.loc[~((sample[help_list_9] == 1).any(axis=1) | (sample[help_list_10] == 1).any(axis=1)), :]  # N = 3925

sample['hobb'].value_counts(dropna=False)  # remove respondents with no internet access at home
sample['HOBB'].value_counts(dropna=False)
sample = sample.loc[(sample['hobb'] == 1) & (sample['HOBB'] == 1), :]  # N = 3603

##### Save data
sample.to_csv(derived_path / 'wave_910_sample.csv', index=False)
