import pandas as pd
import numpy as np
from pathlib import Path
from sspipe import p, px

# Set up paths
main_path = Path('1_sample_selection.py').resolve().parents[2] / 'Data' / 'elsa_10' / 'tab'
derived_path = Path('1_sample_selection.py').resolve().parents[2] / 'Data' / 'derived_10'

# Read data
main_var = ['idauniq', 'W10scout',
            'CaHHa', 'WpDes', 'askinst', # sample selection
            'Hehelf', # self-reported health
            'PScedA', 'PScedB', 'PScedC', 'PScedD', 'PScedE', 'PScedF', 'PScedG', 'PScedH', # ces-d
            'HOBB', 'SCINT'] + \
           ['SCIND'] + [f'SCIND0{number}' for number in range(1, 6)] + \
           ['SCINA'] + [f'SCINA0{number}' for number in range(1, 10)] + [f'SCINA{number}' for number in range(10, 22)] + \
           ['SCINNO'] + [f'SCINNO0{number}' for number in range(1, 10)] + \
           ['HEHaveBP', 'HEHaveHC', 'HEEverDI', # (blood pressure, cholesterol still has, diabetes ever diagnosed)
            'HEHaveAS', 'HEHaveAR', 'HEHaveCA'] + \
           ['CaFam'] + [f'CaFam{number}' for number in range(2, 26)] + \
           ['EXRela', 'exrelefo', 'exreleme', 'exreleou', 'exrelede', 'exreleel', 'exrelefa', 'exrelepr', 'exreleho', 'exreletr', 'EXRele96'] + \
           ['indager', 'indsex', 'fqethnmr', 'dimarr', 'FqMqua', 'fqqumnv5', 'fqqumnv4', 'fqqumnv3', 'fqqumnv2', 'fqqumnv1', 'fqendm'] + \
           ['IaAIm', 'IaAIp', 'IaPPei', 'IasPa', 'IaPAM', 'IaSPp', 'IaPPAm'] + \
           ['IaP'] + [f'IaP{number}' for number in range(2, 43)] + \
           ['IaA'] + [f'IaA{number}' for number in range(2, 43)] + \
           ['IaSint', 'IaNSi', 'IaNPBP', 'IaIsaD', 'IaSSSi', 'Iauiti', 'Iabgi', 'Iaira', 'IafBA', 'IaSiOi']
main_10 = pd.read_table(main_path / 'wave_10_elsa_data_eul_v1.tab',
                        usecols=main_var)

ifs_var = ['idauniq', 'nonwhite', 'edend', 'edqual']
ifs_9 = pd.read_table(main_path / 'wave_9_ifs_derived_variables.tab',
                      usecols=ifs_var)

# Merge data
main_10 = main_10.merge(ifs_9, on='idauniq', how='left')

########## Sample selection
main_10['W10scout'].value_counts(dropna=False)  # remove respondents (-1193)
main_10 = main_10.loc[main_10['W10scout'] == 1, :]

main_10['askinst'].value_counts(dropna=False)  # remove respondents in institution (-6)
main_10 = main_10.loc[main_10['askinst'] == 0, :]

main_10['CaHHa'].value_counts(dropna=False)  # remove respondents living with formal care (e.g. home care worker) (-74)
main_10 = main_10.loc[main_10['CaHHa'] == -1, :]

main_10['CaFam'].value_counts(dropna=False)  # remove respondents living with informal care (e.g. son) (-539)
help_list = ['CaFam'] + [f'CaFam{number}' for number in range(2, 26)]
main_10 = main_10.loc[~((main_10[help_list] == 1).any(axis=1)), :]

main_10['WpDes'].value_counts(dropna=False)  # select respondents who are retired or semi-retired (-2178)
main_10 = main_10.loc[main_10['WpDes'].isin([1, 96]), :]  # N = 3596

########## Save data
main_10.to_csv(derived_path / 'wave_10_sample.csv', index=False)

########## Inspect variables
main_10['HOBB'].value_counts()  # no internet access at home
main_10['SCINT'].value_counts()  # frequency of using internet
