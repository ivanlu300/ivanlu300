import pandas as pd
import numpy as np
from pathlib import Path
from sspipe import p, px

# Set up paths
main_path = Path('1_sample_selection.py').resolve().parents[2] / 'Data' / 'elsa_main' / 'tab'
covid_path = Path('1_sample_selection.py').resolve().parents[2] / 'Data' / 'elsa_covid' / 'tab'
derived_path = Path('1_sample_selection.py').resolve().parents[2] / 'Data' / 'derived'

# Read data
main_var = ['idauniq',
            'scprt', 'chinhh', 'cafam', 'w8nssec5', # controls
            'cahha', # sample selection
            'hehelf', # self-reported health
            'psceda', 'pscedb', 'pscedc', 'pscedd', 'pscede', 'pscedf', 'pscedg', 'pscedh', # ces-d
            'hobb', 'scint', 'scind96',
            'scinaem', 'scinacl', 'scinaed', 'scinabk', 'scinash', 'scinasl', 'scinasn', 'scinact', 'scinanw', 'scinast', 'scinagm', 'scinajb', 'scinaps', 'scina95', 'scina96',
            'hedasbp', 'hedimbp', 'hedasch', 'hedimch', 'hedacdi', 'hedimdi', # three cardiovascular diseases
            'hedbsas', 'hedibas', 'hedbsar', 'hedibar', 'hedbsca', 'hedibca'] + \
    [f'cafam{number}' for number in range(2, 26)] # whether person who helps lives in same household as respondent
main_8 = pd.read_table(main_path / 'wave_8_elsa_data_eul_v2.tab',
                       usecols=main_var)

finance_var = ['idauniq', 'yq10_bu_s', 'totwq10_bu_s']
finance_8 = pd.read_table(main_path / 'wave_8_elsa_financial_dvs_eul_v1.tab',
                          usecols=finance_var)

ifs_var = ['idauniq', 'wgt',
           'age', 'sex', 'nonwhite', 'marstat', 'qual2', 'qual3', 'ndepriv', 'llsill', # controls
           'wselfd', 'inst', # sample selection
           'srh_hrs', 'cesd_sc']
ifs_8 = pd.read_table(main_path / 'wave_8_elsa_ifs_dvs_eul_v1.tab',
                      usecols=ifs_var)

########## Merge data
full_8 = main_8.merge(finance_8, on='idauniq', how='left') \
                .merge(ifs_8, on='idauniq', how='left')

########## Sample selection
full_8['inst'].value_counts(dropna=False) # remove respondents in institution (-58)
full_8 = full_8.loc[full_8['inst'] == 0, :]

full_8['cahha'].value_counts(dropna=False) # remove respondents living with formal care (e.g. home care worker) (-160)
full_8 = full_8.loc[full_8['cahha'] == -1, :]

full_8['cafam'].value_counts(dropna=False)
help_list = ['cafam'] + [f'cafam{number}' for number in range(2, 26)]
full_8 = full_8.loc[~(full_8[help_list] == 1).any(axis=1), :] # remove respondents living with informal care (e.g. son) (-815)

full_8['wselfd'].value_counts(dropna=False) # select respondents who are retired
full_8 = full_8.loc[full_8['wselfd'] == 3, :]

########## Save data
full_8.to_csv(derived_path / 'wave_8.csv', index=False)

########## Inspection
full_8['hobb'].value_counts()

full_8['scprt'].value_counts() # 2803 lives with partner
full_8['chinhh'].value_counts() # 549 do not live with children
