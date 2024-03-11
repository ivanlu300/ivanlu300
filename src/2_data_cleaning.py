import pandas as pd
import numpy as np
from pathlib import Path
from sspipe import p, px

# Set up paths
main_path = Path('1_sample_selection.py').resolve().parents[2] / 'Data' / 'elsa_main' / 'tab'
covid_path = Path('1_sample_selection.py').resolve().parents[2] / 'Data' / 'elsa_covid' / 'tab'
derived_path = Path('1_sample_selection.py').resolve().parents[2] / 'Data' / 'derived'

# Read data
full_8 = pd.read_csv(derived_path / 'wave_8.csv')

########## Treatment
full_8['hobb'].value_counts(dropna=False)
full_8['internet_access'] = np.select(condlist=[full_8['hobb'] == 1, full_8['hobb'] == 2],
                                      choicelist=[1, 0],
                                      default=np.nan)

full_8['scint'].value_counts(dropna=False)
full_8['internet_freq'] = np.where(full_8['scint'] > 0, full_8['scint'], np.nan) # 1 = every day, 6 = never

device_list = ['scinddt', 'scindlt', 'scindtb', 'scindph', 'scind95', 'scind96']
device_list_no96 = ['scinddt', 'scindlt', 'scindtb', 'scindph', 'scind95']
pd.crosstab(full_8['internet_freq'], full_8['scinddt'])
# respondents with scint == 5|6 were not asked about devices (hence coded as -1), but I assume they have no device
full_8[device_list_no96] = full_8[device_list_no96].where(full_8['scint'] < 5, other=0)
pd.crosstab(full_8['internet_freq'], full_8['scind96'])
full_8['scind96'] = full_8['scind96'].where(full_8['scint'] < 5, other=1)
# then deal with NAs
full_8[device_list] = full_8[device_list].where(full_8[device_list] >= 0, other=np.nan) # 1 = yes, 0 = no

activity_list = ['scinaem', 'scinacl', 'scinaed', 'scinabk', 'scinash', 'scinasl', 'scinasn', 'scinact', 'scinanw', 'scinast', 'scinagm', 'scinajb', 'scinaps', 'scina95', 'scina96']
activity_list_no96 = ['scinaem', 'scinacl', 'scinaed', 'scinabk', 'scinash', 'scinasl', 'scinasn', 'scinact', 'scinanw', 'scinast', 'scinagm', 'scinajb', 'scinaps', 'scina95']
pd.crosstab(full_8['internet_freq'], full_8['scinaem'])
# respondents with scint == 5|6 were not asked about activities (hence coded as -1), but I assume they have no activity
full_8[activity_list_no96] = full_8[activity_list_no96].where(full_8['scint'] < 5, other=0)
pd.crosstab(full_8['internet_freq'], full_8['scina96'])
full_8['scina96'] = full_8['scina96'].where(full_8['scint'] < 5, other=1)
# then deal with NAs
full_8[activity_list] = full_8[activity_list].where(full_8[activity_list] >= 0, other=np.nan) # 1 = yes, 0 = no

# Outcome - self-reported health
full_8['hehelf'].value_counts(dropna=False) # 1 = excellent, 5 = poor

# Outcome - cardiovascular diseases
pd.crosstab(full_8['hedasbp'], full_8['hedimbp'])
full_8['high_bp'] = np.select(condlist=[full_8['hedasbp'] == 1, full_8['hedimbp'] == 1, (full_8['hedasbp'].isin([-1, 2])) & (full_8['hedimbp'] == 0)],
                              choicelist=[1, 1, 0],
                              default=np.nan)
full_8['high_bp'].value_counts(dropna=False)

pd.crosstab(full_8['hedasch'], full_8['hedimch'])
full_8['high_chol'] = np.select(condlist=[full_8['hedasch'] == 1, full_8['hedimch'] == 1, (full_8['hedasch'].isin([-1, 2])) & (full_8['hedimch'] == 0)],
                                choicelist=[1, 1, 0],
                                default=np.nan)
full_8['high_chol'].value_counts(dropna=False)

pd.crosstab(full_8['hedacdi'], full_8['hedimdi'])
full_8['diabetes'] = np.select(condlist=[full_8['hedacdi'] == 1, full_8['hedimdi'] == 1, (full_8['hedacdi'].isin([-1, 2])) & (full_8['hedimdi'] == 0)],
                               choicelist=[1, 1, 0],
                               default=np.nan)
full_8['diabetes'].value_counts(dropna=False)

# Outcome - non-cardiovascular diseases
pd.crosstab(full_8['hedbsas'], full_8['hedibas'])
full_8['asthma'] = np.select(condlist=[full_8['hedbsas'] == 1, full_8['hedibas'] == 1, (full_8['hedbsas'].isin([-1, 2])) & (full_8['hedibas'] == 0)],
                             choicelist=[1, 1, 0],
                             default=np.nan)
full_8['asthma'].value_counts(dropna=False)

pd.crosstab(full_8['hedbsar'], full_8['hedibar'])
full_8['arthritis'] = np.select(condlist=[full_8['hedbsar'] == 1, full_8['hedibar'] == 1, (full_8['hedbsar'].isin([-1, 2])) & (full_8['hedibar'] == 0)],
                                choicelist=[1, 1, 0],
                                default=np.nan)
full_8['arthritis'].value_counts(dropna=False)

pd.crosstab(full_8['hedbsca'], full_8['hedibca'])
full_8['cancer'] = np.select(condlist=[full_8['hedbsca'] == 1, full_8['hedibca'] == 1, (full_8['hedbsca'].isin([-1, 2])) & (full_8['hedibca'] == 0)],
                             choicelist=[1, 1, 0],
                             default=np.nan)
full_8['cancer'].value_counts(dropna=False)

# Outcome - mental health
# full_8['psceda'].value_counts(dropna=False)
# cesd_list = [f'psced{chr(letter)}' for letter in range(ord('a'), ord('i'))]
#
# # reverse the score of positive items
# full_8['pscedd'] = full_8['pscedd'].replace({1: 2, 2: 1}) # happy
# full_8['pscedf'] = full_8['pscedf'].replace({1: 2, 2: 1}) # enjoy life
#
# full_8['cesd'] = np.select(condlist=[(full_8[cesd_list] == 1).sum(axis=1) >= 3, (full_8[cesd_list] < 0).any(axis=1)],
#                            choicelist=[1, np.nan],
#                            default=0)
# full_8['cesd'].value_counts(dropna=False) # 658
# # Confirmed: the 'cesd_sc' in the ifs derived dataset has already adjusted for the reverse score of positive items
full_8['cesd_sc'].value_counts(dropna=False)
full_8['cesd'] = np.where(full_8['cesd_sc'] >= 3, 1, 0)
full_8['cesd'].value_counts(dropna=False)

########## Controls
# social class
full_8['w8nssec8'].value_counts(dropna=False)
full_8['social_class'] = np.where(full_8['w8nssec8'].isin([-3, 99]), np.nan, full_8['w8nssec8'])
full_8['social_class'].value_counts(dropna=False) # 1 = high, 8 = low

# income
full_8['yq10_bu_s'].value_counts(dropna=False)
full_8['income_d'] = full_8['yq10_bu_s'].apply(lambda x: int(x) if x.isdigit() else np.nan)
full_8['income_d'].value_counts(dropna=False)

full_8['totwq10_bu_s'].value_counts(dropna=False)
full_8['wealth_d'] = full_8['totwq10_bu_s'].apply(lambda x: int(x) if x.isdigit() else np.nan)
full_8['wealth_d'].value_counts(dropna=False)

# age
full_8['age'].value_counts(dropna=False)

# sex
full_8['sex'].value_counts(dropna=False) # 1 = male, 2 = female

# ethnicity
full_8['nonwhite'].value_counts(dropna=False) # 1 = non-white, 0 = white
full_8['nonwhite'].mean() # 98% white

# marital status
full_8['marstat'].value_counts(dropna=False) # 1-6 categories
full_8['marstat'] = full_8['marstat'].astype(str)

# educational attainment
full_8['qual3'].value_counts(dropna=False) # NAs present
full_8['education'] = np.where(full_8['qual3'] >= 0, full_8['qual3'], np.nan) # 0 = low, 2 = high
full_8['education'].value_counts(dropna=False)

# deprivation
full_8['ndepriv'].value_counts(dropna=False) # 1 = least deprived, 5 = most deprived
full_8['deprive'] = np.select(condlist=[full_8['ndepriv'] == -1, full_8['ndepriv'] < 0],
                              choicelist=[0, np.nan],
                              default=full_8['ndepriv']) # 0 = least deprived, 9 = most deprived
full_8['deprive'].value_counts(dropna=False)

# long-standing illness
full_8['llsill'].value_counts(dropna=False)
full_8['limit_ill'] = np.select(condlist=[full_8['llsill'] == 2, full_8['llsill'].isin([0, 1])],
                                choicelist=[1, 0],
                                default=np.nan)
full_8['limit_ill'].value_counts(dropna=False)

########## Save data
full_8.to_csv(derived_path / 'wave_8_cleaned.csv', index=False)

########## Inspection
full_8['scinddt'].value_counts(dropna=False) # NAs mainly due to scint == -3
