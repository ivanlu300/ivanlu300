import pandas as pd
import numpy as np
from pathlib import Path
from sspipe import p

# Set up paths
main_path = Path().resolve().parents[2] / 'Data' / 'elsa_10' / 'tab'
derived_path = Path().resolve().parents[2] / 'Data' / 'derived_10'

# Read data
sample = pd.read_csv(derived_path / 'wave_910_sample.csv')

########## Treatment (used to distinguish two groups)
sample['scint'].value_counts(dropna=False)
sample['int_freq_9'] = np.where(sample['scint'] > 0, sample['scint'], np.nan)  # 1 = every day, 6 = never
sample['int_freq_9'].value_counts(dropna=False)

sample['SCINT'].value_counts(dropna=False)
sample['int_freq_10'] = np.where(sample['SCINT'] > 0, sample['SCINT'], np.nan)  # 1 = every day, 6 = never
sample['int_freq_10'].value_counts(dropna=False)

########## Wave 9
### devices
device_list_9 = ['scinddt', 'scindlt', 'scindtb', 'scindph', 'scind95']
pd.crosstab(sample['scint'], sample['scinddt'], dropna=False)
# respondents with SCINT == 6 were not asked about devices (hence coded as -1), but I assume they have no device
sample[device_list_9] = sample[device_list_9].where(sample['scint'] != 6, other=0)  # 1 = yes, 0 = no

# respondents with SCINT == 6 were not asked about scind96 (hence coded as -1), but I assume they do not access the internet
pd.crosstab(sample['scint'], sample['scind96'], dropna=False)
sample['scind96'] = sample['scind96'].where(sample['scint'] != 6, other=1)

# then deal with NAs
sample[device_list_9 + ['scind96']] = sample[device_list_9 + ['scind96']].replace(-9, np.nan)

### internet activities
activity_list_9 = ['scinaem', 'scinacl', 'scinaed', 'scinahe', 'scinabk', 'scinash', 'scinasl', 'scinasn', 'scinact',
                   'scinanw', 'scinast', 'scinagm', 'scinajb', 'scinaps', 'scina95']
pd.crosstab(sample['scint'], sample['scinaem'], dropna=False)
# respondents with SCINT == 6 were not asked about activities (hence coded as -1), but I assume they have no activity
sample[activity_list_9] = sample[activity_list_9].where(sample['scint'] != 6, other=0)  # 1 = yes, 0 = no

# respondents with SCINT == 6 were not asked about scina96 (hence coded as -1), but I assume they have no activity
pd.crosstab(sample['scint'], sample['scina96'], dropna=False)
sample['scina96'] = sample['scina96'].where(sample['scint'] != 6, other=1)

# then deal with NAs
sample[activity_list_9 + ['scina96']] = sample[activity_list_9 + ['scina96']].replace(-9, np.nan)

########## Wave 10
### devices
device_list_10 = [f'SCIND0{number}' for number in range(1, 6)]
pd.crosstab(sample['SCINT'], sample['SCIND01'])
# respondents with SCINT == 6 were not asked about devices (hence coded as -1), but I assume they have no device
sample[device_list_10] = sample[device_list_10].where(sample['SCINT'] < 6, other=0)
# then deal with NAs
sample[device_list_10] = sample[device_list_10].where(sample[device_list_10] >= 0, other=np.nan)  # 1 = yes, 0 = no
pd.crosstab(sample['int_freq_10'], sample['SCIND01'])

activity_list_10 = [f'SCINA0{number}' for number in range(1, 10)] + [f'SCINA{number}' for number in range(10, 22)]
activity_list_no96_10 = [f'SCINA0{number}' for number in range(1, 10)] + [f'SCINA{number}' for number in range(10, 21)]
pd.crosstab(sample['SCINT'], sample['SCINA01'])
# respondents with SCINT == 6 were not asked about activities (hence coded as -1), but I assume they have no activity
sample[activity_list_no96_10] = sample[activity_list_no96_10].where(sample['SCINT'] < 6, other=0)
# 21 is none
sample['SCINA21'] = sample['SCINA21'].where(sample['SCINT'] < 6, other=1)
# then deal with NAs
sample[activity_list_10] = sample[activity_list_10].where(sample[activity_list_10] >= 0,
                                                          other=np.nan)  # 1 = yes, 0 = no
pd.crosstab(sample['int_freq_10'], sample['SCINA21'])
pd.crosstab(sample['int_freq_10'], sample['SCINA02'])

########## Outcome (used for DiD)
# self-reported health
sample['hehelf'].value_counts(dropna=False)  # 1 = excellent, 5 = poor
sample['srh_9'] = np.where(sample['hehelf'] > 0, sample['hehelf'], np.nan)
sample['srh_9'].value_counts(dropna=False)

sample['Hehelf'].value_counts(dropna=False)  # 1 = excellent, 5 = poor
sample['srh_10'] = np.where(sample['Hehelf'] > 0, sample['Hehelf'], np.nan)
sample['srh_10'].value_counts(dropna=False)

# cardiovascular disease
sample['hedawbp'].value_counts(dropna=False)  # 1 = yes, -1 = no
sample['hedimbp'].value_counts(dropna=False)  # 1 = yes, 0 = no
sample['high_bp_9'] = np.where((sample['hedawbp'] == 1) | (sample['hedimbp'] == 1), 1, 0)  # 1 = yes, 0 = no

sample['HEEverBP'].value_counts(dropna=False)  # 1 = yes, 2 = no
sample['new_bp'] = np.select(condlist=[sample['high_bp_9'] == 1,
                                       (sample['high_bp_9'] == 0) & (sample['HEEverBP'] == 1),
                                       (sample['high_bp_9'] == 0) & (sample['HEEverBP'] == 2)],
                             choicelist=[0, 1, 0],
                             default=np.nan)
sample['new_bp'].value_counts(dropna=False)

sample['hedawch'].value_counts(dropna=False)  # 9 = yes, -1 = no
sample['hedimch'].value_counts(dropna=False)  # 1 = yes, 0 = no
sample['high_chol_9'] = np.where((sample['hedawch'] == 9) | (sample['hedimch'] == 1), 1, 0)  # 1 = yes, 0 = no

sample['HEEverHC'].value_counts(dropna=False)  # 1 = yes, 2 = no
sample['new_chol'] = np.select(condlist=[sample['high_chol_9'] == 1,
                                         (sample['high_chol_9'] == 0) & (sample['HEEverHC'] == 1),
                                         (sample['high_chol_9'] == 0) & (sample['HEEverHC'] == 2)],
                               choicelist=[0, 1, 0],
                               default=np.nan)
sample['new_chol'].value_counts(dropna=False)

sample['hedawdi'].value_counts(dropna=False)  # 7 = yes, -1 = no
sample['hedimdi'].value_counts(dropna=False)  # 1 = yes, 0 = no
sample['diabetes_9'] = np.where((sample['hedawdi'] == 7) | (sample['hedimdi'] == 1), 1, 0)  # 1 = yes, 0 = no

sample['HEEverDI'].value_counts(dropna=False)  # 1 = yes, 2 = no
sample['new_diabetes'] = np.select(condlist=[sample['diabetes_9'] == 1,
                                             (sample['diabetes_9'] == 0) & (sample['HEEverDI'] == 1),
                                             (sample['diabetes_9'] == 0) & (sample['HEEverDI'] == 2)],
                                   choicelist=[0, 1, 0],
                                   default=np.nan)
sample['new_diabetes'].value_counts(dropna=False)

# non-cardiovascular disease
sample['hedbwas'].value_counts(dropna=False)  # 2 = yes, -1 = no
sample['hedibas'].value_counts(dropna=False)  # 1 = yes, 0 = no
sample['asthma_9'] = np.select(condlist=[(sample['hedbwas'] == 2) | (sample['hedibas'] == 1),
                                         sample['hedibas'] < 0],
                               choicelist=[1, np.nan],
                               default=0)
sample['asthma_9'].value_counts(dropna=False)

sample['HEEverAS'].value_counts(dropna=False)  # 1 = yes, 2 = no
sample['new_asthma'] = np.select(condlist=[sample['asthma_9'] == 1,
                                           (sample['asthma_9'] == 0) & (sample['HEEverAS'] == 1),
                                           (sample['asthma_9'] == 0) & (sample['HEEverAS'] == 2)],
                                 choicelist=[0, 1, 0],
                                 default=np.nan)
sample['new_asthma'].value_counts(dropna=False)

sample['hedbwar'].value_counts(dropna=False)  # 3 = yes, -1 = no
sample['hedibar'].value_counts(dropna=False)  # 1 = yes, 0 = no
sample['arthritis_9'] = np.select(condlist=[(sample['hedbwar'] == 3) | (sample['hedibar'] == 1),
                                            sample['hedibar'] < 0],
                                  choicelist=[1, np.nan],
                                  default=0)
sample['arthritis_9'].value_counts(dropna=False)

sample['HEEverAR'].value_counts(dropna=False)  # 1 = yes, 2 = no
sample['new_arthritis'] = np.select(condlist=[sample['arthritis_9'] == 1,
                                              (sample['arthritis_9'] == 0) & (sample['HEEverAR'] == 1),
                                              (sample['arthritis_9'] == 0) & (sample['HEEverAR'] == 2)],
                                    choicelist=[0, 1, 0],
                                    default=np.nan)
sample['new_arthritis'].value_counts(dropna=False)

sample['hedbwca'].value_counts(dropna=False)  # 5 = yes, -1 = no
sample['hedibca'].value_counts(dropna=False)  # 1 = yes, 0 = no
sample['cancer_9'] = np.select(condlist=[(sample['hedbwca'] == 5) | (sample['hedibca'] == 1),
                                         sample['hedibca'] < 0],
                               choicelist=[1, np.nan],
                               default=0)
sample['cancer_9'].value_counts(dropna=False)

sample['HEEverCA'].value_counts(dropna=False)  # 1 = yes, 2 = no
sample['new_cancer'] = np.select(condlist=[sample['cancer_9'] == 1,
                                           (sample['cancer_9'] == 0) & (sample['HEEverCA'] == 1),
                                           (sample['cancer_9'] == 0) & (sample['HEEverCA'] == 2)],
                                 choicelist=[0, 1, 0],
                                 default=np.nan)
sample['new_cancer'].value_counts(dropna=False)

# mental health
sample['psceda'].value_counts(dropna=False)
cesd_list_9 = ['psceda', 'pscedb', 'pscedc', 'pscedd', 'pscede', 'pscedf', 'pscedg', 'pscedh']

# reverse the score of positive items
sample['pscedd'] = sample['pscedd'].replace({1: 2, 2: 1})  # happy
sample['pscedf'] = sample['pscedf'].replace({1: 2, 2: 1})  # enjoy life

sample['cesd_9'] = np.where((sample[cesd_list_9] < 0).all(axis=1), np.nan, (sample[cesd_list_9] == 1).sum(axis=1))
sample['cesd_9'].value_counts(dropna=False)  # 0 = lowest, 8 = highest

# ces-d diagnosis
sample['cesd_b_9'] = np.select(condlist=[sample['cesd_9'] >= 3, sample['cesd_9'].isna()],
                               choicelist=[1, np.nan],
                               default=0)
sample['cesd_b_9'].value_counts(dropna=False)

cesd_list_10 = [f'PSced{chr(letter)}' for letter in range(ord('A'), ord('I'))]

# reverse the score of positive items
sample['PScedD'] = sample['PScedD'].replace({1: 2, 2: 1})  # happy
sample['PScedF'] = sample['PScedF'].replace({1: 2, 2: 1})  # enjoy life

sample['cesd_10'] = np.where((sample[cesd_list_10] < 0).all(axis=1), np.nan, (sample[cesd_list_10] == 1).sum(axis=1))
sample['cesd_10'].value_counts(dropna=False)  # 0 = lowest, 8 = highest

sample['cesd_b_10'] = np.select(condlist=[sample['cesd_10'] >= 3, sample['cesd_10'].isna()],
                                choicelist=[1, np.nan],
                                default=0)
sample['cesd_b_10'].value_counts(dropna=False)

sample['new_cesd'] = np.select(condlist=[sample['cesd_b_9'] == 1,
                                         (sample['cesd_b_9'] == 0) & (sample['cesd_b_10'] == 1),
                                         (sample['cesd_b_9'] == 0) & (sample['cesd_b_10'] == 0)],
                               choicelist=[0, 1, 0],
                               default=np.nan)
sample['new_cesd'].value_counts(dropna=False)

########## Controls (used for matching)
# employment status
sample['dhwork'].value_counts(dropna=False)  # 1 = in paid employment, 2 = no
sample['employ_status_9'] = np.select(condlist=[sample['dhwork'] == 1, sample['dhwork'] == 2],
                                      choicelist=[1, 0],
                                      default=np.nan)
sample['employ_status_9'].value_counts(dropna=False)

sample['employ_status_10'] = np.select(condlist=[sample['DhWork'] == 1, sample['DhWork'] == 2],
                                       choicelist=[1, 0],
                                       default=np.nan)
sample['employ_status_10'].value_counts(dropna=False)

# income - wave 9
sample['yq10_bu_s'].value_counts(dropna=False)  # too many NAs in the provided decile data, so I will create my own

sample['eqtotinc_bu_s'].value_counts(dropna=False)
sample['total_income_bu_9'] = pd.to_numeric(sample['eqtotinc_bu_s'], errors='coerce')
sample['total_income_bu_9'].value_counts(dropna=False)

sample['total_income_bu_d_9'] = pd.qcut(sample['total_income_bu_9'], q=10, labels=False)
sample['total_income_bu_d_9'].value_counts(dropna=False)  # 0 = lowest, 9 = highest

# income - wave 10
# employment
sample['IaSInc'].value_counts(dropna=False)
sample['employ_income'] = np.select(condlist=[sample['IaSInc'] == -1, sample['IaSInc'] >= 0],
                                    choicelist=[0, sample['IaSInc']],
                                    default=np.nan)
sample['employ_income'].value_counts(dropna=False)

# annuity
sample['IaAIm'].value_counts(dropna=False)  # respondent
sample['annuity'] = np.select(condlist=[sample['IaAIm'] == -1, sample['IaAIm'] >= 0],
                              choicelist=[0, sample['IaAIm']],
                              default=np.nan)
sample['annuity'].value_counts(dropna=False)

pd.crosstab(sample['IaAIp'], sample['IaAIm'])
sample['IaAIp'].value_counts(dropna=False)  # spouse
sample['annuity_s'] = np.select(condlist=[sample['IaAIp'] == -1, sample['IaAIp'] >= 0],
                                choicelist=[0, sample['IaAIp']],
                                default=np.nan)
sample['annuity_s'].value_counts(dropna=False)

# private pension
sample['IaPPei'].value_counts(dropna=False)
sample['IaPPmo'].value_counts(dropna=False)
sample['p_pension_y'] = np.select(condlist=[sample['IaPPei'] == -1, sample['IaPPei'] >= 0],
                                  choicelist=[0, sample['IaPPei']],
                                  default=np.nan)
sample['p_pension_m'] = np.select(condlist=[sample['IaPPmo'] == -1, sample['IaPPmo'] >= 0],
                                  choicelist=[0, sample['IaPPmo'] * 12],
                                  default=np.nan)
sample['p_pension'] = sample['p_pension_y'] + sample['p_pension_m']
sample['p_pension'].value_counts(dropna=False)


# write a function to facilitate period and amount
def period_amount(period, amount):
    return np.select(condlist=[(sample[period] == -1) & (sample[amount] == -1),
                               sample[period].isin([-8, -9]),
                               sample[amount].isin([-8, -9]),
                               sample[period] == 1,
                               sample[period] == 2,
                               sample[period] == 3,
                               sample[period] == 4,
                               sample[period] == 5,
                               sample[period] == 7,
                               sample[period] == 8,
                               sample[period] == 9,
                               sample[period] == 10,
                               sample[period] == 13,
                               sample[period] == 26,
                               sample[period] == 52,
                               sample[period] == 90],
                     choicelist=[0,
                                 np.nan,
                                 np.nan,
                                 sample[amount] * 52,
                                 sample[amount] * (52 / 2),
                                 sample[amount] * (52 / 3),
                                 sample[amount] * (52 / 4),
                                 sample[amount] * 12,
                                 sample[amount] * (12 / 2),
                                 sample[amount] * 8,
                                 sample[amount] * 9,
                                 sample[amount] * 10,
                                 sample[amount] * 4,
                                 sample[amount] * 2,
                                 sample[amount],
                                 sample[amount] * 52],
                     default=np.nan)


# state pension
sample['IasPa'].value_counts(dropna=False)  # period
sample['IaPAM'].value_counts(dropna=False)  # amount
pd.crosstab(sample['IasPa'], sample['IaPAM'])
sample['s_pension'] = period_amount(period='IasPa', amount='IaPAM')  # respondent
sample['s_pension'].value_counts(dropna=False)
sample['s_pension_s'] = period_amount('IaSPp', 'IaPPAm')  # spouse
sample['s_pension_s'].value_counts(dropna=False)

# state benefits
benefit_period_list = ['IaP'] + [f'IaP{number}' for number in range(2, 43)]
benefit_amount_list = ['IaA'] + [f'IaA{number}' for number in range(2, 43)]
pd.crosstab(sample['IaP12'], sample['IaA12'])
for i in range(1, 43):
    sample[f's_benefit_{i}'] = period_amount(period=benefit_period_list[i - 1], amount=benefit_amount_list[i - 1])
sample['s_benefit_12'].value_counts(dropna=False)

# asset income
sample['IaSint'].value_counts(dropna=False)
sample['interest_savings'] = np.select(condlist=[sample['IaSint'] == -1, sample['IaSint'] >= 0],
                                       choicelist=[0, sample['IaSint']],
                                       default=np.nan)
sample['interest_savings'].value_counts(dropna=False)

sample['IaNSi'].value_counts(dropna=False)
sample['interest_national'] = np.select(condlist=[sample['IaNSi'] == -1, sample['IaNSi'] >= 0],
                                        choicelist=[0, sample['IaNSi']],
                                        default=np.nan)
sample['interest_national'].value_counts(dropna=False)

sample['IaNPBP'].value_counts(dropna=False)
sample['interest_premium'] = np.select(condlist=[sample['IaNPBP'] == -1, sample['IaNPBP'] >= 0],
                                       choicelist=[0, sample['IaNPBP']],
                                       default=np.nan)
sample['interest_premium'].value_counts(dropna=False)

sample['IaIsaD'].value_counts(dropna=False)
sample['interest_isa'] = np.select(condlist=[sample['IaIsaD'] == -1, sample['IaIsaD'] >= 0],
                                   choicelist=[0, sample['IaIsaD']],
                                   default=np.nan)
sample['interest_isa'].value_counts(dropna=False)

sample['IaSSSi'].value_counts(dropna=False)
sample['interest_share'] = np.select(condlist=[sample['IaSSSi'] == -1, sample['IaSSSi'] >= 0],
                                     choicelist=[0, sample['IaSSSi']],
                                     default=np.nan)
sample['interest_share'].value_counts(dropna=False)

sample['Iauiti'].value_counts(dropna=False)
sample['interest_trust'] = np.select(condlist=[sample['Iauiti'] == -1, sample['Iauiti'] >= 0],
                                     choicelist=[0, sample['Iauiti']],
                                     default=np.nan)
sample['interest_trust'].value_counts(dropna=False)

sample['Iabgi'].value_counts(dropna=False)
sample['interest_bond'] = np.select(condlist=[sample['Iabgi'] == -1, sample['Iabgi'] >= 0],
                                    choicelist=[0, sample['Iabgi']],
                                    default=np.nan)
sample['interest_bond'].value_counts(dropna=False)

sample['Iaira'].value_counts(dropna=False)
sample['interest_rent'] = np.select(condlist=[sample['Iaira'] == -1, sample['Iaira'] >= 0],
                                    choicelist=[0, sample['Iaira']],
                                    default=np.nan)
sample['interest_rent'].value_counts(dropna=False)

sample['IafBA'].value_counts(dropna=False)
sample['interest_farm'] = np.select(condlist=[sample['IafBA'] == -1, sample['IafBA'] >= 0],
                                    choicelist=[0, sample['IafBA']],
                                    default=np.nan)
sample['interest_farm'].value_counts(dropna=False)

# other income
sample['IaSiOi'].value_counts(dropna=False)
sample['income_other'] = np.select(condlist=[sample['IaSiOi'] == -1, sample['IaSiOi'] >= 0],
                                   choicelist=[0, sample['IaSiOi']],
                                   default=np.nan)
sample['income_other'].value_counts(dropna=False)

# total annual income
benefit_cols = [f's_benefit_{i}' for i in range(1, 43)]
interest_cols = ['interest_savings', 'interest_national', 'interest_premium', 'interest_isa', 'interest_share',
                 'interest_trust', 'interest_bond', 'interest_rent', 'interest_farm', 'income_other']

sample['total_income_bu_10'] = sample[['employ_income', 'annuity', 'annuity_s', 'p_pension', 's_pension',
                                       's_pension_s']].sum(axis=1, min_count=3) + \
                               sample[benefit_cols].sum(axis=1, min_count=21) + \
                               sample[interest_cols].sum(axis=1, min_count=5)
sample['total_income_bu_10'].value_counts(dropna=False)

# deciles
sample['total_income_bu_d_10'] = pd.qcut(sample['total_income_bu_10'], q=10, labels=False)
sample['total_income_bu_d_10'].value_counts(dropna=False)

# age
sample['age'].value_counts(dropna=False)
sample['age_9'] = np.select(condlist=[sample['age'] == 99, sample['age'] >= 0],
                            choicelist=[np.nan, sample['age']],
                            default=np.nan)
sample['age_9'].value_counts(dropna=False)

# sex
sample['sex'].value_counts(dropna=False)
sample['sex_9'] = np.select(condlist=[sample['sex'] == 1, sample['sex'] == 2],
                            choicelist=[0, 1],
                            default=np.nan)  # 1 = female, 0 = male
sample['sex_9'].value_counts(dropna=False)

# ethnicity
sample['nonwhite'].value_counts(dropna=False)
sample['ethnicity_9'] = np.select(condlist=[sample['nonwhite'] == 0, sample['nonwhite'] == 1],
                                  choicelist=[0, 1],
                                  default=np.nan)  # 1 = non-white, 0 = white
sample['ethnicity_9'].value_counts(dropna=False)

# marital status
sample['marstat'].value_counts(dropna=False)  # categorical
sample['marital_status_9'] = sample['marstat'].astype(str)

### cognitive function
# memory
sample['cfmetm'].value_counts(dropna=False)
sample['memory_9'] = np.where(sample['cfmetm'] > 0, sample['cfmetm'], np.nan)  # 1 = excellent, 5 = poor
sample['memory_9'].value_counts(dropna=False)

# numeracy
numeracy_list_9 = ['num_a_9', 'num_b_9', 'num_c_9', 'num_d_9', 'num_e_9']
sample['cfsva'].value_counts(dropna=False)
sample['num_a_9'] = np.select(condlist=[sample['cfsva'] == 93, sample['cfsva'] < 0],
                              choicelist=[1, np.nan],
                              default=0)
sample['cfsvb'].value_counts(dropna=False)
sample['num_b_9'] = np.select(condlist=[sample['cfsvb'] == 86, sample['cfsvb'] < 0],
                              choicelist=[1, np.nan],
                              default=0)
sample['cfsvc'].value_counts(dropna=False)
sample['num_c_9'] = np.select(condlist=[sample['cfsvc'] == 79, sample['cfsvc'] < 0],
                              choicelist=[1, np.nan],
                              default=0)
sample['cfsvd'].value_counts(dropna=False)
sample['num_d_9'] = np.select(condlist=[sample['cfsvd'] == 72, sample['cfsvd'] < 0],
                              choicelist=[1, np.nan],
                              default=0)
sample['cfsve'].value_counts(dropna=False)
sample['num_e_9'] = np.select(condlist=[sample['cfsve'] == 65, sample['cfsve'] < 0],
                              choicelist=[1, np.nan],
                              default=0)
sample['numeracy_9'] = sample[numeracy_list_9].sum(axis=1, min_count=1)  # 0 = least numerate, 5 = most numerate
sample['numeracy_9'].value_counts(dropna=False)

# comprehension (not available in wave 9)

### educational attainment
# age finished full-time education
sample['edend'].value_counts(dropna=False)  # 1 should be removed as it does not make sense in ELSA
sample['edu_age_9'] = np.where(sample['edend'] >= 2, sample['edend'], np.nan)
sample['edu_age_9'].value_counts(dropna=False)  # 8 = 19 or over, 2 = never went to school

# highest qualification
sample['edqual'].value_counts(dropna=False)  # categorical
sample['edu_qual_9'] = np.where(sample['edqual'] >= 1, sample['edqual'], np.nan)
sample['edu_qual_9'].value_counts(dropna=False)

### deprivation
sample['ndepriv'].value_counts(dropna=False)  # 0 = least deprived, 9 = most deprived
sample['n_deprived_9'] = np.select(condlist=[sample['ndepriv'] == -1, sample['ndepriv'] > 0],
                                   choicelist=[0, sample['ndepriv']],
                                   default=np.nan)
sample['n_deprived_9'].value_counts(dropna=False)

########## Save data
sample.to_csv(derived_path / 'wave_910_cleaned.csv', index=False)
