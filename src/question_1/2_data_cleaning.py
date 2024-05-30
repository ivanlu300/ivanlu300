import pandas as pd
import numpy as np
from pathlib import Path
from sspipe import p

# Set up paths
main_path = Path().resolve().parents[2] / 'Data' / 'elsa_10' / 'tab'
derived_path = Path().resolve().parents[2] / 'Data' / 'derived_10'

# Read data
main_10 = pd.read_csv(derived_path / 'wave_10_sample.csv')

########## Treatment
main_10['SCINT'].value_counts(dropna=False)
main_10['int_freq'] = np.where(main_10['SCINT'] > 0, main_10['SCINT'], np.nan)  # 1 = every day, 6 = never
main_10['int_freq'].value_counts(dropna=False)

device_list = [f'SCIND0{number}' for number in range(1, 6)]
pd.crosstab(main_10['SCINT'], main_10['SCIND01'])
# respondents with SCINT == 6 were not asked about devices (hence coded as -1), but I assume they have no device
main_10[device_list] = main_10[device_list].where(main_10['SCINT'] < 6, other=0)
# then deal with NAs
main_10[device_list] = main_10[device_list].where(main_10[device_list] >= 0, other=np.nan)  # 1 = yes, 0 = no
pd.crosstab(main_10['int_freq'], main_10['SCIND01'])

activity_list = [f'SCINA0{number}' for number in range(1, 10)] + [f'SCINA{number}' for number in range(10, 22)]
activity_list_no96 = [f'SCINA0{number}' for number in range(1, 10)] + [f'SCINA{number}' for number in range(10, 21)]
pd.crosstab(main_10['SCINT'], main_10['SCINA01'])
# respondents with SCINT == 6 were not asked about activities (hence coded as -1), but I assume they have no activity
main_10[activity_list_no96] = main_10[activity_list_no96].where(main_10['SCINT'] < 6, other=0)
pd.crosstab(main_10['SCINT'], main_10['SCINA21'])  # 21 is none
main_10['SCINA21'] = main_10['SCINA21'].where(main_10['SCINT'] < 6, other=1)
# then deal with NAs
main_10[activity_list] = main_10[activity_list].where(main_10[activity_list] >= 0, other=np.nan)  # 1 = yes, 0 = no
pd.crosstab(main_10['int_freq'], main_10['SCINA21'])
pd.crosstab(main_10['int_freq'], main_10['SCINA02'])

########## Reverse causality
reason_list = [f'SCINNO0{number}' for number in range(1, 10)]
main_10['SCINNO06'].value_counts(dropna=False)
main_10[reason_list] = main_10[reason_list].where(main_10[reason_list] >= 0, other=np.nan)  # 1 = yes, 0 = no
main_10['SCINNO05'].value_counts(dropna=False)

pd.crosstab(main_10['Heill'], main_10['Helim'], dropna=False)
main_10['HeFunc'].value_counts(dropna=False)

# Outcome - self-reported health
main_10['Hehelf'].value_counts(dropna=False)  # 1 = excellent, 5 = poor
main_10['srh'] = np.where(main_10['Hehelf'] > 0, main_10['Hehelf'], np.nan)
main_10['srh'].value_counts(dropna=False)

# Outcome - cardiovascular diseases
main_10['HEHaveBP'].value_counts(dropna=False)
main_10['high_bp'] = np.select(condlist=[main_10['HEHaveBP'].isin([1, 2]), main_10['HEHaveBP'].isin([-1, 3])],
                               choicelist=[1, 0],
                               default=np.nan)
main_10['high_bp'].value_counts(dropna=False)

main_10['HEHaveHC'].value_counts(dropna=False)
main_10['high_chol'] = np.select(condlist=[main_10['HEHaveHC'].isin([1, 2]), main_10['HEHaveHC'].isin([-1, 3])],
                                 choicelist=[1, 0],
                                 default=np.nan)
main_10['high_chol'].value_counts(dropna=False)

main_10['HEEverDI'].value_counts(dropna=False)
main_10['diabetes'] = np.select(condlist=[main_10['HEEverDI'] == 1, main_10['HEEverDI'] == 2],
                                choicelist=[1, 0],
                                default=np.nan)
main_10['diabetes'].value_counts(dropna=False)

# Outcome - non-cardiovascular diseases
main_10['HEHaveAS'].value_counts(dropna=False)
main_10['asthma'] = np.select(condlist=[main_10['HEHaveAS'].isin([1, 2]), main_10['HEHaveAS'].isin([-1, 3])],
                              choicelist=[1, 0],
                              default=np.nan)
main_10['asthma'].value_counts(dropna=False)

main_10['HEHaveAR'].value_counts(dropna=False)
main_10['arthritis'] = np.select(condlist=[main_10['HEHaveAR'].isin([1, 2]), main_10['HEHaveAR'].isin([-1, 3])],
                                 choicelist=[1, 0],
                                 default=np.nan)
main_10['arthritis'].value_counts(dropna=False)

main_10['HEHaveCA'].value_counts(dropna=False)
main_10['cancer'] = np.select(condlist=[main_10['HEHaveCA'].isin([1, 2]), main_10['HEHaveCA'].isin([-1, 3])],
                              choicelist=[1, 0],
                              default=np.nan)
main_10['cancer'].value_counts(dropna=False)

# Outcome - mental health
main_10['PScedA'].value_counts(dropna=False)
cesd_list = [f'PSced{chr(letter)}' for letter in range(ord('A'), ord('I'))]

# reverse the score of positive items
main_10['PScedD'] = main_10['PScedD'].replace({1: 2, 2: 1})  # happy
main_10['PScedF'] = main_10['PScedF'].replace({1: 2, 2: 1})  # enjoy life

main_10['cesd'] = np.where((main_10[cesd_list] < 0).all(axis=1), np.nan, (main_10[cesd_list] == 1).sum(axis=1))
main_10['cesd'].value_counts(dropna=False)  # 0 = lowest, 8 = highest

main_10['cesd_b'] = np.select(condlist=[((main_10[cesd_list] == 1).sum(axis=1)) >= 3,
                                        (main_10[cesd_list] < 0).all(axis=1)],
                              choicelist=[1, np.nan],
                              default=0)  # 1 = yes, 0 = no
main_10['cesd_b'].value_counts(dropna=False)

# Anxiety disorder
main_10['hepsyan'].value_counts(dropna=False)
main_10['anxiety'] = np.select(condlist=[main_10['hepsyan'] == 1, main_10['hepsyan'] == 0, main_10['hepsyan'] == -1],
                               choicelist=[1, 0, 0],
                               default=np.nan)

# Depression
main_10['hepsyde'].value_counts(dropna=False)
main_10['depression'] = np.select(condlist=[main_10['hepsyde'] == 1, main_10['hepsyde'] == 0, main_10['hepsyde'] == -1],
                                  choicelist=[1, 0, 0],
                                  default=np.nan)

# Mood swings
main_10['hepsymo'].value_counts(dropna=False)
main_10['mood'] = np.select(condlist=[main_10['hepsymo'] == 1, main_10['hepsymo'] == 0, main_10['hepsymo'] == -1],
                            choicelist=[1, 0, 0],
                            default=np.nan)

########## Controls
# employment status
main_10['WpDes'].value_counts(dropna=False)
main_10['DhWork'].value_counts(dropna=False)  # whether in paid employment
main_10['employ_status'] = np.select(condlist=[main_10['DhWork'] == 1, main_10['DhWork'] == 2],
                                     choicelist=[1, 0],
                                     default=np.nan)
main_10['employ_status'].value_counts(dropna=False)

### income
# employment
main_10['IaSInc'].value_counts(dropna=False)
main_10['employ_income'] = np.select(condlist=[main_10['IaSInc'] == -1, main_10['IaSInc'] >= 0],
                                     choicelist=[0, main_10['IaSInc']],
                                     default=np.nan)
main_10['employ_income'].value_counts(dropna=False)

# annuity
main_10['IaAIm'].value_counts(dropna=False)  # respondent
main_10['annuity'] = np.select(condlist=[main_10['IaAIm'] == -1, main_10['IaAIm'] >= 0],
                               choicelist=[0, main_10['IaAIm']],
                               default=np.nan)
main_10['annuity'].value_counts(dropna=False)

pd.crosstab(main_10['IaAIp'], main_10['IaAIm'])
main_10['IaAIp'].value_counts(dropna=False)  # spouse
main_10['annuity_s'] = np.select(condlist=[main_10['IaAIp'] == -1, main_10['IaAIp'] >= 0],
                                 choicelist=[0, main_10['IaAIp']],
                                 default=np.nan)
main_10['annuity_s'].value_counts(dropna=False)

# private pension
main_10['IaPPei'].value_counts(dropna=False)
main_10['IaPPmo'].value_counts(dropna=False)
main_10['p_pension_y'] = np.select(condlist=[main_10['IaPPei'] == -1, main_10['IaPPei'] >= 0],
                                   choicelist=[0, main_10['IaPPei']],
                                   default=np.nan)
main_10['p_pension_m'] = np.select(condlist=[main_10['IaPPmo'] == -1, main_10['IaPPmo'] >= 0],
                                   choicelist=[0, main_10['IaPPmo'] * 12],
                                   default=np.nan)
main_10['p_pension'] = main_10['p_pension_y'] + main_10['p_pension_m']
main_10['p_pension'].value_counts(dropna=False)


# write a function to facilitate period and amount
def period_amount(period, amount):
    return np.select(condlist=[(main_10[period] == -1) & (main_10[amount] == -1),
                               main_10[period].isin([-8, -9]),
                               main_10[amount].isin([-8, -9]),
                               main_10[period] == 1,
                               main_10[period] == 2,
                               main_10[period] == 3,
                               main_10[period] == 4,
                               main_10[period] == 5,
                               main_10[period] == 7,
                               main_10[period] == 8,
                               main_10[period] == 9,
                               main_10[period] == 10,
                               main_10[period] == 13,
                               main_10[period] == 26,
                               main_10[period] == 52,
                               main_10[period] == 90],
                     choicelist=[0,
                                 np.nan,
                                 np.nan,
                                 main_10[amount] * 52,
                                 main_10[amount] * (52 / 2),
                                 main_10[amount] * (52 / 3),
                                 main_10[amount] * (52 / 4),
                                 main_10[amount] * 12,
                                 main_10[amount] * (12 / 2),
                                 main_10[amount] * 8,
                                 main_10[amount] * 9,
                                 main_10[amount] * 10,
                                 main_10[amount] * 4,
                                 main_10[amount] * 2,
                                 main_10[amount],
                                 main_10[amount] * 52],
                     default=np.nan)


# state pension
main_10['IasPa'].value_counts(dropna=False)  # period
main_10['IaPAM'].value_counts(dropna=False)  # amount
pd.crosstab(main_10['IasPa'], main_10['IaPAM'])
main_10['s_pension'] = period_amount(period='IasPa', amount='IaPAM')  # respondent
main_10['s_pension'].value_counts(dropna=False)
main_10['s_pension_s'] = period_amount('IaSPp', 'IaPPAm')  # spouse
main_10['s_pension_s'].value_counts(dropna=False)

# state benefits
benefit_period_list = ['IaP'] + [f'IaP{number}' for number in range(2, 43)]
benefit_amount_list = ['IaA'] + [f'IaA{number}' for number in range(2, 43)]
pd.crosstab(main_10['IaP12'], main_10['IaA12'])
for i in range(1, 43):
    main_10[f's_benefit_{i}'] = period_amount(period=benefit_period_list[i - 1], amount=benefit_amount_list[i - 1])
main_10['s_benefit_12'].value_counts(dropna=False)

# asset income
main_10['IaSint'].value_counts(dropna=False)
main_10['interest_savings'] = np.select(condlist=[main_10['IaSint'] == -1, main_10['IaSint'] >= 0],
                                        choicelist=[0, main_10['IaSint']],
                                        default=np.nan)
main_10['interest_savings'].value_counts(dropna=False)

main_10['IaNSi'].value_counts(dropna=False)
main_10['interest_national'] = np.select(condlist=[main_10['IaNSi'] == -1, main_10['IaNSi'] >= 0],
                                         choicelist=[0, main_10['IaNSi']],
                                         default=np.nan)
main_10['interest_national'].value_counts(dropna=False)

main_10['IaNPBP'].value_counts(dropna=False)
main_10['interest_premium'] = np.select(condlist=[main_10['IaNPBP'] == -1, main_10['IaNPBP'] >= 0],
                                        choicelist=[0, main_10['IaNPBP']],
                                        default=np.nan)
main_10['interest_premium'].value_counts(dropna=False)

main_10['IaIsaD'].value_counts(dropna=False)
main_10['interest_isa'] = np.select(condlist=[main_10['IaIsaD'] == -1, main_10['IaIsaD'] >= 0],
                                    choicelist=[0, main_10['IaIsaD']],
                                    default=np.nan)
main_10['interest_isa'].value_counts(dropna=False)

main_10['IaSSSi'].value_counts(dropna=False)
main_10['interest_share'] = np.select(condlist=[main_10['IaSSSi'] == -1, main_10['IaSSSi'] >= 0],
                                      choicelist=[0, main_10['IaSSSi']],
                                      default=np.nan)
main_10['interest_share'].value_counts(dropna=False)

main_10['Iauiti'].value_counts(dropna=False)
main_10['interest_trust'] = np.select(condlist=[main_10['Iauiti'] == -1, main_10['Iauiti'] >= 0],
                                      choicelist=[0, main_10['Iauiti']],
                                      default=np.nan)
main_10['interest_trust'].value_counts(dropna=False)

main_10['Iabgi'].value_counts(dropna=False)
main_10['interest_bond'] = np.select(condlist=[main_10['Iabgi'] == -1, main_10['Iabgi'] >= 0],
                                     choicelist=[0, main_10['Iabgi']],
                                     default=np.nan)
main_10['interest_bond'].value_counts(dropna=False)

main_10['Iaira'].value_counts(dropna=False)
main_10['interest_rent'] = np.select(condlist=[main_10['Iaira'] == -1, main_10['Iaira'] >= 0],
                                     choicelist=[0, main_10['Iaira']],
                                     default=np.nan)
main_10['interest_rent'].value_counts(dropna=False)

main_10['IafBA'].value_counts(dropna=False)
main_10['interest_farm'] = np.select(condlist=[main_10['IafBA'] == -1, main_10['IafBA'] >= 0],
                                     choicelist=[0, main_10['IafBA']],
                                     default=np.nan)
main_10['interest_farm'].value_counts(dropna=False)

# other income
main_10['IaSiOi'].value_counts(dropna=False)
main_10['income_other'] = np.select(condlist=[main_10['IaSiOi'] == -1, main_10['IaSiOi'] >= 0],
                                    choicelist=[0, main_10['IaSiOi']],
                                    default=np.nan)
main_10['income_other'].value_counts(dropna=False)

# total annual income
benefit_cols = [f's_benefit_{i}' for i in range(1, 43)]
interest_cols = ['interest_savings', 'interest_national', 'interest_premium', 'interest_isa', 'interest_share',
                 'interest_trust', 'interest_bond', 'interest_rent', 'interest_farm', 'income_other']

main_10['total_income_bu'] = main_10[['employ_income', 'annuity', 'annuity_s', 'p_pension', 's_pension',
                                      's_pension_s']].sum(axis=1, min_count=3) + \
                             main_10[benefit_cols].sum(axis=1, min_count=21) + \
                             main_10[interest_cols].sum(axis=1, min_count=5)
main_10['total_income_bu'].value_counts(dropna=False)

# deciles
main_10['total_income_bu_d'] = pd.qcut(main_10['total_income_bu'], q=10, labels=False)
main_10['total_income_bu_d'].value_counts(dropna=False)

# age
main_10['indager'].value_counts(dropna=False)
main_10['age'] = np.where(main_10['indager'] <= 0, np.nan, main_10['indager'])
main_10['age'].value_counts(dropna=False)

# sex
main_10['indsex'].value_counts(dropna=False)  # 1 = male, 2 = female
main_10['sex'] = np.where(main_10['indsex'] == 1, 0, 1)  # 0 = male, 1 = female
main_10['sex'].value_counts(dropna=False)

# ethnicity
pd.crosstab(main_10['fqethnmr'], main_10['nonwhite'], dropna=False)
main_10['ethnicity'] = np.select(condlist=[main_10['fqethnmr'] == 1,
                                           main_10['fqethnmr'] == 2,
                                           (main_10['fqethnmr'] == -1) & (main_10['nonwhite'] == 0),
                                           (main_10['fqethnmr'] == -1) & (main_10['nonwhite'] == 1)],
                                 choicelist=[0, 1, 0, 1],
                                 default=np.nan)  # 0 = white, 1 = non-white
main_10['ethnicity'].value_counts(dropna=False)

# marital status
main_10['dimarr'].value_counts(dropna=False)  # 1-6 categories
main_10['marital_status'] = main_10['dimarr'].astype(str)
type(main_10['marital_status'][0])

### cognitive function
# memory
main_10['CfMetM'].value_counts(dropna=False)
main_10['memory'] = np.where(main_10['CfMetM'] < 0, np.nan, main_10['CfMetM'])  # 1 = excellent, 5 = poor
main_10['memory'].value_counts(dropna=False)

# numeracy
numeracy_list = ['num_a', 'num_b', 'num_c', 'num_d', 'num_e']
main_10['CfSvA'].value_counts(dropna=False)
main_10['num_a'] = np.select(condlist=[main_10['CfSvA'] == 93, main_10['CfSvA'] < 0],
                             choicelist=[1, np.nan],
                             default=0)
main_10['CfSvB'].value_counts(dropna=False)
main_10['num_b'] = np.select(condlist=[main_10['CfSvB'] == 86, main_10['CfSvB'] < 0],
                             choicelist=[1, np.nan],
                             default=0)
main_10['CfSvC'].value_counts(dropna=False)
main_10['num_c'] = np.select(condlist=[main_10['CfSvC'] == 79, main_10['CfSvC'] < 0],
                             choicelist=[1, np.nan],
                             default=0)
main_10['CfSvD'].value_counts(dropna=False)
main_10['num_d'] = np.select(condlist=[main_10['CfSvD'] == 72, main_10['CfSvD'] < 0],
                             choicelist=[1, np.nan],
                             default=0)
main_10['CfSvE'].value_counts(dropna=False)
main_10['num_e'] = np.select(condlist=[main_10['CfSvE'] == 65, main_10['CfSvE'] < 0],
                             choicelist=[1, np.nan],
                             default=0)
main_10['numeracy'] = main_10[numeracy_list].sum(axis=1, min_count=1)  # 0 = least numerate, 5 = most numerate
main_10['numeracy'].value_counts(dropna=False)

# comprehension
comprehension_list = ['CfLitB', 'CfLitC', 'CfLitD', 'CfLitE']
main_10['CfLitC'].value_counts(dropna=False)
main_10[comprehension_list] = main_10[comprehension_list].where(main_10[comprehension_list] > 0, other=np.nan)
main_10[comprehension_list] = main_10[comprehension_list].replace({2: 0})
main_10['comprehension'] = main_10[comprehension_list].sum(axis=1,
                                                           min_count=1)  # 0 = least comprehend, 4 = most comprehend
main_10['comprehension'].value_counts(dropna=False)

### educational attainment
# age finished education
pd.crosstab(main_10['fqendm'], main_10['edend'], dropna=False)
main_10['fqendm'].value_counts(dropna=False)
main_10['edend'].value_counts(dropna=False)  # edend == 1 means the respondent has not finished education
main_10['edu_age'] = np.select(condlist=[main_10['fqendm'] > 0, main_10['fqendm'] == -1],
                               choicelist=[main_10['fqendm'], main_10['edend']],
                               default=np.nan)
main_10['edu_age'] = np.where(main_10['edu_age'] == 1, np.nan, main_10['edu_age'])
main_10['edu_age'].value_counts(dropna=False)

main_10['edqual'].value_counts(dropna=False)
main_10['fqqumnv5'].value_counts(dropna=False)

main_10['edu_qual'] = np.select(condlist=[main_10['fqqumnv5'] == 1,
                                          main_10['fqqumnv4'] == 1,
                                          main_10['fqqumnv3'] == 1,
                                          main_10['fqqumnv2'] == 1,
                                          main_10['fqqumnv1'] == 1,
                                          main_10['edqual'] < 0],
                                choicelist=[1, 1, 3, 4, 5, np.nan],
                                default=main_10['edqual'])
# main_10['edu_qual'] = np.where(main_10['edu_qual'] == 6, np.nan, main_10['edu_qual'])  # edqual == 6 means foreign
main_10['edu_qual'].value_counts(dropna=False)

# deprivation
deprive_list = ['exrelefo', 'exreleme', 'exreleou', 'exrelede', 'exreleel', 'exrelefa', 'exrelepr', 'exreleho',
                'exreletr']
main_10['EXRela'].value_counts(dropna=False)

pd.crosstab(main_10['EXRela'], main_10['exrelefo'], dropna=False)
# respondents with EXRela == 1 were not asked about deprivation (hence coded as -1), but I assume they are not deprived
main_10[deprive_list] = main_10[deprive_list].where(main_10['EXRela'] != 1, other=0)

# remove NAs
main_10[deprive_list] = main_10[deprive_list].where(main_10[deprive_list] >= 0, other=np.nan)

# count the number of deprived items
main_10['n_deprived'] = main_10[deprive_list].sum(axis=1, min_count=1)  # 0 = least deprived, 9 = most deprived
main_10['n_deprived'].value_counts(dropna=False)

########## Save data
main_10.to_csv(derived_path / 'wave_10_cleaned.csv', index=False)

########## Inspection
