import pandas as pd
import numpy as np
import os
from sspipe import p

# Set up paths
main_path = os.path.join(os.path.abspath('..'), 'Data', 'elsa_main', 'tab')
covid_path = os.path.join(os.path.abspath('..'), 'Data', 'elsa_covid', 'tab')

# Read data
main_var = ['idauniq', 'w8xwgt',
            'hehelf', 'heill', 
            'psceda', 'pscedb', 'pscedc', 'pscedd', 'pscede', 'pscedf', 'pscedg', 'pscedh',
            'hobb', 'scint', 'scind96', 
            'scinaem', 'scinacl', 'scinaed', 'scinabk', 'scinash', 'scinasl', 'scinasn', 'scinact', 'scinanw', 'scinast', 'scinagm', 'scinajb', 'scinaps', 'scina95', 'scina96']
main_8 = pd.read_table(os.path.join(main_path, 'wave_8_elsa_data_eul_v2.tab'),
                       usecols=main_var)


nurse_var = ['idauniq', '']
nurse_8 = pd.read_table(os.path.join(main_path, 'wave_8_elsa_nurse_data_eul_v1.tab'),
                        usecols=nurse_var)

# Inspect variables
main_8['hobb'].value_counts() # no internet access at home
main_8['scint'].value_counts() # frequency of using internet

pd.crosstab(main_8['hobb'], main_8['scint'])
# very interesting:
# many respondents have internet access but never use internet 

pd.crosstab(main_8['hobb'], main_8['scinabk'])
# probably not weak instrument for internet access

pd.crosstab(main_8['scinabk'], main_8['scinaed'])
# probably not weak instrument for information and fact finding