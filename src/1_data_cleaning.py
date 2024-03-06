import pandas as pd
import numpy as np
import os
from sspipe import p

# Set up paths
main_path = os.path.join(os.path.abspath('..'), 'Data', 'elsa_main', 'tab')
covid_path = os.path.join(os.path.abspath('..'), 'Data', 'elsa_covid', 'tab')

# Read data
main_8 = pd.read_table(os.path.join(main_path, 'wave_8_elsa_data_eul_v2.tab'),
                       usecols=main_var)

nurse_8 = pd.read_table(os.path.join(main_path, 'wave_8_elsa_nurse_data_eul_v1.tab'),
                        usecols=nurse_var)

# Clean variables
main_8['internet_access'] = np.select(condlist=[main_8['hobb'] == 1, main_8['hobb'] == 2],
                                      choicelist=[1, 0],
                                      default=np.nan)