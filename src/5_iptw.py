import pandas as pd
import numpy as np
from pathlib import Path
from sspipe import p, px

from linearmodels.iv import IV2SLS

# Set up paths
main_path = Path('1_sample_selection.py').resolve().parents[2] / 'Data' / 'elsa_main' / 'tab'
covid_path = Path('1_sample_selection.py').resolve().parents[2] / 'Data' / 'elsa_covid' / 'tab'
derived_path = Path('1_sample_selection.py').resolve().parents[2] / 'Data' / 'derived'
figure_path = Path('1_sample_selection.py').resolve().parents[1] / 'output' / 'figure'
table_path = Path('1_sample_selection.py').resolve().parents[1] / 'output' / 'table'

# Read data
full_8 = pd.read_csv(derived_path / 'wave_8_pca.csv')




# For loop
for j, y in enumerate(y_list):
    model_data = sample_iv[[f'{y}', 'sex_1', 'ex_work_1', 'ex_limit_1', 'treatment', 'spage_2']].dropna(subset=[f'{y}'])
    model_formula = f'{y} ~ 1 + C(sex_1) + C(ex_work_1) + C(ex_limit_1) + [treatment ~ spage_2]'
    model = IV2SLS.from_formula(model_formula, model_data).fit()

    iv_table.iloc[j * 2, 0] = model.params['treatment']
    iv_table.iloc[j * 2 + 1, 0] = model.params['treatment'] / model.std_errors['treatment']