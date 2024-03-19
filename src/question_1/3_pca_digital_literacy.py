import pandas as pd
import numpy as np
import plotnine as pn
from pathlib import Path
from sspipe import p, px

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Set up paths
main_path = Path('1_sample_selection.py').resolve().parents[3] / 'Data' / 'elsa_10' / 'tab'
derived_path = Path('1_sample_selection.py').resolve().parents[3] / 'Data' / 'derived_10'
figure_path = Path('1_sample_selection.py').resolve().parents[2] / 'output' / 'figure'
table_path = Path('1_sample_selection.py').resolve().parents[2] / 'output' / 'table'

# Read data
main_10 = pd.read_csv(derived_path / 'wave_10_cleaned_edu.csv')

# Select variables related to digital literacy
digital_var = ['idauniq',
               'int_freq'] + \
              [f'SCIND0{number}' for number in range(1, 6)] + \
              [f'SCINA0{number}' for number in range(1, 10)] + [f'SCINA{number}' for number in range(10, 22)]

# Remove NAs
main_10['idauniq'].isna().sum()  # no NAs in idauniq
digital_10 = main_10[digital_var].dropna()  # 54 dropped

# Scale the variables before performing PCA
scaler = StandardScaler(with_std=True, with_mean=True)
digital_10_scaled = scaler.fit_transform(digital_10.iloc[:, 1:])

# Perform PCA
pca_digital = PCA()
pca_digital.fit(digital_10_scaled)

# Loadings
pca_loadings = pd.DataFrame(
    pca_digital.components_)  # each row is a principal component, and columns are corresponding variable loadings

########## PC1 loadings
pc1_loadings = pd.DataFrame({'Item': ['Frequency of using the Internet',
                                      'desktop', 'laptop', 'tablet', 'smartphone', 'other',
                                      'emails', 'video calls', 'finding information', 'finances', 'shopping', 'selling',
                                      'social networking', 'news', 'TV/radio', 'music', 'games', 'e-books',
                                      'job application', 'government services', 'checking travel times',
                                      'satellite navigation', 'buying public transport tickets', 'booking a taxi',
                                      'finding local amenities', 'controlling household appliances',
                                      'none of the above'],
                             'Description': np.nan,
                             'Loading': pca_loadings.iloc[0, :] | p(round, 3)
                             })

pc1_loadings.to_latex(index=False, float_format="%.3f") | p(print)

# Explained variance (proportion)
pca_pve = pd.DataFrame({'PC': range(1, digital_10_scaled.shape[1] + 1),
                        'PVE': pca_digital.explained_variance_ratio_})

########## Screeplot
pca_screeplot = (pn.ggplot(pca_pve, pn.aes(x='PC', y='PVE'))
                 + pn.geom_point()
                 + pn.geom_line()
                 + pn.theme_bw()
                 + pn.scale_x_continuous(breaks=range(1, digital_10_scaled.shape[1] + 1))
                 + pn.labs(x='Principal component', y='Proportion of variance explained'))

pn.ggsave(pca_screeplot, filename='pca_screeplot.png', path=figure_path, dpi=300)

# Extract PCA scores
pca_scores = pca_digital.transform(digital_10_scaled) | p(pd.DataFrame)
digital_10['PC1'] = pca_scores.iloc[:, 0].values  # smaller values indicate better digital literacy
digital_10['PC1'].isna().sum()  # no NAs

# Merge PC1 scores with the main data
main_10 = main_10.merge(digital_10[['idauniq', 'PC1']], on='idauniq', how='left')
main_10['PC1'].value_counts(dropna=False)

# binary PC1
main_10['PC1_b'] = np.select(condlist=[main_10['PC1'] < np.nanmean(main_10['PC1']),
                                       main_10['PC1'] >= np.nanmean(main_10['PC1'])],
                             choicelist=[1, 0],
                             default=np.nan)  # 1 = high digital literacy, 0 = low digital literacy
main_10['PC1_b'].value_counts(dropna=False)

# Save data
main_10.to_csv(derived_path / 'wave_10_pca_edu.csv', index=False)

########## Inspection
