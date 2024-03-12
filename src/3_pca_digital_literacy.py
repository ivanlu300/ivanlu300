import pandas as pd
import numpy as np
import plotnine as pn
from pathlib import Path
from sspipe import p, px

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Set up paths
main_path = Path('1_sample_selection.py').resolve().parents[2] / 'Data' / 'elsa_main' / 'tab'
covid_path = Path('1_sample_selection.py').resolve().parents[2] / 'Data' / 'elsa_covid' / 'tab'
derived_path = Path('1_sample_selection.py').resolve().parents[2] / 'Data' / 'derived'
figure_path = Path('1_sample_selection.py').resolve().parents[1] / 'output' / 'figure'
table_path = Path('1_sample_selection.py').resolve().parents[1] / 'output' / 'table'

# Read data
full_8 = pd.read_csv(derived_path / 'wave_8_cleaned.csv')

# Select variables related to digital literacy
digital_var = ['idauniq',
               'internet_freq',
               'scinddt', 'scindlt', 'scindtb', 'scindph', 'scind95', 'scind96',  # devices
               'scinaem', 'scinacl', 'scinaed', 'scinabk', 'scinash', 'scinasl', 'scinasn', 'scinact', 'scinanw',
               'scinast', 'scinagm', 'scinajb', 'scinaps', 'scina95', 'scina96']  # activities

# Remove NAs
full_8['idauniq'].isna().sum()  # no NAs in idauniq
digital_8 = full_8[digital_var].dropna()

# Scale the variables before performing PCA
scaler = StandardScaler(with_std=True, with_mean=True)
digital_8_scaled = scaler.fit_transform(digital_8.iloc[:, 1:])

# Perform PCA
pca_digital = PCA()
pca_digital.fit(digital_8_scaled)

# Loadings
pca_loadings = pd.DataFrame(pca_digital.components_)  # each row is a principal component, and columns are corresponding variable loadings

########## PC1 loadings
pc1_loadings = pd.DataFrame({'Variable': ['Frequency of using the internet',
                                          'desktop', 'laptop', 'tablet', 'smartphone', 'other', 'none',
                                          'emails', 'video calls', 'information searching', 'finances', 'shopping', 'selling', 'social networking', 'content creation', 'news', 'TV/music/ebooks', 'games', 'job application', 'public services', 'other', 'none'],
                             'Loading': pca_loadings.iloc[0, :] | p(round, 3),
                             'Description': (['1 = every day, 6 = never'] + ['1 = yes, 0 = no']*21)})

pc1_loadings.to_latex(float_format="%.3f") | p(print)

# Explained variance (proportion)
pca_pve = pd.DataFrame({'PC': range(1, digital_8_scaled.shape[1] + 1),
                        'PVE': pca_digital.explained_variance_ratio_})

########## Screeplot
pca_screeplot = (pn.ggplot(pca_pve, pn.aes(x='PC', y='PVE'))
                 + pn.geom_point()
                 + pn.geom_line()
                 + pn.theme_bw()
                 + pn.scale_x_continuous(breaks=range(1, digital_8_scaled.shape[1] + 1))
                 + pn.labs(title='Screeplot of PCA',
                           x='Principal component',
                           y='Proportion of variance explained'))

pn.ggsave(pca_screeplot, filename='pca_screeplot.png', path=figure_path, dpi=300)

# Extract PCA scores
pca_scores = pca_digital.transform(digital_8_scaled) | p(pd.DataFrame)
digital_8['PC1'] = pca_scores.iloc[:, 0].values
digital_8['PC1'].isna().sum() # no NAs

# Merge PC1 scores with the main data
full_8 = full_8.merge(digital_8[['idauniq', 'PC1']], on='idauniq', how='left')
full_8['PC1'].isna().sum()

# Save data
full_8.to_csv(derived_path / 'wave_8_pca.csv', index=False)

########## Inspection
full_8['scind96'].value_counts(dropna=False)
pd.crosstab(full_8['internet_freq'], full_8['scind96'])
full_8['scinddt'].value_counts(dropna=False)
pd.crosstab(full_8['internet_freq'], full_8['scinddt'])
