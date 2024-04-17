import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path
from sspipe import p

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Set up paths
main_path = Path().resolve().parents[2] / 'Data' / 'elsa_10' / 'tab'
derived_path = Path().resolve().parents[2] / 'Data' / 'derived_10'
figure_path = Path().resolve().parents[1] / 'output' / 'figure'
table_path = Path().resolve().parents[1] / 'output' / 'table'

# Read data
sample = pd.read_csv(derived_path / 'wave_910_cleaned.csv')

########## Wave 9 PCA
# Select variables related to digital literacy
digital_var_9 = ['idauniq',
                 'int_freq_9'] + \
                ['scinddt', 'scindlt', 'scindtb', 'scindph', 'scind95', 'scind96'] + \
                ['scinaem', 'scinacl', 'scinaed', 'scinahe', 'scinabk', 'scinash', 'scinasl', 'scinasn', 'scinact',
                 'scinanw', 'scinast', 'scinagm', 'scinajb', 'scinaps', 'scina95', 'scina96']

# Remove NAs
sample['idauniq'].isna().sum()  # no NAs in idauniq
digital_9 = sample[digital_var_9].dropna()  # N = 3557

# Scale the variables before performing PCA
scaler_9 = StandardScaler(with_std=True, with_mean=True)
digital_9_scaled = scaler_9.fit_transform(digital_9.iloc[:, 1:])

# Perform PCA
pca_digital_9 = PCA()
pca_digital_9.fit(digital_9_scaled)

# Loadings
pca_loadings_9 = pd.DataFrame(pca_digital_9.components_)
# each row is a principal component, and columns are corresponding variable loadings

# PC1 loadings
pc1_loadings_9 = pd.DataFrame({'Item': ['Frequency of using the Internet',
                                        'desktop', 'laptop', 'tablet', 'smartphone', 'other',
                                        'no device (do not access the Internet)',
                                        'emails', 'video calls', 'finding information (learning)',
                                        'finding information (health)', 'finances', 'shopping', 'selling',
                                        'social networking', 'creating content', 'news', 'music', 'games',
                                        'job application', 'government services', 'other', 'none of the above'],
                               'Description': '',
                               'Loading': pca_loadings_9.iloc[0, :] | p(round, 3)
                               })

pc1_loadings_9.to_latex(index=False, float_format="%.3f") | p(print)

# Explained variance (proportion)
pca_pve_9 = pd.DataFrame({'PC': range(1, digital_9_scaled.shape[1] + 1),
                          'PVE': pca_digital_9.explained_variance_ratio_})

# Screeplot
fig_9 = px.scatter(pca_pve_9, x='PC', y='PVE',
                   labels={'PC': 'Principal component', 'PVE': 'Proportion of variance explained'})
fig_9.add_trace(px.line(pca_pve_9, x='PC', y='PVE').data[0])

fig_9.update_layout(
    xaxis=dict(tickmode='array', tickvals=list(range(1, digital_9_scaled.shape[1] + 1))),
    yaxis=dict(tickmode='array', tickvals=list(np.arange(0, 0.30, 0.05))),
)

fig_9.write_image(figure_path / 'pca_screeplot_q2.png')
fig_9.show()

# Extract PCA scores
pca_scores_9 = pca_digital_9.transform(digital_9_scaled) | p(pd.DataFrame)
digital_9['PC1_9'] = pca_scores_9.iloc[:, 0].values  # smaller values indicate better digital literacy
digital_9['PC1_9'].isna().sum()  # no NAs

# Merge PC1 scores with the main data
sample = sample.merge(digital_9[['idauniq', 'PC1_9']], on='idauniq', how='left')
sample['PC1_9'].value_counts(dropna=False)

# binary PC1
sample['PC1_b_9'] = np.select(condlist=[sample['PC1_9'] < np.nanmedian(sample['PC1_9']),
                                        sample['PC1_9'] >= np.nanmedian(sample['PC1_9'])],
                              choicelist=[1, 0],
                              default=np.nan)  # 1 = high digital literacy, 0 = low digital literacy
sample['PC1_b_9'].value_counts(dropna=False)

########## Wave 10 PCA
# Select variables related to digital literacy
digital_var_10 = ['idauniq',
                  'int_freq_10'] + \
                 [f'SCIND0{number}' for number in range(1, 6)] + \
                 [f'SCINA0{number}' for number in range(1, 10)] + [f'SCINA{number}' for number in range(10, 22)]

# Remove NAs
sample['idauniq'].isna().sum()  # no NAs in idauniq
digital_10 = sample[digital_var_10].dropna()  # N = 3557

# Scale the variables before performing PCA
scaler_10 = StandardScaler(with_std=True, with_mean=True)
digital_10_scaled = scaler_10.fit_transform(digital_10.iloc[:, 1:])

# Perform PCA
pca_digital_10 = PCA()
pca_digital_10.fit(digital_10_scaled)

# Loadings
pca_loadings_10 = pd.DataFrame(pca_digital_10.components_)
# each row is a principal component, and columns are corresponding variable loadings

# PC1 loadings
pc1_loadings_10 = pd.DataFrame({'Item': ['Frequency of using the Internet',
                                         'desktop', 'laptop', 'tablet', 'smartphone', 'other',
                                         'no device (do not access the Internet)',
                                         'emails', 'video calls', 'finding information (learning)',
                                         'finding information (health)', 'finances', 'shopping', 'selling',
                                         'social networking', 'creating content', 'news', 'music', 'games',
                                         'job application', 'government services', 'other', 'none of the above'],
                                'Description': '',
                                'Loading': pca_loadings_10.iloc[0, :] | p(round, 3)
                                })

pc1_loadings_10.to_latex(index=False, float_format="%.3f") | p(print)

# Explained variance (proportion)
pca_pve_10 = pd.DataFrame({'PC': range(1, digital_10_scaled.shape[1] + 1),
                           'PVE': pca_digital_10.explained_variance_ratio_})

# Screeplot
fig_10 = px.scatter(pca_pve_10, x='PC', y='PVE',
                    labels={'PC': 'Principal component', 'PVE': 'Proportion of variance explained'})
fig_10.add_trace(px.line(pca_pve_10, x='PC', y='PVE').data[0])

fig_10.update_layout(
    xaxis=dict(tickmode='array', tickvals=list(range(1, digital_10_scaled.shape[1] + 1))),
    yaxis=dict(tickmode='array', tickvals=list(np.arange(0, 0.30, 0.05))),
)

fig_10.write_image(figure_path / 'pca_screeplot_q2.png')
fig_10.show()

# Extract PCA scores
pca_scores_10 = pca_digital_10.transform(digital_10_scaled) | p(pd.DataFrame)
digital_10['PC1_10'] = pca_scores_10.iloc[:, 0].values  # smaller values indicate better digital literacy
digital_10['PC1_10'].isna().sum()  # no NAs

# Merge PC1 scores with the main data
sample = sample.merge(digital_10[['idauniq', 'PC1_10']], on='idauniq', how='left')
sample['PC1_10'].value_counts(dropna=False)

# binary PC1
sample['PC1_b_10'] = np.select(condlist=[sample['PC1_10'] < np.nanmedian(sample['PC1_10']),
                                         sample['PC1_10'] >= np.nanmedian(sample['PC1_10'])],
                               choicelist=[1, 0],
                               default=np.nan)  # 1 = high digital literacy, 0 = low digital literacy
sample['PC1_b_10'].value_counts(dropna=False)

########## Save data
sample.to_csv(derived_path / 'wave_910_pca.csv', index=False)

########## Inspection
pd.crosstab(sample['PC1_b_9'], sample['PC1_b_10'])
