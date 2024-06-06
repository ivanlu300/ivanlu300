import pandas as pd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
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
# Select the intersection (i.e., inner join) of the features sets for Wave 9 and Wave 10
digital_var_9 = ['idauniq',
                 'int_freq_9'] + \
                ['scinddt', 'scindlt', 'scindtb', 'scindph', 'scind95'] + \
                ['scinaem', 'scinacl', 'scinaed', 'scinabk', 'scinash', 'scinasl', 'scinasn',
                 'scinanw', 'scinast', 'scinagm', 'scinajb', 'scinaps', 'scina96']

# Remove NAs
sample['idauniq'].isna().sum()  # no NAs in idauniq
digital_9 = sample[digital_var_9].dropna()  # N = 3603

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
pc1_loadings_9 = pd.DataFrame({'Item': ['Frequency of Internet usage',
                                        'desktop', 'laptop', 'tablet', 'smartphone', 'other',
                                        'emails', 'video calls', 'finding information', 'finances', 'shopping',
                                        'selling',
                                        'social networking', 'news', 'music', 'games',
                                        'job application', 'government services', 'no online activities'],
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

# Calculate the standard deviation of PC1
digital_9['PC1_9'].std()  # 2.15

# Merge PC1 scores with the main data
sample = sample.merge(digital_9[['idauniq', 'PC1_9']], on='idauniq', how='left')
sample['PC1_9'].value_counts(dropna=False)

# binary PC1
sample['PC1_b_9'] = np.select(condlist=[sample['PC1_9'] <= sample['PC1_9'].quantile(0.25),
                                        sample['PC1_9'] >= sample['PC1_9'].quantile(0.75)],
                              choicelist=[1, 0],
                              default=np.nan)  # 1 = high digital literacy, 0 = low digital literacy
sample['PC1_b_9'].value_counts(dropna=False)  # 1 = 891, 0 = 891, NaN = 1821

########## Wave 10 PCA
# Select the intersection (i.e., inner join) of the features sets for Wave 9 and Wave 10
digital_var_10 = ['idauniq',
                  'int_freq_10'] + \
                 [f'SCIND0{number}' for number in range(1, 6)] + \
                 ['SCINA01', 'SCINA02', 'SCINA03', 'SCINA04', 'SCINA05', 'SCINA06', 'SCINA07', 'SCINA08'] + \
                 ['SCINA10', 'SCINA11', 'SCINA13', 'SCINA14', 'SCINA21']

# Remove NAs
sample['idauniq'].isna().sum()  # no NAs in idauniq
digital_10 = sample[digital_var_10].dropna()  # N = 3566

# Scale the variables before performing PCA
scaler_10 = StandardScaler(with_std=True, with_mean=True)
digital_10_scaled = scaler_10.fit_transform(digital_10.iloc[:, 1:])
#
# # Perform PCA
# pca_digital_10 = PCA()
# pca_digital_10.fit(digital_10_scaled)
#
# # Loadings
# pca_loadings_10 = pd.DataFrame(pca_digital_10.components_)
# # each row is a principal component, and columns are corresponding variable loadings
#
# # PC1 loadings
# pc1_loadings_10 = pd.DataFrame({'Item': ['Frequency of using the Internet',
#                                          'desktop', 'laptop', 'tablet', 'smartphone', 'other devices',
#                                          'emails', 'video calls', 'finding information', 'finances', 'shopping', 'selling',
#                                          'social networking', 'news', 'TV/radio', 'music', 'games', 'e-books',
#                                          'job application', 'government services', 'checking travel times',
#                                          'satellite navigation', 'buying public transport tickets', 'booking a taxi',
#                                          'finding local amenities', 'controlling household appliances',
#                                          'no online activities'],
#                                 'Description': '',
#                                 'Loading': pca_loadings_10.iloc[0, :] | p(round, 3)
#                                 })
#
# pc1_loadings_10.to_latex(index=False, float_format="%.3f") | p(print)
#
# # Explained variance (proportion)
# pca_pve_10 = pd.DataFrame({'PC': range(1, digital_10_scaled.shape[1] + 1),
#                            'PVE': pca_digital_10.explained_variance_ratio_})
#
# # Screeplot
# fig_10 = px.scatter(pca_pve_10, x='PC', y='PVE',
#                     labels={'PC': 'Principal component', 'PVE': 'Proportion of variance explained'})
# fig_10.add_trace(px.line(pca_pve_10, x='PC', y='PVE').data[0])
#
# fig_10.update_layout(
#     xaxis=dict(tickmode='array', tickvals=list(range(1, digital_10_scaled.shape[1] + 1))),
#     yaxis=dict(tickmode='array', tickvals=list(np.arange(0, 0.30, 0.05))),
# )
#
# # fig_10.write_image(figure_path / 'pca_screeplot_w10_q2.png')
# # fig_10.show()
#
# # combine the two plots
# fig = make_subplots(rows=1, cols=2, subplot_titles=('Wave 9', 'Wave 10'))
#
# for trace in fig_9.data:
#     fig.add_trace(trace, row=1, col=1)
#
# fig.update_xaxes(title='Principal component', row=1, col=1)
# fig.update_yaxes(title='Proportion of variance explained', row=1, col=1)
#
# for trace in fig_10.data:
#     fig.add_trace(trace, row=1, col=2)
#
# fig.update_xaxes(title='Principal component', row=1, col=2)
#
# fig.write_image(figure_path / 'pca_screeplot_q2.png')
# fig.show()

# Use the same loadings from Wave 9 to construct digital literacy index for Wave 10
digital_10['PC1_10'] = pd.DataFrame(digital_10_scaled).apply(lambda x: sum([a * b for a, b in zip(pca_loadings_9.iloc[0, :], x)]), axis=1)

# Merge PC1 scores with the main data
sample = sample.merge(digital_10[['idauniq', 'PC1_10']], on='idauniq', how='left')
sample['PC1_10'].value_counts(dropna=False)

# difference in digital literacy index between Wave 9 and Wave 10
sample['PC1_diff'] = sample['PC1_10'] - sample['PC1_9']

#
sample = sample.loc[sample['PC1_b_9'].notnull(), :]

# remove respondents whose difference in digital literacy index between waves is greater or equal to one standard deviation
sample = sample.loc[abs(sample['PC1_diff']) < digital_9['PC1_9'].std(), :]

# check the distribution of group assignment
sample['PC1_b_9'].value_counts(dropna=False)  # 1 = 589, 0 = 291

########## Save data
sample.to_csv(derived_path / 'wave_910_pca.csv', index=False)

########## Inspection
