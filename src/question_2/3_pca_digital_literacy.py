import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
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

# Select variables related to digital literacy
digital_var = ['idauniq',
               'int_freq_9'] + \
              ['scinddt', 'scindlt', 'scindtb', 'scindph', 'scind95', 'scind96'] + \
              ['scinaem', 'scinacl', 'scinaed', 'scinahe', 'scinabk', 'scinash', 'scinasl', 'scinasn', 'scinact',
               'scinanw', 'scinast', 'scinagm', 'scinajb', 'scinaps', 'scina95', 'scina96']

# Remove NAs
sample['idauniq'].isna().sum()  # no NAs in idauniq
digital_9 = sample[digital_var].dropna()  # N = 3557

# Scale the variables before performing PCA
scaler = StandardScaler(with_std=True, with_mean=True)
digital_9_scaled = scaler.fit_transform(digital_9.iloc[:, 1:])

# Perform PCA
pca_digital = PCA()
pca_digital.fit(digital_9_scaled)

# Loadings
pca_loadings = pd.DataFrame(
    pca_digital.components_)  # each row is a principal component, and columns are corresponding variable loadings

########## PC1 loadings
pc1_loadings = pd.DataFrame({'Item': ['Frequency of using the Internet',
                                      'desktop', 'laptop', 'tablet', 'smartphone', 'other',
                                      'no device (do not access the Internet)',
                                      'emails', 'video calls', 'finding information (learning)',
                                      'finding information (health)', 'finances', 'shopping', 'selling',
                                      'social networking', 'creating content', 'news', 'music', 'games',
                                      'job application', 'government services', 'other', 'none of the above'],
                             'Description': '',
                             'Loading': pca_loadings.iloc[0, :] | p(round, 3)
                             })

pc1_loadings.to_latex(index=False, float_format="%.3f") | p(print)

# Explained variance (proportion)
pca_pve = pd.DataFrame({'PC': range(1, digital_9_scaled.shape[1] + 1),
                        'PVE': pca_digital.explained_variance_ratio_})

########## Screeplot
# seaborn
sns.lineplot(x='PC', y='PVE', data=pca_pve)
sns.scatterplot(x='PC', y='PVE', data=pca_pve)

plt.xlabel('Principal component')
plt.ylabel('Proportion of variance explained')
plt.xticks(range(1, digital_9_scaled.shape[1] + 1))
plt.yticks(np.arange(0, 0.30, 0.05))

plt.savefig(figure_path / 'pca_screeplot_q2.png')
plt.show()

# plotly
fig = px.scatter(pca_pve, x='PC', y='PVE')
fig.add_trace(px.line(pca_pve, x='PC', y='PVE').data[0])

fig.update_layout(
    xaxis_title='Principal component',
    yaxis_title='Proportion of variance explained',
    xaxis=dict(tickmode='array', tickvals=list(range(1, digital_9_scaled.shape[1] + 1))),
    yaxis=dict(tickmode='array', tickvals=list(np.arange(0, 0.30, 0.05))),
)

fig.write_image(figure_path / 'pca_screeplot_q2.png')
fig.show()

########## Extract PCA scores
pca_scores = pca_digital.transform(digital_9_scaled) | p(pd.DataFrame)
digital_9['PC1'] = pca_scores.iloc[:, 0].values  # smaller values indicate better digital literacy
digital_9['PC1'].isna().sum()  # no NAs

# Merge PC1 scores with the main data
sample = sample.merge(digital_9[['idauniq', 'PC1']], on='idauniq', how='left')
sample['PC1'].value_counts(dropna=False)

# binary PC1
sample['PC1_b'] = np.select(condlist=[sample['PC1'] < np.nanmean(sample['PC1']),
                                      sample['PC1'] >= np.nanmean(sample['PC1'])],
                            choicelist=[1, 0],
                            default=np.nan)  # 1 = high digital literacy, 0 = low digital literacy
sample['PC1_b'].value_counts(dropna=False)

########## Save data
sample.to_csv(derived_path / 'wave_910_pca.csv', index=False)

########## Inspection
