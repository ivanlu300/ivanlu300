import pandas as pd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from pathlib import Path
from sspipe import p

# Set up paths
main_path = Path().resolve().parents[2] / 'Data' / 'elsa_10' / 'tab'
derived_path = Path().resolve().parents[2] / 'Data' / 'derived_10'
figure_path = Path().resolve().parents[1] / 'output' / 'figure'
table_path = Path().resolve().parents[1] / 'output' / 'table'

# Read data
sample = pd.read_csv(derived_path / 'wave_910_pca.csv')

# Select plot data
plot_data = sample[['PC1_b_9',
                    'srh_9', 'srh_10',
                    'high_bp_9', 'high_bp_10', 'high_chol_9', 'high_chol_10', 'diabetes_9', 'diabetes_10', 'asthma_9', 'asthma_10', 'arthritis_9', 'arthritis_10', 'cancer_9', 'cancer_10',
                    'cesd_9', 'cesd_10', 'anxiety_9', 'anxiety_10', 'mood_9', 'mood_10']]

# Prepare data for plotting
def calculate_95CI(x):
    mean = x.mean()
    count = x.count()
    std_err = x.std() / np.sqrt(count)
    lower_bound = mean - (mean - 1.96 * std_err)
    upper_bound = (mean + 1.96 * std_err) - mean
    return pd.Series([mean, lower_bound, upper_bound], index=['mean', 'lower_bound', 'upper_bound'])

# List of outcomes
outcome_list = ['srh', 'high_bp', 'high_chol', 'diabetes', 'asthma', 'arthritis', 'cancer', 'cesd', 'anxiety', 'mood']
plot_data['group'] = np.where(plot_data['PC1_b_9'] == 0, 'Low', 'High')

# For loop to draw error bar plots
fig = make_subplots(rows=5, cols=2, subplot_titles=['Self-rated health', 'High blood pressure', 'High cholesterol', 'Diabetes', 'Asthma', 'Arthritis', 'Cancer', 'Depression score', 'Anxiety disorder', 'Mood swings'])

for i, outcome in enumerate(outcome_list):
    wave_9 = plot_data.groupby('group')[f'{outcome}_9'].apply(calculate_95CI).unstack().reset_index()
    wave_10 = plot_data.groupby('group')[f'{outcome}_10'].apply(calculate_95CI).unstack().reset_index()

    # combine wave 9 and wave 10 data
    wave_9['wave'] = 9
    wave_10['wave'] = 10
    combined = pd.concat([wave_9, wave_10], axis=0)
    combined['group'] = combined['group'].astype(str)

    # subplot
    fig_sub = px.scatter(combined, x='wave', y='mean', error_y='upper_bound', error_y_minus='lower_bound', color='group')
    fig_sub.add_trace(px.line(combined, x='wave', y='mean', color='group').data[0])
    fig_sub.add_trace(px.line(combined, x='wave', y='mean', color='group').data[1])
    
    for trace in fig_sub.data:
        fig.add_trace(trace, row=(i // 2) + 1, col=(i % 2) + 1)
    
    fig.update_xaxes(tickvals=[9, 10], row=(i // 2) + 1, col=(i % 2) + 1)
    fig.update_traces(showlegend=True if i == 0 else False, row=(i // 2) + 1, col=(i % 2) + 1)

fig.update_layout(height=1200, width=800, showlegend=True, legend=dict(xanchor='center', x=0.5, yanchor='bottom', y=-0.1, orientation='h'), legend_title_text='Digital literacy')
fig.show()

# Save figure
fig.write_image(figure_path / 'desc_stats_q2.png')

# ########## GPT
# fig = make_subplots(rows=5, cols=2, subplot_titles=['Self-rated health', 'High blood pressure', 'High cholesterol', 'Diabetes', 'Asthma', 'Arthritis', 'Cancer', 'Depression score', 'Anxiety disorder', 'Mood swings'])

# for i, outcome in enumerate(outcome_list):
#     wave_9 = plot_data.groupby('group')[f'{outcome}_9'].apply(calculate_95CI).unstack().reset_index()
#     wave_10 = plot_data.groupby('group')[f'{outcome}_10'].apply(calculate_95CI).unstack().reset_index()

#     # combine wave 9 and wave 10 data
#     wave_9['wave'] = 9
#     wave_10['wave'] = 10
#     combined = pd.concat([wave_9, wave_10])

#     # Create scatter plot with error bars and line plot in subplots
#     fig.add_trace(go.Scatter(
#         x=combined['wave'],
#         y=combined['mean'],
#         mode='lines+markers',
#         error_y=dict(
#             type='data',
#             symmetric=True,
#             array=combined['upper_bound']
#         ),
#         name=f'{outcome}',
#         legendgroup=outcome,
#         showlegend=True if i == 0 else False,  # Show legend only once
#     ), row=(i // 2) + 1, col=(i % 2) + 1)

# fig.update_layout(height=1200, width=800, showlegend=True)
# fig.show()
