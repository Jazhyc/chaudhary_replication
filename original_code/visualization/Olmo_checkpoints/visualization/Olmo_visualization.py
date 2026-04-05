"""This file visualizes OLMo checkpoint AUROC scores across training steps."""

from dash import Dash, html, dcc
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import os
import glob

# Read all CSV files in the datasets folder (aggregated data)
data_folder = './visualization/Olmo_checkpoints/datasets'
all_data = []

for csv_file in glob.glob(os.path.join(data_folder, '*.csv')):
    df_temp = pd.read_csv(csv_file)
    # Filter out rows with errors or missing AUROC values
    if 'status' in df_temp.columns:
        df_temp = df_temp[df_temp['status'] == 'success']
    if 'best_auroc' in df_temp.columns:
        df_temp = df_temp.dropna(subset=['best_auroc'])
        all_data.append(df_temp)

# Read specific layer datasets and compute best AUROC for each step
specific_layer_folder = './visualization/Olmo_checkpoints/specific_layer_datasets'
specific_layer_data = []

for csv_file in glob.glob(os.path.join(specific_layer_folder, '*.csv')):
    df_layers = pd.read_csv(csv_file)
    # Group by revision and find the maximum AUROC across all layers
    df_grouped = df_layers.groupby('revision').agg({
        'auroc': 'max',
        'layer': lambda x: x.iloc[df_layers.loc[x.index, 'auroc'].argmax()]
    }).reset_index()
    df_grouped.rename(columns={'auroc': 'best_auroc', 'layer': 'best_layer'}, inplace=True)
    specific_layer_data.append(df_grouped)

# Combine all dataframes
if all_data:
    df_aggregated = pd.concat(all_data, ignore_index=True)
else:
    df_aggregated = pd.DataFrame()

if specific_layer_data:
    df_specific = pd.concat(specific_layer_data, ignore_index=True)
    # Remove duplicates, keeping the first occurrence
    df_specific = df_specific.drop_duplicates(subset=['revision'], keep='first')
else:
    df_specific = pd.DataFrame()

# Combine both datasets
if not df_aggregated.empty and not df_specific.empty:
    df = pd.concat([df_aggregated, df_specific], ignore_index=True)
elif not df_specific.empty:
    df = df_specific
else:
    df = df_aggregated

# Remove any duplicate entries (same step), keeping first
df = df.drop_duplicates(subset=['revision'], keep='first')

# Extract step number and tokens from revision column
df['step'] = df['revision'].str.extract(r'step(\d+)').astype(int)
df['tokens'] = df['revision'].str.extract(r'tokens(\d+)').astype(int)

# Convert step to thousands (e.g., 557000 -> 557)
df['step_k'] = df['step'] / 1000

# Sort by step number
df = df.sort_values('step')

print(f"Loaded {len(df)} checkpoints")
print(f"Step range: {df['step_k'].min():.0f}k to {df['step_k'].max():.0f}k")
print(f"Token range: {df['tokens'].min()}B to {df['tokens'].max()}B")
print(f"AUROC range: {df['best_auroc'].min():.4f} to {df['best_auroc'].max():.4f}")

app = Dash()

fig = go.Figure()

# Main scatter plot for AUROC vs step
fig.add_trace(go.Scatter(
    x=df['step_k'],
    y=df['best_auroc'],
    mode='markers',
    marker=dict(
        color='#FF6B6B',
        size=10,
        line=dict(width=1.5, color='black'),
        opacity=0.8
    ),
    name='AUROC',
    hovertemplate="<b>Step %{customdata[0]}</b><br>" +
                 "Tokens: %{customdata[1]}B<br>" +
                 "AUROC: %{y:.4f}<br>" +
                 "Best Layer: %{customdata[2]}<extra></extra>",
    customdata=np.column_stack((df['step'], df['tokens'], df['best_layer']))
))

# Add a trend line
z = np.polyfit(df['step_k'], df['best_auroc'], 2)
p = np.poly1d(z)
x_trend = np.linspace(df['step_k'].min(), df['step_k'].max(), 100)
y_trend = p(x_trend)

fig.add_trace(go.Scatter(
    x=x_trend,
    y=y_trend,
    mode='lines',
    line=dict(color='rgba(100, 100, 100, 0.3)', width=2, dash='dash'),
    name='Trend',
    hoverinfo='skip'
))

# Update layout
fig.update_layout(
    title=dict(
        text="OLMo Checkpoint AUROC Performance",
        font=dict(size=20, color='black'),
        x=0.5,
        xanchor='center'
    ),
    width=1000,
    height=600,
    plot_bgcolor='#FAFAF7',
    paper_bgcolor='#FAFAF7',
    font=dict(color='black', size=11),
    legend=dict(
        font=dict(size=13, color='black', family='Arial'),
        orientation="h",
        yanchor="top",
        y=1.0,
        xanchor="left",
        x=0,
        bgcolor='#EDEDE8',
        borderwidth=0
    ),
    margin=dict(l=80, r=80, t=80, b=80),
    xaxis=dict(
        title=dict(text="Training Step (thousands)", font=dict(size=18, color='black', family='Arial')),
        tickfont=dict(size=13, color='black', family='Arial'),
        linecolor='black',
        linewidth=1.5,
        gridcolor='#E5E5E5',
        gridwidth=0.8,
        showgrid=True,
        zeroline=False,
        mirror=False,
        ticks='outside',
        ticklen=5
    ),
    yaxis=dict(
        title=dict(text="Best AUROC", font=dict(size=18, color='black', family='Arial')),
        tickfont=dict(size=13, color='black', family='Arial'),
        linecolor='black',
        linewidth=1.5,
        gridcolor='#E5E5E5',
        gridwidth=0.8,
        showgrid=True,
        zeroline=False,
        mirror=False,
        ticks='outside',
        ticklen=5,
        range=[0.55, 0.80]  # Adjust based on your data range
    )
)

# Add plot borders
fig.update_xaxes(mirror=False, ticks='outside', showline=True, linecolor='black')
fig.update_yaxes(mirror=False, ticks='outside', showline=True, linecolor='black')

app.layout = [
    html.H1(children='OLMo Checkpoint AUROC Performance', 
            style={'textAlign':'center', 'color': 'black', 'fontSize': '24px', 'fontFamily': 'Arial, sans-serif'}),
    dcc.Graph(figure=fig)
]

if __name__ == '__main__':
    app.run(debug=True, port=8053)

