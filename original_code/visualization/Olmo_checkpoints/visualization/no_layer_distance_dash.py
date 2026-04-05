"""Interactive Dash visualization of OLMo checkpoint AUROC absolute distance from 0.5."""

from dash import Dash, html, dcc
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import os
import glob

# Read specific layer datasets and compute highest absolute distance from 0.5 for each step
specific_layer_folder = './visualization/Olmo_checkpoints/specific_layer_datasets'
specific_layer_data = []

print("Loading data from specific_layer_datasets...")
for csv_file in glob.glob(os.path.join(specific_layer_folder, '*.csv')):
    df_layers = pd.read_csv(csv_file)
    # Calculate absolute distance from 0.5 for each layer
    df_layers['auroc_distance'] = np.abs(df_layers['auroc'] - 0.5)
    # Group by revision and find the maximum distance across all layers
    df_grouped = df_layers.groupby('revision').agg({
        'auroc_distance': 'max',
        'auroc': lambda x: x.iloc[df_layers.loc[x.index, 'auroc_distance'].argmax()],
        'layer': lambda x: x.iloc[df_layers.loc[x.index, 'auroc_distance'].argmax()]
    }).reset_index()
    df_grouped.rename(columns={'auroc_distance': 'max_distance', 'auroc': 'auroc_at_max_distance', 'layer': 'best_layer'}, inplace=True)
    specific_layer_data.append(df_grouped)

# Combine all dataframes
df = pd.concat(specific_layer_data, ignore_index=True)

# Remove any duplicate entries (same step), keeping first
df = df.drop_duplicates(subset=['revision'], keep='first')

# Extract step number and tokens from revision column
df['step'] = df['revision'].str.extract(r'step(\d+)').astype(int)
df['tokens'] = df['revision'].str.extract(r'tokens(\d+)').astype(int)

# Convert step to thousands (e.g., 557000 -> 557)
df['step_k'] = df['step'] / 1000

# Sort by step number
df = df.sort_values('step')

print(f"\nLoaded {len(df)} checkpoints")
print(f"Step range: {df['step_k'].min():.0f}k to {df['step_k'].max():.0f}k")
print(f"Token range: {df['tokens'].min()}B to {df['tokens'].max()}B")
print(f"AUROC at max distance range: {df['auroc_at_max_distance'].min():.4f} to {df['auroc_at_max_distance'].max():.4f}")
print(f"Max distance range: {df['max_distance'].min():.4f} to {df['max_distance'].max():.4f}")

app = Dash(__name__)

fig = go.Figure()

# Main scatter plot for max distance vs step
fig.add_trace(go.Scatter(
    x=df['step_k'],
    y=df['max_distance'],
    mode='markers',
    marker=dict(
        color='#4ECDC4',
        size=10,
        line=dict(width=1.5, color='black'),
        opacity=0.8
    ),
    name='Max Distance',
    hovertemplate="<b>Step %{customdata[0]}</b><br>" +
                 "Tokens: %{customdata[1]}B<br>" +
                 "Max Distance from 0.5: %{y:.4f}<br>" +
                 "AUROC at Max Distance: %{customdata[2]:.4f}<br>" +
                 "Layer: %{customdata[3]}<extra></extra>",
    customdata=np.column_stack((df['step'], df['tokens'], df['auroc_at_max_distance'], df['best_layer']))
))

# Add a trend line
z = np.polyfit(df['step_k'], df['max_distance'], 2)
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
        text="OLMo Checkpoint AUROC Distance from Random Chance",
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
        title=dict(text="Highest AUROC Absolute Distance from 0.5", font=dict(size=18, color='black', family='Arial')),
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
        range=[0.10, 0.45]  # Adjust based on your data range
    )
)

# Add plot borders
fig.update_xaxes(mirror=False, ticks='outside', showline=True, linecolor='black')
fig.update_yaxes(mirror=False, ticks='outside', showline=True, linecolor='black')

app.layout = [
    html.H1(children='OLMo Checkpoint AUROC Distance from Random Chance', 
            style={'textAlign':'center', 'color': 'black', 'fontSize': '24px', 'fontFamily': 'Arial, sans-serif'}),
    html.P(children='Showing the highest absolute distance from 0.5 (random chance) across all layers for each checkpoint.',
           style={'textAlign':'center', 'color': '#666', 'fontSize': '14px', 'fontFamily': 'Arial, sans-serif', 'marginTop': '-10px'}),
    html.P(children='Note: High distance can come from either very high AUROC (>0.9) or very low AUROC (<0.1).',
           style={'textAlign':'center', 'color': '#666', 'fontSize': '12px', 'fontFamily': 'Arial, sans-serif', 'marginTop': '-15px', 'fontStyle': 'italic'}),
    dcc.Graph(figure=fig)
]

if __name__ == '__main__':
    app.run(debug=True, port=8054)

