"""
Visualize OLMo AUROC distance from 0.5 averaged across layers.

Creates two visualizations:
1. Average over ALL layers
2. Average over MIDDLE layers only (excluding first 10 and last 10 layers)
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import os
import glob

# Read all CSV files in the specific_layer_datasets folder
data_folder = './visualization/Olmo_checkpoints/specific_layer_datasets'
all_data = []

print("Loading data from specific_layer_datasets...")
for csv_file in glob.glob(os.path.join(data_folder, '*.csv')):
    df_temp = pd.read_csv(csv_file)
    all_data.append(df_temp)
    print(f"Loaded {os.path.basename(csv_file)}: {len(df_temp)} rows")

# Combine all dataframes
df = pd.concat(all_data, ignore_index=True)

# Extract step number and tokens from revision column
df['step'] = df['revision'].str.extract(r'step(\d+)').astype(int)
df['tokens'] = df['revision'].str.extract(r'tokens(\d+)').astype(int)

# Convert step to thousands (e.g., 557000 -> 557)
df['step_k'] = df['step'] / 1000

# Calculate absolute distance from 0.5 for each layer
df['auroc_distance'] = np.abs(df['auroc'] - 0.5)

# Ensure layer is integer
df['layer'] = df['layer'].astype(int)

print(f"\nTotal rows loaded: {len(df)}")
print(f"Unique steps: {df['step'].nunique()}")
print(f"Unique layers: {sorted(df['layer'].unique())}")
print(f"Layer range: {df['layer'].min()} to {df['layer'].max()}")

# Get layer range info
min_layer = df['layer'].min()
max_layer = df['layer'].max()
n_layers = len(df['layer'].unique())

# Define middle layers (excluding first 10 and last 10)
# If layers are 1-31, middle layers would be 11-21
middle_layer_start = min_layer + 10
middle_layer_end = max_layer - 10

print(f"\nAll layers: {min_layer} to {max_layer}")
print(f"Middle layers (excluding first 10 and last 10): {middle_layer_start} to {middle_layer_end}")

# Calculate averages per step
# 1. Average over ALL layers
all_layers_avg = df.groupby(['step', 'step_k', 'tokens']).agg({
    'auroc_distance': 'mean',
    'auroc': 'mean'
}).reset_index()
all_layers_avg.columns = ['step', 'step_k', 'tokens', 'avg_distance', 'avg_auroc']
all_layers_avg = all_layers_avg.sort_values('step_k')

# 2. Average over MIDDLE layers only
middle_layers_df = df[(df['layer'] >= middle_layer_start) & (df['layer'] <= middle_layer_end)]
middle_layers_avg = middle_layers_df.groupby(['step', 'step_k', 'tokens']).agg({
    'auroc_distance': 'mean',
    'auroc': 'mean'
}).reset_index()
middle_layers_avg.columns = ['step', 'step_k', 'tokens', 'avg_distance', 'avg_auroc']
middle_layers_avg = middle_layers_avg.sort_values('step_k')

print(f"\nData points for all layers average: {len(all_layers_avg)}")
print(f"Data points for middle layers average: {len(middle_layers_avg)}")

# Create figure with two subplots
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=[
        f'Average Over ALL Layers (1-{max_layer})',
        f'Average Over MIDDLE Layers ({middle_layer_start}-{middle_layer_end})'
    ],
    horizontal_spacing=0.1
)

# Colors
all_layers_color = '#2E86AB'  # Blue
middle_layers_color = '#A23B72'  # Magenta/Pink

# Plot 1: Average over ALL layers - scatter points only
fig.add_trace(
    go.Scatter(
        x=all_layers_avg['step_k'],
        y=all_layers_avg['avg_distance'],
        mode='markers',
        marker=dict(
            size=8,
            color=all_layers_color,
            opacity=0.6,
            line=dict(width=1, color='white')
        ),
        name='All Layers (points)',
        hovertemplate="<b>Step %{customdata[0]}</b><br>" +
                     "Tokens: %{customdata[1]}B<br>" +
                     "Avg Distance: %{y:.4f}<br>" +
                     "Avg AUROC: %{customdata[2]:.4f}<extra></extra>",
        customdata=np.column_stack((all_layers_avg['step'], all_layers_avg['tokens'], all_layers_avg['avg_auroc']))
    ),
    row=1, col=1
)

# Add trend line for all layers (more prominent)
if len(all_layers_avg) > 3:
    z = np.polyfit(all_layers_avg['step_k'], all_layers_avg['avg_distance'], 2)
    p = np.poly1d(z)
    x_trend = np.linspace(all_layers_avg['step_k'].min(), all_layers_avg['step_k'].max(), 100)
    y_trend = p(x_trend)
    
    fig.add_trace(
        go.Scatter(
            x=x_trend,
            y=y_trend,
            mode='lines',
            line=dict(color=all_layers_color, width=3),
            name='All Layers (trend)',
            hoverinfo='skip'
        ),
        row=1, col=1
    )

# Plot 2: Average over MIDDLE layers - scatter points only
fig.add_trace(
    go.Scatter(
        x=middle_layers_avg['step_k'],
        y=middle_layers_avg['avg_distance'],
        mode='markers',
        marker=dict(
            size=8,
            color=middle_layers_color,
            opacity=0.6,
            line=dict(width=1, color='white')
        ),
        name='Middle Layers (points)',
        hovertemplate="<b>Step %{customdata[0]}</b><br>" +
                     "Tokens: %{customdata[1]}B<br>" +
                     "Avg Distance: %{y:.4f}<br>" +
                     "Avg AUROC: %{customdata[2]:.4f}<extra></extra>",
        customdata=np.column_stack((middle_layers_avg['step'], middle_layers_avg['tokens'], middle_layers_avg['avg_auroc']))
    ),
    row=1, col=2
)

# Add trend line for middle layers (more prominent)
if len(middle_layers_avg) > 3:
    z = np.polyfit(middle_layers_avg['step_k'], middle_layers_avg['avg_distance'], 2)
    p = np.poly1d(z)
    x_trend = np.linspace(middle_layers_avg['step_k'].min(), middle_layers_avg['step_k'].max(), 100)
    y_trend = p(x_trend)
    
    fig.add_trace(
        go.Scatter(
            x=x_trend,
            y=y_trend,
            mode='lines',
            line=dict(color=middle_layers_color, width=3),
            name='Middle Layers (trend)',
            hoverinfo='skip'
        ),
        row=1, col=2
    )

# Update layout
fig.update_layout(
    title=dict(
        text="OLMo AUROC Distance from 0.5: Layer Averages Across Training Steps",
        font=dict(size=22, color='black', family='Arial'),
        x=0.5,
        xanchor='center'
    ),
    width=1400,
    height=550,
    plot_bgcolor='#FAFAF7',
    paper_bgcolor='#FAFAF7',
    font=dict(color='black', size=12),
    margin=dict(l=80, r=60, t=100, b=80),
    showlegend=True,
    legend=dict(
        orientation='h',
        yanchor='bottom',
        y=1.02,
        xanchor='center',
        x=0.5
    )
)

# Update x-axes
for col in [1, 2]:
    fig.update_xaxes(
        title_text="Training Step (k)",
        title_font=dict(size=14),
        tickfont=dict(size=11),
        linecolor='black',
        linewidth=1.5,
        gridcolor='#E5E5E5',
        gridwidth=0.8,
        showgrid=True,
        zeroline=False,
        mirror=False,
        ticks='outside',
        ticklen=4,
        showline=True,
        row=1, col=col
    )

# Update y-axes
for col in [1, 2]:
    fig.update_yaxes(
        title_text="Avg Distance from 0.5" if col == 1 else "",
        title_font=dict(size=14),
        tickfont=dict(size=11),
        linecolor='black',
        linewidth=1.5,
        gridcolor='#E5E5E5',
        gridwidth=0.8,
        showgrid=True,
        zeroline=False,
        mirror=False,
        ticks='outside',
        ticklen=4,
        showline=True,
        range=[0.0, 0.5],  # Reasonable range for averaged values
        row=1, col=col
    )

# Save as PDF
try:
    output_path = './visualization/Olmo_checkpoints/olmo_averaged_layers_visualization.pdf'
    fig.write_image(output_path, width=1400, height=550)
    print(f"\nSaved visualization to: {output_path}")
except Exception as e:
    print(f"\nNote: Could not save static image: {e}")
    print("To enable image export, install kaleido: pip install kaleido")

# Also save as HTML for interactive viewing
try:
    html_path = './visualization/Olmo_checkpoints/olmo_averaged_layers_visualization.html'
    fig.write_html(html_path)
    print(f"Saved interactive HTML to: {html_path}")
except Exception as e:
    print(f"Note: Could not save HTML: {e}")

# Print summary statistics
print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)

print(f"\nAll Layers Average:")
print(f"  Mean distance: {all_layers_avg['avg_distance'].mean():.4f}")
print(f"  Min distance:  {all_layers_avg['avg_distance'].min():.4f}")
print(f"  Max distance:  {all_layers_avg['avg_distance'].max():.4f}")
print(f"  Std distance:  {all_layers_avg['avg_distance'].std():.4f}")

print(f"\nMiddle Layers Average (layers {middle_layer_start}-{middle_layer_end}):")
print(f"  Mean distance: {middle_layers_avg['avg_distance'].mean():.4f}")
print(f"  Min distance:  {middle_layers_avg['avg_distance'].min():.4f}")
print(f"  Max distance:  {middle_layers_avg['avg_distance'].max():.4f}")
print(f"  Std distance:  {middle_layers_avg['avg_distance'].std():.4f}")

# Show the figure
fig.show()

