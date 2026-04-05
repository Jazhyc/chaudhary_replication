"""This file visualizes OLMo checkpoint AUROC scores per layer across training steps."""

from dash import Dash, html, dcc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import os
import glob

# Read all CSV files in the specific_layer_datasets folder
data_folder = '/Users/iansu/Maheep-INN/visualization/Olmo_checkpoints/specific_layer_datasets'
all_data = []

for csv_file in glob.glob(os.path.join(data_folder, '*.csv')):
    df_temp = pd.read_csv(csv_file)
    all_data.append(df_temp)
    print(f"Loaded {csv_file}: {len(df_temp)} rows")

# Combine all dataframes
df = pd.concat(all_data, ignore_index=True)

# Extract step number and tokens from revision column
df['step'] = df['revision'].str.extract(r'step(\d+)').astype(int)
df['tokens'] = df['revision'].str.extract(r'tokens(\d+)').astype(int)

# Convert step to thousands (e.g., 557000 -> 557)
df['step_k'] = df['step'] / 1000

# Ensure layer is integer
df['layer'] = df['layer'].astype(int)

print(f"\nTotal rows loaded: {len(df)}")
print(f"Unique steps: {df['step'].nunique()}")
print(f"Unique layers: {sorted(df['layer'].unique())}")
print(f"Layer range: {df['layer'].min()} to {df['layer'].max()}")
print(f"Step range: {df['step_k'].min():.0f}k to {df['step_k'].max():.0f}k")
print(f"AUROC range: {df['auroc'].min():.4f} to {df['auroc'].max():.4f}")

# Get unique layers sorted
layers = sorted(df['layer'].unique())
n_layers = len(layers)

# Create color palette for layers
colors = [
    '#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8',
    '#F7DC6F', '#BB8FCE', '#85C1E2', '#F8B739', '#52B788',
    '#E63946', '#457B9D', '#A8DADC', '#F4A261', '#2A9D8F'
]

# Calculate grid dimensions - using 5 columns for better space utilization
n_cols = 5
n_rows = (n_layers + n_cols - 1) // n_cols

# Create subplots
fig = make_subplots(
    rows=n_rows, 
    cols=n_cols,
    subplot_titles=[f'Layer {layer}' for layer in layers],
    vertical_spacing=0.10,
    horizontal_spacing=0.06
)

# Add traces for each layer
for idx, layer in enumerate(layers):
    row = idx // n_cols + 1
    col = idx % n_cols + 1
    
    # Filter data for this layer
    layer_data = df[df['layer'] == layer].sort_values('step_k')
    
    # Skip if no data
    if len(layer_data) == 0:
        continue
    
    # Main line plot (no markers for cleaner view)
    fig.add_trace(
        go.Scatter(
            x=layer_data['step_k'],
            y=layer_data['auroc'],
            mode='lines',
            line=dict(
                color=colors[idx % len(colors)],
                width=2,
                dash='solid'
            ),
            name=f'Layer {layer}',
            showlegend=False,
            hovertemplate="<b>Step %{customdata[0]}</b><br>" +
                         "Tokens: %{customdata[1]}B<br>" +
                         "AUROC: %{y:.4f}<extra></extra>",
            customdata=np.column_stack((layer_data['step'], layer_data['tokens']))
        ),
        row=row, 
        col=col
    )
    
    # Add a trend line if we have enough data points
    if len(layer_data) > 3:
        try:
            z = np.polyfit(layer_data['step_k'], layer_data['auroc'], 2)
            p = np.poly1d(z)
            x_trend = np.linspace(layer_data['step_k'].min(), layer_data['step_k'].max(), 50)
            y_trend = p(x_trend)
            
            fig.add_trace(
                go.Scatter(
                    x=x_trend,
                    y=y_trend,
                    mode='lines',
                    line=dict(color='rgba(100, 100, 100, 0.2)', width=1.5, dash='dash'),
                    showlegend=False,
                    hoverinfo='skip'
                ),
                row=row,
                col=col
            )
        except:
            pass  # Skip trend line if fitting fails

# Update layout
fig.update_layout(
    title=dict(
        text="OLMo AUROC Performance by Layer Across Training Steps",
        font=dict(size=24, color='black', family='Arial'),
        x=0.5,
        xanchor='center'
    ),
    width=2000,
    height=350 * n_rows,
    plot_bgcolor='#FAFAF7',
    paper_bgcolor='#FAFAF7',
    font=dict(color='black', size=10),
    margin=dict(l=60, r=60, t=100, b=60),
    showlegend=False
)

# Update all x and y axes
for i in range(1, n_layers + 1):
    fig.update_xaxes(
        title_text="Step (k)" if i > (n_rows - 1) * n_cols else "",
        title_font=dict(size=12),
        tickfont=dict(size=10),
        linecolor='black',
        linewidth=1.5,
        gridcolor='#E5E5E5',
        gridwidth=0.8,
        showgrid=True,
        zeroline=False,
        mirror=False,
        ticks='outside',
        ticklen=3,
        showline=True,
        row=(i - 1) // n_cols + 1,
        col=(i - 1) % n_cols + 1
    )
    
    fig.update_yaxes(
        title_text="AUROC" if (i - 1) % n_cols == 0 else "",
        title_font=dict(size=12),
        tickfont=dict(size=10),
        linecolor='black',
        linewidth=1.5,
        gridcolor='#E5E5E5',
        gridwidth=0.8,
        showgrid=True,
        zeroline=False,
        mirror=False,
        ticks='outside',
        ticklen=3,
        showline=True,
        range=[0.0, 1.0],
        row=(i - 1) // n_cols + 1,
        col=(i - 1) % n_cols + 1
    )

app = Dash()

app.layout = [
    html.H1(
        children='OLMo AUROC Performance by Layer', 
        style={
            'textAlign': 'center', 
            'color': 'black', 
            'fontSize': '28px', 
            'fontFamily': 'Arial, sans-serif',
            'marginBottom': '20px'
        }
    ),
    html.P(
        children=f'Showing AUROC across {df["step"].nunique()} training steps for {n_layers} layers',
        style={
            'textAlign': 'center',
            'color': '#666',
            'fontSize': '14px',
            'fontFamily': 'Arial, sans-serif',
            'marginBottom': '30px'
        }
    ),
    dcc.Graph(figure=fig)
]

# Optionally save the figure as a static image
try:
    output_path = '/Users/iansu/Maheep-INN/visualization/Olmo_checkpoints/olmo_layers_visualization.pdf'
    fig.write_image(output_path, width=2000, height=350 * n_rows)
    print(f"\nSaved static visualization to: {output_path}")
except Exception as e:
    print(f"\nNote: Could not save static image (this is optional): {e}")
    print("To enable image export, install kaleido: pip install kaleido")

if __name__ == '__main__':
    app.run(debug=True, port=8054)

