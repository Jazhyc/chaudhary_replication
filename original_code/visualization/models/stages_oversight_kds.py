"""This file is used to plot the lowest KDS score vs model size for stages_oversight results."""

from dash import Dash, html, dcc
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import os
import glob

# Define model sizes (in billions of parameters)
model_sizes = {
    'qwen2.5-0.5b': 0.5,
    'qwen2.5-1.5b': 1.5,
    'qwen2.5-3b': 3.0,
    'qwen2.5-7b': 7.0,
    'qwen2.5-32b': 32.0,
    'gemma-3-0.27b-it': 0.27,
    'gemma-3-1b-it': 1.0,
    'gemma-3-4b-it': 4.0,
    'gemma-3-12b-it': 12.0,
    'gemma-3-27b-it': 27.0,
    'phi-3-mini-4k-instruct': 3.8,
    'phi-3-medium-4k-instruct': 14.0,
}

# Define model families
def get_model_family(model_name):
    if 'qwen' in model_name.lower():
        return 'Qwen-2.5'
    elif 'gemma' in model_name.lower():
        return 'Gemma-3'
    elif 'phi' in model_name.lower():
        return 'Phi-3'
    else:
        return 'Other'

# Read all TSV files in the stages_oversight_folder
data_folder = './stages_oversight_folder'
results = []
processed_models = set()  # Track which models we've already processed

for tsv_file in glob.glob(os.path.join(data_folder, '*.tsv')):
    # Read the TSV file
    df_temp = pd.read_csv(tsv_file, sep='\t', header=None)
    
    # Extract model name from filename or from the data
    if len(df_temp) > 0:
        # Extract model name from the second column
        first_entry = df_temp[1].iloc[0]
        model_name = first_entry.split('_')[0]
        
        # Skip if we've already processed this model
        if model_name in processed_models:
            continue
        
        # Extract contamination levels and KDS scores
        contamination_levels = []
        kds_scores = []
        
        for _, row in df_temp.iterrows():
            # Parse the contamination level from the experiment name (column 1)
            exp_name = row[1]
            # Extract contamination value (e.g., "0.0", "0.05", etc.)
            if 'contamination=' in str(row[3]):
                # Parse from the Namespace string
                import re
                match = re.search(r'contamination=([\d.]+)', str(row[3]))
                if match:
                    contamination = float(match.group(1))
                    contamination_levels.append(contamination)
                    kds_scores.append(row[2])
        
        # Sort by contamination level
        if len(contamination_levels) > 0:
            sorted_pairs = sorted(zip(contamination_levels, kds_scores))
            contamination_levels = [x[0] for x in sorted_pairs]
            kds_scores = [x[1] for x in sorted_pairs]
            
            # Calculate AUC using trapezoidal rule
            # AUC = sum of trapezoid areas between consecutive points
            auc = 0
            for i in range(len(contamination_levels) - 1):
                x1, x2 = contamination_levels[i], contamination_levels[i + 1]
                y1, y2 = kds_scores[i], kds_scores[i + 1]
                # Trapezoid area = (x2 - x1) * (y1 + y2) / 2
                auc += (x2 - x1) * (y1 + y2) / 2
            
            # Try to match with known model sizes
            size = None
            for known_model, known_size in model_sizes.items():
                if known_model in model_name.lower() or model_name.lower() in known_model:
                    size = known_size
                    break
            
            if size is not None:
                results.append({
                    'model': model_name,
                    'size': size,
                    'auc': auc,
                    'model_family': get_model_family(model_name),
                    'num_points': len(kds_scores)  # Track how many data points
                })
                processed_models.add(model_name)
                print(f"Processed {model_name}: {len(kds_scores)} points, AUC = {auc:.3f}")

# Create DataFrame
df = pd.DataFrame(results)
df = df.sort_values('size')

# Add light jittering for better visualization
np.random.seed(42)
df['size_jittered'] = df['size'] + np.random.normal(0, 0.15, len(df))
df['auc_jittered'] = df['auc'] + np.random.normal(0, 0.1, len(df))

app = Dash()

# Define colors for model families
color_map = {
    'Qwen-2.5': '#FF6B6B',
    'Gemma-3': '#4ECDC4',
    'Phi-3': '#95E1D3'
}

# Define symbol mapping for model families
symbol_map = {
    'Qwen-2.5': 'triangle-up',
    'Gemma-3': 'circle',
    'Phi-3': 'square'
}

fig = go.Figure()

# Add scatter plots - one trace per model family
for model_family in df['model_family'].unique():
    subset = df[df['model_family'] == model_family]
    
    fig.add_trace(go.Scatter(
        x=subset['size_jittered'],
        y=subset['auc'],
        mode='markers',
        marker=dict(
            color=color_map.get(model_family, 'gray'),
            symbol=symbol_map.get(model_family, 'circle'),
            size=20,
            line=dict(width=1.5, color='black'),
            opacity=1.0
        ),
        name=model_family,
        showlegend=True,
        hovertemplate=f"<b>{model_family}</b><br>" +
                     "Size: %{x:.2f}B<br>" +
                     "AUC: %{y:.3f}<extra></extra>"
    ))

# Add text labels showing the actual AUC values
text_offset_x = 1.5  # Move labels further to the right
text_offset_y = 0.0  # Keep at same vertical level as points
fig.add_trace(go.Scatter(
    x=df['size_jittered'] + text_offset_x,
    y=df['auc'] + text_offset_y,
    mode='text',
    text=[f"{val:.1f}" for val in df['auc']],
    textfont=dict(color='black', size=12, family='Arial', weight='bold'),
    showlegend=False,
    hoverinfo='skip'
))

# Update layout
fig.update_layout(
    title=dict(
        text="AUC (KDS Score) vs Model Size (Stages Oversight)",
        font=dict(size=20, color='black'),
        x=0.5,
        xanchor='center'
    ),
    width=800,
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
        borderwidth=0,
        itemsizing="constant",
        itemwidth=50,
        traceorder="normal",
        entrywidth=0.3,
        entrywidthmode='fraction'
    ),
    margin=dict(l=80, r=120, t=80, b=80),
    xaxis=dict(
        range=[-1, 35],
        title=dict(text="Model Parameters (Billions)", font=dict(size=18, color='black', family='Arial')),
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
        minor=dict(
            ticklen=3,
            showgrid=True,
            gridcolor='#F0F0F0',
            gridwidth=0.5
        )
    ),
    yaxis=dict(
        range=[-9, 0],  # Start at -9 (bottom) and go to 0 (top)
        title=dict(text="AUC (Area Under Curve)", font=dict(size=18, color='black', family='Arial')),
        tickfont=dict(size=13, color='black', family='Arial'),
        linecolor='black',
        linewidth=1.5,
        gridcolor='#E5E5E5',
        gridwidth=0.8,
        showgrid=True,
        zeroline=True,
        zerolinecolor='black',
        zerolinewidth=2,
        mirror=False,
        ticks='outside',
        ticklen=5,
        minor=dict(
            ticklen=3,
            showgrid=True,
            gridcolor='#F0F0F0',
            gridwidth=0.5
        )
    )
)

# Add plot borders - only x and y axis (bottom and left)
fig.update_xaxes(mirror=False, ticks='outside', showline=True, linecolor='black')
fig.update_yaxes(mirror=False, ticks='outside', showline=True, linecolor='black')

app.layout = [
    html.H1(children='AUC (KDS Score) vs Model Size (Stages Oversight)', 
            style={'textAlign':'center', 'color': 'black', 'fontSize': '24px', 'fontFamily': 'Arial, sans-serif'}),
    dcc.Graph(figure=fig)
]

if __name__ == '__main__':
    app.run(debug=True, port=8051)

