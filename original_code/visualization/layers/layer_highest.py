"""This file is used to plot the AUROC score minus 0.5 for all layers of a particular model."""
from dash import Dash, html, dcc
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import os

# Configuration - Change this line to use different CSV files
CSV_FILE_PATH = '/Users/iansu/Maheep-INN/visualization/layers/datasets/Phi-3/Phi-3-medium-4k-instruct.csv'
# /Users/iansu/Maheep-INN/visualization/layers/datasets/Gemma-3/gemma-3-270m-it.csv
# Read the data
df = pd.read_csv(CSV_FILE_PATH)

# Check if file has headers by looking at first row
first_cell = str(df.iloc[0, 0]).strip()
if 'model' in first_cell.lower() or first_cell.lower() == 'model':
    # File has headers - clean them up
    df.columns = df.columns.str.strip()
    df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
else:
    # File doesn't have headers - assign column names based on number of columns
    if len(df.columns) == 5:
        df.columns = ['Model', 'Size', 'Layer', 'AUROC', 'Max_Layer']
    elif len(df.columns) == 3:
        df.columns = ['Model', 'Layer', 'AUROC']
    else:
        # Default fallback
        df.columns = ['Model', 'Layer', 'AUROC'] + [f'Col_{i}' for i in range(3, len(df.columns))]
    
    # Clean up data
    df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

# Auto-detect model information from file path
def detect_model_info(file_path):
    if 'Qwen-2.5-Instruct' in file_path:
        return 'Qwen-2.5-Instruct'
    elif 'Llama-3.1-Instruct' in file_path:
        return 'Llama-3.1-Instruct'
    elif 'GPT-OSS' in file_path:
        return 'GPT-OSS'
    elif 'Gemma-3' in file_path:
        return 'Gemma-3'
    elif 'Phi-3' in file_path:
        return 'Phi-3'
    else:
        return 'Unknown'

model_family = detect_model_info(CSV_FILE_PATH)

# Calculate score minus 0.5
df['score_minus_half'] = df['AUROC'] - 0.5

# Get max layer for dynamic x-axis
max_layer = df['Layer'].max()

app = Dash()

# Define colors and symbols for each model family
model_colors = {
    'GPT-OSS': '#FFA500',  # Orange
    'Qwen-2.5-Instruct': '#FF7F50',  # Orange-red  
    'Llama-3.1-Instruct': '#FF6B6B',  # Red
    'Gemma-3': '#9370DB',  # Purple
    'Phi-3': '#00CED1'  # Dark Turquoise
}

model_symbols = {
    'GPT-OSS': 'square',
    'Qwen-2.5-Instruct': 'triangle-up', 
    'Llama-3.1-Instruct': 'circle',
    'Gemma-3': 'diamond',
    'Phi-3': 'star'
}

fig = go.Figure()

# Add trace for the detected model
color = model_colors.get(model_family, '#1f77b4')  # Default blue if unknown
symbol = model_symbols.get(model_family, 'circle')  # Default circle if unknown

fig.add_trace(go.Scatter(
    x=df['Layer'],
    y=df['score_minus_half'],
    mode='lines+markers',
    name=model_family,
    line=dict(color=color, width=3),
    marker=dict(
        color=color,
        symbol=symbol,
        size=8,
        line=dict(width=1, color='white')
    ),
    hovertemplate="Layer: %{x}<br>Score - 0.5: %{y:.3f}<extra></extra>"
))

# Add confidence bands (simulated for demo - replace with actual data if available)
# Create confidence intervals around the line
upper_bound = df['score_minus_half'] + 0.05
lower_bound = df['score_minus_half'] - 0.05
# Allow negative values since we're no longer using absolute distance

# Convert color to rgba format for confidence band
def hex_to_rgba(hex_color, alpha=0.2):
    hex_color = hex_color.lstrip('#')
    rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return f'rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {alpha})'

# Add upper bound (invisible line for fill)
fig.add_trace(go.Scatter(
    x=df['Layer'],
    y=upper_bound,
    mode='lines',
    line=dict(width=0),
    showlegend=False,
    hoverinfo='skip'
))

# Add lower bound with fill
fig.add_trace(go.Scatter(
    x=df['Layer'],
    y=lower_bound,
    mode='lines',
    line=dict(width=0),
    fill='tonexty',
    fillcolor=hex_to_rgba(color),
    showlegend=False,
    hoverinfo='skip'
))

# Update layout to match the style in the image
fig.update_layout(
    title=dict(
        text="Phi-3-medium-4k-instruct",
        font=dict(size=20, color='black', family='Arial'),
        x=0.5,
        xanchor='center'
    ),
    width=1400,
    height=600,
    plot_bgcolor='#FAFAF7',
    paper_bgcolor='#FAFAF7',
    font=dict(color='black', size=14, family='Arial'),
    legend=dict(
        font=dict(size=14, color='black', family='Arial'),
        orientation="h",
        yanchor="top",
        y=1.02,
        xanchor="center",
        x=0.5,
        bgcolor='rgba(255, 255, 255, 0.8)',
        borderwidth=0
    ),
    margin=dict(l=80, r=80, t=100, b=80),
    xaxis=dict(
        range=[0.5, max_layer + 0.5],
        title=dict(text="Layer", font=dict(size=18, color='black', family='Arial')),
        tickfont=dict(size=12, color='black', family='Arial'),
        linecolor='black',
        linewidth=1,
        gridcolor='lightgray',
        gridwidth=1,
        showgrid=True,
        zeroline=False,
        mirror=False,
        ticks='outside',
        ticklen=5,
        dtick=5,
        tickangle=0
    ),
    yaxis=dict(
        range=[-0.5, 0.5],
        title=dict(text="AUROC Score - 0.5", font=dict(size=18, color='black', family='Arial')),
        tickfont=dict(size=14, color='black', family='Arial'),
        linecolor='black',
        linewidth=1,
        gridcolor='lightgray',
        gridwidth=1,
        showgrid=True,
        zeroline=False,
        mirror=False,
        ticks='outside',
        ticklen=5
    )
)

# Add plot borders
fig.update_xaxes(mirror=False, ticks='outside', showline=True, linecolor='black')
fig.update_yaxes(mirror=False, ticks='outside', showline=True, linecolor='black')

app.layout = [
    html.H1(children='AUROC Score - 0.5 on Simple Contrastive', 
            style={'textAlign':'center', 'color': 'black', 'fontSize': '24px', 'fontFamily': 'Arial, sans-serif'}),
    dcc.Graph(figure=fig)
]

if __name__ == '__main__':
    app.run(debug=True)