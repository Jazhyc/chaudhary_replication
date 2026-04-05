"""This file is used to plot the AUROC absolute distance for all models we tested.
These models come from the csv file 'all_models.csv'"""
from dash import Dash, html, dcc
import plotly.graph_objects as go
import pandas as pd
import numpy as np

df = pd.read_csv('/Users/iansu/Maheep-INN/visualization/models/datasets/no_qwen_absolute.csv')

# Clean up whitespace in all string columns
df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

# Convert size to numerical values for proper sorting
df['size_numeric'] = df['size'].str.replace('B', '').astype(float)
df = df.sort_values('size_numeric')

# Add model family column
def get_model_family(model_name):
    if 'LLaMA' in model_name:
        return 'Llama-3.1-Instruct'
    elif 'Qwen' in model_name:
        return 'Qwen-2.5-Instruct'
    elif 'Gpt-oss' in model_name:
        return 'GPT-OSS'
    elif 'gemma' in model_name:
        return 'Gemma-3'
    elif 'Phi' in model_name:
        return 'Phi-3'
    else:
        return 'Other'

df['model_family'] = df['model'].apply(get_model_family)

# Calculate layer ratio and scale it for marker sizes (2x bigger due to fewer data points)
df['layer_ratio'] = df['best_layer'] / df['Layers']
min_size, max_size = 14, 14
df['marker_size'] = min_size + (df['layer_ratio'] * (max_size - min_size))

# Add light horizontal jittering only
np.random.seed(42)
df['size_jittered'] = df['size_numeric'] # + np.random.normal(0, 0.2, len(df))
df['AUROC_jittered'] = df['AUROC']  # No vertical jitter

# app = Dash()

# CHANGED: Warm color map for model families instead of single orange
# model_colors = {
#     'Llama-3.1-Instruct': 'rgb(139, 69, 19)',      # Orange
#     'Qwen-2.5-Instruct': 'rgb(255, 100, 20)',       # Red-orange
#     'GPT-OSS': 'rgb(255, 165, 65)',                   # Brown
#     'Gemma-3': 'rgb(255, 200, 150)',                 # Light peach
#     'Phi-3': 'rgb(222, 184, 135)'                    # Burlywood
# }

model_colors = {
    'Llama-3.1-Instruct': '#8B4513',    # Saddle brown (darkest for contrast)
    'Qwen-2.5-Instruct': '#CD853F',    # Peru (medium brown)
    'GPT-OSS': '#D2691E',              # Chocolate (warm orange-brown)
    'Gemma-3': '#DAA520',              # Goldenrod (gold tone)
    'Phi-3': '#B8860B'                 # Dark goldenrod (deeper gold)
}

# Enhanced symbol mapping with more distinct shapes
symbol_map = {
    'Llama-3.1-Instruct': 'circle',
    'Qwen-2.5-Instruct': 'triangle-up',
    'GPT-OSS': 'x',                    # X symbol
    'Gemma-3': 'cross',                # Plus/cross symbol
    'Phi-3': 'star'
}

fig = go.Figure()

import statsmodels.api as sm 

x_original = df['size_numeric'].values
y = df['AUROC'].values


from scipy.optimize import curve_fit

# Define logarithmic function: y = a * log(x) + b
def log_func(x, a, b):
    return a * np.log10(x) + b

from scipy.optimize import curve_fit

# Define square root function: y = a * sqrt(x) + b
def sqrt_func(x, a, b):
    return a * np.sqrt(x) + b

# Fit the function
popt, _ = curve_fit(sqrt_func, x_original, y)
# x_smooth = np.linspace(x_original.min(), x_original.max(), 200)
# y_smooth = sqrt_func(x_smooth, *popt)
# Since your x-axis range is [-0.699, 2.3] in log scale (which is ~0.2 to ~200)
x_smooth = np.linspace(0.2, 200, 200)
y_smooth = sqrt_func(x_smooth, *popt)

fig.add_trace(go.Scatter(
    x=x_smooth,
    y=y_smooth,
    mode='lines',
    line=dict(color='#8B2635', width=5, dash='dash'),
    name='Square-root Fit ( y = a√x + b)',
    showlegend=True
))


# Set x-axis type to linear
fig.update_xaxes(type='linear')

# Add scatter plots - one trace per model family
for model_family in df['model_family'].unique():
    subset = df[df['model_family'] == model_family]
    
    fig.add_trace(go.Scatter(
        x=subset['size_jittered'],
        y=subset['AUROC'],
        mode='markers',
        marker=dict(
            color=model_colors.get(model_family, 'gray'),
            symbol=symbol_map[model_family],
            size=22,
            line=dict(width=2.5, color='white'),  # ENHANCED: Slightly thicker white border
            opacity=0.9                           # ENHANCED: Slightly higher opacity
        ),
        name=model_family,
        showlegend=True,
        hovertemplate=f"Model: {model_family}<br>" +
                    "Size: %{x:.1f}B<br>" +
                    "AUROC Distance: %{y:.3f}<extra></extra>"
    ))

# Add text labels (no legend entry)
# Use proportional offset for log scale - multiply by 1.3 to position labels to the right
# text_offset_multiplier = 1.5
# fig.add_trace(go.Scatter(
#     x=df['size_jittered'] * text_offset_multiplier,
#     y=df['AUROC_jittered'],
#     mode='text',
#     text=[f"L{layer}" for layer in df['best_layer']],
#     textfont=dict(color='black', size=14, family='Times New Roman'),  # CHANGED: Times New Roman
#     showlegend=False,
#     hoverinfo='skip'
# ))

# Update layout
fig.update_layout(
    # title=dict(
    #     text="Scaling Law vs Model's Evaluation Awareness Analysis",
    #     font=dict(size=26, color='black', family='Times New Roman'),  # CHANGED: Times New Roman
    #     x=0.5,
    #     xanchor='center'
    # ),
    width=600,
    height=600,
    plot_bgcolor='#FFFEF7',
    paper_bgcolor='white',
    font=dict(color='black', size=30, family='Times New Roman'),  # CHANGED: Times New Roman
    legend=dict(
        font=dict(size=18, color='black', family='Times New Roman, serif'),
        bgcolor='rgba(255, 255, 255, 0.95)',
        bordercolor='#CCCCCC',
        borderwidth=1,
        orientation="v",
        yanchor="top",
        y=0.95,
        xanchor="left", 
        x=0.02,
        itemsizing="constant",
        itemwidth=60,
        tracegroupgap=15,           # ENHANCED: Increased from 10 to 15 for more spacing
        itemclick="toggleothers",
        itemdoubleclick="toggle"
    ),
    margin=dict(l=80, r=0, t=0, b=80),
    # Enhanced x-axis with subtle grid
    xaxis=dict(
        type="log",
        range=[-0.699, 2.3],
        title=dict(
            text="<b>Model Parameters (Billions)</b>", 
            font=dict(size=22, color='black', family='Times New Roman, serif')
        ),
        tickfont=dict(size=18, color='black', family='Times New Roman, serif'),
        linecolor='black',
        gridcolor='#F0F0F0',      # ENHANCED: Very subtle light gray grid
        linewidth=2,
        gridwidth=0.8,            # ENHANCED: Thin grid lines
        showgrid=True,            # ENHANCED: Enable grid
        zeroline=False,
        mirror=False,
        ticks='outside',
        ticklen=8,
        tickvals=[0.2, 1, 10, 100],
        ticktext=['0.2', '10⁰', '10¹', '10²'],
        dtick=1,
        minor=dict(
            ticklen=4,
            showgrid=True,
            gridcolor='#F8F8F8',     # ENHANCED: Even lighter for minor grid
            gridwidth=0.4
        )
    ),

    # Enhanced y-axis with subtle grid
    yaxis=dict(
        range=[-0.02, 0.5],
        title=dict(
            text="<b>AUROC Absolute Distance from 0.5</b>", 
            font=dict(size=22, color='black', family='Times New Roman, serif')
        ),
        tickfont=dict(size=18, color='black', family='Times New Roman, serif'),
        linecolor='black',
        gridcolor='#F0F0F0',      # ENHANCED: Subtle grid lines
        linewidth=2,
        gridwidth=0.8,            # ENHANCED: Thin grid lines
        showgrid=True,            # ENHANCED: Enable grid
        zeroline=False,
        mirror=False,
        ticks='outside',
        ticklen=8,
        minor=dict(
            ticklen=4,
            showgrid=True,
            gridcolor='#F8F8F8',     # ENHANCED: Very light minor grid
            gridwidth=0.4
        )
    )
)

# Add plot borders - only x and y axis (bottom and left)
fig.update_xaxes(mirror=False, ticks='outside', showline=True, linecolor='black')
fig.update_yaxes(mirror=False, ticks='outside', showline=True, linecolor='black')

fig.write_image("model_auroc_absolute_distance_new.pdf", 
                   width=800, height=500, engine="kaleido")

