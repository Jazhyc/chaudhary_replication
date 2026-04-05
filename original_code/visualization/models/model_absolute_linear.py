"""This file is used to plot the AUROC absolute distance for all models we tested.
These models come from the csv file 'all_models.csv'"""
from dash import Dash, html, dcc
import plotly.graph_objects as go
import pandas as pd
import numpy as np

df = pd.read_csv('./visualization/models/datasets/no_qwen_absolute.csv')

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

# Calculate layer ratio and scale it for marker sizes (smaller points)
df['layer_ratio'] = df['best_layer'] / df['Layers']
min_size, max_size = 14, 14
df['marker_size'] = min_size + (df['layer_ratio'] * (max_size - min_size))

# Add light horizontal jittering only
np.random.seed(42)
df['size_jittered'] = df['size_numeric'] + np.random.normal(0, 0.2, len(df))
df['AUROC_jittered'] = df['AUROC']  # No vertical jitter

app = Dash()

# Use single color for all points
single_color = 'orange'

# Define symbol mapping for model families
symbol_map = {
    'Llama-3.1-Instruct': 'circle',
    'Qwen-2.5-Instruct': 'triangle-up',
    'GPT-OSS': 'square',
    'Gemma-3': 'diamond',
    'Phi-3': 'star'
}

fig = go.Figure()

# Add scatter plots - one trace per model family
for model_family in df['model_family'].unique():
    subset = df[df['model_family'] == model_family]
    
    fig.add_trace(go.Scatter(
        x=subset['size_jittered'],
        y=subset['AUROC'],
        mode='markers',
        marker=dict(
            color=single_color,
            symbol=symbol_map[model_family],
            size=subset['marker_size'],
            line=dict(width=1.5, color='black'),
            opacity=1.0
        ),
        name=model_family,
        showlegend=True,
        hovertemplate=f"Model: {model_family}<br>" +
                     "Size: %{x:.1f}B<br>" +
                     "AUROC Distance: %{y:.3f}<extra></extra>"
    ))

# Add text labels (no legend entry)
# Use additive offset for linear scale - add constant value to position labels to the right
text_offset = 4.5  # Add 4.5 billion parameters to position labels to the right

# Create custom y positions - lower the Phi-3-mini-4k-instruct label
text_y_positions = []
for i, model in enumerate(df['model']):
    if 'Phi-3-mini-4k-instruct' in model:
        # Lower this specific label by 0.03
        text_y_positions.append(df['AUROC_jittered'].iloc[i] - 0.012)
    else:
        text_y_positions.append(df['AUROC_jittered'].iloc[i])

fig.add_trace(go.Scatter(
    x=df['size_jittered'] + text_offset,
    y=text_y_positions,
    mode='text',
    text=[f"L{layer}" for layer in df['best_layer']],
    textfont=dict(color='black', size=14, family='Arial'),
    showlegend=False,
    hoverinfo='skip'
))

# Update layout
fig.update_layout(
    title=dict(
        text="AUROC Absolute Distance from 0.5 on Simple Contrastive",
        font=dict(size=20, color='black'),
        x=0.5,
        xanchor='center'
    ),
    width=600,
    height=600,
    plot_bgcolor='#FAFAF7',
    paper_bgcolor='#FAFAF7',
    font=dict(color='black', size=11),
    legend=dict(
        font=dict(size=13, color='black', family='Arial'),
        orientation="h",
        yanchor="top",
        y=1,
        xanchor="left",
        x=0,
        bgcolor='#EDEDE8',
        borderwidth=0,
        itemsizing="constant",
        itemwidth=50,
        traceorder="normal",
        entrywidth=0.475, #This is the width of the legend entry
        entrywidthmode='fraction'
    ),
    margin=dict(l=80, r=120, t=80, b=80),
    xaxis=dict(
        type="linear",
        range=[0, 80],  # Linear range from 0 to 80 billion parameters
        title=dict(text="Model Parameters (Billions)", font=dict(size=18, color='black', family='Arial')),
        tickfont=dict(size=13, color='black', family='Arial'),
        linecolor='black',
        linewidth=1.5,
        gridcolor='#E5E5E5',
        gridwidth=1,
        showgrid=True,
        zeroline=False,
        mirror=False,
        ticks='outside',
        ticklen=5,
        dtick=10,  # Show grid lines every 10 billion parameters
        minor=dict(
            ticklen=0,
            showgrid=False,  # Disable minor grid lines for cleaner look
            gridcolor='#F0F0F0',
            gridwidth=0.5
        )
    ),
    yaxis=dict(
        range=[-0.02, 0.5],
        title=dict(text="AUROC Absolute Distance from 0.5", font=dict(size=18, color='black', family='Arial')),
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
    )
)

# Add plot borders - only x and y axis (bottom and left)
fig.update_xaxes(mirror=False, ticks='outside', showline=True, linecolor='black')
fig.update_yaxes(mirror=False, ticks='outside', showline=True, linecolor='black')

app.layout = [
    html.H1(children='AUROC Absolute Distance from 0.5 on Simple Contrastive', 
            style={'textAlign':'center', 'color': 'black', 'fontSize': '24px', 'fontFamily': 'Arial, sans-serif'}),
    dcc.Graph(figure=fig)
]

if __name__ == '__main__':
    app.run(debug=True)