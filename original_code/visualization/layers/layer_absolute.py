"""Clean visualization approaches for AUROC data"""
import plotly.graph_objects as go
import plotly.subplots as sp
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import os
import re
from pathlib import Path
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d

def detect_model_info(file_path):
    """Auto-detect model information from file path"""
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

def extract_model_size(filename):
    """Extract model size for sorting"""
    filename_str = str(filename).lower()
    
    # Look for patterns like 0.5B, 1.5B, 32B, etc.
    match = re.search(r'(\d+\.?\d*)b', filename_str)
    if match:
        return float(match.group(1))
    
    # Look for patterns like 410m, 270m, etc. (convert to B)
    match = re.search(r'(\d+)m', filename_str)
    if match:
        return float(match.group(1)) / 1000
    
    # Look for patterns like 20B_Simple, etc.
    match = re.search(r'(\d+)b_', filename_str)
    if match:
        return float(match.group(1))
    
    return 0

def get_model_colors(num_models):
    """Generate color palette based on number of models"""
    # Define full color progression from peach to red
    full_palette = [
        'rgb(255, 218, 185)',  # Peach
        'rgb(255, 200, 150)',  # Light peach
        'rgb(255, 180, 120)',  # Peach-orange
        'rgb(255, 165, 65)',   # Orange
        'rgb(255, 140, 40)',   # Orange-red
        'rgb(255, 100, 20)',   # Red-orange
        'rgb(255, 50, 10)',    # Deep red-orange
        'rgb(255, 0, 0)'       # Pure red
    ]
    

    # Multiple models get evenly distributed colors from the palette
    # Take colors from index 0 to -1 (excluding the last one for single model case)
    available_colors = full_palette[-num_models:]  
    
    if num_models <= len(available_colors):
        # If we have enough predefined colors, select evenly spaced ones
        step = len(available_colors) / num_models
        selected_indices = [int(i * step) for i in range(num_models)]
        return [available_colors[i] for i in selected_indices]
    

def smooth_data(y_data, sigma=1.5):
    """Apply Gaussian smoothing to reduce noise"""
    return gaussian_filter1d(y_data, sigma=sigma)

def process_csv_data(csv_path):
    """Load and process CSV data"""
    df = pd.read_csv(csv_path)
    
    # Check if file has headers
    first_cell = str(df.iloc[0, 0]).strip()
    if 'model' in first_cell.lower() or first_cell.lower() == 'model':
        df.columns = df.columns.str.strip()
        df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    else:
        if len(df.columns) == 5:
            df.columns = ['Model', 'Size', 'Layer', 'AUROC', 'Max_Layer']
        elif len(df.columns) == 3:
            df.columns = ['Model', 'Layer', 'AUROC']
        else:
            df.columns = ['Model', 'Layer', 'AUROC'] + [f'Col_{i}' for i in range(3, len(df.columns))]
        
        df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    
    # Calculate metrics
    df['score_minus_half'] = abs(df['AUROC'] - 0.5)
    max_layer = df['Layer'].max()
    df['relative_depth'] = df['Layer'] / max_layer
    
    return df

def create_heatmap(model_families, output_dir):
    """Create heatmap showing all families and sizes"""
    
    # Collect all data with size information
    all_entries = []
    
    for family, files in model_families.items():
        for i, csv_path in enumerate(files):
            df = process_csv_data(csv_path)
            model_name = Path(csv_path).stem
            size = extract_model_size(model_name)
            
            # Create interpolated data for consistent grid
            depth_points = np.linspace(0, 1, 50)  # 50 points from 0 to 1
            interp_func = interp1d(df['relative_depth'], df['score_minus_half'], 
                                 kind='linear', fill_value='extrapolate')
            values = interp_func(depth_points)
            
            all_entries.append({
                'family': family,
                'size': size,
                'values': values,
                'label': f"{family}-{size}B" if size > 0 else f"{family}-Unknown"
            })
    
    # Sort by size first, then by family
    all_entries.sort(key=lambda x: (x['size'], x['family']))
    
    # Extract sorted data
    all_data = [entry['values'] for entry in all_entries]
    model_sizes = [entry['label'] for entry in all_entries]
    
    # Create custom colorscale to match your warm theme
    warm_colorscale = [
        [0.0, 'rgb(255, 248, 235)'],    # Very light cream (low values)
        [0.2, 'rgb(255, 230, 210)'],    # Light peach
        [0.4, 'rgb(255, 200, 150)'],    # Light orange
        [0.6, 'rgb(255, 165, 65)'],     # Orange  
        [0.8, 'rgb(255, 100, 20)'],     # Red-orange
        [1.0, 'rgb(139, 69, 19)']       # Brown (matching your Family Mean color)
    ]
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=all_data,
        x=depth_points,
        y=model_sizes,
        colorscale=warm_colorscale,
        hoverongaps=False,
        colorbar=dict(
            title=dict(
                text="AUROC Distance",
                font=dict(family="Times New Roman", size=18, color='black')
            ),
            tickfont=dict(family="Times New Roman", size=16, color='black')
            # Removed 'titleside' - not a valid property
        )
    ))
    
    fig.update_layout(
        title=dict(
            text="AUROC Distance Heatmap: All Model Families and Sizes",
            font=dict(family="Times New Roman", size=26, color='black')
        ),
        xaxis=dict(
            title=dict(
                text="Relative Layer Depth",
                font=dict(family="Times New Roman", size=20, color='black')
            ),
            tickfont=dict(family="Times New Roman", size=18, color='black')
        ),
        yaxis=dict(
            title=dict(
                text="Model Family-Size",
                font=dict(family="Times New Roman", size=20, color='black')
            ),
            tickfont=dict(family="Times New Roman", size=18, color='black')
        ),
        width=1200, 
        height=800,
        font=dict(family="Times New Roman", size=16),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    fig.write_image(str(output_dir / "heatmap_all_models.pdf"), width=1200, height=800, engine="kaleido")
    print("Saved: heatmap_all_models.pdf")



def create_small_multiples(csv_files, family_name, output_dir):
    """Create small multiples - one subplot per model"""
    
    files = sorted(csv_files, key=lambda x: extract_model_size(x.stem))
    n_models = len(files)
    
    # Calculate grid dimensions
    cols = min(3, n_models)
    rows = (n_models + cols - 1) // cols
    
    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=[Path(f.stem).stem for f in files],
        shared_xaxes=True, shared_yaxes=True,
        vertical_spacing=0.08, horizontal_spacing=0.05
    )
    
    # Get hardcoded colors
    colors = get_model_colors(n_models)
    
    for i, csv_path in enumerate(files):
        df = process_csv_data(csv_path)
        color = colors[i] if i < len(colors) else colors[-1]
        
        # Smooth the data
        smoothed_y = smooth_data(df['score_minus_half'].values)
        
        row = (i // cols) + 1
        col = (i % cols) + 1
        
        fig.add_trace(
            go.Scatter(
                x=df['relative_depth'],
                y=smoothed_y,
                mode='lines',
                line=dict(color=color, width=3),
                name=Path(csv_path).stem,
                showlegend=False
            ),
            row=row, col=col
        )
    
    fig.update_layout(
        title=f"{family_name} Family: Individual Model Patterns",
        height=200 * rows + 100,
        width=1200,
        plot_bgcolor='white'
    )
    
    fig.update_xaxes(title_text="Relative Layer Depth", row=rows)
    fig.update_yaxes(title_text="AUROC Distance", col=1)
    
    fig.write_image(str(output_dir / f"{family_name}_small_multiples.pdf"), 
                   width=1200, height=200*rows+100, engine="kaleido")
    print(f"Saved: {family_name}_small_multiples.pdf")

def create_summary_stats(csv_files, family_name, output_dir):
    """Create summary statistics plot showing mean and trend"""
    
    files = sorted(csv_files, key=lambda x: extract_model_size(x.stem))
    
    fig = go.Figure()
    
    # Collect all data for statistics
    all_depths = []
    all_scores = []
    model_data = {}
    
    # Get hardcoded colors
    colors = get_model_colors(len(files))
    
    for i, csv_path in enumerate(files):
        df = process_csv_data(csv_path)
        model_name = Path(csv_path).stem
        size = extract_model_size(model_name)
        
        # Use hardcoded color
        color = colors[i] if i < len(colors) else colors[-1]
        
        # Store for individual lines
        model_data[model_name] = {
            'depth': df['relative_depth'].values,
            'score': smooth_data(df['score_minus_half'].values),
            'size': size,
            'color': color
        }
        
        all_depths.extend(df['relative_depth'].values)
        all_scores.extend(df['score_minus_half'].values)
    
    # Calculate overall statistics
    depth_bins = np.linspace(0, 1, 20)
    bin_means = []
    bin_stds = []
    
    for i in range(len(depth_bins)-1):
        mask = (np.array(all_depths) >= depth_bins[i]) & (np.array(all_depths) < depth_bins[i+1])
        if np.any(mask):
            bin_means.append(np.mean(np.array(all_scores)[mask]))
            bin_stds.append(np.std(np.array(all_scores)[mask]))
        else:
            bin_means.append(0)
            bin_stds.append(0)
    
    bin_centers = (depth_bins[:-1] + depth_bins[1:]) / 2
    
    # Add confidence band for mean
    upper_bound = np.array(bin_means) + np.array(bin_stds)
    lower_bound = np.maximum(np.array(bin_means) - np.array(bin_stds), 0)
    
    fig.add_trace(go.Scatter(
        x=bin_centers, y=upper_bound,
        mode='lines', line=dict(width=0),
        showlegend=False, hoverinfo='skip'
    ))
    
    fig.add_trace(go.Scatter(
        x=bin_centers, y=lower_bound,
        mode='lines', line=dict(width=0),
        # fill='tonexty', fillcolor='rgba(128, 128, 128, 0.2)',
        fill='tonexty', fillcolor='rgba(255, 230, 210, 0.3)',  # Very light peach
        showlegend=False, hoverinfo='skip'
    ))
    
    # Add mean line in deep navy blue
    fig.add_trace(go.Scatter(
        x=bin_centers, y=bin_means,
        mode='lines+markers',
        line=dict(color='#8B4513', width=4),  # Deep navy blue color
        marker=dict(color='#8B4513', size=8),
        name='Family Mean'
    ))
    
    # Add individual model lines (thinner)
    for model_name, data in model_data.items():
        if data['size'] > 0:
            size_label = f"{data['size']}B"
        elif "mini" in model_name.lower():
            size_label = "Mini"
        elif "medium" in model_name.lower():
            size_label = "Medium"
        else:
            size_label = "Unknown"
        fig.add_trace(go.Scatter(
            x=data['depth'], y=data['score'],
            mode='lines',
            line=dict(color=data['color'], width=2, dash='dot'),
            name=size_label,
            opacity=0.7
        ))
        
    fig.update_layout(
        title=dict(
            text=f"{family_name} Model Family Analysis",
            font=dict(family="Times New Roman", size=26, color='black')  # Increased from 20
        ),
        xaxis=dict(
            title=dict(
                text="Relative Layer Depth",
                font=dict(family="Times New Roman", size=20, color='black')  # Increased from 16
            ),
            tickfont=dict(family="Times New Roman", size=18, color='black')  # Increased from 14
        ),
        yaxis=dict(
            title=dict(
                text="AUROC Distance from 0.5",
                font=dict(family="Times New Roman", size=20, color='black')  # Increased from 16
            ),
            tickfont=dict(family="Times New Roman", size=18, color='black')  # Increased from 14
        ),
        width=1000, 
        height=600,
        plot_bgcolor='white',
        legend=dict(
            x=0.02, 
            y=0.98,
            font=dict(family="Times New Roman", size=16, color='black'),
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='#8B4513',
            borderwidth=1,
            # Add these lines to change symbols to lines:
            itemsizing='constant',
            itemwidth=40,        # Width of legend line symbols
            tracegroupgap=5      # Spacing between legend items
        ),
        font=dict(family="Times New Roman")
    )


    
    fig.write_image(str(output_dir / f"{family_name}_summary_stats.pdf"), 
                   width=1000, height=600, engine="kaleido")
    print(f"Saved: {family_name}_summary_stats.pdf")

def main():
    datasets_dir = Path("visualization/layers/datasets")
    figures_dir = Path("figures")
    figures_dir.mkdir(exist_ok=True)
    
    # Group files by model family
    model_families = {
        "Gemma-3-it": [],
        "Llama-3.1-Instruct": [],
        "Phi-3": [],
        "GPT-OSS": [], 
        # "Qwen-2.5-Instruct": []
    }

    # Look for CSV files in ALL subdirectories recursively
    csv_files = list(datasets_dir.rglob("*.csv"))
    print(f"Found {len(csv_files)} CSV files total")  # DEBUG
    
    for csv_file in csv_files:
        # Check the parent directory name instead of just filename
        parent_dir = csv_file.parent.name
        filename = csv_file.name.lower()
        
        print(f"Processing file: {csv_file.relative_to(datasets_dir)}")  # DEBUG
        print(f"Parent directory: {parent_dir}")  # DEBUG
        
        # Match based on parent directory name
        if "gemma" in parent_dir.lower():
            model_families["Gemma-3-it"].append(csv_file)
        elif "llama" in parent_dir.lower():
            model_families["Llama-3.1-Instruct"].append(csv_file)
        elif "phi" in parent_dir.lower():
            model_families["Phi-3"].append(csv_file)
        elif "gpt" in parent_dir.lower():
            print(f"Found GPT file: {csv_file.relative_to(datasets_dir)}")  # DEBUG
            model_families["GPT-OSS"].append(csv_file)
    
    # Print what was found for each family
    for family, files in model_families.items():
        print(f"{family}: {len(files)} files")  # DEBUG
        for f in files:
            print(f"  - {f.relative_to(datasets_dir)}")
    
    # Filter out empty families
    model_families = {k: v for k, v in model_families.items() if v}
    
    print("Creating visualizations...")
    # ... rest of your code remains the same

    
    # 1. Create heatmap for all models
    print("Creating heatmap...")
    create_heatmap(model_families, figures_dir)
    
    # 2. Create small multiples and summary stats for each family
    for family, files in model_families.items():
        if family == 'Unknown':
            continue
            
        print(f"Processing {family} family...")
        
        # Small multiples
        create_small_multiples(files, family, figures_dir)
        
        # Summary statistics
        create_summary_stats(files, family, figures_dir)
    
    print(f"\nAll visualizations created! Check the '{figures_dir}' folder for outputs.")

if __name__ == "__main__":
    main()
