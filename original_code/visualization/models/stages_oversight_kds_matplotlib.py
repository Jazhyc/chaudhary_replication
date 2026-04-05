"""This file generates a static plot of the lowest KDS score vs model size for stages_oversight results using matplotlib."""

import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

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

for tsv_file in glob.glob(os.path.join(data_folder, '*.tsv')):
    # Read the TSV file
    df_temp = pd.read_csv(tsv_file, sep='\t', header=None)
    
    # Extract model name from filename or from the data
    if len(df_temp) > 0:
        # The third column (index 2) contains the KDS score
        kds_scores = df_temp[2].values
        min_kds = min(kds_scores)  # Get the most negative (lowest) KDS score
        
        # Extract model name from the second column
        first_entry = df_temp[1].iloc[0]
        model_name = first_entry.split('_')[0]
        
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
                'min_kds': min_kds,
                'model_family': get_model_family(model_name)
            })

# Create DataFrame
df = pd.DataFrame(results)
df = df.sort_values('size')

print(f"Found {len(df)} models:")
for _, row in df.iterrows():
    print(f"  {row['model']}: {row['size']}B, KDS: {row['min_kds']:.3f}")

# Save to CSV
csv_output = './visualization/models/datasets/stages_oversight_kds.csv'
df.to_csv(csv_output, index=False)
print(f"\nData saved to: {csv_output}")

# Add light jittering for better visualization
np.random.seed(42)
df['size_jittered'] = df['size'] + np.random.normal(0, 0.15, len(df))

# Convert min_kds to absolute value for better visualization
df['abs_min_kds'] = abs(df['min_kds'])

# Define colors for model families
color_map = {
    'Qwen-2.5': '#FF6B6B',
    'Gemma-3': '#4ECDC4',
    'Phi-3': '#95E1D3'
}

# Define marker mapping for model families
marker_map = {
    'Qwen-2.5': '^',  # triangle up
    'Gemma-3': 'o',   # circle
    'Phi-3': 's'      # square
}

# Create the plot
fig, ax = plt.subplots(figsize=(10, 7))
fig.patch.set_facecolor('#FAFAF7')
ax.set_facecolor('#FAFAF7')

# Plot each model family separately
for model_family in df['model_family'].unique():
    subset = df[df['model_family'] == model_family]
    ax.scatter(
        subset['size_jittered'],
        subset['abs_min_kds'],
        c=color_map.get(model_family, 'gray'),
        marker=marker_map.get(model_family, 'o'),
        s=200,
        edgecolors='black',
        linewidths=1.5,
        label=model_family,
        alpha=1.0,
        zorder=3
    )
    
    # Add text labels
    for _, row in subset.iterrows():
        ax.text(
            row['size_jittered'] + 0.5,
            row['abs_min_kds'] + 0.3,
            f"{row['abs_min_kds']:.1f}",
            fontsize=10,
            ha='left',
            va='bottom',
            fontfamily='Arial'
        )

# Customize the plot
ax.set_xlabel('Model Parameters (Billions)', fontsize=16, fontfamily='Arial', fontweight='bold')
ax.set_ylabel('Absolute Minimum KDS Score', fontsize=16, fontfamily='Arial', fontweight='bold')
ax.set_title('Lowest KDS Score vs Model Size (Stages Oversight)', fontsize=18, fontfamily='Arial', fontweight='bold', pad=20)

# Set axis limits
ax.set_xlim(-1, 35)
ax.set_ylim(0, max(df['abs_min_kds']) * 1.15)

# Grid
ax.grid(True, color='#E5E5E5', linewidth=0.8, zorder=0)
ax.set_axisbelow(True)

# Spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('black')
ax.spines['left'].set_linewidth(1.5)
ax.spines['bottom'].set_color('black')
ax.spines['bottom'].set_linewidth(1.5)

# Tick parameters
ax.tick_params(axis='both', which='major', labelsize=12, length=5, width=1.5, color='black')

# Legend
legend = ax.legend(
    loc='upper left',
    fontsize=12,
    frameon=True,
    fancybox=False,
    edgecolor='black',
    facecolor='#EDEDE8',
    framealpha=1.0
)
legend.get_frame().set_linewidth(0)

plt.tight_layout()

# Save as PDF
output_file = './visualization/models/stages_oversight_kds_vs_size.pdf'
plt.savefig(output_file, format='pdf', dpi=300, bbox_inches='tight', facecolor='#FAFAF7')
print(f"Plot saved to: {output_file}")

# Also save as PNG for easy viewing
output_png = './visualization/models/stages_oversight_kds_vs_size.png'
plt.savefig(output_png, format='png', dpi=300, bbox_inches='tight', facecolor='#FAFAF7')
print(f"Plot also saved to: {output_png}")

plt.close()

