"""This file visualizes OLMo checkpoint AUROC scores using only specific_layer_datasets."""

import pandas as pd
import numpy as np
import os
import glob
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Read specific layer datasets and compute best AUROC for each step
specific_layer_folder = './visualization/Olmo_checkpoints/specific_layer_datasets'
specific_layer_data = []

print("Loading data from specific_layer_datasets...")
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
print(f"AUROC range: {df['best_auroc'].min():.4f} to {df['best_auroc'].max():.4f}")

# Create the plot
fig, ax = plt.subplots(figsize=(12, 7))

# Set background color
fig.patch.set_facecolor('#FAFAF7')
ax.set_facecolor('#FAFAF7')

# Main scatter plot
ax.scatter(df['step_k'], df['best_auroc'], 
          color='#FF6B6B', s=80, alpha=0.8, 
          edgecolors='black', linewidths=1.5, 
          zorder=3, label='Best AUROC')

# Add a polynomial trend line
z = np.polyfit(df['step_k'], df['best_auroc'], 2)
p = np.poly1d(z)
x_trend = np.linspace(df['step_k'].min(), df['step_k'].max(), 100)
y_trend = p(x_trend)

ax.plot(x_trend, y_trend, 
       color='gray', linewidth=2, linestyle='--', 
       alpha=0.5, label='Trend', zorder=2)

# Styling
ax.set_xlabel('Training Step (thousands)', fontsize=16, fontweight='bold', family='Arial')
ax.set_ylabel('Best AUROC', fontsize=16, fontweight='bold', family='Arial')
ax.set_title('OLMo Checkpoint AUROC Performance', fontsize=20, fontweight='bold', 
            family='Arial', pad=20)

# Grid
ax.grid(True, alpha=0.3, linewidth=0.8, color='#E5E5E5', zorder=1)

# Tick styling
ax.tick_params(axis='both', which='major', labelsize=12, length=5, width=1.5, 
              direction='out', colors='black')

# Spine styling
for spine in ax.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(1.5)

# Legend
ax.legend(fontsize=12, frameon=True, facecolor='#EDEDE8', 
         edgecolor='black', loc='lower right')

# Set y-axis limits with some padding
y_min = max(0.4, df['best_auroc'].min() - 0.05)
y_max = min(1.0, df['best_auroc'].max() + 0.05)
ax.set_ylim(y_min, y_max)

plt.tight_layout()

# Save the figure
output_path = './visualization/Olmo_checkpoints/no_layer_auroc_vs_step.pdf'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='#FAFAF7')
print(f"\nPlot saved to: {output_path}")

# Also save as PNG
output_path_png = './visualization/Olmo_checkpoints/no_layer_auroc_vs_step.png'
plt.savefig(output_path_png, dpi=300, bbox_inches='tight', facecolor='#FAFAF7')
print(f"Plot saved to: {output_path_png}")

plt.close()

print("\n✅ Visualization complete!")

