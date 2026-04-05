import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('OLMo_7B_all_layers.csv')

# Create the plot
plt.figure(figsize=(12, 6))

# Plot AUROC vs Layer
plt.plot(df['layer'], df['auroc'], marker='o', linewidth=2, markersize=6, color='#E63946')

# Add a horizontal line at 0.5 (random chance)
plt.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, label='Random chance (0.5)')

# Highlight the best layer
best_idx = df['auroc'].idxmax()
best_layer = df.loc[best_idx, 'layer']
best_auroc = df.loc[best_idx, 'auroc']
plt.scatter([best_layer], [best_auroc], color='#2A9D8F', s=150, zorder=5, 
            label=f'Best: Layer {best_layer} (AUROC={best_auroc:.4f})')

# Labels and title
plt.xlabel('Layer', fontsize=12, fontweight='bold')
plt.ylabel('AUROC', fontsize=12, fontweight='bold')
plt.title('OLMo-7B: AUROC by Layer', fontsize=14, fontweight='bold')

# Grid and legend
plt.grid(True, alpha=0.3)
plt.legend(loc='lower right')

# Set axis limits
plt.xlim(0, df['layer'].max() + 1)
plt.ylim(0.3, 0.7)

# Set x-axis ticks
plt.xticks(range(0, int(df['layer'].max()) + 1, 2))

# Tight layout
plt.tight_layout()

# Save the figure
plt.savefig('OLMo_7B_auroc_by_layer.pdf', format='pdf', dpi=300, bbox_inches='tight')
plt.savefig('OLMo_7B_auroc_by_layer.png', format='png', dpi=300, bbox_inches='tight')

print(f"Plot saved to OLMo_7B_auroc_by_layer.pdf and OLMo_7B_auroc_by_layer.png")
print(f"\nBest Layer: {best_layer}")
print(f"Best AUROC: {best_auroc:.4f}")

