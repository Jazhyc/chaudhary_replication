# OLMo Checkpoint AUROC Visualization

This folder contains visualizations of AUROC performance across OLMo training checkpoints.

## Data Sources

The visualization combines data from two sources:

1. **`datasets/`**: Aggregated checkpoint data with pre-computed best AUROC values
2. **`specific_layer_datasets/`**: Layer-by-layer AUROC data for each checkpoint

The `specific_layer_datasets` provide more comprehensive coverage, especially for early training steps (0-130k), and the visualization automatically computes the best AUROC across all layers for each checkpoint.

## Data Coverage

- **Total checkpoints**: 557
- **Step range**: 0k to 557k (0 to 557,000 training steps)
- **Token range**: 0B to 2464B
- **AUROC range**: 0.4648 to 0.9207

### Coverage by Training Stage

| Step Range | Checkpoints | AUROC Range |
|------------|-------------|-------------|
| 0k-50k     | 50          | 0.4648 - 0.8389 |
| 50k-100k   | 50          | 0.5761 - 0.9207 |
| 100k-150k  | 50          | 0.5462 - 0.7984 |
| 150k-200k  | 49          | 0.5108 - 0.8317 |
| 200k-300k  | 100         | 0.5446 - 0.9160 |
| 300k-400k  | 100         | 0.6048 - 0.8131 |
| 400k-500k  | 100         | 0.5847 - 0.8381 |
| 500k-600k  | 58          | 0.5929 - 0.7991 |

## Visualization Scripts

### 1. Static Visualization (Recommended)

**File**: `visualization/Olmo_visualization_static.py`

Generates static PDF and PNG plots using matplotlib.

**Usage**:
```bash
cd /path/to/Maheep-INN
python3 visualization/Olmo_checkpoints/visualization/Olmo_visualization_static.py
```

**Output**:
- `olmo_auroc_vs_step.pdf` - High-quality vector plot
- `olmo_auroc_vs_step.png` - Raster image (300 DPI)

### 2. Interactive Visualization (Dash)

**File**: `visualization/Olmo_visualization.py`

Launches an interactive Dash web application with hover tooltips showing detailed information for each checkpoint.

**Usage**:
```bash
cd /path/to/Maheep-INN
python3 visualization/Olmo_checkpoints/visualization/Olmo_visualization.py
```

Then open your browser to `http://localhost:8053`

**Features**:
- Interactive hover tooltips showing step number, token count, AUROC, and best layer
- Zoomable and pannable plot
- Polynomial trend line

## Key Findings

The visualization reveals:

1. **Early Training Variability**: High variance in AUROC during the first 100k steps, with some checkpoints achieving surprisingly high performance (>0.9 AUROC)
2. **Mid-Training Plateau**: AUROC stabilizes around 0.65-0.75 between 100k-400k steps
3. **Late Training Consistency**: More consistent performance in later checkpoints (400k+), though with slightly lower peak values
4. **Overall Trend**: The polynomial trend line shows a slight decrease in average AUROC over training, suggesting potential overfitting or task-specific adaptation

## Technical Details

### Data Processing

For each checkpoint in `specific_layer_datasets/`:
1. Load layer-by-layer AUROC scores
2. Group by revision (checkpoint identifier)
3. Select the maximum AUROC across all layers
4. Record which layer achieved the best performance

### Visualization Style

- **Color scheme**: Red markers (#FF6B6B) on light background (#FAFAF7)
- **Markers**: 80pt circles with black edges
- **Trend line**: Gray dashed polynomial (degree 2)
- **Grid**: Light gray for readability
- **Font**: Arial, bold labels

## Files Generated

- `olmo_auroc_vs_step.pdf` - Vector plot for publications
- `olmo_auroc_vs_step.png` - Raster plot for presentations/web

## Dependencies

- pandas
- numpy
- matplotlib (for static visualization)
- plotly + dash (for interactive visualization)

Install with:
```bash
pip install pandas numpy matplotlib plotly dash
```

