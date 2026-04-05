"""This file extracts the lowest KDS score data from stages_oversight results and saves to CSV."""

import pandas as pd
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
                'abs_min_kds': abs(min_kds),
                'model_family': get_model_family(model_name)
            })

# Create DataFrame
df = pd.DataFrame(results)
df = df.sort_values('size')

print(f"Found {len(df)} models:")
print("\n{:<30} {:>10} {:>15} {:>15}".format("Model", "Size (B)", "Min KDS", "Abs Min KDS"))
print("-" * 75)
for _, row in df.iterrows():
    print("{:<30} {:>10.2f} {:>15.3f} {:>15.3f}".format(
        row['model'], row['size'], row['min_kds'], row['abs_min_kds']
    ))

# Save to CSV
csv_output = './visualization/models/datasets/stages_oversight_kds.csv'
os.makedirs('./visualization/models/datasets', exist_ok=True)
df.to_csv(csv_output, index=False)
print(f"\n✓ Data saved to: {csv_output}")
print(f"  Columns: {list(df.columns)}")
print(f"  Total rows: {len(df)}")

