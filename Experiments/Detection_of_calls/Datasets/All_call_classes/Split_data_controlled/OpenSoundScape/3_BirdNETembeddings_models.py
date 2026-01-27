# import bioacoustics_model_zoo as bmz

# Commented out: Embedding generation already done
# call_audios_dir = "/home/Shelby/blackbird_calls/Experiments/Detection_of_calls/Datasets/All_call_classes/Split_data_controlled/OpenSoundScape/Call_audios"
# padded_output_dir = os.path.join(call_audios_dir, "padded_3s")
# dataset_dir = Path(padded_output_dir)
# model = bmz.BirdNET()
# wav_files = list(dataset_dir.glob("*.WAV"))
# embeddings = model.embed(wav_files)
# embeddings.to_csv("/home/Shelby/blackbird_calls/Experiments/Detection_of_calls/Datasets/All_call_classes/Split_data_controlled/OpenSoundScape/birds_embeddings_birdnet_padded_3s.csv")

# Add call type and model columns to existing embeddings
import pandas as pd

# Load the embeddings CSV
csv_path = "/home/Shelby/blackbird_calls/Experiments/Detection_of_calls/Datasets/All_call_classes/Split_data_controlled/OpenSoundScape/birds_embeddings_birdnet_padded_3s.csv"
df = pd.read_csv(csv_path)

print("Original columns:", df.columns.tolist())
print(f"Shape: {df.shape}")
print("\nFirst few rows:")
print(df.head())

# Extract call type from the filename in the 'file' column
# Filename format: file_id_begin_time_end_time_call_type.WAV
# Extract call type (last part before .WAV)
def extract_call_type(filepath):
    # Get just the filename without path
    filename = filepath.split('/')[-1]
    # Remove .WAV extension and split by underscore
    filename_without_ext = filename.rsplit('.', 1)[0]
    # Get the last part (call type)
    call_type = filename_without_ext.rsplit('_', 1)[-1]
    return call_type

# Apply to the 'file' column
df['calltype'] = df['file'].apply(extract_call_type)

# Reorder columns to put 'calltype' first
cols = ['calltype'] + [col for col in df.columns if col != 'calltype']
df = df[cols]

print("\n\nAfter adding calltype column:")
print("New columns:", df.columns.tolist())
print("\nFirst few rows with calltype:")
print(df.head())

# Extract file_id from the filename to match with annotations CSV
def extract_file_id(filepath):
    # Get just the filename without path
    filename = filepath.split('/')[-1]
    # Remove .WAV extension
    filename_without_ext = filename.rsplit('.', 1)[0]
    # Format: file_id_begin_time_end_time_call_type
    # Example: SL07_Trial_1_trim_37.44715902_37.57169857_D.WAV
    # file_id should be: SL07_Trial_1_trim
    # Split by underscore and reconstruct file_id (everything before the timestamps)
    parts = filename_without_ext.split('_')
    
    # Find where the numeric parts start (timestamps are floats with decimals)
    file_id_parts = []
    for part in parts:
        # Check if this part looks like a float timestamp (has decimal point)
        if '.' in part:
            try:
                float(part)
                break  # Stop when we hit the first timestamp
            except ValueError:
                file_id_parts.append(part)
        else:
            file_id_parts.append(part)
    
    return '_'.join(file_id_parts)

# Extract file_id for each row
df['file_id'] = df['file'].apply(extract_file_id)

print("\n\nExtracted file_ids:")
print(df[['file', 'file_id']].head(10))

# Load annotations CSV to get Model information
annotations_csv = "/home/Shelby/blackbird_calls/Dataset_processing/cv4e_calls_channel1_v2.csv"
annotations_df = pd.read_csv(annotations_csv)

print(f"\n\nLoaded annotations CSV with {len(annotations_df)} rows")
print(f"Unique Models in annotations: {annotations_df['Model'].unique()}")

# Create a mapping of file_id to Model
file_id_to_model = annotations_df.groupby('file_id')['Model'].first().to_dict()

print(f"\nCreated mapping for {len(file_id_to_model)} unique file_ids")

# Map Model to each row in embeddings
df['Model'] = df['file_id'].map(file_id_to_model)

# Check for any missing mappings
missing_models = df['Model'].isna().sum()
if missing_models > 0:
    print(f"\n⚠ Warning: {missing_models} rows have no matching Model in annotations")
    print("Sample rows with missing Model:")
    print(df[df['Model'].isna()][['file', 'file_id']].head())

# Get unique models and create binary columns for each
unique_models = df['Model'].dropna().unique()
print(f"\n\nCreating binary columns for models: {list(unique_models)}")

for model in unique_models:
    # Create binary column: 1 if this model was used, 0 otherwise
    df[model] = (df['Model'] == model).astype(int)

# Reorder columns: calltype, then model binary columns, then Model, file_id, file, then embeddings
model_columns = [col for col in unique_models]
embedding_columns = [col for col in df.columns if col not in ['calltype', 'file', 'file_id', 'Model'] + model_columns]
cols = ['calltype'] + model_columns + ['Model', 'file_id', 'file'] + embedding_columns
df = df[cols]

print("\n\nAfter adding Model binary columns:")
print("New columns:", df.columns.tolist()[:10], "... (and embedding features)")
print("\nFirst few rows with Model columns:")
print(df[['calltype'] + model_columns + ['Model', 'file_id']].head(10))

# Save the updated CSV with new filename
output_csv = "/home/Shelby/blackbird_calls/Experiments/Detection_of_calls/Datasets/All_call_classes/Split_data_controlled/OpenSoundScape/birds_embeddings_birdnet_padded_3s_model.csv"
df.to_csv(output_csv, index=False)
print(f"\n✓ Updated CSV with Model columns saved to {output_csv}")
