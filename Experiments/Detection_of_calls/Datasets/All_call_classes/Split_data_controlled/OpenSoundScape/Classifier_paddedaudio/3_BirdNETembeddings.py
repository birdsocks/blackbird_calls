import bioacoustics_model_zoo as bmz
import os
from pathlib import Path
import pandas as pd

call_audios_dir = "/home/Shelby/blackbird_calls/Experiments/Detection_of_calls/Datasets/All_call_classes/Split_data_controlled/OpenSoundScape/Call_audios"
padded_output_dir = os.path.join(call_audios_dir, "padded_3s")

dataset_dir = Path(padded_output_dir)
model = bmz.BirdNET()
# glob for all the audio (.wav) files in the directory
wav_files = list(dataset_dir.glob("*.WAV"))
embeddings = model.embed(wav_files)
#print(embeddings)
embeddings.to_csv("/home/Shelby/blackbird_calls/Experiments/Detection_of_calls/Datasets/All_call_classes/Split_data_controlled/OpenSoundScape/birds_embeddings_birdnet_padded_3s.csv")

#rename file to include call type column

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

# Save the updated CSV
df.to_csv(csv_path, index=False)
print(f"\n✓ Updated CSV saved to {csv_path}")
