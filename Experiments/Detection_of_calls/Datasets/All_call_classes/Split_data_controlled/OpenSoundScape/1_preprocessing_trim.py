import os
from pathlib import Path

# Directory containing the audio files
audio_dir = "/mnt/class_data/Shelby/One_Minute_Audio"

# Find all WAV files with spaces in their names
wav_files = list(Path(audio_dir).glob("*.WAV"))
renamed_count = 0

for wav_file in wav_files:
    if ' ' in wav_file.name:
        # Create new filename by replacing spaces with underscores
        new_name = wav_file.name.replace(' ', '_')
        new_path = wav_file.parent / new_name
        
        # Rename the file
        os.rename(wav_file, new_path)
        renamed_count += 1
        print(f"Renamed: {wav_file.name} -> {new_name}")

print(f"\nTotal files renamed: {renamed_count}")

import pandas as pd
from opensoundscape import Audio
from pathlib import Path
import os

# Define paths
audio_dir = "/mnt/class_data/Shelby/One_Minute_Audio"
csv_path = "/home/Shelby/blackbird_calls/Dataset_processing/cv4e_calls_channel1_v2.csv"
output_dir = "/home/Shelby/blackbird_calls/Experiments/Detection_of_calls/Datasets/All_call_classes/Split_data_controlled/OpenSoundScape/Call_audios"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Read the CSV with annotations
df = pd.read_csv(csv_path)
print(f"Found {len(df)} annotations in CSV")

# Get unique file IDs to process
unique_files = df['file_id'].unique()
print(f"Processing {len(unique_files)} unique audio files")

# Process each annotation
processed_count = 0
for idx, row in df.iterrows():
    file_id = row['file_id']
    begin_time = row['begin_time_s']
    end_time = row['end_time_s']
    call_type = row['call_type']
    
    # Construct the input audio file path (assuming .WAV extension)
    audio_file_path = Path(audio_dir) / f"{file_id}.WAV"
    
    # Check if the audio file exists
    if not audio_file_path.exists():
        print(f"Warning: Audio file not found: {audio_file_path}")
        continue
    
    # Load the full audio file for each trim
    audio = Audio.from_file(audio_file_path)
    
    # Trim the audio according to begin and end times
    trimmed_audio = audio.trim(start_time=begin_time, end_time=end_time)
    
    # Create output filename: file_id_begin_time_end_time_call_type.wav
    output_filename = f"{file_id}_{begin_time}_{end_time}_{call_type}.WAV"
    output_path = os.path.join(output_dir, output_filename)
    
    # Save the trimmed audio
    trimmed_audio.save(output_path)
    
    processed_count += 1
    if processed_count % 100 == 0:
        print(f"Processed {processed_count}/{len(df)} annotations...")

print(f"Completed! Processed {processed_count} audio clips and saved to {output_dir}")

import os
from pathlib import Path
from collections import Counter

# Path to the trimmed audio files
call_audios_dir = "/home/Shelby/blackbird_calls/Experiments/Detection_of_calls/Datasets/All_call_classes/Split_data_controlled/OpenSoundScape/Call_audios"

# Get all WAV files
wav_files = list(Path(call_audios_dir).glob("*.WAV"))
print(f"Found {len(wav_files)} trimmed audio files\n")

# Extract call types from filenames
call_types = []
nan_files = []

for wav_file in wav_files:
    # Filename format: file_id_begin_time_end_time_call_type.WAV
    # Extract call type (last part before .WAV)
    filename_parts = wav_file.stem.rsplit('_', 1)  # Split from the right, only once
    if len(filename_parts) == 2:
        call_type = filename_parts[1]
        
        # Check for nan call types
        if call_type.lower() == 'nan':
            nan_files.append(wav_file)
        else:
            call_types.append(call_type)

# Count occurrences of each call type
call_type_counts = Counter(call_types)

print("Call type counts:")
for call_type, count in sorted(call_type_counts.items(), key=lambda x: x[1], reverse=True):
    print(f"  {call_type}: {count}")

print(f"\nTotal valid call types: {len(call_types)}")
print(f"Files with 'nan' call type: {len(nan_files)}")

# Remove files with nan call types
if nan_files:
    print("\nRemoving files with 'nan' call type...")
    for nan_file in nan_files:
        os.remove(nan_file)
        print(f"  Removed: {nan_file.name}")
    print(f"\nRemoved {len(nan_files)} files with 'nan' call type")
else:
    print("\nNo files with 'nan' call type found")