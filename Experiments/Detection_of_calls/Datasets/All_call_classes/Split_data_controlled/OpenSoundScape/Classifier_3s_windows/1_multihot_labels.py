from opensoundscape.annotations import BoxedAnnotations
import pandas as pd

box_labels = pd.read_csv('/home/Shelby/blackbird_calls/Dataset_processing/cv4e_calls_channel1_v2.csv')
print(box_labels.columns)
cols_we_need = ["begin_time_s", "end_time_s", "low_freq_hz", "high_freq_hz", "call_type","wav_fname"]
clean_labels = box_labels[cols_we_need]
# rename columns to start_time, end_time, low_f, high_f, annotation, audio_file
clean_labels = clean_labels.rename(columns={
    "begin_time_s": "start_time",
    "end_time_s": "end_time",
    "low_freq_hz": "low_f",
    "high_freq_hz": "high_f",
    "call_type": "annotation",
    "wav_fname": "audio_file"
})
# I need to replace all the spaces with _ in the audio_file column
clean_labels['audio_file'] = clean_labels['audio_file'].str.replace(' ', '_')
# drop any nans in clean_labels
clean_labels = clean_labels.dropna()
# Convert annotation column to object dtype for OpenSoundScape compatibility
clean_labels['annotation'] = clean_labels['annotation'].astype('object')

clean_labels["annotation"].value_counts()

# Display the resulting labels dataframe
# Each row is a 3-second clip, columns are call types with 0/1 for absence/presence
print(f"Shape: {labels.shape} (rows = 3s clips, columns = call types)")
print(f"\nCall types found: {list(labels.columns)}")
print(f"\nSample of multi-hot encoded labels:")
labels.head(20)

# check for duplicate index values in labels
duplicate_indices = labels.index[labels.index.duplicated()].unique()
if len(duplicate_indices) > 0:
    print(f"Found {len(duplicate_indices)} duplicate index values in labels.")
    for idx in duplicate_indices:
        print(f"Duplicate index: {idx}")
else:
    print("No duplicate index values found in labels.")

# drop any duplicate indices, keeping the first occurrence
labels = labels[~labels.index.duplicated(keep='first')] 


# Check label distribution
print("Number of clips with each call type:")
print(labels.sum().sort_values(ascending=False))
print(f"\nTotal clips: {len(labels)}")
print(f"Clips with at least one call: {(labels.sum(axis=1) > 0).sum()}")
print(f"Clips with no calls: {(labels.sum(axis=1) == 0).sum()}")


# Filter and rename call types before saving
# Rename call types (merge similar types together)
calltype_mapping = {
    'M': 'A',  # Merge M into A
    'O': 'E',  # Merge O into E
    'I': 'D',  # Merge I into D
    'L': 'C',  # Merge L into C
}

# Apply renaming by combining columns
for old_name, new_name in calltype_mapping.items():
    if old_name in labels.columns and new_name in labels.columns:
        # Merge: if either old or new has a 1, the result should be 1
        labels[new_name] = (labels[old_name] | labels[new_name]).astype(int)
        labels = labels.drop(columns=[old_name])
        print(f"Merged column '{old_name}' into '{new_name}'")
    elif old_name in labels.columns:
        # Just rename if new name doesn't exist
        labels = labels.rename(columns={old_name: new_name})
        print(f"Renamed column '{old_name}' to '{new_name}'")

# Filter out specific call types
call_types_to_exclude = ['F', 'Growl', 'H', 'N']
columns_to_drop = [col for col in call_types_to_exclude if col in labels.columns]
if columns_to_drop:
    labels = labels.drop(columns=columns_to_drop)
    print(f"\nRemoved columns: {columns_to_drop}")

print(f"\nFinal call types: {list(labels.columns)}")
print(f"Call type distribution after filtering:")
print(labels.sum().sort_values(ascending=False))

# Save the multi-hot encoded labels to CSV
output_path = '/home/Shelby/blackbird_calls/Experiments/Detection_of_calls/Datasets/All_call_classes/Split_data_controlled/OpenSoundScape/Classifier_3s_windows/clip_labels_3s.csv'
labels.to_csv(output_path)
print(f"Saved multi-hot encoded labels to: {output_path}")
print(f"\nThis file contains {len(labels)} rows (3-second clips)")
print(f"and {len(labels.columns)} columns (call types with binary 0/1 labels)")