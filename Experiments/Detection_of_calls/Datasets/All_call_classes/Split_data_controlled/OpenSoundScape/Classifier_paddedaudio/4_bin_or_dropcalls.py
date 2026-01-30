import pandas as pd

csv_path = "/home/Shelby/blackbird_calls/Experiments/Detection_of_calls/Datasets/All_call_classes/Split_data_controlled/OpenSoundScape/birds_embeddings_birdnet_padded_3s.csv"
df = pd.read_csv(csv_path)

# Rename call types (merge similar types together)
# Add/modify mappings as needed - format: 'old_name': 'new_name'
calltype_mapping = {
    'M': 'A',  # Merge M into A
    'O': 'E',  # Merge O into E
    'I': 'D',  # Rename X to Y
    'L': 'C',  # Merge L into C
}

# Apply renaming
df['calltype'] = df['calltype'].replace(calltype_mapping)
print(f"After renaming - Call types: {sorted(df['calltype'].unique())}")

# Filter out specific call types for analysis
# Add/remove call types from this list as needed
call_types_to_exclude = ['F', 'Growl', 'H', 'N']  # Replace with actual call types you want to exclude

# Create filtered dataframe (original df remains unchanged)
df_filtered = df[~df['calltype'].isin(call_types_to_exclude)].copy()

print(f"\nOriginal dataset: {len(df)} samples with {df['calltype'].nunique()} call types")
print(f"Filtered dataset: {len(df_filtered)} samples with {df_filtered['calltype'].nunique()} call types")
print(f"\nRemaining call types:\n{df_filtered['calltype'].value_counts()}")

# Save the filtered and renamed data to a new CSV
output_path = "/home/Shelby/blackbird_calls/Experiments/Detection_of_calls/Datasets/All_call_classes/Split_data_controlled/OpenSoundScape/birds_embeddings_birdnet_padded_3s_condensed.csv"
df_filtered.to_csv(output_path, index=False)
print(f"\nSaved filtered data to: {output_path}")