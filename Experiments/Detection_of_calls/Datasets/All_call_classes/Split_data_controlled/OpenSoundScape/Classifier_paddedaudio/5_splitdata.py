import pandas as pd

# Load the clip labels created from Process_boxes_to_classifier_labels.ipynb
clip_labels = pd.read_csv('/home/Shelby/blackbird_calls/Experiments/Detection_of_calls/Datasets/All_call_classes/Split_data_controlled/OpenSoundScape/birds_embeddings_birdnet_padded_3s_condensed.csv')
# , index_col=[0,1,2]
# take only first 4 columns
clip_labels = clip_labels.iloc[:, :4]

# make calltype A and M the same = A
#clip_labels['calltype'] = clip_labels['calltype'].replace('M', 'A')
# make calltype N and chits the same = Chits
#clip_labels['calltype'] = clip_labels['calltype'].replace('N', 'Chits')
# make calltype L and C the same = C
#clip_labels['calltype'] = clip_labels['calltype'].replace('L', 'C')

# sort columns like file, start_time, end_time, call_type
clip_labels = clip_labels[['file', 'start_time', 'end_time', 'calltype']]
# make call_type one hot encoding (true or false)
clip_labels = pd.get_dummies(clip_labels, columns=['calltype'])
clip_labels

classes = clip_labels.columns[3:].tolist()
classes

import os
import pandas as pd

# Path to YOLO dataset split files
split_dir = '/home/Shelby/blackbird_calls/Experiments/Detection_of_calls/Datasets/All_call_classes/Split_data_controlled/YOLO/dataset_split/'

# Read the split files and extract unique trial identifiers
def extract_trial_ids(txt_file):
    """Extract unique trial identifiers from YOLO split file"""
    with open(txt_file, 'r') as f:
        paths = [line.strip() for line in f]
    
    # Extract trial IDs from image filenames
    # Format: .../AZ02_Trial_1_trim_spec_00-05.png -> AZ02_Trial_1_trim
    trial_ids = set()
    for path in paths:
        filename = os.path.basename(path)
        # Remove _spec_XX-XX.png to get trial ID
        trial_id = filename.rsplit('_spec_', 1)[0]
        trial_ids.add(trial_id)
    
    return trial_ids

# Extract trial IDs for each split
test_trials = extract_trial_ids(os.path.join(split_dir, 'test.txt'))
val_trials = extract_trial_ids(os.path.join(split_dir, 'validate.txt'))
train_trials = extract_trial_ids(os.path.join(split_dir, 'train.txt'))

print(f"Test trials: {sorted(test_trials)}")
print(f"Validation trials: {sorted(val_trials)}")
print(f"Train trials ({len(train_trials)} total): {sorted(list(train_trials)[:5])}...")

# Create masks based on trial IDs in the audio file paths
def create_mask(clip_labels, trial_ids):
    """Create boolean mask for clips from specified trial IDs"""
    return clip_labels.reset_index()["file"].apply(
        lambda x: any(trial_id in x for trial_id in trial_ids)
    ).values

# Split the data
mask_test = create_mask(clip_labels, test_trials)
test_set = clip_labels[mask_test]

mask_val = create_mask(clip_labels, val_trials)
val_set = clip_labels[mask_val]

mask_train = create_mask(clip_labels, train_trials)
train_set = clip_labels[mask_train]

# Verify the split
print(f"\nSplit summary:")
print(f"Training set: {len(train_set)} clips")
print(f"Validation set: {len(val_set)} clips")
print(f"Test set: {len(test_set)} clips")
print(f"Total: {len(train_set) + len(val_set) + len(test_set)} clips")
print(f"Original total: {len(clip_labels)} clips")

# make index file and start time and end time
train_set = train_set.reset_index(drop=True)
val_set = val_set.reset_index(drop=True)
test_set = test_set.reset_index(drop=True)

# train_set.set_index(['file', 'start_time', 'end_time'], inplace=True)
# val_set.set_index(['file', 'start_time', 'end_time'], inplace=True)
# test_set.set_index(['file', 'start_time', 'end_time'], inplace=True)

# bar plot number of calltype files in each split
import matplotlib.pyplot as plt
calltype_columns = [col for col in classes if col.startswith('calltype_')]
train_counts = train_set[calltype_columns].sum()
val_counts = val_set[calltype_columns].sum()
test_counts = test_set[calltype_columns].sum()
x = range(len(calltype_columns))
plt.bar(x, train_counts, width=0.2, label='Train')
plt.bar([i + 0.2 for i in x], val_counts, width=0.2, label='Validation')
plt.bar([i + 0.4 for i in x], test_counts, width=0.2, label='Test')
plt.xticks([i + 0.2 for i in x], calltype_columns, rotation=90)
plt.ylabel('Number of clips')
plt.title('Number of clips per call type in each split')
plt.legend()
plt.tight_layout()
plt.show()

# print table of number of calltype files in each split
split_summary = pd.DataFrame({
    'Train': train_counts,
    'Validation': val_counts,
    'Test': test_counts
})
split_summary

# save splits to csv
train_set.to_csv('train_set_onehot.csv', index=False)
val_set.to_csv('val_set_onehot.csv', index=False)
test_set.to_csv('test_set_onehot.csv', index=False)