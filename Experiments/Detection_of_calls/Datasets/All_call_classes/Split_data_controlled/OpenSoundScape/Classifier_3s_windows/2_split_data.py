import pandas as pd

# Load the clip labels created from Process_boxes_to_classifier_labels.ipynb
clip_labels = pd.read_csv('/home/Shelby/blackbird_calls/Experiments/Detection_of_calls/Datasets/All_call_classes/Split_data_controlled/OpenSoundScape/Classifier_3s_windows/clip_labels_3s.csv')
# , index_col=[0,1,2]

print(f"Loaded {len(clip_labels)} clips with {len(clip_labels.columns)} call types")
print(f"Call types: {list(clip_labels.columns)}")
clip_labels.head()

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

# Save .csv tables of the training, validation, and test sets
os.makedirs("./annotated_data", exist_ok=True)
train_set.to_csv("./annotated_data/train_set.csv")
val_set.to_csv("./annotated_data/val_set.csv")
test_set.to_csv("./annotated_data/test_set.csv")

print(f"\nSaved splits to ./annotated_data/")


#count how many of each call type are in each set
print("Training set call type counts:")
print(train_set.sum())

print("Validation set call type counts:")
print(val_set.sum())

print("Test set call type counts:")
print(test_set.sum())