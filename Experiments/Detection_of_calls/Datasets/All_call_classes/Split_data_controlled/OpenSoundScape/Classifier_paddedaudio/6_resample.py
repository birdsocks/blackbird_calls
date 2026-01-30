import pandas as pd
import matplotlib.pyplot as plt

RESAMPLE = True
RESAMPLE_VAL = True
DOWNSAMPLE = True
UPSAMPLE = True
N_SAMPLES_TRAIN = 100
N_SAMPLES_VAL = 100

# Load the clip labels created from Process_boxes_to_classifier_labels.ipynb
train_set = pd.read_csv('train_set_onehot.csv')
val_set = pd.read_csv('val_set_onehot.csv')
test_set = pd.read_csv('test_set_onehot.csv')

# concat all subsets in dataset df
dataset_df = pd.concat([train_set, val_set, test_set], ignore_index=True)

classes = dataset_df.columns[3:].tolist()

# make index file and start time and end time
train_set = train_set.reset_index(drop=True)
val_set = val_set.reset_index(drop=True)
test_set = test_set.reset_index(drop=True)

calltype_columns = [col for col in classes if col.startswith('calltype_')]
train_counts = train_set[calltype_columns].sum()
val_counts = val_set[calltype_columns].sum()
test_counts = test_set[calltype_columns].sum()

# train_set.set_index(['file', 'start_time', 'end_time'], inplace=True)
# val_set.set_index(['file', 'start_time', 'end_time'], inplace=True)
# test_set.set_index(['file', 'start_time', 'end_time'], inplace=True)

# print table of number of calltype files in each split
split_summary = pd.DataFrame({
    'Train': train_counts,
    'Validation': val_counts,
    'Test': test_counts
})

print(split_summary)

# resample train set by upsampling minority classes
seed = 42

def resample_set(df, n_samples_per_class, seed=42, downsample=True, upsample=True):
    """
    Resample a multi-label dataset to have n_samples_per_class for each class.
    For upsampling: duplicates rows as needed (keeps duplicates)
    For downsampling: randomly selects subset of samples
    """
    resampled_dfs = []
    
    # Get label columns (exclude metadata columns)
    label_cols = [col for col in df.columns if col not in ['file', 'start_time', 'end_time']]
    
    for class_label in label_cols:
        # Get rows that have this class
        class_df = df[df[class_label] == 1].copy()
        
        if len(class_df) == 0:
            print(f"Warning: No samples found for {class_label}")
            continue
        
        # Sample with or without replacement depending on class size
        if len(class_df) < n_samples_per_class and upsample:
            # Upsample with replacement - this creates duplicates
            sampled_df = class_df.sample(
                n=n_samples_per_class, replace=True, random_state=seed
            )
            new_number = len(sampled_df)
            print(f"Upsampled {class_label} from {len(class_df)} to {new_number} clips.")
        elif len(class_df) > n_samples_per_class and downsample:
            # Downsample
            sampled_df = class_df.sample(
                n=n_samples_per_class, replace=False, random_state=seed
            )

            new_number = len(sampled_df)
            print(f"Downsampled {class_label} from {len(class_df)} to {new_number} clips.")
        else:
            sampled_df = class_df
        resampled_dfs.append(sampled_df)
    
    # Concatenate all resampled dataframes and remove duplicates
    # (a clip can appear multiple times if it belongs to multiple classes that were all upsampled)
    resampled_set = pd.concat(resampled_dfs, axis=0)
    resampled_set = resampled_set.reset_index(drop=True)
    
    print(f"\nTotal unique clips after resampling: {len(resampled_set)}")
    
    # Show per-class counts after resampling
    print("\nPer-class counts after resampling:")
    for col in label_cols:
        count = resampled_set[col].sum()
        print(f"  {col}: {count}")
    
    return resampled_set

if RESAMPLE:
    resampled_train_set = resample_set(train_set, N_SAMPLES_TRAIN, seed, downsample=DOWNSAMPLE, upsample=UPSAMPLE)
    resampled_train_set.set_index(['file', 'start_time', 'end_time'], inplace=True)
    print("\nResampled training set created.")
    if RESAMPLE_VAL:
        resampled_val_set = resample_set(val_set, N_SAMPLES_VAL, seed, downsample=DOWNSAMPLE, upsample=False)
        print("\nResampled validation set created.")
    else:
        resampled_val_set = val_set.copy()
    resampled_val_set.set_index(['file', 'start_time', 'end_time'], inplace=True)  
    test_set.set_index(['file', 'start_time', 'end_time'], inplace=True)
else:
    resampled_train_set = train_set.set_index(['file', 'start_time', 'end_time'])
    resampled_val_set = val_set.set_index(['file', 'start_time', 'end_time'])
    test_set.set_index(['file', 'start_time', 'end_time'], inplace=True)

# bar plot number of calltype files in each split resampled

# Save resampled datasets for next script
print("\nSaving resampled datasets...")
resampled_train_set.to_csv('resampled_train_set.csv')
resampled_val_set.to_csv('resampled_val_set.csv')
test_set.to_csv('test_set.csv')
print("Saved: resampled_train_set.csv, resampled_val_set.csv, test_set.csv")

# bar plot number of calltype files in each split
calltype_columns = [col for col in classes if col.startswith('calltype_')]
train_counts = resampled_train_set.reset_index()[calltype_columns].sum()
val_counts = resampled_val_set.reset_index()[calltype_columns].sum()
test_counts = test_set.reset_index()[calltype_columns].sum()
x = range(len(calltype_columns))
plt.bar(x, train_counts, width=0.2, label='Train (Resampled)')
plt.bar([i + 0.2 for i in x], val_counts, width=0.2, label='Validation (Resampled)')
plt.bar([i + 0.4 for i in x], test_counts, width=0.2, label='Test')
plt.xticks([i + 0.2 for i in x], calltype_columns, rotation=90)
plt.ylabel('Number of clips')
plt.title('Number of clips per call type in each split (After Resampling)')
plt.legend()
plt.tight_layout()
plt.show()

print("\nResampling complete. Ready for training (run 7_train_paddedaudio.py)")