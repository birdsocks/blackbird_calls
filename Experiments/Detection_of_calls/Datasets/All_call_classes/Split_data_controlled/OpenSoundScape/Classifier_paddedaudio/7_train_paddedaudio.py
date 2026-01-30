# read in the training, validation and test set
import pandas as pd
import bioacoustics_model_zoo as bmz
import torch
import numpy as np
import pickle
import os
import sys
from io import StringIO
import matplotlib.pyplot as plt

print("Loading resampled datasets from script 6...")
# Load the RESAMPLED datasets created by 6_resample.py
resampled_train_set = pd.read_csv('resampled_train_set.csv', index_col=[0,1,2])
resampled_val_set = pd.read_csv('resampled_val_set.csv', index_col=[0,1,2])
test_set = pd.read_csv('test_set.csv', index_col=[0,1,2])

print(f"Loaded resampled_train_set: {len(resampled_train_set)} samples")
print(f"Loaded resampled_val_set: {len(resampled_val_set)} samples")
print(f"Loaded test_set: {len(test_set)} samples")

# Allow numpy types to be loaded
torch.serialization.add_safe_globals([np.float32, np.float64, np.int32, np.int64])

print("\nInitializing BirdNET model...")
birdnet = bmz.BirdNET()

num_workers = 4

print("\nGenerating embeddings for validation and test sets...")
emb_val = birdnet.embed(resampled_val_set, return_dfs=False, batch_size=128, num_workers=num_workers)
print(f"Generated {len(emb_val)} validation embeddings")
emb_test = birdnet.embed(test_set, return_dfs=False, batch_size=128, num_workers=num_workers)
print(f"Generated {len(emb_test)} test embeddings")

# We want to train the classifier on the 'A' class here, corresponding to the primary R. sierrae call type.
# Let's replace fc output layer with 1-output layer for class 'A'
classes = list(resampled_train_set.columns)
birdnet.change_classes(classes)

# Convert all label columns to float to avoid object type errors
resampled_train_set = resampled_train_set.astype(float)
resampled_val_set = resampled_val_set.astype(float)
test_set = test_set.astype(float)
# Convert labels to float (in case they're boolean or object type)
train_labels = resampled_train_set.astype(float).values
val_labels = resampled_val_set.astype(float).values
test_labels = test_set.astype(float).values

# fit the classification head with embeddings and labels
# birdnet.network.fit(emb_train, train_labels, emb_val, val_labels)

# Create training directory if it doesn't exist
os.makedirs('training', exist_ok=True)

# Capture stdout to parse training metrics
captured_output = StringIO()
old_stdout = sys.stdout
sys.stdout = captured_output

try:
    # Train the model
    history = birdnet.train(
        train_df=resampled_train_set,
        validation_df=resampled_val_set,
        embedding_batch_size=128,
        embedding_num_workers=num_workers,
        # n_augmentation_variants=2,
        steps=100,
        device=0
    )
finally:
    # Restore stdout
    sys.stdout = old_stdout

# Print captured output
output = captured_output.getvalue()
print(output)

# Parse the captured output to extract metrics
import re
train_losses = []
val_losses = []
val_aurocs = []
val_maps = []
epochs = []

for line in output.split('\n'):
    # Match lines like: "Epoch 100/500, Loss: 0.03169373422861099, Val Loss: 0.11118992418050766"
    epoch_match = re.search(r'Epoch (\d+)/\d+, Loss: ([\d.]+), Val Loss: ([\d.]+)', line)
    if epoch_match:
        epochs.append(int(epoch_match.group(1)))
        train_losses.append(float(epoch_match.group(2)))
        val_losses.append(float(epoch_match.group(3)))
    
    # Match lines like: "val AU ROC: 0.923"
    auroc_match = re.search(r'val AU ROC: ([\d.]+)', line)
    if auroc_match:
        val_aurocs.append(float(auroc_match.group(1)))
    
    # Match lines like: "val MAP: 0.657"
    map_match = re.search(r'val MAP: ([\d.]+)', line)
    if map_match:
        val_maps.append(float(map_match.group(1)))

# Create history dictionary
history = {
    'epochs': epochs,
    'train_loss': train_losses,
    'val_loss': val_losses,
    'val_auroc': val_aurocs,
    'val_map': val_maps
}

print(f"\nParsed {len(epochs)} training checkpoints")

# Save training history
with open('training/training_history.pkl', 'wb') as f:
    pickle.dump(history, f)

print("Training complete. History saved to 'training/training_history.pkl'")

# Plot training curves
# Load training history
with open('training/training_history.pkl', 'rb') as f:
    history = pickle.load(f)

# Create figure with 3 subplots
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Plot training and validation loss
axes[0].plot(history['epochs'], history['train_loss'], 'o-', label='Train Loss')
axes[0].plot(history['epochs'], history['val_loss'], 'o-', label='Validation Loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Training and Validation Loss')
axes[0].legend()
axes[0].grid(True)

# Plot validation AU ROC
axes[1].plot(history['epochs'], history['val_auroc'], 'o-', color='green', label='Val AU ROC')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('AU ROC')
axes[1].set_title('Validation AU ROC')
axes[1].legend()
axes[1].grid(True)

# Plot validation MAP
axes[2].plot(history['epochs'], history['val_map'], 'o-', color='purple', label='Val MAP')
axes[2].set_xlabel('Epoch')
axes[2].set_ylabel('MAP')
axes[2].set_title('Validation Mean Average Precision')
axes[2].legend()
axes[2].grid(True)

plt.tight_layout()
plt.show()

# Print final metrics
print(f"\nFinal Metrics (Epoch {history['epochs'][-1]}):")
print(f"  Train Loss: {history['train_loss'][-1]:.4f}")
print(f"  Val Loss: {history['val_loss'][-1]:.4f}")
print(f"  Val AU ROC: {history['val_auroc'][-1]:.4f}")
print(f"  Val MAP: {history['val_map'][-1]:.4f}")


# Save the trained model
model_path = "/home/Shelby/blackbird_calls/Experiments/Detection_of_calls/Datasets/All_call_classes/Split_data_controlled/OpenSoundScape/blackbirdcallv1.model"
birdnet.save(model_path)
print(f"\nModel saved to: {model_path}")

# Save embeddings and variables needed for visualization scripts
print("\nSaving embeddings and metadata for visualization scripts...")
np.save('emb_val.npy', emb_val)
np.save('emb_test.npy', emb_test)

# Save classes list
with open('classes.pkl', 'wb') as f:
    pickle.dump(classes, f)

print("Saved: emb_val.npy, emb_test.npy, classes.pkl")
print("\nTraining complete! Ready to run visualization scripts (8 and 9)")