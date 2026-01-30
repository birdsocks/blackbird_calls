# Prepare results dataframe for visualization
import librosa
import soundfile as sf
import re
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import torch
import bioacoustics_model_zoo as bmz
import pickle
from scipy.stats import pearsonr

print("Loading model, embeddings, and data from previous scripts...")

# Load the trained model
birdnet = bmz.BirdNET()
birdnet.load("/home/Shelby/blackbird_calls/Experiments/Detection_of_calls/Datasets/All_call_classes/Split_data_controlled/OpenSoundScape/blackbirdcallv1.model")

# Load embeddings
emb_test = np.load('emb_test.npy')

# Load classes
with open('classes.pkl', 'rb') as f:
    classes = pickle.load(f)

# Load test dataset
test_set = pd.read_csv('test_set.csv', index_col=[0,1,2])

print(f"Loaded model with {len(classes)} classes")
print(f"Test set: {len(test_set)} samples")

# Generate test predictions
print("\nGenerating test predictions...")
preds = birdnet.network(torch.tensor(emb_test).cuda()).detach().cpu()

# Get predictions and true labels
pred_class_indices = np.argmax(preds.numpy(), axis=1)
true_class_indices = np.argmax(test_set.values, axis=1)
confidence_scores = np.max(preds.numpy(), axis=1)

# Get file paths from test_set index
test_set_reset = test_set.reset_index()

# Function to extract original call duration from filename
def extract_original_duration(filename):
    """
    Extract original call duration from padded filename.
    Filename format: ...start_time_end_time_calltype.WAV
    Example: SL07_Trial_1_trim_37.44715902_37.57169857_D.WAV
    Original duration = end_time - start_time
    """
    # Extract the two timestamps before the calltype
    match = re.search(r'_(\d+\.\d+)_(\d+\.\d+)_[A-Za-z]+\.WAV', filename)
    if match:
        start = float(match.group(1))
        end = float(match.group(2))
        return end - start
    return None

# Create results dataframe
results_list = []
for i in range(len(test_set_reset)):
    file_path = test_set_reset.iloc[i]['file']
    filename = file_path.split('/')[-1]
    original_duration = extract_original_duration(filename)
    
    results_list.append({
        'File': file_path,
        'Filename': filename,
        'StartTime': test_set_reset.iloc[i]['start_time'],
        'EndTime': test_set_reset.iloc[i]['end_time'],
        'OriginalDuration': original_duration,
        'TrueClassIdx': true_class_indices[i],
        'PredClassIdx': pred_class_indices[i],
        'TrueClass': classes[true_class_indices[i]],
        'PredictedClass': classes[pred_class_indices[i]],
        'Confidence': confidence_scores[i],
        'Correct': true_class_indices[i] == pred_class_indices[i]
    })

df_results = pd.DataFrame(results_list)

print(f"Created results dataframe with {len(df_results)} samples")
print(f"Correct predictions: {df_results['Correct'].sum()} ({100*df_results['Correct'].mean():.1f}%)")
print(f"\nOriginal duration stats:")
print(f"  Min: {df_results['OriginalDuration'].min():.3f}s")
print(f"  Max: {df_results['OriginalDuration'].max():.3f}s")
print(f"  Mean: {df_results['OriginalDuration'].mean():.3f}s")
print(f"  Median: {df_results['OriginalDuration'].median():.3f}s")
df_results.head()


#Visualize high and low confidence predictions for correct and incorrect classifications
def plot_spectrogram_results(df_results, mode='random', filter_classes=None, n_samples=25):
    """
    Plot spectrograms of predictions with varying confidence levels.
    
    Parameters:
    - df_results: DataFrame with columns 'File', 'StartTime', 'EndTime', 'TrueClass', 'PredictedClass', 'Confidence', 'Correct'
    - mode: 'max' (highest confidence), 'min' (lowest confidence), 'random', 'incorrect' (wrong predictions)
    - filter_classes: List of class names to filter by (None = all classes)
    - n_samples: Number of samples to plot (default 25)
    """
    # Filter by classes if specified
    df_filtered = df_results.copy()
    if filter_classes:
        df_filtered = df_filtered[df_filtered['PredictedClass'].isin(filter_classes)]
    
    # Select samples based on mode
    if mode == 'max':
        samples = df_filtered.sort_values(by='Confidence', ascending=False).head(n_samples)
        title_suffix = "(Highest Confidence)"
    elif mode == 'min':
        samples = df_filtered.sort_values(by='Confidence', ascending=True).head(n_samples)
        title_suffix = "(Lowest Confidence)"
    elif mode == 'incorrect':
        samples = df_filtered[df_filtered['Correct'] == False].head(n_samples)
        title_suffix = "(Incorrect Predictions)"
    else:  # random
        samples = df_filtered.sample(min(n_samples, len(df_filtered)), random_state=42)
        title_suffix = "(Random)"
    
    if len(samples) == 0:
        print("No samples found matching criteria!")
        return
    
    # Calculate grid size
    n_cols = 5
    n_rows = int(np.ceil(len(samples) / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4*n_rows))
    axes = axes.ravel() if n_rows > 1 else [axes] if n_cols == 1 else axes
    
    for i, (_, row) in enumerate(samples.iterrows()):
        if i >= len(axes):
            break
            
        # Load audio segment
        try:
            audio, sr = librosa.load(row['File'], sr=48000, 
                                    offset=row['StartTime'], 
                                    duration=row['EndTime']-row['StartTime'])
            
            # Generate mel spectrogram
            mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Plot
            axes[i].imshow(mel_spec_db, aspect='auto', origin='lower', cmap='viridis')
            
            # Color based on correctness
            color = 'green' if row['Correct'] else 'red'
            axes[i].set_title(f"True: {row['TrueClass']}\nPred: {row['PredictedClass']} ({row['Confidence']:.2f})", 
                             color=color, fontsize=10)
            axes[i].axis('off')
        except Exception as e:
            axes[i].text(0.5, 0.5, f"Error loading:\n{str(e)[:30]}", 
                        ha='center', va='center', fontsize=8)
            axes[i].axis('off')
    
    # Hide empty subplots
    for i in range(len(samples), len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(f"Spectrograms {title_suffix}", fontsize=16)
    plt.tight_layout()
    plt.show()
    
    print(f"\nDisplayed {len(samples)} samples")
    print(f"Correct: {samples['Correct'].sum()} | Incorrect: {(~samples['Correct']).sum()}")

#Plot high confidence
plot_spectrogram_results(df_results, mode='max', n_samples=25)

#Plot low confidence
plot_spectrogram_results(df_results, mode='min', n_samples=25)

# Plot incorrect predictions only
plot_spectrogram_results(df_results, mode='incorrect', n_samples=25)



# Plot confidence vs original call duration
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Scatter plot: Confidence vs Original Duration
ax1 = axes[0]
# Color by correctness
correct = df_results[df_results['Correct'] == True]
incorrect = df_results[df_results['Correct'] == False]

ax1.scatter(correct['OriginalDuration'], correct['Confidence'], 
           alpha=0.5, c='green', label='Correct', s=30)
ax1.scatter(incorrect['OriginalDuration'], incorrect['Confidence'], 
           alpha=0.5, c='red', label='Incorrect', s=30)

ax1.set_xlabel('Original Call Duration (seconds)')
ax1.set_ylabel('Confidence Score')
ax1.set_title('Confidence vs Original Call Duration')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Add correlation coefficient
from scipy.stats import pearsonr
corr, p_value = pearsonr(df_results['OriginalDuration'], df_results['Confidence'])
ax1.text(0.05, 0.95, f'Correlation: {corr:.3f}\np-value: {p_value:.3e}', 
        transform=ax1.transAxes, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Binned analysis: Average confidence by duration bins
ax2 = axes[1]
# Create duration bins
df_results['DurationBin'] = pd.cut(df_results['OriginalDuration'], 
                                    bins=10, 
                                    labels=[f'{i}' for i in range(10)])

# Calculate mean confidence and accuracy for each bin
bin_stats = df_results.groupby('DurationBin').agg({
    'Confidence': 'mean',
    'Correct': 'mean',
    'OriginalDuration': 'mean'
}).reset_index()

ax2_twin = ax2.twinx()
ax2.bar(bin_stats.index, bin_stats['Confidence'], alpha=0.6, color='blue', label='Avg Confidence')
ax2_twin.plot(bin_stats.index, bin_stats['Correct'], color='red', marker='o', 
             linewidth=2, markersize=8, label='Accuracy')

ax2.set_xlabel('Duration Bin (shorter → longer)')
ax2.set_ylabel('Average Confidence', color='blue')
ax2_twin.set_ylabel('Accuracy', color='red')
ax2.set_title('Confidence & Accuracy by Call Duration')
ax2.tick_params(axis='y', labelcolor='blue')
ax2_twin.tick_params(axis='y', labelcolor='red')
ax2.set_xticklabels([f'{d:.2f}s' for d in bin_stats['OriginalDuration']], rotation=45)

# Add legends
lines1, labels1 = ax2.get_legend_handles_labels()
lines2, labels2 = ax2_twin.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

plt.tight_layout()
plt.show()

print(f"\nCorrelation between original duration and confidence: {corr:.4f}")
print(f"P-value: {p_value:.4e}")
if corr > 0:
    print("→ Longer calls tend to have HIGHER confidence")
elif corr < 0:
    print("→ Shorter calls tend to have HIGHER confidence")
else:
    print("→ No clear relationship")