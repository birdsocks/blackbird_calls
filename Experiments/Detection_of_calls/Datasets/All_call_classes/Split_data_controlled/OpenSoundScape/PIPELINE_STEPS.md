# Complete Pipeline: From Raw Audio to Trained Classifier

This document outlines the complete processing pipeline you followed to train your bird call classifier.

---

## 📋 Overview
**Start:** Raw 1-minute audio files + bounding box annotations  
**End:** Trained BirdNET classifier on padded 3-second audio clips

---

## 🔄 Processing Steps (In Order)

### **Step 1: Trim Audio to Call Boxes**
**File:** `1_preprocessing_trim.py`

**Input:**
- Raw 1-minute WAV files (`/mnt/class_data/Shelby/One_Minute_Audio/`)
- Bounding box annotations CSV (`cv4e_calls_channel1_v2.csv`)

**What it does:**
- Extracts individual call segments from 1-minute audio files based on bounding box timestamps
- Trims audio to exact call duration (varies per call)
- Saves trimmed files with naming format: `{trial}_{start_time}_{end_time}_{calltype}.WAV`

**Output:**
- Individual trimmed call audio files in `Call_audios/` folder

---

### **Step 2: Pad Audio to 3 Seconds**
**File:** `2_padaudio_3s.py`

**Input:**
- Trimmed audio files from Step 1 (various durations)

**What it does:**
- Pads all audio clips to exactly 3.0 seconds duration
- Adds zero-padding before and after the original audio (centered)
- If clip is already ≥3s, trims to exactly 3s

**Output:**
- 3-second padded audio files in `Call_audios/padded_3s/` folder

---

### **Step 3: Generate BirdNET Embeddings**
**File:** `3_BirdNETembeddings.py`

**Input:**
- 3-second padded audio files from Step 2

**What it does:**
- Uses pre-trained BirdNET model to extract audio embeddings
- Creates high-dimensional feature vectors (embeddings) representing each audio clip
- Links embeddings with file paths and call types

**Output:**
- `birds_embeddings_birdnet_padded_3s.csv`
  - Contains: file path, calltype, and 1024 embedding dimensions

---

### **Step 4: Create Clip Labels from Bounding Boxes**
**File:** `Process_boxes_to_classifier_labels.ipynb`

**Input:**
- Original bounding box CSV (`cv4e_calls_channel1_v2.csv`)
- 1-minute audio file paths

**What it does:**
- Creates 3-second sliding windows over 1-minute audio files
- Determines which call types are present in each 3-second clip
- Applies overlap thresholds (min_label_overlap=0.2, min_label_fraction=0.5)
- Creates multi-hot encoded labels (multiple call types per clip possible)

**Output:**
- `clip_labels_3s.csv`
  - Index: (file, start_time, end_time)
  - Columns: Binary 0/1 for each call type present in clip

**Note:** This file wasn't directly used in your final pipeline - you used the condensed embeddings instead.

---

### **Step 5: Filter and Prepare Data (Optional - done in ClusterembeddingsTSNE)**
**File:** `ClusterembeddingsTSNE.ipynb` (Cell 2)

**Input:**
- `birds_embeddings_birdnet_padded_3s.csv`

**What it does:**
- Renames similar call types (M→A, O→E, I→D, L→C)
- Filters out unwanted call types (F, Growl, H, N)
- Reduces dataset to focus on key call types

**Output:**
- `birds_embeddings_birdnet_padded_3s_condensed.csv`
  - Filtered and renamed version with ~1865 samples
  - Contains: file, start_time, end_time, calltype, and embeddings

---

### **Step 6: Split Data into Train/Val/Test Sets**
**File:** `padded3s_classifier/a01_resplitting.ipynb`

**Input:**
- `birds_embeddings_birdnet_padded_3s_condensed.csv`
- YOLO dataset split files (train.txt, validate.txt, test.txt from YOLO folder)

**What it does:**
- Loads the condensed embeddings CSV
- Extracts trial IDs from each audio file path
- Matches trial IDs to YOLO splits to maintain consistent train/val/test splits
- Creates one-hot encoded labels (converts calltype column to binary columns)
- Ensures no data leakage by splitting at trial level (not individual clips)

**Output:**
- `padded3s_classifier/train_set_onehot.csv` (~70% of data)
- `padded3s_classifier/val_set_onehot.csv` (~15% of data)
- `padded3s_classifier/test_set_onehot.csv` (~15% of data)

Each CSV contains:
- `file`: Path to padded 3s audio file
- `start_time`, `end_time`: Always 0.0, 3.0 (since files are 3s)
- `calltype_A`, `calltype_B`, ... : One-hot encoded labels (0 or 1)

---

### **Step 7: Train BirdNET Classifier**
**File:** `padded3s_classifier/b02_train_on_embeddings_100epochs_fulldataset.ipynb`

**Input:**
- `train_set_onehot.csv`
- `val_set_onehot.csv`
- `test_set_onehot.csv`

**What it does:**
1. Loads the one-hot encoded train/val/test splits
2. Optionally resamples training data to balance classes
3. Uses BirdNET to generate embeddings from audio files on-the-fly
4. Trains a classification head on top of BirdNET embeddings
5. Monitors validation performance (loss, AU ROC, MAP)
6. Evaluates on test set with classification reports and confusion matrices
7. Saves the trained model

**Parameters:**
- `steps=100` epochs
- No resampling (`RESAMPLE=False`)
- Uses full dataset without augmentation

**Output:**
- Trained model: `blackbirdcallv1.model`
- Training history: `training/training_history.pkl`
- Performance metrics, confusion matrices, classification reports

---

## 📊 Data Flow Diagram

```
Raw Audio (1-min WAV) + Bounding Boxes (CSV)
    ↓
[1_preprocessing_trim.py]
    ↓
Individual Call Segments (various durations)
    ↓
[2_padaudio_3s.py]
    ↓
3-Second Padded Audio Clips
    ↓
[3_BirdNETembeddings.py]
    ↓
birds_embeddings_birdnet_padded_3s.csv
    ↓
[ClusterembeddingsTSNE.ipynb - Cell 2] (Optional filtering)
    ↓
birds_embeddings_birdnet_padded_3s_condensed.csv
    ↓
[a01_resplitting.ipynb]
    ↓
train_set_onehot.csv, val_set_onehot.csv, test_set_onehot.csv
    ↓
[b02_train_on_embeddings_100epochs_fulldataset.ipynb]
    ↓
Trained Model (blackbirdcallv1.model)
```

---

## 📁 Key Files Summary

### Scripts (Run Once)
1. `1_preprocessing_trim.py` - Extract calls from 1-min audio
2. `2_padaudio_3s.py` - Pad to 3 seconds
3. `3_BirdNETembeddings.py` - Generate BirdNET embeddings

### Notebooks (Interactive Processing)
4. `Process_boxes_to_classifier_labels.ipynb` - Create clip labels (not used in final pipeline)
5. `ClusterembeddingsTSNE.ipynb` (Cell 2) - Filter and rename call types
6. `padded3s_classifier/a01_resplitting.ipynb` - Create train/val/test splits
7. `padded3s_classifier/b02_train_on_embeddings_100epochs_fulldataset.ipynb` - **Final training**

### Intermediate Data Files
- `birds_embeddings_birdnet_padded_3s.csv` - Raw embeddings (all call types)
- `birds_embeddings_birdnet_padded_3s_condensed.csv` - Filtered embeddings
- `train_set_onehot.csv`, `val_set_onehot.csv`, `test_set_onehot.csv` - Split datasets

### Final Output
- `blackbirdcallv1.model` - Trained classifier

---

## 🔑 Key Design Decisions

1. **3-Second Clips:** Standardized duration for model input
2. **Zero-Padding:** Preserves original call centered in clip
3. **Trial-Level Splits:** Prevents data leakage between train/val/test
4. **Call Type Filtering:** Focused on reliable, well-represented call types
5. **No Resampling:** Used natural class distribution for final model

---

## 🚀 To Reproduce This Pipeline

Run in order:
```bash
# 1. Extract calls from raw audio
python 1_preprocessing_trim.py

# 2. Pad all calls to 3 seconds
python 2_padaudio_3s.py

# 3. Generate BirdNET embeddings
python 3_BirdNETembeddings.py

# 4. Filter data (run Cell 2 in ClusterembeddingsTSNE.ipynb)
# Creates: birds_embeddings_birdnet_padded_3s_condensed.csv

# 5. Split data (run padded3s_classifier/a01_resplitting.ipynb)
# Creates: train_set_onehot.csv, val_set_onehot.csv, test_set_onehot.csv

# 6. Train classifier (run b02_train_on_embeddings_100epochs_fulldataset.ipynb)
# Creates: blackbirdcallv1.model
```

---

Generated: 2026-01-29
