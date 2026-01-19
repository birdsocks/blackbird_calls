import os
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

# === PARAMETERS ===
images_dir = "/home/Shelby/blackbird_calls/Experiments/Detection_of_calls/Datasets/Images_all_classes"
crops_dir = "/home/Shelby/blackbird_calls/Experiments/Detection_of_calls/Crops_all_classes"
os.makedirs(crops_dir, exist_ok=True)

csv_path = "/home/Shelby/blackbird_calls/Dataset_processing/Datasets/cv4e_calls_channel1_v2.csv"
annotations_df = pd.read_csv(csv_path)

img_w_px, img_h_px = 1280, 720
segment_length = 5  # seconds
freq_min, freq_max = 0, 24000  # Hz (adjust if needed for your spectrograms)

# Helper: Map time/freq to pixel coordinates
def data_to_pixels(time_s, freq_hz, time_range, freq_range, img_shape):
    x_px = (time_s - time_range[0]) / (time_range[1] - time_range[0]) * img_shape[1]
    y_px = img_shape[0] - (freq_hz - freq_range[0]) / (freq_range[1] - freq_range[0]) * img_shape[0]
    return int(x_px), int(y_px)

# Loop through all images
for img_name in os.listdir(images_dir):
    if not img_name.endswith('.png'):
        continue
    # Parse file info
    stem = Path(img_name).stem
    # Example: SL54 Trial 1_trim_spec_00-05.png
    if '_spec_' not in stem:
        continue
    wav_part, seg_part = stem.split('_spec_')
    # Get segment start/end in seconds
    try:
        seg_start, seg_end = [int(x) for x in seg_part.split('-')]
    except Exception:
        continue
    time_range = (seg_start, seg_end)
    freq_range = (freq_min, freq_max)

    # Find matching annotations for this wav file and segment
    matches = annotations_df[annotations_df['wav_fname'].str.replace('.WAV','').str.replace('.wav','') == wav_part.strip()]
    # Filter to calls within this segment
    matches = matches[(matches['begin_time_s'] < seg_end) & (matches['end_time_s'] > seg_start)]
    if matches.empty:
        continue

    # Load image
    img_path = os.path.join(images_dir, img_name)
    img = Image.open(img_path)
    img_shape = img.size[::-1]  # (height, width)
    
    for idx, row in matches.iterrows():
        # Clip call to segment bounds
        call_start = max(row['begin_time_s'], seg_start)
        call_end = min(row['end_time_s'], seg_end)
        call_low = max(row['low_freq_hz'], freq_min)
        call_high = min(row['high_freq_hz'], freq_max)
        # Map to pixel coordinates
        x0, y1 = data_to_pixels(call_start, call_high, time_range, freq_range, img_shape)
        x1, y0 = data_to_pixels(call_end, call_low, time_range, freq_range, img_shape)
        # Ensure valid crop box
        x0, x1 = sorted([max(0, x0), min(img_shape[1], x1)])
        y0, y1 = sorted([max(0, y0), min(img_shape[0], y1)])
        if x1 - x0 < 5 or y1 - y0 < 5:
            continue  # Skip tiny crops
        crop = img.crop((x0, y0, x1, y1))
        crop_fname = f"{stem}_call_{idx}.png"
        crop.save(os.path.join(crops_dir, crop_fname))
        print(f"Saved crop: {crop_fname}")