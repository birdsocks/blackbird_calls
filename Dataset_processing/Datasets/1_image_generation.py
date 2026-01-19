import os
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import librosa
import librosa.display

# === SET UP ===
trimmed_audio_dir = "/mnt/class_data/Shelby/One_Minute_Audio"
output_dir = "/home/Shelby/blackbird_calls/Experiments/Detection_of_calls/Datasets/Images_all_classes"
os.makedirs(output_dir, exist_ok=True)

audio_files = [f for f in os.listdir(trimmed_audio_dir) if f.endswith('.WAV') or f.endswith('.wav')]
print(f"Found {len(audio_files)} total files")

csv_path = "/home/Shelby/blackbird_calls/Dataset_processing/Datasets/cv4e_calls_channel1_v2.csv"
annotations_df = pd.read_csv(csv_path)

img_w_px, img_h_px = 1280, 720
dpi = 120
duration_limit = 60  # seconds
segment_length = 5   # seconds

for file_to_visualize in audio_files:
    print(f"Processing: {file_to_visualize}")
    current_wav_fname = file_to_visualize
    matching_annotations = annotations_df[annotations_df["wav_fname"] == current_wav_fname]
    if len(matching_annotations) == 0:
        matching_annotations = annotations_df[annotations_df["wav_fname"].str.lower() == current_wav_fname.lower()]

    audio_path = os.path.join(trimmed_audio_dir, file_to_visualize)
    try:
        y, sr = librosa.load(audio_path, sr=None, mono=False)
        if y.ndim > 1:
            y = y[0]

        total_samples = int(sr * duration_limit)
        y = y[:total_samples]

        num_segments = duration_limit // segment_length
        for i in range(int(num_segments)):
            start_sample = int(i * segment_length * sr)
            end_sample = int((i + 1) * segment_length * sr)
            y_segment = y[start_sample:end_sample]

            fig, ax = plt.subplots(figsize=(img_w_px / dpi, img_h_px / dpi), dpi=dpi)

            n_fft = 1024
            win_length = 1024
            hop_length = 256
            window = "hann"

            D = librosa.stft(y_segment, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, center=False)
            S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

            img = librosa.display.specshow(S_db, sr=sr, hop_length=hop_length, ax=ax, x_axis="off", y_axis="off", shading="nearest", antialiased=False)
            ax.axis('off')

            out_png_fname = f"{Path(file_to_visualize).stem}_spec_{i*segment_length:02d}-{(i+1)*segment_length:02d}.png"
            out_png_path = os.path.join(output_dir, out_png_fname)
            plt.savefig(out_png_path, dpi=dpi, bbox_inches="tight", pad_inches=0)
            plt.close(fig)
            print(f"Saved spectrogram: {out_png_path}")

    except Exception as e:
        print(f"✗ Error processing {file_to_visualize}: {str(e)}")