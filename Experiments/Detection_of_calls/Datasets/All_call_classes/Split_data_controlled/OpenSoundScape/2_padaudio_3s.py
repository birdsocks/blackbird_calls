from opensoundscape import Audio
from pathlib import Path
import os
import numpy as np

# Paths
call_audios_dir = "/home/Shelby/blackbird_calls/Experiments/Detection_of_calls/Datasets/All_call_classes/Split_data_controlled/OpenSoundScape/Call_audios"
padded_output_dir = os.path.join(call_audios_dir, "padded_3s")

# Create output directory for padded files
os.makedirs(padded_output_dir, exist_ok=True)

# Get all trimmed WAV files
wav_files = list(Path(call_audios_dir).glob("*.WAV"))
print(f"Found {len(wav_files)} trimmed audio files to pad")

# Target duration in seconds
target_duration = 3.0

# Process each file
padded_count = 0
for wav_file in wav_files:
    # Load the audio
    audio = Audio.from_file(wav_file)
    
    # Get current duration and sample rate
    current_duration = audio.duration
    sample_rate = audio.sample_rate
    
    # Calculate total samples needed for 3 seconds
    target_samples = int(target_duration * sample_rate)
    current_samples = len(audio.samples)
    
    # Only pad if duration is less than 3 seconds
    if current_duration < target_duration:
        # Calculate padding needed in samples
        padding_samples = target_samples - current_samples
        pad_before_samples = padding_samples // 2
        pad_after_samples = padding_samples - pad_before_samples  # Handle odd numbers
        
        # Create zero arrays for padding
        zeros_before = np.zeros(pad_before_samples)
        zeros_after = np.zeros(pad_after_samples)
        
        # Concatenate: zeros + original audio + zeros
        padded_samples = np.concatenate([zeros_before, audio.samples, zeros_after])
        
        # Create new Audio object from padded samples
        padded_audio = Audio(samples=padded_samples, sample_rate=sample_rate)
    else:
        # If already 3 seconds or longer, trim to exactly 3 seconds
        padded_audio = audio.trim(start_time=0, end_time=target_duration)
    
    # Save the padded audio with the same filename
    output_path = os.path.join(padded_output_dir, wav_file.name)
    padded_audio.save(output_path)
    
    padded_count += 1
    if padded_count % 100 == 0:
        print(f"Processed {padded_count}/{len(wav_files)} files...")

print(f"\nCompleted! Padded {padded_count} audio files to 3 seconds and saved to {padded_output_dir}")
