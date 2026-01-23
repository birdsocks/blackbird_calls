import bioacoustics_model_zoo as bmz
from pathlib import Path
model = bmz.Perch2()
model.device = "cpu"

audio_dir = "/mnt/class_data/Shelby/One_Minute_Audio"
directory = Path(audio_dir)

# find all the .WAV files in the directory
wav_files = list(directory.rglob("*.WAV"))
print(f"Found {len(wav_files)} .WAV files.")
example = wav_files[0]
print(example)

model.predict([example])
