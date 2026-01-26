import bioacoustics_model_zoo as bmz
from pathlib import Path
model = bmz.Perch2(version=1)
print(f"Model device is: {model.device}")
print(f"Model system is: {model.system}")
print(f"Setting model device and system")
model.device = "cpu"
model.system = "CPU"
print(f"Model device is: {model.device}")
print(f"Model system is: {model.system}")
padded_5s = "/home/Shelby/blackbird_calls/Experiments/Detection_of_calls/Datasets/All_call_classes/Split_data_controlled/OpenSoundScape/Call_audios/padded_5s"
wav_files = list(Path(padded_5s).glob("*.WAV"))
print(f"Found {len(wav_files)} wav files")
#print(f"Running on {model.system}")
model.predict(wav_files)
