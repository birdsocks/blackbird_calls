import bioacoustics_model_zoo as bmz
from pathlib import Path
model = bmz.Perch2()
model.device = "cpu"
padded_5s = "/home/Shelby/blackbird_calls/Experiments/Detection_of_calls/Datasets/All_call_classes/Split_data_controlled/OpenSoundScape/Call_audios/padded_5s"
wav_files = list(Path(padded_5s).glob("*.WAV"))
print(f"Found {len(wav_files)} wav files")
model.predict(wav_files)
