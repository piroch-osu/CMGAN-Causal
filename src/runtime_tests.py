import os
import numpy as np
import soundfile as sf
import torchaudio
import torch
from models import generator
from tools.compute_metrics import compute_metrics
import torchaudio.transforms as T
import time

from evaluation import enhance_one_track

torch.set_num_threads(4)  # or num of CPU cores on the node


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


TEST_DURATIONS = [0.5, 1, 2, 5, 10]  # seconds to test
SAMPLE_RATE = 16000 # Required for PESQ


# # Pre-create resampler for torchaudio
# resampler = T.Resample(orig_freq=0, new_freq=SAMPLE_RATE)  # orig_freq will be set dynamically


def load_and_trim(audio_path, seconds, target_sr=SAMPLE_RATE):
    # Load full WAV
    audio, sr = torchaudio.load(audio_path)

    # Check that sample rate is 16k
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        audio = resampler(audio)
        sr = target_sr
    
    N = int(seconds * SAMPLE_RATE)

    # pad audio if too short
    if audio.size(-1) < N:
        padding = N - audio.size(-1)
        audio = torch.cat([audio, torch.zeros(1, padding)], dim=-1)

    return audio[:, :N], sr

if __name__ == "__main__":
    # Initialize model
    model = generator.TSCNet(num_channel=64, num_features=201).to(device)
    model.load_state_dict(torch.load("./best_ckpt/ckpt_80", map_location=device))
    model.eval()

    noisy_path = "/users/PAS3239/hamilton1565/CMGAN-Causal/VCTK-DEMAND/test/noisy/p232_001.wav"
    clean_path = "/users/PAS3239/hamilton1565/CMGAN-Causal/VCTK-DEMAND/test/clean/p232_001.wav"

    # Load clean reference audio
    clean_audio, clean_sr = sf.read(clean_path)
    clean_audio = clean_audio.astype(np.float32)

    if clean_sr != SAMPLE_RATE:
        clean_audio_t = torch.from_numpy(clean_audio).unsqueeze(0)
        resampler = torchaudio.transforms.Resample(orig_freq=clean_sr, new_freq=SAMPLE_RATE)
        clean_audio_t = resampler(clean_audio_t)
        clean_audio = clean_audio_t.squeeze().numpy()
        clean_sr = SAMPLE_RATE

    os.makedirs("./temp", exist_ok=True)

    # Loop through durations
    for dur in TEST_DURATIONS:
        noisy_trimmed, _ = load_and_trim(noisy_path, dur)
        temp_path = f"./temp/temp_{dur}s.wav" # Save trimmed audio to temp WAV
        sf.write(temp_path, noisy_trimmed.squeeze().numpy(), SAMPLE_RATE)

        # Enhance trimmed audio
        est_audio, length, runtime = enhance_one_track(
            model, temp_path, "./temp", cut_len=16000*16, save_tracks=False
        )

        # Compute metrics
        metrics = compute_metrics(clean_audio[:length], est_audio, int(SAMPLE_RATE), 0)
        clip_dur = length / SAMPLE_RATE
        rtf = runtime / clip_dur


        print(f"\n=== {dur} seconds ===")
        print("Runtime:", runtime)
        print("RTF: ", rtf)
        print("PESQ:", metrics[0])
        print("STOI:", metrics[5])
