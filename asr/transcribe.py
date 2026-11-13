import argparse
from pathlib import Path

import soundfile as sf
import torch

from .data import log_mel_spectrogram
from .model import SmallCNNGRUASR
from .text import Vocabulary


def main() -> None:
    parser = argparse.ArgumentParser(description="Transcribe a 16 kHz audio file.")
    parser.add_argument("audio", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=Path("asr-checkpoint.pt"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    vocabulary = Vocabulary()
    if checkpoint["vocabulary_characters"] != vocabulary.characters:
        raise ValueError("Checkpoint vocabulary does not match this inference code")
    model = SmallCNNGRUASR(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    waveform, sample_rate = sf.read(args.audio, dtype="float32")
    if sample_rate != 16_000:
        raise ValueError(f"Expected 16 kHz audio, received {sample_rate} Hz")
    waveform = torch.from_numpy(waveform)
    if waveform.ndim == 2:
        waveform = waveform.mean(dim=1)
    if waveform.numel() == 0:
        raise ValueError(f"Audio file is empty: {args.audio}")
    features = log_mel_spectrogram(waveform).unsqueeze(0).to(device)
    input_lengths = torch.tensor([features.size(1)], dtype=torch.long)
    with torch.inference_mode():
        logits, output_lengths = model(features, input_lengths)
    print(vocabulary.decode(logits[0, : output_lengths.item()].argmax(dim=-1).tolist()))


if __name__ == "__main__":
    main()
