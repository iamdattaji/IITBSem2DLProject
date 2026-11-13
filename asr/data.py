from pathlib import Path

import soundfile as sf
import torch
from torch.utils.data import Dataset

from .text import Vocabulary, normalize_text


class LibriSpeechDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str,
        vocabulary: Vocabulary,
        sample_rate: int = 16_000,
        max_utterances: int | None = None,
    ) -> None:
        root = Path(root).expanduser()
        split_root = root / split
        if not split_root.is_dir():
            raise FileNotFoundError(
                f"Split '{split}' was not found under {root}. "
                "Pass the directory containing the LibriSpeech split folders."
            )

        audio_paths = {
            path.stem: path
            for pattern in ("*.flac", "*.wav")
            for path in split_root.rglob(pattern)
        }
        self.samples: list[tuple[Path, str]] = []
        for transcript_path in sorted(split_root.rglob("*.trans.txt")):
            for line_number, line in enumerate(
                transcript_path.read_text(encoding="utf-8").splitlines(), start=1
            ):
                utterance_id, separator, transcript = line.partition(" ")
                if not separator or not transcript.strip():
                    raise ValueError(
                        f"Malformed transcript at {transcript_path}:{line_number}"
                    )
                audio_path = audio_paths.get(utterance_id)
                if audio_path is None:
                    raise FileNotFoundError(
                        f"No FLAC or WAV file found for {utterance_id} "
                        f"from {transcript_path}"
                    )
                transcript = normalize_text(transcript)
                if transcript:
                    self.samples.append((audio_path, transcript))
                    if max_utterances is not None and len(self.samples) >= max_utterances:
                        break
            if max_utterances is not None and len(self.samples) >= max_utterances:
                break

        if not self.samples:
            raise ValueError(f"No usable utterances found under {split_root}")
        self.vocabulary = vocabulary
        self.sample_rate = sample_rate

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        audio_path, transcript = self.samples[index]
        waveform, sample_rate = sf.read(audio_path, dtype="float32")
        if sample_rate != self.sample_rate:
            raise ValueError(
                f"{audio_path} has sample rate {sample_rate}; expected "
                f"{self.sample_rate} Hz"
            )
        waveform = torch.from_numpy(waveform)
        if waveform.ndim == 2:
            waveform = waveform.mean(dim=1)
        if waveform.numel() == 0:
            raise ValueError(f"Audio file is empty: {audio_path}")

        features = log_mel_spectrogram(waveform, self.sample_rate)
        target = torch.tensor(self.vocabulary.encode(transcript), dtype=torch.long)
        return features, target, transcript


def _mel_filterbank(
    sample_rate: int, n_fft: int, n_mels: int, device: torch.device
) -> torch.Tensor:
    def hz_to_mel(frequency: torch.Tensor) -> torch.Tensor:
        return 2595.0 * torch.log10(1.0 + frequency / 700.0)

    def mel_to_hz(mel: torch.Tensor) -> torch.Tensor:
        return 700.0 * (torch.pow(10.0, mel / 2595.0) - 1.0)

    minimum_mel = hz_to_mel(torch.tensor(0.0, device=device))
    maximum_mel = hz_to_mel(torch.tensor(sample_rate / 2, device=device))
    mel_points = torch.linspace(minimum_mel, maximum_mel, n_mels + 2, device=device)
    hz_points = mel_to_hz(mel_points)
    frequencies = torch.linspace(0, sample_rate / 2, n_fft // 2 + 1, device=device)
    lower = hz_points[:-2, None]
    center = hz_points[1:-1, None]
    upper = hz_points[2:, None]
    rising = (frequencies[None, :] - lower) / (center - lower)
    falling = (upper - frequencies[None, :]) / (upper - center)
    return torch.minimum(rising, falling).clamp(min=0.0)


def log_mel_spectrogram(
    waveform: torch.Tensor,
    sample_rate: int = 16_000,
    n_mels: int = 80,
    n_fft: int = 400,
    hop_length: int = 160,
) -> torch.Tensor:
    window = torch.hann_window(n_fft, device=waveform.device)
    spectrum = torch.stft(
        waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=window,
        center=True,
        return_complex=True,
    ).abs().square()
    filters = _mel_filterbank(sample_rate, n_fft, n_mels, waveform.device)
    features = torch.log(torch.matmul(filters, spectrum).clamp_min(1e-10)).transpose(0, 1)
    mean = features.mean(dim=0, keepdim=True)
    standard_deviation = features.std(
        dim=0, keepdim=True, correction=0
    ).clamp_min(1e-5)
    return (features - mean) / standard_deviation


def collate_batch(
    batch: list[tuple[torch.Tensor, torch.Tensor, str]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    features, targets, transcripts = zip(*batch)
    input_lengths = torch.tensor([item.size(0) for item in features], dtype=torch.long)
    target_lengths = torch.tensor([item.numel() for item in targets], dtype=torch.long)
    padded_features = torch.nn.utils.rnn.pad_sequence(features, batch_first=True)
    concatenated_targets = torch.cat(targets)
    return (
        padded_features,
        input_lengths,
        concatenated_targets,
        target_lengths,
        list(transcripts),
    )
