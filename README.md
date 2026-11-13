# Lightweight LibriSpeech ASR

A small, from-scratch speech recognizer built with PyTorch. It converts 16 kHz
audio to per-utterance normalized log-Mel features, encodes them with a compact
CNN and bidirectional GRU, and trains with CTC loss. Validation reports word
error rate (WER) and character error rate (CER).

## Setup

Use Python 3.10 or newer and install the dependencies:

```bash
python -m pip install -r requirements.txt
```

Download and extract the Kaggle dataset
[`pypiahmad/librispeech-asr-corpus`](https://www.kaggle.com/datasets/pypiahmad/librispeech-asr-corpus).
The loader expects the standard LibriSpeech layout, with transcript files and
audio under split folders, for example:

```text
data/LibriSpeech/
  train-clean-100/<speaker>/<chapter>/*.flac
  train-clean-100/<speaker>/<chapter>/*.trans.txt
  dev-clean/<speaker>/<chapter>/*.flac
  dev-clean/<speaker>/<chapter>/*.trans.txt
```

Each `.trans.txt` row must contain an utterance ID followed by its transcript.
The loader also accepts WAV files. Audio must be 16 kHz; stereo audio is
converted to mono.

## Train

Run a small smoke-sized experiment first, then increase the utterance limits,
epochs, or use the full split:

```bash
python -m asr.train \
  --dataset-root data/LibriSpeech \
  --max-train-utterances 500 \
  --max-valid-utterances 100 \
  --epochs 3 \
  --output checkpoints/asr.pt
```

For a longer run, omit the `--max-*-utterances` options. The default splits are
`train-clean-100` and `dev-clean`; override them with `--train-split` and
`--valid-split` if the Kaggle copy uses different split names. Training saves
the checkpoint with the best validation WER.

## Transcribe

```bash
python -m asr.transcribe path/to/audio.wav --checkpoint checkpoints/asr.pt
```

The recognizer is intentionally small and trained from scratch, so a tiny
training subset is useful for checking the pipeline but is not expected to
match pretrained or large-scale ASR accuracy.
