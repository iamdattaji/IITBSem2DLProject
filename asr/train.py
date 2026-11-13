import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import LibriSpeechDataset, collate_batch
from .metrics import character_error_rate, word_error_rate
from .model import SmallCNNGRUASR
from .text import Vocabulary


def _run_epoch(
    model: SmallCNNGRUASR,
    loader: DataLoader,
    vocabulary: Vocabulary,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    references: list[str] = []
    hypotheses: list[str] = []
    batches = 0

    for features, input_lengths, targets, target_lengths, transcripts in loader:
        features = features.to(device)
        targets = targets.to(device)
        with torch.set_grad_enabled(training):
            logits, output_lengths = model(features, input_lengths)
            loss = F.ctc_loss(
                logits.log_softmax(dim=-1).transpose(0, 1),
                targets,
                output_lengths,
                target_lengths,
                blank=vocabulary.blank_id,
                zero_infinity=True,
            )
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

        predictions = logits.argmax(dim=-1)
        for prediction, length, transcript in zip(
            predictions, output_lengths.tolist(), transcripts
        ):
            references.append(transcript)
            hypotheses.append(vocabulary.decode(prediction[:length].tolist()))
        total_loss += loss.item()
        batches += 1

    return (
        total_loss / max(batches, 1),
        word_error_rate(references, hypotheses),
        character_error_rate(references, hypotheses),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a compact LibriSpeech CTC model.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--train-split", default="train-clean-100")
    parser.add_argument("--valid-split", default="dev-clean")
    parser.add_argument("--output", type=Path, default=Path("asr-checkpoint.pt"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-train-utterances", type=int)
    parser.add_argument("--max-valid-utterances", type=int)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    vocabulary = Vocabulary()
    train_data = LibriSpeechDataset(
        args.dataset_root,
        args.train_split,
        vocabulary,
        max_utterances=args.max_train_utterances,
    )
    valid_data = LibriSpeechDataset(
        args.dataset_root,
        args.valid_split,
        vocabulary,
        max_utterances=args.max_valid_utterances,
    )
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_batch,
        pin_memory=device.type == "cuda",
    )
    valid_loader = DataLoader(
        valid_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_batch,
        pin_memory=device.type == "cuda",
    )

    model = SmallCNNGRUASR(vocabulary.size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    best_valid_wer = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_wer, train_cer = _run_epoch(
            model, train_loader, vocabulary, device, optimizer
        )
        valid_loss, valid_wer, valid_cer = _run_epoch(
            model, valid_loader, vocabulary, device
        )
        print(
            f"epoch={epoch} train_loss={train_loss:.4f} "
            f"train_wer={train_wer:.4f} train_cer={train_cer:.4f} "
            f"valid_loss={valid_loss:.4f} valid_wer={valid_wer:.4f} "
            f"valid_cer={valid_cer:.4f}",
            flush=True,
        )
        if valid_wer < best_valid_wer:
            best_valid_wer = valid_wer
            args.output.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "vocabulary_characters": vocabulary.characters,
                    "model_config": {"vocabulary_size": vocabulary.size},
                    "epoch": epoch,
                    "valid_wer": valid_wer,
                    "valid_cer": valid_cer,
                },
                args.output,
            )
    print(f"Best checkpoint saved to {args.output}")


if __name__ == "__main__":
    main()
