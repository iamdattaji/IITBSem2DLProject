import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class SmallCNNGRUASR(nn.Module):
    """A compact CNN + bidirectional GRU acoustic model trained with CTC."""

    def __init__(
        self,
        vocabulary_size: int,
        n_mels: int = 80,
        cnn_channels: int = 48,
        hidden_size: int = 128,
        recurrent_layers: int = 2,
    ) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 24, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(24),
            nn.GELU(),
            nn.Conv2d(24, cnn_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(cnn_channels),
            nn.GELU(),
        )
        reduced_mels = (n_mels + 3) // 4
        self.recurrent = nn.GRU(
            input_size=cnn_channels * reduced_mels,
            hidden_size=hidden_size,
            num_layers=recurrent_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.1 if recurrent_layers > 1 else 0.0,
        )
        self.classifier = nn.Linear(hidden_size * 2, vocabulary_size)

    def forward(
        self, features: torch.Tensor, input_lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(features.unsqueeze(1))
        batch, channels, time_steps, frequency_bins = encoded.shape
        encoded = encoded.permute(0, 2, 1, 3).reshape(
            batch, time_steps, channels * frequency_bins
        )
        output_lengths = torch.div(input_lengths + 3, 4, rounding_mode="floor")
        packed = pack_padded_sequence(
            encoded,
            output_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_output, _ = self.recurrent(packed)
        encoded, _ = pad_packed_sequence(
            packed_output, batch_first=True, total_length=time_steps
        )
        return self.classifier(encoded), output_lengths
