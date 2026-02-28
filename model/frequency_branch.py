import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft


class FrequencyBranch(nn.Module):
    """
    Processes the FFT magnitude spectrum of the input image through a small
    conv stack instead of a single giant linear layer.

    Key improvements over the original:
    - log1p compression brings the huge dynamic range of FFT magnitudes
      (DC component can be 1000× larger than high-freq bins) to a sensible
      scale before any learnable layers see it.
    - fftshift centres the low-frequency content so the conv filters can
      learn spatially-meaningful frequency patterns (e.g. GAN grid artefacts
      always appear at specific positions in the centred spectrum).
    - Conv layers share weights spatially → far fewer parameters than the
      original 150 k→128 linear layer (~19 M params), which badly overfit.
    - BatchNorm after each conv stabilises training.
    """

    def __init__(self):
        super(FrequencyBranch, self).__init__()

        # 3-channel magnitude spectrum → progressively richer features
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.pool  = nn.MaxPool2d(2, 2)

        # After two pool(2,2): 224 → 56  →  64 × 56 × 56
        self.fc = nn.Linear(64 * 56 * 56, 128)
        self.bn_fc = nn.BatchNorm1d(128)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute 2-D FFT and take the magnitude spectrum
        magnitude = torch.abs(torch.fft.fft2(x))

        # Shift DC to the centre so low/high-freq layout is spatially consistent
        magnitude = torch.fft.fftshift(magnitude)

        # Log-compress to tame the extreme dynamic range
        magnitude = torch.log1p(magnitude)

        out = self.pool(F.relu(self.bn1(self.conv1(magnitude))))
        out = self.pool(F.relu(self.bn2(self.conv2(out))))
        out = out.view(out.size(0), -1)
        out = F.relu(self.bn_fc(self.fc(out)))

        return out