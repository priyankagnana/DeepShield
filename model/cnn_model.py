import torch
import torch.nn as nn
import torch.nn.functional as F
from model.frequency_branch import FrequencyBranch


class DeepfakeCNN(nn.Module):
    """
    Dual-branch deepfake detector.

    Spatial branch: 4 conv blocks (32→64→128→256) with BatchNorm + MaxPool
    Frequency branch: FFT magnitude processed by a separate conv stack
    Fusion: concat → BN → Dropout → classifier

    Changes from v1:
    - 4 conv blocks instead of 2 → richer spatial features, smaller FC input
      (256 × 14 × 14 = 50 176 vs original 64 × 56 × 56 = 200 704)
    - Dropout(0.5) on the spatial FC and on the fused representation
    - BatchNorm1d on the fused 256-dim vector normalises the two branches to
      the same scale before the classifier sees them
    """

    def __init__(self):
        super(DeepfakeCNN, self).__init__()

        # --- Spatial branch ---
        self.conv1 = nn.Conv2d(3,   32,  3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32,  64,  3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64,  128, 3, padding=1)
        self.bn3   = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn4   = nn.BatchNorm2d(256)
        self.pool  = nn.MaxPool2d(2, 2)

        # After 4 × pool(2,2): 224 → 14  →  256 × 14 × 14
        self.spatial_fc      = nn.Linear(256 * 14 * 14, 256)
        self.spatial_bn      = nn.BatchNorm1d(256)
        self.spatial_dropout = nn.Dropout(0.5)

        # --- Frequency branch ---
        self.frequency_branch = FrequencyBranch()

        # --- Fusion ---
        # spatial(256) + frequency(128) = 384
        self.fusion_bn      = nn.BatchNorm1d(384)
        self.fusion_dropout = nn.Dropout(0.5)
        self.classifier     = nn.Linear(384, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Spatial branch
        s = self.pool(F.relu(self.bn1(self.conv1(x))))
        s = self.pool(F.relu(self.bn2(self.conv2(s))))
        s = self.pool(F.relu(self.bn3(self.conv3(s))))
        s = self.pool(F.relu(self.bn4(self.conv4(s))))
        s = s.view(s.size(0), -1)
        s = self.spatial_dropout(F.relu(self.spatial_bn(self.spatial_fc(s))))

        # Frequency branch
        f = self.frequency_branch(x)

        # Fusion
        combined = torch.cat((s, f), dim=1)
        combined = self.fusion_dropout(self.fusion_bn(combined))
        return self.classifier(combined)