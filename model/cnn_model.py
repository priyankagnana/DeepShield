import torch
import torch.nn as nn
import torch.nn.functional as F
from model.frequency_branch import FrequencyBranch

class DeepfakeCNN(nn.Module):
    def __init__(self):
        super(DeepfakeCNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)

        self.spatial_fc = nn.Linear(64 * 56 * 56, 128)

        self.frequency_branch = FrequencyBranch()

        self.classifier = nn.Linear(256, 1)

    def forward(self, x):

        # Spatial branch (conv → BN → ReLU → pool)
        s = self.pool(F.relu(self.bn1(self.conv1(x))))
        s = self.pool(F.relu(self.bn2(self.conv2(s))))
        s = s.view(s.size(0), -1)
        s = self.spatial_fc(s)

        # Frequency branch
        f = self.frequency_branch(x)

        # Fusion
        combined = torch.cat((s, f), dim=1)
        output = self.classifier(combined)

        return output