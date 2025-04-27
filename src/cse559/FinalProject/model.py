import torch
import torch.nn as nn
import torch.nn.functional as F

class facial_expressionCNN(nn.Module):
    """
    PyTorch model for real-time facial expression CNN.
    Input: 1×48×48  ──► 7-class soft-max.
    """
    def __init__(self, n_classes: int = 7):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=0),  # valid
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=5, stride=2),                 # valid

            # Block 2
            nn.Conv2d(64, 64, kernel_size=3),
            nn.ReLU(inplace=True),

            # Block 3
            nn.Conv2d(64, 64, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=3, stride=2),

            # Block 4
            nn.Conv2d(64, 128, kernel_size=3),
            nn.ReLU(inplace=True),

            # Block 5
            nn.Conv2d(128, 128, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=3, stride=2)                  # output 128×1×1
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),               # 128
            nn.Linear(128, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(1024, n_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)
