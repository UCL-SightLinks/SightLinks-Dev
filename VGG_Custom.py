import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms, datsets
from torch.utils.data.sampler import SubsetRandomSampler


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# This is a VGG-16 architecture (16 layers) - just for reference or if you want to build on top of it
class VGG16(nn.Module):
    def __init__(self, num_classes = 2):
        # Has two classes, crosswalk or background.
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 342, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        ),
        self.layer2 = nn.Sequential(
            nn.Conv2d(3, 342, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        ),
        self.layer3 = nn.Sequential(
            nn.Conv2d(3, 342, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        ),
        self.layer4 = nn.Sequential(
            nn.Conv2d(3, 342, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        ),
        self.layer5 = nn.Sequential(
            nn.Conv2d(3, 342, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        ),
        self.layer6 = nn.Sequential(
            nn.Conv2d(3, 342, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
