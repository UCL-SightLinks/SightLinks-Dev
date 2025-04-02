# Besides the usual feature extraction/ preprocessing methods, or statistical ones - we can also employ ML models to automatically
# build feature extraction layers, using self-supervised learning for example.

# This is an implementation of simCLR

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Subset
import numpy as np
import random
import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

import ClassUtils

# Applies random data augmentation techniques to the image, following the "simCLR" methodology
# https://arxiv.org/pdf/2002.05709 
class SimCLRAugmentationTransform:
    def __init__(self):
        # Following the default augmentations listed out in their paper's Data Augmentation Details
        # If you want to add in more transforms, remember to use transforms.RandomApply
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(size=224,),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.25)],
                  p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,), inplace=False),
        ])

    # We want to produce two contrasting views of the image
    def __call__(self, x):
        return self.transform(x), self.transform(x)
    

# See: https://medium.com/data-science/nt-xent-normalized-temperature-scaled-cross-entropy-loss-explained-and-implemented-in-pytorch-cc081f69848
# for a pretty in-depth and clear explanation of how it works. TLDR: Normalise, temperature scale, Cross-Entropy loss.
class NTXentLoss(nn.Module):
    def __init__(self, temperature = 0.5):
        super().__init__()
        self.temperature = temperature
        self.cosSimilarity = nn.CosineSimilarity(dim=-1)
        self.CEL = nn.CrossEntropyLoss(reduction="mean")

    def forward(self, z_i, z_j, temperature=None):
        if temperature is None:
            temperature = self.temperature
        x = torch.cat([z_i, z_j], dim=0)
        xcs = self.cosSimilarity(x[None,:,:], x[:,None,:])

        # Naturally, each embedding will have a 1-1 similarity with itself, so will contribute nothing useful,
        # so we zero these contributions.
        xcs[torch.eye(xcs.shape[0], dtype=torch.bool)] = -float('inf')

        # Math works out on these I think, because they're in contigious pairs as an input.
        target = torch.arange(len(z_i) * 2).to(device)
        target[0::2] += 1
        target[1::2] -= 1

        return self.CEL(xcs/ temperature, target)


# A SimCLR model - to use just override the getModel function to whatever model you want to use, 
# and post-training remove the projection layer and add in your classifier head (overwrite model's fully connected layer)
class SimCLR(nn.Module):
    def __init__(self, encoderModel="MobileNet_V3", outDim=128, projectionHeadInputSize=1000):
        super().__init__()
        self.encoder = self.getModel(encoderModel)
        self.projectionHead = nn.Sequential(
            nn.Linear(projectionHeadInputSize, 256),
            nn.LeakyReLU(),
            nn.Linear(256, outDim)
        )


    def getModel(self, modelName):
        modelDictionairy = {
            "ResNet-50": models.resnet50(progress=True),
            "ResNet-18": models.resnet18(progress=True),
            "MobileNet_V3": models.mobilenet_v3_small(progress=True),
        }
        # This should only be evaluated upon the getting of a particular model (calling of a particular index)
        # - shouldn't have to install all of these until used unless there's a bug
        model = modelDictionairy[modelName]
        model.fc = nn.Identity()  # This should be overwritten in your downstream application

        return model
    
    def forward(self, x):
        features = self.encoder(x)
        featureVector = self.projectionHead(features)
        return featureVector


# This is dataset dependent, and works with how we currently do things with our dataset. Adapt to your usage.
if __name__ == "__main__":
    batchSize = 128
    data_size = 5280
    transform = SimCLRAugmentationTransform()

    dataset = ClassUtils.CrosswalkDataset("zebra_annotations/classification_data",transform=transform)
    dataloader = DataLoader(
    Subset(dataset, random.sample(list(range(0, int(len(dataset) * 0.95))), data_size)),
      batch_size=batchSize, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimCLR().to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-2)
    lossFunction = NTXentLoss(temperature=0.5)

    epochs = 25
    lossTracker = []
    print(len(dataloader))
    for epoch in range(epochs):
        eraLoss = 0.0
        model.train()

        for (x_i, x_j), _ in dataloader:
            # Discards labels (image it was unlabelled data...)
            x_i = x_i.to(device)
            x_j = x_j.to(device)

            optimiser.zero_grad()

            z_i, z_j = model(x_i), model(x_j)

            loss = lossFunction(z_i, z_j)
            loss.backward()
            optimiser.step()

            eraLoss += loss.item()
        print("\n")

        avgLoss = eraLoss / len(dataloader)
        lossTracker.append(avgLoss)
        print(f"In epoch {epoch+1} of {epochs+1}, there was a loss of {avgLoss:.4f}")

    print("Completed Training!!!")
    torch.save(model.state_dict(), "Feature Extractor")
    