import torch
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
import os
import sys
import numpy as np
import random
import time

"""
This is required to access development tool functions while this file is separated in the ExperimentalMethods folder,
If this file is moved to the stable development tool directory, please make sure to adjust or remove this statement.
"""

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from ClassUtils import CrosswalkDataset

import warnings
# Torchvision's models utils has a depreciation warning for the pretrained parameter in its instantiation but we don't use that
warnings.filterwarnings(
    action='ignore',
    category=DeprecationWarning,
    module=r'.*'
)

learning_rate = 4e-3
epoch_num = 25

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vgg16 = models.vgg16(weights = models.VGG16_Weights)
# Modifies fully connected layer to output binary class predictions
vgg16.classifier[6] = nn.Linear(vgg16.classifier[6].in_features, 2)

# Freeze as you see fit depending on the application you want to design. This will prevent further training of feature extraction layers in our code.
for param in vgg16.features.parameters():
     param.requires_grad = False
# for param in vgg16.classifier[:6].parameters():
#     param.requires_grad = False

vgg16 = vgg16.to(device)
loss_function = nn.BCELoss()

# Prevents accidental loading of the whole training process in the background
if __name__ == "__main__":
    # Takes only the classifier layers, which have not been frozen
    optimiser = torch.optim.Adam(params=
                                filter(lambda p: p.requires_grad, vgg16.parameters()),
                                lr=learning_rate)


    training_dataset = CrosswalkDataset("zebra_annotations/classification_data")  
    training_loader = DataLoader(Subset(training_dataset, random.sample(range(len(training_dataset)-1), 25000)), batch_size=128, shuffle=True)

    for param in vgg16.features.parameters():
        param.requires_grad = False


    vgg16.train()
    print(len(training_dataset))
    for epoch in range(epoch_num):
        running_loss = 0.0
        start_time = time.time()
        last_time = start_time
        for images, gt in training_loader:
            images, gt = images.to(device), gt.to(device)

            classifications = torch.sigmoid(vgg16(images))
            loss = loss_function(classifications, gt)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            batch_time = time.time()

            running_loss += loss.item()

            last_time = batch_time
            print(",,, ---")

        
        print(f"\nEpoch {epoch + 1} of {epoch_num} has a per image loss of [{running_loss/len(training_loader):.4f}]")
        print(f"{(last_time - start_time):.6f}")

    # Includes the feature extraction layers
    torch.save(vgg16.state_dict(), "VGG16_Full_State_Dict.pth")
    # Only includes the classifier layer 
    # - the 'head' whose weights you can use to overwrite if you don't want to store the whole state dict file
    torch.save(vgg16.classifier[6].state_dict(), "vgg16_binary_classifier_onlyHead.pth")
