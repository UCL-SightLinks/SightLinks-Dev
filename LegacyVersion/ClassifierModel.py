import numpy as np
import torch.nn as nn


# A really basic classifier model to detect points of interest (any likely crosswalks) for further investigation
class BasicClassificationModel(nn.Module):
    def __init__(self, image_size):
        super().__init__()
        self.image_size = image_size

        # In channels are 3 - the RGB colours.
        self.first_convolutional_layer = nn.Conv2d(3, 16, 5, padding=0)
        self.second_convolutional_layer = nn.Conv2d(16, 32, 5, padding=0)
        self.third_convolutional_layer = nn.Conv2d(32, 64, 5, padding=0)

        self.fully_connected_layer = self.create_dynamic_output_layer()
        # Actually predicts class probability

        self.pooling_layer = nn.MaxPool2d(5)
        self.activation_layer = nn.LeakyReLU(0.01)

    # It is necessary to be able to take in images of a dynamic size, since may rescale depending on local regulations
    # for crosswalk size. We can also assume squareness since this takes in the results of the segmentation model.
    def create_dynamic_output_layer(self):
        output_image_size = self.image_size

        for layer in range(2):
            output_image_size = ((output_image_size - 4) // 5)
            # Convolution, then padding

        output_image_size = output_image_size - 4
        fully_connected_layer = nn.Linear(output_image_size * output_image_size * 64, 2)
        return fully_connected_layer

    def forward(self, x):
        x = self.first_convolutional_layer(x)
        x = self.pooling_layer(x)

        x = self.second_convolutional_layer(x)
        x = self.pooling_layer(x)

        x = self.third_convolutional_layer(x)
        x = self.activation_layer(x)

        x = x.view(x.size(0), -1)
        # Flattens the feature maps to pass into the fully connected layer - from 4D to 2D (batch, lin_tensor)

        class_predictions = self.fully_connected_layer(x)

        return class_predictions


classifierModel = BasicClassificationModel(image_size=416)  # assumed square
print(classifierModel)
