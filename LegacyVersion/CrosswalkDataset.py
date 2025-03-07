import torch
import pandas as pd
import numpy as np
from PIL import Image


class CrosswalkDataset:
    def __init__(self, annotation_path, image_path, transform=None):
        self.annotations = pd.read_csv(annotation_path)
        self.image_dir = image_path

        self.transform = transform
        # In case later one we want to do normalisation, pre-processing etc. --> Someone else will look at this

        self.unique_labels = sorted(self.annotations['class'].unique())
        # There might be a more efficient method to do this -- come back to
        self.type_mapping = {type_value: 1 + idx for idx, type_value in enumerate(self.unique_labels)}
        # Labels have been converted to numerical class labels mapped by type mapping - for tensor conversion, 0 is
        # the background object/ none type. For the current binary classifier everything other than zebra is 0

        self.image_data = []
        self.labels = []  # One-to-One mapping with bounding boxes by the way

        self.process_annotations()

    def __len__(self):
        return len(self.image_data)

    def __getitem__(self, index):
        image = self.image_data[index]
        class_label = self.labels[index]

        tensor_image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
        # From (H, W, C) to (C, H, W)  --> This is the format that pytorch uses.
        tensor_label = torch.tensor(class_label, dtype=torch.float32)

        if self.transform:
            tensor_image = self.transform(tensor_image)

        return tensor_image, tensor_label

    def process_annotations(self):
        for filename, group in self.annotations.groupby('filename'):
            completed_image_path = f"{self.image_dir}/{filename}"
            image = Image.open(completed_image_path)
            image_array = np.array(image)
            entity_annotations = [1, 0]

            for _, row in group.iterrows():
                # We can add additional classes in here - come back to later
                if row['class'] == "ZebraStyle":
                    # numerical_class = self.type_mapping[row['class']]
                    entity_annotations = [0, 1]
                    # print([numerical_class, (row['xmin'], row['ymin'], row['xmax'], row['ymax'])])

                else:
                    pass
                    # Classified as a background object

            self.image_data.append(image_array)
            self.labels.append(entity_annotations)


crosswalk_dataset = CrosswalkDataset("Crosswalk.v7-crosswalk-t3.tensorflow/train/_annotations.csv",
                           "Crosswalk.v7-crosswalk-t3.tensorflow/train")

