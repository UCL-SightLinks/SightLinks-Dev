from ultralytics import YOLO
import torch
from albumentations import (
    Compose, CLAHE, GaussianBlur, MedianBlur,
    HorizontalFlip, RandomBrightnessContrast, ToGray,
    ShiftScaleRotate, RandomShadow, RandomFog
)
from albumentations.pytorch import ToTensorV2
from ultralytics.data.dataset import YOLODataset

# -----------------------------------------------------------------------------
# Custom YOLO dataset class that applies Albumentations during training
# -----------------------------------------------------------------------------
class CustomYOLODataset(YOLODataset):
    """
    A subclass of YOLODataset that integrates Albumentations for advanced
    data augmentation. This allows us to apply diverse transformations
    (lighting changes, blurring, geometric rotations, etc.) for more
    robust model training.
    """
    def __init__(self, *args, **kwargs):
        # Initialize the parent class (YOLODataset)
        super().__init__(*args, **kwargs)

        # Compose a series of Albumentations-based transformations
        self.transforms = Compose([
            # Enhances local contrast and details
            CLAHE(clip_limit=(1, 4), tile_grid_size=(8, 8), p=0.3),

            # Adds shadows and fog effects to simulate real-world weather conditions
            RandomShadow(shadow_roi=(0, 0.5, 1, 1), p=0.3),
            RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=0.2),

            # Introduces blur, both Gaussian and Median
            GaussianBlur(blur_limit=(3, 7), p=0.2),
            MedianBlur(blur_limit=5, p=0.1),

            # Applies geometric transformations (shift, scale, rotate)
            # border_mode=0 => pad with 0 (black) if the image is rotated/scaled beyond edges
            ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.2,
                rotate_limit=15,
                border_mode=0,
                p=0.3
            ),
            HorizontalFlip(p=0.5),  # 50% chance to flip image horizontally

            # Adjusts brightness and contrast, slight chance to convert to grayscale
            RandomBrightnessContrast(
                brightness_limit=0.3,
                contrast_limit=0.3,
                p=0.4
            ),
            ToGray(p=0.01),  # Very small chance to convert image entirely to grayscale

            # Finally, convert augmented images to PyTorch tensors
            ToTensorV2()
        ])

    def __getitem__(self, idx):
        """
        Overridden method that:
        1) Fetches an image and its bounding boxes from the parent YOLODataset.
        2) Applies our Albumentations transforms (if any).
        3) Returns the augmented image (tensor) and updated labels.
        """
        # Retrieve (img, labels) from YOLODataset
        img, labels = super().__getitem__(idx)

        # If we have transforms, apply them
        if self.transforms:
            transformed = self.transforms(
                image=img,
                bboxes=labels[:, 1:],        # YOLODataset format: class_idx, x, y, w, h
                class_labels=labels[:, 0]    # The first column = class IDs
            )
            img = transformed['image']

            # Recombine labels to match the expected format:
            # The 'bboxes' come from Albumentations, and 'class_labels' are the class IDs
            if len(transformed['bboxes']):
                labels = torch.cat([
                    torch.tensor(transformed['class_labels']).unsqueeze(1),
                    torch.tensor(transformed['bboxes'])
                ], dim=1)

        return img, labels


# -----------------------------------------------------------------------------
# Model and Training Configuration
# -----------------------------------------------------------------------------
epochs = 30
batch = 48
optimizer = 'SGD'
yolo11 = 'yolo11n-obb.pt'

if __name__ == '__main__':
    # Clear GPU cache to free memory before training
    torch.cuda.empty_cache()

    # Load the YOLO model
    model = YOLO(yolo11)

    # Define a callback to replace YOLO's default dataset
    # with our custom dataset class that includes Albumentations.
    def custom_dataset_callback(trainer):
        """
        Callback function to override the default YOLO training/validation datasets
        with our CustomYOLODataset, allowing us to apply Albumentations transforms.
        """
        # Replace training dataset
        if hasattr(trainer.data, 'train_dataset'):
            trainer.data.train_dataset = CustomYOLODataset(
                trainer.data.train_dataset.path,
                **trainer.data.train_dataset.args
            )
        # Replace validation dataset
        if hasattr(trainer.data, 'val_dataset'):
            trainer.data.val_dataset = CustomYOLODataset(
                trainer.data.val_dataset.path,
                **trainer.data.val_dataset.args
            )

    # Register the callback so it runs at the start of training
    model.add_callback('on_train_start', custom_dataset_callback)

    # Train the model with additional data augmentation parameters
    model.train(
        data='dataset.yaml',
        epochs=epochs,
        imgsz=1024,
        batch=batch,
        optimizer=optimizer,

        # YOLO-specific augmentation hyperparameters
        mosaic=1.0,
        mixup=0.3,
        copy_paste=0.3,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        shear=2.0,
        perspective=0.0015,
        fliplr=0.5,
        flipud=0.1,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,

        # Output and logging configs
        project='runs/train',
        name=f'exp_{yolo11}_{epochs}_{batch}_{optimizer}',
        cache=False,  # Don't cache images for speed
        amp=True,     # Use Automatic Mixed Precision for faster training
        device='cuda' # Train on GPU
    )
