from ultralytics import YOLO
import torch
from albumentations import (
    Compose, CLAHE, GaussianBlur, MedianBlur, 
    HorizontalFlip, RandomBrightnessContrast, ToGray,
    ShiftScaleRotate, RandomShadow, RandomFog
)
from albumentations.pytorch import ToTensorV2
from ultralytics.data.dataset import YOLODataset

# Custom dataset class to use Albumentations
class CustomYOLODataset(YOLODataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.transforms = Compose([
            # Enhance contrast and details
            CLAHE(clip_limit=(1, 4), tile_grid_size=(8, 8), p=0.3),
            
            # Weather and lighting variations
            RandomShadow(shadow_roi=(0, 0.5, 1, 1), p=0.3),
            RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=0.2),
            
            # Blur effects
            GaussianBlur(blur_limit=(3, 7), p=0.2),
            MedianBlur(blur_limit=5, p=0.1),
            
            # Geometric transformations
            ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.2,
                rotate_limit=15,
                border_mode=0,
                p=0.3
            ),
            HorizontalFlip(p=0.5),
            
            # Color adjustments
            RandomBrightnessContrast(
                brightness_limit=0.3,
                contrast_limit=0.3,
                p=0.4
            ),
            ToGray(p=0.01),
            
            ToTensorV2()
        ])

    def __getitem__(self, idx):
        img, labels = super().__getitem__(idx)
        if self.transforms:
            transformed = self.transforms(image=img, bboxes=labels[:, 1:], class_labels=labels[:, 0])
            img = transformed['image']
            if len(transformed['bboxes']):
                labels = torch.cat([
                    torch.tensor(transformed['class_labels']).unsqueeze(1),
                    torch.tensor(transformed['bboxes'])
                ], dim=1)
        return img, labels

# Model and training configuration
epochs = 30
batch = 48
optimizer = 'SGD'
yolo11 = 'yolo11n-obb.pt'

if __name__ == '__main__':
    # Clear GPU cache
    torch.cuda.empty_cache()
    
    # Load model
    model = YOLO(yolo11)
    
    # Register custom dataset using the correct method
    def custom_dataset_callback(trainer):
        if hasattr(trainer.data, 'train_dataset'):
            trainer.data.train_dataset = CustomYOLODataset(trainer.data.train_dataset.path, **trainer.data.train_dataset.args)
        if hasattr(trainer.data, 'val_dataset'):
            trainer.data.val_dataset = CustomYOLODataset(trainer.data.val_dataset.path, **trainer.data.val_dataset.args)

    model.add_callback('on_train_start', custom_dataset_callback)

    # Train with enhanced settings
    model.train(
        data='dataset.yaml',
        epochs=epochs,
        imgsz=1024,
        batch=batch,
        optimizer=optimizer,
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
        project='runs/train',
        name=f'exp_{yolo11}_{epochs}_{batch}_{optimizer}',
        cache=False,
        amp=True,
        device='cuda'
    )
