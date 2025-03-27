from ultralytics import YOLO
import torch
epochs = 200
batch = 8
optimizer = 'SGD'
yolo11 = 'yolo11l-obb.pt'
 
if __name__ == '__main__':
    # Load a model
    torch.cuda.empty_cache()
    model = YOLO(yolo11)  # load a pretrained model

    # Train the model with custom settings
    model.train(
        data='dataset.yaml',      # path to data config file
        epochs=epochs,              # number of epochs
        imgsz=1024,             # image size
        batch=batch,               # batch size
        device='cuda',             # cuda device (use 'cpu' for CPU)
        project='runs/train',   # save results to project/name
        name=f'exp_{yolo11}_{epochs}_{batch}_{optimizer}'           # experiment name
    )
