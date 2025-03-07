import torch

import CrosswalkDataset as Dataset
import ClassifierModel as Model
import Utilities as Utils


def train_model_v0(model_to_train, dataset, epoch_number=25, loss_func=Utils.BasicClassificationLoss,
                   batch_size=16, save=False):
    optimiser = torch.optim.Adam(model_to_train.parameters(), lr=0.001)
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=batch_size)

    loss_function = loss_func()

    for epoch in range(epoch_number):
        model_to_train.train()
        running_loss = 0.0

        for images, gt_labels in dataloader:
            optimiser.zero_grad()

            predictions = model_to_train(images)

            batch_loss = loss_function(predictions, gt_labels)
            batch_loss.backward()

            running_loss += batch_loss
            optimiser.step()

        print(f"Epoch [{epoch + 1} of {epoch_number}] finished, with loss {running_loss / len(dataloader)} in "
              f"len {len(dataloader) * batch_size}")

    Utils.save_model(model_to_train, optimiser)
    return model_to_train


# Additionally incorporated a learning rate scheduler
def train_model_v1(model_to_train, dataset, epoch_number=10, loss_func=Utils.BasicClassificationLoss,
                   batch_size=16, save=False):
    optimiser = torch.optim.Adam(model_to_train.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimiser, gamma=0.95)
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=batch_size)
    loss_function = loss_func()

    for epoch in range(epoch_number):
        model_to_train.train()
        running_loss = 0.0

        for images, gt_labels in dataloader:
            optimiser.zero_grad()

            predictions = model_to_train(images)

            batch_loss = loss_function(predictions, gt_labels)
            batch_loss.backward()

            running_loss += batch_loss
            optimiser.step()

        scheduler.step()

        print(f"Epoch [{epoch + 1} of {epoch_number}] finished, with loss {running_loss / len(dataloader)} in "
              f"len {len(dataloader) * batch_size}")

    Utils.save_model(model_to_train, optimiser)
    return model_to_train


model = Model.BasicClassificationModel(image_size=416)
# size should be dynamically obtained later on
crosswalk_dataset = Dataset.CrosswalkDataset("Crosswalk.v7-crosswalk-t3.tensorflow/train/_annotations.csv",
                                             "Crosswalk.v7-crosswalk-t3.tensorflow/train")
model = train_model_v1(model, crosswalk_dataset, save=True)
