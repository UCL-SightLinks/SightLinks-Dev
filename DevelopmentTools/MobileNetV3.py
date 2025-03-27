# This is unquantised - for comparision

import ClassUtils
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset
import random
from torchvision import models
from torch.utils.data import DataLoader
import time

import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# These define the model that will be trained
num_classes = 2
batch_size = 256
epochs = 25
learning_rate = 5e-4
train_data_size = 25000
saved_state_dict_path = "MobileNetV3_test.pth"

model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)

model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
model = model.to(device)

dataset = ClassUtils.CrosswalkDataset("zebra_annotations/classification_data")

train_loader = DataLoader(
    Subset(dataset, random.sample(list(range(0, int(len(dataset) * 0.95))), train_data_size)),
      batch_size=batch_size, shuffle=True)
test_loader = DataLoader(
    Subset(dataset, random.sample(list(range(int(len(dataset) * 0.95), len(dataset))), 12)),
      batch_size=batch_size, shuffle=False)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# Generalised training function that uses a training-testing split defined in the training variables.
# Works best with transfer learning as is shown in the 'MobileNetV3.py' function where it is defined - check it out.
def train_model():
    model.train()
    start_time = time.time()
    for epoch in range(epochs):
        to_do = train_data_size
        running_loss = 0.0
        for inputs, labels in train_loader:
            try:
                inputs, labels = inputs.to(device), labels.to(device)
            except:
                continue
            
            optimizer.zero_grad()
            outputs = torch.sigmoid(model(inputs))
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            to_do -= batch_size

        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader)}, time {time.time()- start_time}")
        start_time = time.time()

# Do not use to actually evaluate performance, this is for quick checks - 'EvaluatePerformance.py' has actual quantified evaluation tools.
def test_model():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            try:
                inputs, labels = inputs.to(device), labels.to(device)
            except:
                continue
            outputs = torch.sigmoid(model(inputs))

            predicted = (outputs/100) > 0.5
            for i in range(len(inputs)):
                    plt.close()
                    plt.imshow(torch.permute(inputs[i], (1, 2, 0)).cpu().detach().numpy())
                    plt.title(f"prediction of {outputs[i].tolist()[0]:.3f}%, {100 * predicted[i].tolist()[0]:.3f}%,\nactual: {labels[i].tolist()}")
                    plt.axis("off")
                    plt.show()
            
            total += labels.size(0)
            # print(predicted, labels)

            for prediction, label in zip(predicted, labels):
                correct += ((prediction[0]>50) == label[0])
    
    print(f"Accuracy: {100 * correct / total}%")



train = True
if __name__ == "__main__":
    if train:
        train_model()
        torch.save(model.state_dict(), "mn3_vs55.pth")
    else:
        state_dictionairy = torch.load(saved_state_dict_path, weights_only=True)
        print(type(state_dictionairy))
        model.load_state_dict(state_dictionairy)

    test_model()

else:
    state_dictionairy = torch.load(saved_state_dict_path, weights_only=True)
    model.load_state_dict(state_dictionairy)
    print(f"Module: [{__name__}] has been loaded")
