import matplotlib.pyplot as plt
import torch
import Utilities as Utils
import classifierModel
import CrosswalkDataset as Dataset


model = Utils.load_model_for_eval('trainedClassifier_weights.pth', classifierModel.BasicClassificationModel)
dataset = Dataset.CrosswalkDataset("Crosswalk.v7-crosswalk-t3.tensorflow/test/_annotations.csv",
                                   "Crosswalk.v7-crosswalk-t3.tensorflow/test")


with torch.no_grad():
    loss = 0.0
    batch_size = 3
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=batch_size)
    loss_function = Utils.BasicClassificationLoss()
    count, notCount = 0, 0
    for images, gt_labels in dataloader:
        predictions = model(images)
        softmax_probabilities = Utils.softmax(predictions)
        for i in range(len(images)):
            plt.imshow(images[i].permute(1, 2, 0).numpy() / 255.0)
            classif = False
            if (gt_labels[i][1] > gt_labels[i][0] and softmax_probabilities[i][1] > softmax_probabilities[i][0]) or (gt_labels[i][1] <= gt_labels[i][0] and softmax_probabilities[i][1] <= softmax_probabilities[i][0]):
                classif = True
                count += 1
            else:
                notCount += 1
            plt.title(str(softmax_probabilities[i]) + " " + str(gt_labels[i]) + str(classif))
            plt.show()
            print(softmax_probabilities[i])

        batch_loss = loss_function(predictions, gt_labels)
        loss += batch_loss

    print("Loss is: ", loss / (len(dataloader) * batch_size))
    print(count, notCount)
