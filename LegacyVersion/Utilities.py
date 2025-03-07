# This just contains loss functions and other things required for the classifier training process
import numpy as np
import torch
import torch.nn as nn


class BasicClassificationLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.classification_loss = nn.CrossEntropyLoss()

    def forward(self, pred_labels, gt_labels):
        return self.classification_loss(pred_labels, gt_labels)


def save_model(trained_model, optimiser_used):
    torch.save(trained_model, 'trainedClassifier.pth')
    print(",")
    torch.save(trained_model.state_dict(), 'trainedClassifier_weights.pth')
    torch.save(optimiser_used, 'optimiserUsed.pth')


def load_model_for_eval(file_path, model_type):
    model_template = model_type(416)
    model_template.load_state_dict(torch.load(file_path, weights_only=True))
    model_template.eval()
    return model_template


def softmax(unprocessed_logits):
    logits = np.array(unprocessed_logits)
    exponentials = np.exp(logits)
    softmax_arr = exponentials / sum(exponentials)
    return softmax_arr
