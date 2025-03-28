import ClassUtils
import LoadUtils

import torch
import torchvision
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np
import random

import warnings

# Torchvision's models utils has a depreciation warning for the pretrained parameter in its instantiation but we don't use that
warnings.filterwarnings(
    action='ignore',
    category=DeprecationWarning,
    module=r'.*'
)

vgg16_state_path = "VGG16_Full_State_Dict.pth"
# A prototype v2 version that's only been trained on 2000 images by transfer learning
mobileNet_path = "MobileNetV3_state_dict_big_train.pth"
data_path = "zebra_annotations/classification_data"

classify = None
transform = None

# Loads a given VGG binary classifier state dictionairy into a model, for transfer learning or immediate use
def load_vgg_classifier(state_dict_path):
    """
    Initialises the weights of the Mobile Net v3 model architecture to the pre-trained weights
    stored in the model state dictionairy in the 'models' directory
    
    Args:
        state_dict_path (string): The path to the state dictionairy relative to the function call
    
    Returns:
        model (MobileNetV3 object): An initialised mobilenet v3 model with the saved weights,
          in evaluation mode so the weights will not be changed
    """
    model = models.vgg16()

    # Modifies fully connected layer to output binary class predictions
    model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, 2)
    state_dict = torch.load(state_dict_path, weights_only=True)
    model.load_state_dict(state_dict)

    model.eval()

    return model

# Only loads the classifier weights, in the case where it is transfer learning on only the top 
# or the feature extraction has been frozen during training
# This saves a significant amount of space
def partial_vgg_load(classifier_state_dict_path):
    """
    Loads a pre-trained VGG16 model, modifies its final classifier layer to output two classes, 
    and applies the provided state dictionary.

    Args:
        classifier_state_dict_path (dict):
            The state dictionary for the modified classifier layer, typically loaded from a file.

    Returns:
        torch.nn.Module:
            The VGG16 model in evaluation mode, ready for inference or additional fine-tuning.
    """
    model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)

    model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, 2)
    model.classifier.load_state_dict(classifier_state_dict_path)

    model.eval()

    return model

# Loads a given ResNet binary classifier state dictionairy into a model, for transfer learning or immediate use
def load_resnet_classifier(state_dict_path):
    """
    Loads a pre-trained ResNet18 model, modifies its final fully connected layer to output a single
    binary classification value, and applies the provided state dictionary.

    Args:
        state_dict_path (str):
            The file path to the state dictionary containing the trained weights for the ResNet model.

    Returns:
        torch.nn.Module:
            The ResNet18 model in evaluation mode, ready for binary classification tasks.
    """
    # Ignore depreciation warnings --> It works fine for our needs
    resnet = models.resnet18(pretrained=True)
    resnet.fc = torch.nn.Linear(resnet.fc.in_features, 1)

    state_dict = torch.load(state_dict_path, weights_only=True)
    resnet.load_state_dict(state_dict)
    
    resnet.eval()
    return resnet

# Loads a given MN3 binary classifier state dictionairy into a model, for transfer learning or immediate use
# We use this for our current version
def load_mobileNet_classifier(state_dict_path):
    """
    Loads a MobileNetV3 Small model, modifies its final classifier layer to output two classes,
    and applies the provided state dictionary.

    Args:
        state_dict_path (str):
            The file path to the state dictionary containing the trained weights for the MobileNet model.

    Returns:
        torch.nn.Module:
            The MobileNetV3 Small model in evaluation mode, configured for two-class classification.
    """
    # Ignore depreciation warnings --> It works fine for our needs
    model = models.mobilenet_v3_small()
    model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, 2)

    state_dict = torch.load(state_dict_path, weights_only=True)
    model.load_state_dict(state_dict)

    model.eval()
    return model

# classify = load_vgg_classifier(vgg16_state_path)
# transform = ClassUtils.vgg_transform

classify = load_mobileNet_classifier(mobileNet_path)
transform = ClassUtils.mob3_transform



# Takes a numpy array representing an image and returns a boolean classification based on the given threshold value
# Defaults to 0.35 as it errs on the side of letting images through.
def infer(image, infer_model=classify, infer_transform=transform):
    """
    Applies a binary classification model to an input image array, generalising the inference
    process to be model-independent. 

    Args:
        image (numpy array): The input image(s) in the form of a numpy array of form
            [channels: [width: [height:]]] or [batch: [channels: [width: [height:]]]]
        infer_model (pytorch model object): Any binary classification pytorch model object with a forward method
        infer_transform (pytorch transform object): Any pre-processing required for the model to run
    
    Returns:
        probs (tensor): A list of binary classifaction confidences in the range [0, 1], 
            where each member of the batch sums to 1.

    """
    if infer_model is None or infer_transform is None:
        raise TypeError("Error: The inference classes have not been initialised properly.")
    if not torch.is_tensor(image):
        image = infer_transform(image)
    
    # Expects batches - this adds another dimensions to properly format the data
    if len(image.shape) <= 3:
        image = image.unsqueeze(0)

    logit_pred = infer_model(image)

    probs = 1 / (1 + np.exp(-logit_pred.detach().numpy()))
    # prob = max(0, min(np.exp(logit_pred.detach().numpy())[0], 1))
    return probs


# Takes a PIL image and returns a boolean classification based on the given threshold value
# Defaults to 0.35 as it errs on the side of letting images through.
def PIL_infer(image, threshold=0.35):
    """
    A wrapper function for the infer function that ensures compatibility with the PIL image library format
    and that works for a single image only. Uses the system default model.

    Args:
        image (PIL image): The image that will be classified in PIL format
        threshold (float): The confidence threshold for postiive classification
    
    Returns:
        classification (boolean): Whether the image is likely to be the target object.
    """
    tensor_im = torchvision.transforms.functional.pil_to_tensor(image).float()/ 255
    prediction = infer(tensor_im)
    classification = prediction[0][0] > threshold
    return classification

# For testing and demo purposes
def infer_and_display(image, threshold, actual_label, onlyWrong=False):
    """
    A wrapper for the infer function that allows a visual representation of the model's classification, for 
    demonstration, debuggin and model fine-tuning. Uses the system default model.

    Args:
        image (numpy array): The batch of images that will be classified
        threshold (float): The minimum probability required for a positive classification
        actual label (boolean array): The ground truth class labels of the input image array
        onlyWrong (boolean): Whether to exclusively display incorrect classifications

    Returns:
        prediction (boolean array): The set of classifications made by the model, only returned if they were all correct
        probability (tensor array): The set of probabilities assigned to each class by the binary classification model
    """
    probability = infer(image)
    prediction = probability > threshold
    is_correct = (actual_label[0] == 1) == prediction

    if onlyWrong and is_correct:
        return prediction
    
    plt.imshow(torch.permute(image, (1, 2, 0)).detach().numpy())
    plt.title(f"Prediction: {prediction[0][0]} with confidence {probability[0][0]}%, Actual: {actual_label[0] == 1}")
    plt.axis("off")
    plt.show()

    return probability


# Template code for how to create an inference code
def example_init(examples=20, display=True):
    """
    An example function for the classification process, for demonstration purposes and as a tutorial for usage

    Args:
        examples (int): The number of images to be loaded from the training dataset.
        display (boolean): Whether to display the results of each classification, which pauses the program until
            each individual classification display window is closed.

    Returns:
        None
    """
    dataset = ClassUtils.CrosswalkDataset(data_path)
    
    random_points = [random.randint(0, len(dataset)-1) for i in range(examples)]
    correct, incorrect, falsepos, falseneg = 0, 0, 0, 0
    for point in random_points:
        image, label = dataset[point]

        class_guess = [0, 1]
        if infer(image)[0][0] > 0.5:
            class_guess = [1, 0]
        if class_guess == label.tolist():
            correct += 1
        else:
            if class_guess[0]:
                falsepos += 1
            else:
                falseneg += 1
            incorrect += 1
        
        if display:
            print(f"Prediction of {infer_and_display(image, 0.4, label)}% of a crosswalk (Crosswalk: {label[0]==1})")
    print(f"correct: {correct}, incorrect: {incorrect}, of which false positives were {falsepos} and false negatives were {falseneg}")

if __name__ == "__main__":
    example_init(examples=200,display=False)

else:
    print(f"Module: [{__name__}] has been loaded")


