import time
import numpy as np
import Classify

# We import each model class inside of their respective test method
# to prevent long load times from initializing all models at once.

# Stores metrics such as accuracy, precision, recall, and F1 for a given test run
class Results:
    """
    A helper class for storing and computing common binary classification metrics.

    Attributes:
        accuracy (float or None): The proportion of correct predictions among all predictions.
        precision (float or None): Among the predictions labeled as positive, how many were correct.
        recall (float or None): Among the actual positive samples, how many were correctly predicted.
        F1 (float or None): The harmonic mean of precision and recall, providing a balanced measure.
    """
    def __init__(self):
        self.accuracy = None
        self.precision = None
        self.recall = None
        self.F1 = None

    def calculate_accuracy(self, correct_pos, correct_neg, total):
        """
        Calculates the overall accuracy:
            (correct positives + correct negatives) / total samples
        Returns 0 if total is 0 to avoid division by zero.
        """
        return (correct_pos + correct_neg) / total if total > 0 else 0

    def calculate_precision(self, correct_pos, false_pos):
        """
        Calculates precision:
            correct_pos / (correct_pos + false_pos)
        Returns 0 if there are no predicted positives (i.e., correct_pos + false_pos == 0).
        """
        return correct_pos / (correct_pos + false_pos) if (correct_pos + false_pos) > 0 else 0

    def calculate_recall(self, correct_pos, false_neg):
        """
        Calculates recall:
            correct_pos / (correct_pos + false_neg)
        Returns 0 if there are no actual positives (i.e., correct_pos + false_neg == 0).
        """
        return correct_pos / (correct_pos + false_neg) if (correct_pos + false_neg) > 0 else 0

    def calculateF1(self, precision, recall):
        """
        Calculates the F1 score:
            2 * (precision * recall) / (precision + recall)
        Returns 0 if (precision + recall) is 0.
        """
        return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0


class Evaluate:
    """
    A class for evaluating binary classification models. It manages the model,
    a dataloader, and the threshold required for interpreting predictions.

    Attributes:
        model (torch.nn.Module or None): The model used for inference.
        dataLoader (DataLoader or None): The DataLoader that supplies the test data.
        threshold (float): The classification threshold, above which a prediction is considered positive.
    """
    def __init__(self):
        self.model = None
        self.dataLoader = None
        self.threshold = 0.5  # Default threshold for converting probabilities to binary predictions

    def run_test(self, verbose=True, visual=False):
        """
        Runs an inference test over the dataset provided by dataLoader, using the stored model.
        Gathers common classification metrics (true positives, false positives, etc.) and prints
        the results if verbose is True.

        Args:
            verbose (bool): Whether to print out accuracy, false positives/negatives, etc.
            visual (bool): If True, displays incorrect classifications using a visualization function.

        Returns:
            (tuple): A tuple of (correct_pos, correct_neg, false_pos, false_neg, total) summarizing
                     classification performance.
        """
        # Make sure the model and data loader have been set up
        if self.model is None or self.dataLoader is None:
            raise AttributeError("Please choose a model and dataloader before running the test")

        self.model.eval()  # Put the model in evaluation mode

        total, correct_pos, correct_neg = 0, 0, 0
        false_pos, false_neg = 0, 0
        running_average_time = 0.0  # Tracks average inference time
        collated_results = Results()  # Stores final evaluation metrics (unused locally)
        incorrect = []  # Holds incorrectly classified images if we want to visualize them

        # Iterate over batches from the dataloader
        for image, gt in self.dataLoader:
            current_start_time = time.time()

            # Classify the batch using the "infer" function from Classify
            prediction = Classify.infer(image, self.model)

            # Calculate how long this batch took
            running_average_time += time.time() - current_start_time

            # Separate predictions into positives or negatives based on threshold
            positive = prediction[prediction[:, 0] > self.threshold]
            negative = prediction[prediction[:, 0] <= self.threshold]

            # Ground truth for positive and negative subsets
            positive_gt = gt[prediction[:, 0] > self.threshold]
            negative_gt = gt[prediction[:, 0] <= self.threshold]

            # Count correct positives and negatives
            correct_pos += len(positive[positive_gt[:, 0] == 1])
            correct_neg += len(negative[negative_gt[:, 0] == 0])

            # Count false positives and false negatives
            false_pos += len(positive[positive_gt[:, 0] == 0])
            false_neg += len(negative[negative_gt[:, 0] == 1])

            # Update total samples processed
            total += min(self.dataLoader.batch_size, len(image))

            # Identify which samples are false positives/negatives for potential visualization
            false_pos_mask = (prediction[:, 0] > self.threshold) & (gt[:, 0].detach().numpy() == 0)
            false_neg_mask = (prediction[:, 0] < self.threshold) & (gt[:, 0].detach().numpy() == 1)

            # Store incorrect images if any exist in this batch
            if len(false_pos_mask) > 0:
                incorrect.append((image[false_pos_mask], gt[false_pos_mask]))
            if len(false_neg_mask) > 0:
                incorrect.append((image[false_neg_mask], gt[false_neg_mask]))

        # Print summary if requested
        if verbose:
            print(f"Total Images Processed: [{total}],"
                  f"\nAccuracy: [{((correct_pos+correct_neg)/total)*100:.2f}%],"
                  f"\nCorrect Positives: [{correct_pos}], Correct Negatives: [{correct_neg}],"
                  f"\nFalse Positives: [{false_pos}], False Negatives: [{false_neg}],"
                  f"\nAverage Running Time (s) per image: [{running_average_time / total}]")

        # If visual parameter is true, display the misclassified images
        if visual and incorrect:
            for (img_set, lab_set) in incorrect:
                for (img, lab) in zip(img_set, lab_set):
                    if len(img) > 0:
                        Classify.infer_and_display(img, 0.5, lab)

        return (correct_pos, correct_neg, false_pos, false_neg, total)

    def test_MobileNet3_default(self, model_state_dict, test_num=1, verbose=True, visual=False) -> Results:
        """
        Loads a MobileNetV3 model from a given state dictionary, sets up a test dataset of the specified size,
        and evaluates the model using run_test.

        Args:
            model_state_dict (str):
                The file path to the MobileNetV3 state dictionary to load.
            test_num (int):
                The number of samples from the dataset to test on (the last 5% of the dataset).
            verbose (bool):
                If True, prints the performance metrics after testing.
            visual (bool):
                If True, displays incorrectly classified images with their labels.

        Returns:
            Results:
                An object containing accuracy, precision, recall, and F1 scores for the test run.
        """
        import MobileNetV3 as mn3

        # We excluded the last 5% of data samples from training, so test must use that portion
        if test_num > len(mn3.dataset) * 0.05:
            test_num = int((len(mn3.dataset) - 1) * 0.05)

        # Creates a DataLoader of randomly selected samples from the last 5% of the dataset
        test_loader = mn3.DataLoader(
            mn3.Subset(
                mn3.dataset,
                mn3.random.sample(
                    list(range(int(len(mn3.dataset) * 0.95), len(mn3.dataset))),
                    test_num
                )
            ),
            batch_size=mn3.batch_size,
            shuffle=False
        )
        
        # Load the specified MobileNet classifier
        test_model = Classify.load_mobileNet_classifier(model_state_dict)

        # Assign the loaded model and DataLoader to the Evaluate instance
        self.model = test_model
        self.dataLoader = test_loader

        # Run the test and gather metrics
        correct_pos, correct_neg, false_pos, false_neg, total = self.run_test(verbose=verbose, visual=visual)

        # Clear out the model and dataloader from this Evaluate instance
        self.model = None
        self.dataLoader = None

        # Prepare the Results object with the final metrics
        test_results = Results()
        test_results.accuracy = test_results.calculate_accuracy(correct_pos, correct_neg, total)
        test_results.precision = test_results.calculate_precision(correct_pos, false_pos)
        test_results.recall = test_results.calculate_recall(correct_pos, false_neg)
        test_results.F1 = test_results.calculateF1(test_results.precision, test_results.recall)

        return test_results


# Instantiate the Evaluate class
eval = Evaluate()

# If this file is run as a script, run a default test on MobileNetV3 with a specified state dict
if __name__ == "__main__":
    mn3_test_results = eval.test_MobileNet3_default(
        "MobileNetV3_state_dict_big_train.pth",
        test_num=10000,
        visual=True
    )
