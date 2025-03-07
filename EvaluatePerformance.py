import time
import numpy as np

import Classify
# We import each model class inside of their respective test method to prevent long load times of initialising all of them


# These are the default functions used for evaluation - overwrite or add more parameters as is required for your testing
# We did not use these for our testing - we calculated it manually - but this is a much easier way
class Results:
    def __init__(self):
        self.accuracy = None
        self.precision = None
        self.recall = None
        self.F1 = None

    # Proportion of all predictions that were right - basically what did it get right
    def calculate_accuracy(self, correct_pos, correct_neg, total):
        return (correct_pos + correct_neg) / total if total > 0 else 0

    # Proportion of all positive predictions that were actually positive - aka if it predicted positive, how often was it
    # actually right
    def calculate_precision(self, correct_pos, false_pos):
        return correct_pos / (correct_pos + false_pos) if (correct_pos + false_pos) > 0 else 0

    # Proportion of all positive cases that were predicted positive - aka how many positive images did it correctly predict
    def calculate_recall(self, correct_pos, false_neg):
        return correct_pos / (correct_pos + false_neg) if (correct_pos + false_neg) > 0 else 0

    # A combination of precision and recall that takes both of them into consideration - a decent 'summary' accuracy metric
    def calculateF1(self, precision, recall): 
        return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0


# Purely for comparing the performance of Binary Classification models
class Evaluate:
    def __init__(self):
        self.model = None
        self.dataLoader=None
        self.threshold=0.5
        pass

    # GENERALISED EVALUATION FUNCTION FOR COMPARISION BETWEEN MODEL ARCHITECTURES - USE SPECIFIC EVAL FUNC FOR TESTING TRAINING SETUPS
    # Tests performance of a model on unseen data - this is the function we used to evaluate our classification models during training
    # to determine the best model architecture to use.
    def run_test(self, verbose=True, visual=False):
        if self.model is None or self.dataLoader is None:
            raise AttributeError("Please choose a model to test before running the test")
        
        self.model.eval()

        total, correct_pos, correct_neg, false_pos, false_neg = 0, 0, 0, 0, 0
        running_average_time = 0.0
        collated_results = Results()
        incorrect = []

        for image, gt in self.dataLoader:
            current_start_time = time.time()
            prediction = Classify.infer(image, self.model)
            running_average_time += time.time() - current_start_time
            positive, negative = prediction[prediction[:, 0] > self.threshold], prediction[prediction[:, 0] <= self.threshold]
            positive_gt, negative_gt = gt[prediction[:, 0] > self.threshold], gt[prediction[:, 0] <= self.threshold]

            correct_pos += len(positive[positive_gt[:, 0]==1])
            correct_neg += len(negative[negative_gt[:, 0]==0])
            false_pos += len(positive[positive_gt[:, 0]==0])
            false_neg += len(negative[negative_gt[:, 0]==1])
            total += min(self.dataLoader.batch_size, len(image))

            false_pos_mask = (prediction[:, 0] > self.threshold) & (gt[:, 0].detach().numpy() == 0)
            false_neg_mask = (prediction[:, 0] < self.threshold) & (gt[:, 0].detach().numpy() == 1)

            if len(false_pos_mask) > 0:
                incorrect.append((image[false_pos_mask], gt[false_pos_mask]))
            if len(false_neg_mask) > 0:
                incorrect.append((image[false_neg_mask], gt[false_neg_mask]))

        if verbose:
            print(f"Total Images Processed: [{total}],"
                  f" \nAccuracy: [{((correct_pos+correct_neg)/total)*100:.2f}%],"
                  f" \nCorrect Positives: [{correct_pos}], Correct Negatives: [{correct_neg}],"
                  f" \nFalse Positives: [{false_pos}], False Negatives [{false_neg}],"
                  f" \nAverage Running Time (s) per image: [{running_average_time / total}]")

        if visual and incorrect:
            for (img_set, lab_set) in incorrect:
                for (img, lab) in zip(img_set, lab_set):
                    if len(img) > 0:
                        Classify.infer_and_display(img, 0.5, lab)

        return (correct_pos, correct_neg, false_pos, false_neg, total)

    # What we used to test how changes in our training paramters and input data affected performance
    def test_MobileNet3_default(self, model_state_dict, test_num=1, verbose=True, visual=False) -> Results:
        import MobileNetV3 as mn3

        # We excluded the last 5% of data samples from training
        if test_num > len(mn3.dataset) * 0.05:
            test_num = int((len(mn3.dataset) - 1) * 0.05)

        test_loader = mn3.DataLoader(
            mn3.Subset(mn3.dataset, mn3.random.sample(list(range(int(len(mn3.dataset) * 0.95), len(mn3.dataset))), test_num)),
            batch_size=mn3.batch_size, shuffle=False)
        
        test_model = Classify.load_mobileNet_classifier(model_state_dict)

        self.model = test_model
        self.dataLoader = test_loader

        correct_pos, correct_neg, false_pos, false_neg, total = self.run_test(verbose=verbose, visual=visual)

        self.model=None
        self.dataLoader=None
        test_results = Results()

        test_results.accuracy = test_results.calculate_accuracy(correct_pos, correct_neg, total)
        test_results.precision = test_results.calculate_precision(correct_pos, false_pos)
        test_results.recall = test_results.calculate_recall(correct_pos, false_neg)
        test_results.F1 = test_results.calculateF1(test_results.precision, test_results.recall)

        return test_results
        

eval = Evaluate()
if __name__ == "__main__":
    mn3_test_results = eval.test_MobileNet3_default("MobileNetV3_state_dict_big_train.pth", test_num=10000, visual=True)
