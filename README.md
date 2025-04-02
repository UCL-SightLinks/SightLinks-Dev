# SightLinks Development Repository

Welcome to the SightLinks Development Repository! We hope you can use these tools to build accessible systems and make the world a more welcoming place. This repository contains various tools and resources designed to facilitate the development of the SightLinks project, focusing on data generation, annotation, model training, feature extraction and much more. These are the tools used to develop the SightLinks system, as well as some addiitional Beta features that can bring the project even further. 

#### Note: There is a significant overlap between SightLinks-Main and SightLinks-Dev, with many of the functions overlapping.

## Project Overview

SightLinks is an initiative aimed at enhancing accessibility mapping through automated analysis of satellite imagery. By accurately detecting features relevant to wheelchair users and individuals with mobility needs, SightLinks contributes to the development of more inclusive navigation solutions. The system achieves high reliability in feature detection while maintaining strict safety standards to prevent false positives. ([students.cs.ucl.ac.uk](https://students.cs.ucl.ac.uk/2024/group15/index.html))

## Repository Structure

- **DevelopmentTools/**: This directory includes tools and scripts used for data generation, annotation, model training, and feature pre-processing. These resources are essential for processing satellite imagery and training machine learning models to detect accessibility features, but that's not the only thing they're limited to being used for!

- **LegacyVersion/**: This directory contains previous versions of the tools and scripts, preserved for reference and potential reuse.

## Getting Started

To begin using the tools in this repository:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/UCL-SightLinks/SightLinks-Dev.git
   ```
2. **Install Dependencies**: Ensure you have Python installed, then install the necessary packages:
   ```bash
   pip install -r requirements.txt
   ```
3. **Navigate to the Development Tools Directory**:
   ```bash
   cd SightLinks-Dev/DevelopmentTools
   ```
4. **Run the Tools**: Follow the instructions provided in each file's documentation to generate data, annotate images, or train models. Remember to include your dataset, and cross-reference with the documentation provided in SightLinks-Main! 

## Types of tools:

This directory contains three distinct types of development tools: one for training the MobileNetV3 classifier, one for training the YOLOV11-OBB image detection model, and one for use in the experimental phase of our system. The different components are explained below briefly so that our experimental process can be recreated, but the methodology and implementation are described in more detail within the report. 

#### Automate Annotations.py
Contains code required to automatically retrieve and annotate crosswalk data for classification training, now using MapBox API over the previously used Google Static Tiles which is more expensive. Image are retrieved based on datasets retrieved from OpenStreetMap, these datasets are not complete and very sparse but sufficient for training models as long as multiple regions are sampled.

#### Evaluate Performance.py
Contains a set of evaluatory metrics for the objective comparision of different classification models using unseen data testing. Currently specialised for MobileNet models, but simple to expand as a template class is provided to extend.

#### Classify.py
Contains the code for loading and executing model inference. Required in development for the evaluation of model performance after training.

#### Load Utils.py
Contains the code for loading well-structured datasets such as yaml, as well as code to generate new classification data by breaking down object bounding box annotations. This provides a much larger, high quality, dataset for training the classification model.

#### MobileNetV3.py
Contains the code required for training and testing the MobileNetV3 classification model. This was the method used to train the classification layer of our program.

#### object detection augmentation
Contains the code for applying a series of transforms and noise to the dataset in order to improve the resistance to occlusion and obstruction of our YOLO model without requiring explicit annotations.

#### yolo train
Contains the code for training the YOLOv11-OBB model, which is used as the rotated object detection model in our crosswalk detection layer.

### DevelopmentTools/ExperimentalMethods:
This folder holds a collection of functions that were not incorporated into our final pipeline for a variety of reasons, but that are relevant to the development process especially for retraining. These form the resources we used for our experiments during the research process - discussed in more detail in our report. 

#### VGG Custom and VGG TL
Two different implementations of the VGG16 classifier architecture. Both were used during development but later replaced as we encountered better model architectures for our use.

#### Self-Supervised Feature Extraction
A self-supervised feature extraction model that we experimented with using due to the abundance of unlabelled data we had. Unfortunately, we did not apply this in the final version, as in the end it did not provide a significant enough accuracy improvement since we had already generated a large annotated dataset, and we did not have enough computational resources to use more data than that. It remains as part of the experimental methodology, but also because for retraining for new transport features it offers great potential.

#### Quantised Mobile Net
A Quantisation Aware Training implementation of the Mobile Net V3 architecture that we attempted, as a potential direction of improvement. The version we implemented is working, however is not optimised and requires much more data with much more training due to the QAT procedure, which we were unable to do with our current resources. Improvements are welcome for future development!


#### Feature Extraction Methods
Different feature extraction methods we investigated as potentially being useful, although determined it was not optimal for our system in the end. They are included as references for the experiment process, as well as for use in developing other applications which focus more on occluded data.


## Contributing

We welcome contributions to enhance the functionality and efficiency of the tools in this project, in fact we encourage it. To contribute:

1. Fork the repository.
2. Create a new branch for your feature or bug fix:
   ```bash
   git checkout -b feature-name
   ```
3. Commit your changes:
   ```bash
   git commit -m "Description of feature or fix"
   ```
4. Push to your branch:
   ```bash
   git push origin feature-name
   ```
5. Open a pull request detailing your changes. (we will most likely accept, there's a lot of improvements to be made!)

## License

This project is licensed under the MIT License, and owned by the UCL Computer Science department.

## Contact

For questions or further information, please contact the SightLinks development team at [zcabkde@ucl.ac.uk], or the university's computer science department as they have taken over this project ([UCL CS Department Staff Email List](https://www.ucl.ac.uk/computer-science/people/computer-science-professional-services-staff)).

---

*Note: For a comprehensive overview of the SightLinks project, including key features, functionalities, and team members, please visit our [project website](https://students.cs.ucl.ac.uk/2024/group15/index.html)!!* ([students.cs.ucl.ac.uk](https://students.cs.ucl.ac.uk/2024/group15/index.html))

