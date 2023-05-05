# DSCI6612-02_Term-Project - Pneumonia Detection from CXR (chest x-ray)

Intro:
Deep learning and computer vision advancements have made it possible to design software that helps doctors today. In this research, we're employing a convolutional neural network (CNN) model trained on chest X-ray pictures to identify whether a patient has pneumonia or not. This classification task was carried out using the VGG16 CNN model. To save radiologists' time, this model is used to pre-screen chest X-ray pictures before their review.
Two Jupyter Notebooks are used in this project:

1) Exploratory Data Analysis (EDA)
NIH X-Ray Dataset and X-ray image pixel-level analysis.

2) Build and Train Model
    Here we Used Scikit-Learn to split the dataset, Keras ImageDataGenerator for image pre-processing and build & train a Keras Sequential model and transform its probabilistic outputs to binary predictions and test the model with random images.


DATASET:

This NIH Chest X-ray Dataset contains 30,805 distinct patients with 112,120 X-ray images labeled with diseases. The related radiological reports were text-mined for disease classifications by the authors using Natural Language Processing to generate these labels. The labels should be > 90% accurate and adequate for learning under weak supervision. 
Here we used 74 X-ray images labeled with diseases.

PROJECT OVERVIEW:

Exploratory Data Analysis Building and Training Model

1)EDA.ipynb with Anaconda for exploratory data analysis

Each X-Ray image file's metadata includes information about the associated disease findings, patient gender, age, patient position during X-ray, and image shape. Pixel level evaluation of X-Ray picture files using Intensity Profiles of normalized image pixels graphed. X-Rays are also displayed using scikit-image. 

2)Building and Training Your Model, Fine Tuning Convolutional Neural Network VGG16 for Pneumonia Detection from X-Rays 

Inputs:
	ChestX-ray dataset containing 74 X-Ray images (.png) in data/images and metadata in data folder. 

Output:
	CNN model with model weights trained to categorize a chest X-ray image for the presence or absence of pneumonia.


Open Build_and_Train_Model with Jupyter Notebook.


	Create training data and validation data splits with scikit-learn train_test_split function.

Check that the training data split is equal between positive and negative cases. Ensure that the validation data split has a positive to negative case ratio that is representative of clinical scenarios. Also, ensure that the demographics of each split are representative of the overall dataset.

Using Keras ImageDataGenerator, prepare image preprocessing for each data split.

Create a new Keras Sequential model by adding VGG16 model layers and freezing their ImageNet-trained weights to fine-tune the ImageNet VGG16 model. Add Dense and Dropout layers after that, with their weights trained for identifying chest X-Ray pictures for pneumonia.

The model training will include a history that displays loss metrics for each training epoch. At each training epoch, the best model weights are also captured.

Model predictions are first returned as probabilities ranging from 0 to 1. The probabilistic outcomes were compared to the ground truth labels.

A threshold analysis was completed to select the boundary at which probabilistic results are converted into binary results of either pneumonia presence or absence. The CheXNet algorithm achieved an F1 score of 0.435, while a panel of four independent Radiologists averaged an F1 score of 0.387 [2]. This project's final F1 score is 0.36, which is similar in performance to the panel of Radiologist.

Prediction vs. Actual value:
	 At the end of the conclusion, we receive the model's accuracy and recall, and then we give it new input images to see how it performs, and it gives 70% correct responses.

