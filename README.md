# Convolutional-Recurrent Architectures for Art Classification

## Project Overview

This project implements a convolutional-recurrent model for classifying artwork attributes such as **Style**, **Artist**, **Genre**, and other related categories. The model utilizes a CNN-RNN architecture, where the **ResNet-50** CNN acts as a feature extractor, followed by an **LSTM** layer to capture the sequential nature of art styles and genres. 

The task involves using the **ArtGAN dataset**, which contains images of artwork from different genres and artists. The goal is to classify the artwork into predefined categories, improving the classification performance with each training epoch.

In addition to classification, the project also evaluates the model based on accuracy and explores **outliers**—paintings that might not fit a particular artist or genre despite being assigned to them.

## Dataset

The dataset used for training is the **ArtGAN dataset**, which can be downloaded from the following link:

[ArtGAN Dataset](https://drive.google.com/file/d/1vTChp3nU5GQeLkPwotrybpUGUXj12BTK/view)

### Dataset Details:
- The dataset consists of images from various genres, artists, and styles.
- The images are categorized into different classes, including **Abstract Expressionism**, **Cubism**, **Pop Art**, and more.
- This dataset is used for training and evaluating the model's ability to classify artworks based on style, artist, and other attributes.

## Files

### 1. `train_model.py`
- This script is responsible for training the **CNN-RNN** model on the **ArtGAN dataset**.
- It includes data preprocessing steps such as resizing images and normalization.
- The dataset is split into **train**, **validation**, and **test** subsets.
- The model architecture is based on a **ResNet-50** backbone, with an **LSTM** layer for sequential learning. 
- After each epoch, the model's weights are saved:
  - `model_best.pth` for the best model based on validation accuracy.
  - `model_last.pth` for the model at the end of the training process.

### 2. `test_model.py`
- This script tests the trained model.
- It loads the best model (`model_best.pth`) and evaluates its performance on the test dataset.
- The script computes the **test accuracy** and identifies outliers (i.e., artworks that were incorrectly classified or don't fit the assigned artist/genre).

### 3. `class_names.txt`
- This text file contains the names of the classes in the dataset (one per line), which corresponds to the different artists, genres, and styles.

### 4. Model Files
- The `.pth` files (e.g., `model_best.pth`) contain the saved weights of the trained model.
- **Note:** These model files are too large to upload directly to GitHub. They can be generated by running the `train_model.py` script or requested via a separate link.

## How to Run

### 1. Install Dependencies:
To run the scripts, first install the required Python libraries:
```bash
pip install -r requirements.txt
```

### 2. Train the Model
To train the model, simply run the following script:
```bash
python train_model.py
```
This will start the training process, saving the model's weights after each epoch. The number of epochs can be adjusted within the script based on the desired level of training.

### 3. Test the Model
Once the model is trained, performance can be evaluated on the test set by running the following script:
```bash
python test_model.py
```
This will load the best model (model_best.pth) and output the test accuracy and outlier information.

## Evaluation Metrics

**1. Accuracy:**

Accuracy is the primary evaluation metric used to assess the model’s performance on the test dataset. It is calculated as the ratio of correctly predicted samples to the total number of samples. For example, after running the model for 2 epochs, the accuracy obtained was around 33.28%, which is relatively low. However, if the number of epochs is increased, the model will likely achieve better accuracy, as it has more opportunities to learn from the dataset.

**2. Outlier Detection:**

Outlier detection is used to identify artworks that do not fit the expected artist, style, or genre, even though they may have been assigned to them based on the dataset labels. This can happen due to various reasons, such as ambiguous classifications or unusual features in the artwork. Outlier detection is implemented by checking the confidence scores of the model's predictions. If the confidence (calculated using softmax) for a particular class is significantly low, it indicates that the model is uncertain about its prediction, which could suggest an outlier.

Outliers are identified based on the following criteria:

Softmax Confidence Threshold: If the confidence score for the predicted class is lower than a predefined threshold (e.g., 0.6), it may indicate that the model is unsure about the classification, making the sample an outlier.
    
Incorrect Classifications: If the predicted class doesn't match the actual label and the confidence is low, the sample is considered an outlier.

### Impact of Epochs on Accuracy and Outliers:

Since the model was only trained for 2 epochs, the accuracy is relatively low (about 33.28%) and there are many outliers identified. This is expected because the model hasn't had enough time to learn the complex patterns in the dataset.
    
Training for more epochs would allow the model to learn better feature representations, likely improving the accuracy and reducing the number of outliers. The more epochs the model is trained, the more robust the predictions will become.

## Conclusion

This project demonstrates how convolutional-recurrent models can be applied to the classification of artwork from different genres and artists. The implementation includes:

Data preprocessing and augmentation.
    
A convolutional-recurrent architecture combining ResNet-50 and LSTM.
    
Evaluation using accuracy and outlier detection.

By increasing the training epochs, the model's performance can be improved, leading to higher accuracy and fewer outliers.
