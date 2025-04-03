# Convolutional-Recurrent Architectures for Art Classification

## Project Overview

This project implements a convolutional-recurrent model for classifying artwork attributes such as **Style**, **Artist**, **Genre**, and other related categories. The model utilizes a CNN-RNN architecture, where the ResNet-50 CNN acts as a feature extractor, followed by an LSTM layer to capture the sequential nature of art styles and genres.

The task involves using the **ArtGAN dataset**, which contains images of artwork from different genres and artists. The goal is to classify the artwork into predefined categories, improving the classification performance with each training epoch.

The project also includes evaluating the model based on various metrics such as accuracy and exploring outliers, i.e., paintings that might not fit a particular artist or genre despite being assigned to them.

## Dataset

The dataset used for training is the **ArtGAN dataset**, which can be downloaded from the following link:

[ArtGAN Dataset](https://drive.google.com/file/d/1vTChp3nU5GQeLkPwotrybpUGUXj12BTK/view)

### Dataset Details:
- The dataset consists of images from various genres, artists, and styles.
- The images are categorized into different classes, including Abstract Expressionism, Cubism, Pop Art, etc.
- This dataset is used for training and evaluating the model's ability to classify artworks based on style and other attributes.

## Files

### 1. `train_model.py`
- This script trains the CNN-RNN model on the ArtGAN dataset.
- It includes data preprocessing (resizing and normalization), dataset splitting (train, validation, test), and model training.
- The model is based on a ResNet-50 backbone with an LSTM layer for sequence learning.
- After each epoch, the model's weights are saved to disk (`model_best.pth` for the best model and `model_last.pth` for the latest trained model).

### 2. `test_model.py`
- This script is used to test the trained model.
- It loads the best model (`model_best.pth`) and evaluates its performance on the test dataset.
- The script computes the test accuracy and outputs the result.

### 3. `class_names.txt`
- This text file contains the names of the classes in the dataset (one per line).

### 4. Model Files
- The `.pth` files (e.g., `model_best.pth`) are the saved weights of the trained model.
- **Note:** These files may be too large to upload directly to GitHub. You can train the model locally using the `train_model.py` script or request the model weights via a separate link.

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
This will begin the training process, saving the model's weights after each epoch. You can adjust the number of epochs in the script if you wish to train the model for more epochs.

### 3. Test the Model
Once the model is trained, you can evaluate its performance on the test set by running the following script:
```bash
python test_model.py
```
This will load the best model and output the test accuracy.

## Evaluation Metrics
*Accuracy*: The primary evaluation metric used to assess the model's performance on the test set.
