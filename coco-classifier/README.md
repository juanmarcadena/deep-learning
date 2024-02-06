# Image Classification with PyTorch and COCO Dataset

This project showcases the creation of a custom image classification dataset from the COCO dataset and the implementation of Convolutional Neural Networks (CNNs) for image classification using PyTorch. The classification targets five categories: airplane, bus, cat, dog, and pizza.

## Overview

The project is conducted on Google Colab to leverage its advanced GPUs, enhancing the training speed and performance of our models. The dataset, once prepared, is stored on Google Drive for easy access and management within the Colab environment.

## Setup and Installation

To run this project, you will need access to Google Colab and a Google Drive account for dataset storage. Ensure the following libraries are installed in your Colab environment:

- PyTorch
- torchvision
- pycocotools
- PIL (Python Imaging Library)
- NumPy
- Matplotlib
- scikit-learn
- seaborn
- pandas

Most of these should be pre-installed in Google Colab; you can install any missing libraries using `!pip install`.

## Data Preparation

The data is collected from the COCO dataset, focusing on images that fall into the categories of 'airplane,' 'bus,' 'cat,' 'dog,' and 'pizza.' A script filters out images with multiple classes to simplify classification and selects a fixed number of images for each category. The images are resized to 64x64 pixels and saved to Google Drive, structured for easy access during training.

## Model Training and Evaluation

Three CNN models of varying complexity are implemented and trained on the dataset:

- **Net1**: A basic model with two convolutional layers.
- **Net2**: An improved version of Net1 with padding added to the convolutional layers.
- **Net3**: A more complex model featuring three convolutional layers and additional depth.

The training process, implemented in Google Colab, takes advantage of the platform's GPUs. After training, the models are evaluated using accuracy metrics and confusion matrices to understand their performance.

## Running the Notebook

1. Mount your Google Drive to Colab to access the dataset and save model outputs:

    ```python
    from google.colab import drive
    drive.mount('/content/gdrive')
    ```

2. Navigate to the notebook in Google Colab and run the cells sequentially to train and evaluate the models.

3. The dataset should be located in your Google Drive, in a path accessible to the notebook for loading and processing the images.

## Results

The performance of each CNN model is assessed based on accuracy and loss over the training and validation datasets. The confusion matrix provides insight into the classification capabilities of each model across the different categories.