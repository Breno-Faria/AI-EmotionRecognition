# AI-EmotionRecognition
# Comp 472 Summer 2024 AI project

# Group members
 - Breno Faria
 - Sydney Stoddart
 - Peter Howe

# Content

## dataset.py

Dataset.py defines our dataset class EmotionDataset, inheriting from torch.utils.data.Dataset. This allows us to use our dataset in a consistent way with our models.

## model.py

Model.py defines our main and variant models, both inheriting from torch.nn.Module. They contain the standard architecture for a class inheriting from a Module, including defining the layers as nn.Sequential layers as well as overriding the forward method to allow for its use. Since the kernel size defines how the layers scale from two dimentional down to one, they contain functionality to dynamically scale the width of the first forwards layer on initiation. Analysis of the design decisions are available in our report.

## train.py

This script is the one responsible for training our models. It uses 10 epocs, saving the best version from each epoc to models/{model_name.pth}. When using this script, you must provide a filename ending in .pth. Usage is as follows:

python3 train.py {filename.pth} {variant?} {kernel_size}

The optional variant flag, -v, decides whether we use the variant model or not. The optional kernel size, either 4 or 7 (default 7), decides the size of the kernel for the model. 

## evaluation.py

This script calculates the micro and macro confusion matrices, which it then uses to calculate accuracy, precision, recall, and f1 measure. It takes the models that are decided on calling of the script and outputs these measures. Usage is as follows:

python3 evaluation.py {micro/macro} {variant_number}

micro/macro: -i for micro statistics, -a for macro statistics
variant number: 0, 1, 2, or 3. If none are provided it will display statistics for all four.

## collect_data.py


This script collect all of our data together, formatted with JSON, and aggregating them into data.json. It collects all of the ~370 focused images from chat-gpt, the remaining sorted by hand, and 500 neutral images. It then converts the first 500 happy and angry images from the dataset we retrieved online. Once they are collected into a single array, these 2000 json objects are dumped into data.json. 

The main function of this file contains all the function calls to aggregate our data. Simply running the code performs the data aggregation (assuming our dataset is placed in the main directory).

This file will not run without the dataset included. To test this script, download the dataset from https://www.kaggle.com/datasets/ananthu017/emotion-detection-fer and place the archive folder in the main directory of this project.

## plot.py

Plot.py collects images from our JSON formatted dataset and plots these using matplotlib. It also contains some helper functions to collect the data from our dataset, either sorted into four arrays per each emotion or as one large array. 

Running plot.py without arguments will create every plot required for part 1 of the project. The argument options are as follows:

-f: display the frequencies of emotions, i.e. the number of each emotion in the dataset

-h: display pixel density histograms for each class
 
-r: display pixel density histograms for 15 random images from each category

-d: collect the first 25 images from each of the four classes, placing them in a directory for easy zipping and submitting.

## data_cleaning

This folder contains the scripts that will go through the "neutral images" folder, and check for any "engaged" pictures. The first time the file is ran, images_to_string goes over the folder containing the png images and encodes them using python's library b64. After that, unknown_emotion_cleanup is ran to properly divide everything into 3 folders, one for unknown emoitons, one for neutral and another one for engaged. Finally, evaluate_images is the file inteted to be ran multiple times, which will go over the unknown_emotion images and label as many as possible. After that, it updates all three files. There is also a fourth file containing helper function

The results folder has the images separated by emotion, the neutral folder has the images in png format

## decoding_images

This folder contains the image_decoder script, that as the name suggests, receives as input an array of json objects and then goes over them writing them to a folder, converting them into PNG images
