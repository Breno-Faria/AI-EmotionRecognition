# AI-EmotionRecognition
# Comp 472 Summer 2024 AI project

# Group members
 - Breno Faria
 - Sydney Stoddart
 - Peter Howe

# Content

## collect_data.py


This script collect all of our data together, formatted with JSON, and aggregating them into data.json. It collects all of the ~370 focused images from chat-gpt, the remaining sorted by hand, and 500 neutral images. It then converts the first 500 happy and angry images from the dataset we retrieved online. Once they are collected into a single array, these 2000 json objects are dumped into data.json. 

## plot.py

Plot.py collects images from our JSON formatted dataset and plots these using matplotlib. It also contains some helper functions to collect the data from our dataset, either sorted into four arrays per each emotion or as one large array. 

Running plot.py without arguments will create every plot required for part 1 of the project. The argument options are as follows:

-f: display the frequencies of emotions, i.e. the number of each emotion in the dataset

-h: display pixel density histograms for each class

-r: display pixel density histograms for 15 random images from each category

## data_cleaning

This folder contains the scripts that will go through the "neutral images" folder, and check for any "engaged" pictures. The first time the file is ran, images_to_string goes over the folder containing the png images and encodes them using python's library b64. After that, unknown_emotion_cleanup is ran to properly divide everything into 3 folders, one for unknown emoitons, one for neutral and another one for engaged. Finally, evaluate_images is the file inteted to be ran multiple times, which will go over the unknown_emotion images and label as many as possible. After that, it updates all three files. There is also a fourth file containing helper function

The results folder has the images separated by emotion, the neutral folder has the images in png format

## decoding_images

This folder contains the image_decoder script, that as the name suggests, receives as input an array of json objects and then goes over them writing them to a folder, converting them into PNG images
