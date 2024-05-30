import json
import os
import base64
import matplotlib.pyplot as plt
import numpy as np
import random
from io import BytesIO
from PIL import Image
import jsonify_images

    
def get_data(file_path):
    data = []
    with open(file_path, "r") as f:
        try:
            data = json.load(f)
        except:
            print("json load failed on file " + file_path)
            return
    return data


def get_sorted_data(file_path):
    data = get_data(file_path)
    if data == None:
        return
    sorted_data = []
    emotions = get_emotion_stats(file_path)
    if emotions == None:
        return
    for emotion in emotions.keys():
        emotion_data = []
        for img in data:
            if img["emotion"] == emotion:
                emotion_data.append(img)
        sorted_data.append(emotion_data)
    return sorted_data
     


def get_emotion_stats(file_path):
    data = get_data(file_path)
    if data == None:
        return

    emotions = {}
    for image in data:
        emotion = image["emotion"]
        if emotion not in emotions:
            emotions[emotion] = 1
        else:
            emotions[emotion] = emotions[emotion] + 1

    return emotions


def get_img_array(file_path):
    data = get_data(file_path)
    if data == None:
        return

    imgs = []
    for image in data:
        imgs.append(image["img"])

    return imgs


def plot_emotions(emotions):
    plt.bar(emotions.keys(), emotions.values(), color='skyblue', width=0.6)
    plt.title("Emotion frequencies")
    plt.xlabel("Emotions")
    plt.ylabel("Data points")
    plt.show()



def random_image_sample(dir, num_files):
    all_files = os.listdir(dir)
    all_files = [f for f in all_files if os.path.isfile(os.path.join(dir, f))]
    if len(all_files) < num_files:
        raise ValueError("Not enough files for given sample number")
    sample_files = random.sample(all_files, num_files)
    for file in sample_files:
        encoded_image = jsonify_images.encode_image(os.path.join(dir, file))
        plot_pixel_frequency(encoded_image, file)


def plot_pixel_frequency(img, title):
    binary_image = base64.b64decode(img)
    image = Image.open(BytesIO(binary_image))
    img_arr = np.array(image).flatten()
    plt.figure(figsize=(10, 6))
    plt.hist(img_arr, bins=256, range=(0, 255), edgecolor='black', alpha=0.75)
    plt.title(f'Pixel Intensity Distribution: {title}')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()


def plot_class_frequency(imgs, image_class):
    images = np.array([])
    for img in imgs:
        i=0
        if img["emotion"] == image_class:
            i += 1
            binary_image = base64.b64decode(img["img"])
            image = Image.open(BytesIO(binary_image))
            img_arr = np.array(image).flatten()
            print(len(img_arr))
            images = np.concatenate((images, img_arr))
            if i >= 10:
                break

    plt.figure(figsize=(10, 6))
    plt.hist(images, bins=256, range=(0, 255), edgecolor='black', alpha=0.75)
    plt.title(f'Pixel Intensity Distribution: {image_class}')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.grid(True)


def plot_all_class_frequencies(sorted_imgs):
    for images in sorted_imgs:
        img_class = images[0]["emotion"]
        plot_class_frequency(images, img_class)

if __name__ == "__main__":
    frequencies = False
    histograms = {
        "neutral": False,
        "happy": False,
        "engaged": False,
        "angry": False
    }


    if len(sys.argv) < 2:
        
    emotions = ["neutral", "happy", "angry"]
    filename = "data/data.json"
    plot_emotions(get_emotion_stats(filename))
    #img_array = get_img_array(filename)
    # np.set_printoptions(threshold=100)
    # data = get_sorted_data(filename)
    # if data != None:
    #     plot_all_class_frequencies(data)
        # for emotion in emotions:
        #     plot_class_frequency(data, emotion)
    #random_image_sample("archive/test/neutral", 10)
    plt.show()
