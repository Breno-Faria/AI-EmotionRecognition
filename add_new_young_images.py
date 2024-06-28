import json
import os
import collect_data
import random

def collect_from_new_images(emotion_txt, emotion_dir):
    numbers = []
    with open(emotion_txt) as f:
        lines = f.readlines()
        numbers = [int(line.strip()) for line in lines]
    if len(numbers) != 30:
        print("numbers is not the right length, its", len(numbers))
        quit()
    emotion_from_numbers = []
    for num in numbers:
        filename = emotion_dir + os.sep + "im" + str(num) + ".png"
        image_encoding = collect_data.encode_image(filename)
        image_class = {
            "img": image_encoding,
            "emotion": "engaged"
        }
        emotion_from_numbers.append(image_class)

    print("length of new emotions is", len(emotion_from_numbers))
    return emotion_from_numbers

def make_new_dataset():
    data = []
    with open("data.json") as f:
        data = json.load(f)
    data += collect_from_new_images("./unbias_dataset/angry.txt", "unbias_dataset/test/angry")
    data += collect_from_new_images("./unbias_dataset/happy.txt", "unbias_dataset/test/happy")
    data += collect_from_new_images("./unbias_dataset/neutral.txt", "unbias_dataset/test/neutral")
    data += collect_from_new_images("./unbias_dataset/engaged.txt", "unbias_dataset/test/neutral")
    random.shuffle(data)
    with open("new_data.json", "w") as f:
        json.dump(data, f, indent=6)

def make_new_young_json():
    data = []
    with open("bias_data/json/young/old_encoded_images.json") as f:
        data = json.load(f)
    data += collect_from_new_images("./unbias_dataset/angry.txt", "unbias_dataset/test/angry")
    data += collect_from_new_images("./unbias_dataset/happy.txt", "unbias_dataset/test/happy")
    data += collect_from_new_images("./unbias_dataset/neutral.txt", "unbias_dataset/test/neutral")
    data += collect_from_new_images("./unbias_dataset/engaged.txt", "unbias_dataset/test/neutral")
    with open("bias_data/json/young/encoded_images.json", "w") as f:
        json.dump(data, f, indent=6)

if __name__ == "__main__":
    make_new_dataset()
