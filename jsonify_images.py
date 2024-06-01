import base64
import json
import os


def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')


def write_json(image_directory, write_directory, emotions, num_per_emotion):

    for emotion in emotions:
        emotion_read_path = image_directory + emotion + "/"
        emotion_write_path = write_directory + emotion + ".json"
        for subdir, dirs, files in os.walk(emotion_read_path):
            with open(emotion_write_path, "w") as f:
                data = []
                for file in files[:num_per_emotion]:
                    image_path = subdir + os.sep + file
                    image_encoding = encode_image(image_path)
                    image_class = {
                        "img": image_encoding,
                        "emotion": emotion
                    }
                    data.append(image_class)
                json.dump(data, f, indent=6)


def write_json_as_one_file(image_directory, write_directory, emotions, num_per_emotion):


    emotion_write_path = write_directory + "data" + ".json"
    with open(emotion_write_path, "w") as f:
        data = []
        for emotion in emotions:
            emotion_read_path = image_directory + emotion + "/"
            for subdir, dirs, files in os.walk(emotion_read_path):
                i = 0
                for file in files[:num_per_emotion]:
                    image_path = subdir + os.sep + file
                    image_encoding = encode_image(image_path)
                    image_class = {
                        "img": image_encoding,
                        "emotion": emotion
                    }
                    data.append(image_class)
        json.dump(data, f, indent=6)

def collect_data_from_archive():
    data = []
    emotions = ["angry", "happy"]
    for emotion in emotions:
        for subdir, dirs, files in os.walk("archive/train/" + emotion + "/"):
            print("inside directory walk loop")
            for file in files[:500]:
                image_path = subdir + os.sep + file
                image_encoding = encode_image(image_path)
                image_class = {
                    "img": image_encoding,
                    "emotion": emotion
                }
                data.append(image_class)
    if len(data) == 0:
        print("didn't get any data from angry or happy, quitting")
        quit()

    return data

def collect_data_from_neutral():
    data = []
    for i in range(3000, 3500):
        filename = "archive/train/neutral/im" + str(i) + ".png"
        image_encoding = encode_image(filename)
        image_class = {
            "img": image_encoding,
            "emotion": "neutral" 
        }
        data.append(image_class)
    return data

def collect_data_from_engaged():
    engaged_from_gpt = []
    with open("data_cleaning/results/engaged_images.txt") as f:
        try:
            data = json.load(f)
        except:
            print("json load failed on engaged_images.txt")
            quit()
        else:
            engaged_from_gpt = data
    numbers = []
    with open("data_cleaning/results/engaged_numbers.txt") as f:
        lines = f.readlines()
        numbers = [int(line.strip()) for line in lines]
    if len(numbers) != 179:
        print("numbers is not the right length, its", len(numbers))
        quit()
    engaged_from_numbers = []
    for num in numbers:
        filename = "archive/train/neutral/im" + str(num) + ".png"
        image_encoding = encode_image(filename)
        image_class = {
            "img": image_encoding,
            "emotion": "engaged"
        }
        engaged_from_numbers.append(image_class)

    engaged = engaged_from_gpt + engaged_from_numbers
    print("length of engaged is", len(engaged))
    return engaged[:500]


def write_data(data):
    with open("data.json", "w") as f:
        json.dump(data, f, indent=6)
if __name__ == "__main__":
    data = []
    data += collect_data_from_archive()
    print("length of data after collect from happy and angry:", len(data))
    data += collect_data_from_neutral()
    data += collect_data_from_engaged()
    print("length of data is", len(data))
    with open("data.json", "w") as f:
        json.dump(data, f, indent=6)

