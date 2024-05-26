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

    

if __name__ == "__main__":
    image_directory = "archive/train/"
    write_directory = "data/"
    emotions = ["neutral", "angry", "happy"]
    write_json(image_directory, write_directory, emotions, 500)
    write_json_as_one_file(image_directory, write_directory, emotions, 500)
    #read_json(write_directory, emotions)

