import json
import os
import base64

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

path_json = "bias_data/json/"
path_png = "bias_data/png/"

for emotion in os.listdir(path_json):
    if emotion != ".DS_Store":
        path_2_json = path_json + emotion
        path_2_png = path_png + emotion
        for category in os.listdir(path_2_png):
            if category != ".DS_Store":
                path_3_png = path_2_png + "/" + category
                path_3_json = path_2_json + "/" + category
                fo = open(path_3_json+"/encoded_images.txt", "w")
                tmp_arr = []
                for img in os.listdir(path_3_png):
                    if img != ".DS_Store":
                        imgStr = encode_image(path_3_png+"/"+img)
                        tmp_arr.append({"img": imgStr, "emotion": emotion.lower()})
                json.dump(tmp_arr, fo, indent=6)
                fo.close()