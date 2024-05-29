import json
from image_processing_helper_functions import getResponse
import os

# Tries to open the files containing the evaluated images. If no such file exists, it initializes an empty array and creates the file. Otherwise,
# it reads the file, saves it to an array (and then overwrites currently file)

neutral_output = open("./results/neutral_images.txt", "w")
neutral_arr = []

engaged_output = open("./results/engaged_images.txt", "w")
engaged_arr = []

unknown_output = open("./results/unknown_emotion_images_updated.txt", "w")
unknown_arr = []


# Iterates over file with images that haven't been evaluated yet. Tries getting a response, if it is succesful, it appends it to the correspondent
# emotion array. Otherwise, it remains in the "unknown" list.
with open('./results/unknown_emotion_images.txt') as file:
    picsArray = json.load(file)
    for pic in picsArray:
        if pic["emotion"].lower() == "engaged":
            engaged_arr.append(pic)
            
        elif pic["emotion"].lower() == "neutral":
            neutral_arr.append(pic)
            
        elif pic["emotion"].lower() == "error":
            unknown_arr.append(pic)

# After iterating over the file attempting to classify the emotions, it checks if there is an non-empty array. If so, it writes it to a file.
if len(engaged_arr) > 0:
    engaged_output.seek(0)
    json.dump(engaged_arr, engaged_output,indent=6)
if len(neutral_arr) > 0:
    neutral_output.seek(0)
    json.dump(neutral_arr, neutral_output,indent=6)
if len(unknown_arr) > 0:
    json.dump(unknown_arr, unknown_output,indent=6)
neutral_output.close()
engaged_output.close()
unknown_output.close()

# This step is done to make sure the evaluated images aren't kept in the "unknown emotions" file, to avoid duplicates.
os.remove("./results/unknown_emotion_images.txt")
os.rename("./results/unknown_emotion_images_updated.txt", "./results/unknown_emotion_images.txt")