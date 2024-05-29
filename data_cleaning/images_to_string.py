import json
import os
from image_processing_helper_functions import encode_image, getResponse

# This script goes over the "neutral" folder, and attempts to get a response and label it with an emotion (engaged or neutral).
# If it fails for any reason (such as, timeout from the API, inconclusive results, etc), it labels it as ERROR to be reprocessed later

# It is intenteded to be ran once, in order to convert it into a file with json objects, with the string-fied images
# Has to be ran from "data_cleaning" directory (type "cd data_cleaning" on terminal)
neutral_arr = []
fo = open("./results/unknown_emotion_images.txt", "w")

for img in os.listdir("./img/neutral"):
    imgStr=encode_image('./img/neutral/'+str(img))
    resp = getResponse(imgStr)
    emotionStr = "ERROR"
    if 'choices' in resp and isinstance(resp['choices'], list) and len(resp['choices']) > 0:
            emotionStr = resp['choices'][0]['message']['content']
    neutral_arr.append({"img": imgStr, "emotion": emotionStr})

json.dump(neutral_arr, fo, indent=6)
fo.close()