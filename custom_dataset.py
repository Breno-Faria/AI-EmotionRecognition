import os
import collect_data

directory = "custom/"
directories = ["angry", "happy", "neutral", "engaged"]
for class_directory in directories:
    path = directory + class_directory
    for root, dirs, files in os.walk(path):
        for file in files:
            print(file)
            print(collect_data.encode_image(path + "/" + file))
