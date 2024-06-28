import json
import random
import sys
import os

def read_and_randomize(input_filename: str, output_filename: str):
    data = []
    with open(input_filename) as f:
        data = json.load(f)
    
    random.shuffle(data)
    with open(output_filename, "w") as f:
        json.dump(data, f, indent=6)


def randomize_biased_data(biased_dir: str):
    directory_walker = os.walk(biased_dir)
    next(directory_walker)
    for dir, _, _ in directory_walker:
        inp_file = dir + os.sep + "encoded_images.json"
        out_file = dir + os.sep + "randomized_data.json"
        read_and_randomize(inp_file, out_file)

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Usage: python3 randomize_dataset [ -m | -b ]")
        print("-m: randomizes main dataset (overwrites randomized_data.json)")
        print("-b: randomizes biased datasets (overwrites all randomized data in bias_data/json/)")
        sys.exit(1)
    elif sys.argv[1] == "-m":
        read_and_randomize("data.json", "randomized_data_2.json")
    elif sys.argv[1] == "-b":
        randomize_biased_data("bias_data/json")



