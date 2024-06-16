import json
import random

def read_and_randomize(input_filename: str, output_filename: str):
    data = []
    with open(input_filename) as f:
        data = json.load(f)
    
    random.shuffle(data)
    with open(output_filename, "w") as f:
        json.dump(data, f, indent=6)


if __name__ == "__main__":
    read_and_randomize("data.json", "randomized_data.json")



