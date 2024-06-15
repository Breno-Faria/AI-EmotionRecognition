import json
import random
import base64
import torch
from torch.utils.data import Dataset
from PIL import Image
from io import BytesIO

class EmotionDataset(Dataset):
    def __init__(self, json_array, transform=None):
        self.transform = transform
        self.data = json_array
        self.label_map = {
            "happy": 0,
            "angry": 1,
            "engaged": 2,
            "neutral": 3,
        }
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img = item["img"]
        emotion = self.label_map[item["emotion"]]

        img_bytes = base64.b64decode(img)

        img = Image.open(BytesIO(img_bytes))
        if img.mode == 'L':
            img = img.convert('RGB')
        if self.transform:
            img = self.transform(img)
        emotion = torch.tensor(emotion, dtype=torch.long)
        return img, emotion

def loadData(json_file, num_training=1400, num_validation=300, num_testing=300):
    data = []
    with open(json_file) as f:
        data = json.load(f)
    random.shuffle(data)
    validation_idx = num_training
    testing_idx = num_training + num_validation
    training = data[:validation_idx]
    validation = data[validation_idx:testing_idx]
    testing = data[testing_idx:]
    return training, validation, testing


