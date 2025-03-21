import json
import random
import base64
import torch
from torch.utils.data import Dataset
from PIL import Image
from io import BytesIO
from sklearn.model_selection import KFold

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

def loadTestingData(json_file, section):
    data = []
    with open(json_file) as f:
        data = json.load(f)
    if section == 0:
        return data[0:400]
    if section == 1:
        return data[400:800]
    if section == 2:
        return data[800:1200]
    if section == 3:
        return data[1200:1600]
    return data[1600:2000]


def loadData(json_file, num_training=1400, num_validation=300, num_testing=300, biased=False):
    data = []
    with open(json_file) as f:
        data = json.load(f)
    if not biased:
        validation_idx = num_training
        testing_idx = num_training + num_validation

    else:
        return [], [], data
        #validation_idx = (len(data)//10)*8
        #testing_idx = validation_idx+(len(data)//10)
    training = data[:validation_idx]
    validation = data[validation_idx:testing_idx]
    testing = data[testing_idx:]
    print(len(testing))
    return training, validation, testing


def k_loadData(json_file, k=10):
    data = []
    with open(json_file) as f:
        data = json.load(f)
    
    kf = KFold(n_splits=k)
    
    fold_data = []
    for train_index, test_index in kf.split(data):
        training = [data[i] for i in train_index]
        testing = [data[i] for i in test_index]
        fold_data.append((training, testing))
    
    return fold_data
