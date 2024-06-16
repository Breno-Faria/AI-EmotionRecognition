#accuracy = (confusion_matrix[0][0] + confusion_matrix[1][1]) / 2
from multiprocessing import freeze_support
import train
import numpy as np
import torch
from model import BrenoPeterandSydneysSuperCoolConvolutionalNeuralNetworkVeryAccurateAndGood
from PIL import Image
import torchvision.transforms as transforms
import os
from dataset import loadData
import base64
import io

def generate_micro_metrics_row(confusion_matrix_array):

    confusion_matrix_sum = np.sum(confusion_matrix_array, axis=0)
    return generate_metrics_row(confusion_matrix_sum)

def generate_macro_metrics_row(confusion_matrix_array):

    metrics_results = np.array([0.0, 0.0, 0.0])

    for confusion_matrix in confusion_matrix_array:
        metrix_results += generate_metrics_row(confusion_matrix)
    metrics_results /= len(confusion_matrix_array)

    return metrics_results



def generate_metrics_row(confusion_matrix):

    precision = confusion_matrix[0][0] / (confusion_matrix[0][0]+confusion_matrix[1][0] + 1e-10)
    recall = confusion_matrix[0][0] / (confusion_matrix[0][0] + confusion_matrix[0][1] + 1e-10)
    f1 = 2 * ((precision*recall) / precision + recall + 1e-10)

    res = [precision, recall, f1]
    return res


def predict_emotion(model, img):

    model.eval()

    # Define preprocessing transformations (same as in training)
    preprocess = transforms.Compose([
        transforms.Resize((48, 48)),  # Resize image to match training input size
        transforms.ToTensor(),  # Convert image to tensor
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize image
    ])

    image = Image.open(io.BytesIO(img)).convert('RGB')
    # Load and preprocess the image
    input_tensor = preprocess(image)
    input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension

    # Perform inference
    with torch.no_grad():  # Disable gradient computation
        output = model(input_tensor)

    _, predicted_class = torch.max(output, 1)
    return predicted_class.item()
    print(f'Predicted class: {predicted_class.item()}')


def generate_confusion_matrix():

    labels = {
            "happy": 0,
            "angry": 1,
            "engaged": 2,
            "neutral": 3,
        }
    
    model = BrenoPeterandSydneysSuperCoolConvolutionalNeuralNetworkVeryAccurateAndGood()
    model.load_state_dict(torch.load('model.pth'))

    tmp, tmp, testing_data_arr = loadData("./data.json")

    confusion_matrix = [

        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ]

    for labeled_data in testing_data_arr:
        predicted_class = predict_emotion(model, base64.b64decode(labeled_data['img']))
        confusion_matrix[predicted_class][labels[labeled_data['emotion']]] +=1


    return confusion_matrix

conf = generate_confusion_matrix()
print(conf)





# # Load the model
# model = BrenoPeterandSydneysSuperCoolConvolutionalNeuralNetworkVeryAccurateAndGood()
# model.load_state_dict(torch.load('model.pth'))
# model.eval()

# # Define preprocessing transformations (same as in training)
# preprocess = transforms.Compose([
#     transforms.Resize((48, 48)),  # Resize image to match training input size
#     transforms.ToTensor(),  # Convert image to tensor
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize image
# ])

# # Load and preprocess the image
# image_path = './dataset/happy/img0.png'
# image = Image.open(image_path).convert('RGB')
# input_tensor = preprocess(image)
# input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension

# # Perform inference
# with torch.no_grad():  # Disable gradient computation
#     output = model(input_tensor)

# _, predicted_class = torch.max(output, 1)
# print(f'Predicted class: {predicted_class.item()}')