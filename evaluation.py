#accuracy = (confusion_matrix[0][0] + confusion_matrix[1][1]) / 2
from multiprocessing import freeze_support
import numpy as np
import torch
from model import BrenoPeterandSydneysSuperCoolConvolutionalNeuralNetworkVeryAccurateAndGood, BrenoPeterandSydneysSuperCoolConvolutionalNeuralNetworkVeryAccurateAndGood_Variant1
from PIL import Image
import torchvision.transforms as transforms
from dataset import loadData
import base64
import io



def _generate_micro_metrics_row(confusion_matrix):


    num_classes = len(confusion_matrix)


    precision = [0.0] * num_classes
    recall = [0.0] * num_classes
    f1_measure = [0.0] * num_classes

    for i in range(num_classes):
        # Precision for class i
        true_positives = confusion_matrix[i][i]
        predicted_positives = sum(confusion_matrix[j][i] for j in range(num_classes))
        precision[i] = true_positives / predicted_positives if predicted_positives > 0 else 0.0
        
        # Recall for class i
        actual_positives = sum(confusion_matrix[i][j] for j in range(num_classes))
        recall[i] = true_positives / actual_positives if actual_positives > 0 else 0.0
        
        # F1-Measure for class i
        if precision[i] + recall[i] > 0:
            f1_measure[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i])
        else:
            f1_measure[i] = 0.0

    # Micro-averaged Precision, Recall, and F1-Measure
    micro_averaged_precision = sum(precision) / num_classes
    micro_averaged_recall = sum(recall) / num_classes
    micro_averaged_f1_measure = sum(f1_measure) / num_classes

    return micro_averaged_precision, micro_averaged_f1_measure, micro_averaged_recall

def generate_macro_metrics_row(confusion_matrix):

    precision = [0.0] * 4
    recall = [0.0] * 4
    f1_measure = [0.0] * 4

    for i in range(4):

        true_positives = confusion_matrix[i][i]
        predicted_positives = sum(confusion_matrix[j][i] for j in range(4))
        precision[i] = true_positives / predicted_positives if predicted_positives > 0 else 0.0
        
        actual_positives = sum(confusion_matrix[i][j] for j in range(4))
        recall[i] = true_positives / actual_positives if actual_positives > 0 else 0.0
        
        if precision[i] + recall[i] > 0:
            f1_measure[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i])
        else:
            f1_measure[i] = 0.0

    macro_averaged_precision = 0.0
    for i in range(4):
        macro_averaged_precision += precision[i]
    macro_averaged_precision /= 4

    macro_averaged_f1_measure = 0.0
    for i in range(4):
        macro_averaged_f1_measure += f1_measure[i]
    macro_averaged_f1_measure /= 4

    macro_averaged_recall = 0.0
    for i in range(4):
        macro_averaged_recall += recall[i]
    macro_averaged_recall /= 4

    return macro_averaged_precision, macro_averaged_f1_measure, macro_averaged_recall



def generate_metrics_row(confusion_matrix):

    # Accuracy
    accuracy = 0.0
    correct_predictions = sum(confusion_matrix[i][i] for i in range(4))
    total_predictions = sum(sum(confusion_matrix[i][j] for j in range(4)) for i in range(4))
    accuracy = correct_predictions / total_predictions

    # Precision, Recall, and F1-Measure for each class
    precision = [0.0] * 4
    recall = [0.0] * 4
    f1_measure = [0.0] * 4

    for i in range(4):
        # Precision for class i
        true_positives = confusion_matrix[i][i]
        predicted_positives = sum(confusion_matrix[j][i] for j in range(4))
        precision[i] = true_positives / predicted_positives if predicted_positives > 0 else 0.0
        
        # Recall for class i
        actual_positives = sum(confusion_matrix[i][j] for j in range(4))
        recall[i] = true_positives / actual_positives if actual_positives > 0 else 0.0
        
        # F1-Measure for class i
        if precision[i] + recall[i] > 0:
            f1_measure[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i])
        else:
            f1_measure[i] = 0.0

    return precision, f1_measure, recall, accuracy
    
    

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


def generate_confusion_matrices(variant=0):
    labels = {
            "happy": 0,
            "angry": 1,
            "engaged": 2,
            "neutral": 3,
        }
   
    if variant == 0:
        model = BrenoPeterandSydneysSuperCoolConvolutionalNeuralNetworkVeryAccurateAndGood()
        model.load_state_dict(torch.load('models/model.pth'))

    if variant == 1:
        model = BrenoPeterandSydneysSuperCoolConvolutionalNeuralNetworkVeryAccurateAndGood_Variant1()
        model.load_state_dict(torch.load('models/model_variant1.pth'))

    if variant == 2:
        model = BrenoPeterandSydneysSuperCoolConvolutionalNeuralNetworkVeryAccurateAndGood(kernel_size=7)
        model.load_state_dict(torch.load('models/model_variant2.pth'))

    if variant == 3:
        model = BrenoPeterandSydneysSuperCoolConvolutionalNeuralNetworkVeryAccurateAndGood_Variant1(kernel_size=7)
        model.load_state_dict(torch.load('models/model_variant3.pth'))


    _, _, testing_data_arr = loadData("./data.json")

    confusion_matrix = [

        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ]

    micro_confusion_matrices = {
        0: [
            [0, 0],
            [0, 0],
        ],
        1: [
            [0, 0],
            [0, 0],
        ],
        2: [
            [0, 0],
            [0, 0],
        ],
        3: [
            [0, 0],
            [0, 0],
        ],
    }

    for labeled_data in testing_data_arr:
        predicted_class = int(predict_emotion(model, base64.b64decode(labeled_data['img'])))
        true_class = labels[labeled_data['emotion']] 
        confusion_matrix[predicted_class][true_class] +=1

        if predicted_class == true_class: # Correct evaluation
            for emotion in micro_confusion_matrices.keys():
                emotion_micro_matrix = micro_confusion_matrices[emotion]
                if emotion == predicted_class:
                    emotion_micro_matrix[0][0] += 1 # Increase true positive
                else:
                    emotion_micro_matrix[1][1] += 1 # Increase true negative
        else: # Incorrect evaluation
            for emotion in micro_confusion_matrices.keys():
                emotion_micro_matrix = micro_confusion_matrices[emotion]
                if emotion == true_class:
                    emotion_micro_matrix[1][0] += 1 # Increase false negative
                elif emotion == predicted_class:
                    emotion_micro_matrix[0][1] += 1 # Increase false positive
                else:
                    emotion_micro_matrix[1][1] += 1 # Increase true negative
                    


    
    return confusion_matrix, micro_confusion_matrices

if __name__ == "__main__":

    for i in range(4):
        print('\n')
        print(f"Variant {i}")
        general_matrix, micro_matrices = generate_confusion_matrices(variant=i)
        precision_vector, f1_vector, recall_vector, accuracy= generate_metrics_row(general_matrix)
        for emotion in micro_matrices.keys():
            for row in micro_matrices[emotion]:
                print(row)
            print()

        for row in general_matrix:
            print(row)
        print()
        print("Accuracy:", accuracy)
        print("Precision:", precision_vector) 
        print("F1:", f1_vector) 
        print("Recall:", recall_vector)
        precision = sum(precision_vector) / 4
        f1 = sum(f1_vector) / 4
        recall = sum(recall_vector) / 4
        print(f'Precision {precision}')
        print(f'f1 {f1}')
        print(f'recall {recall}')
   
