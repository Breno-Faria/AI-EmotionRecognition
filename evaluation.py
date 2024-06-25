from multiprocessing import freeze_support
import sys
import numpy as np
import torch
from model import BrenoPeterandSydneysSuperCoolConvolutionalNeuralNetworkVeryAccurateAndGood, BrenoPeterandSydneysSuperCoolConvolutionalNeuralNetworkVeryAccurateAndGood_Variant1
from PIL import Image
import torchvision.transforms as transforms
from dataset import loadData
import base64
import io


def generate_micro_metrics(micro_matrix):
    TP = micro_matrix[0][0]
    TN = micro_matrix[1][1]
    FN = micro_matrix[1][0]
    FP = micro_matrix[0][1]

    precision = (TP) / (TP + FP)
    recall = TP / (TP + FN)
    f1 = (2*precision*recall)/(precision+recall)
    accuracy = (TP + TN) / (TP + TN + FN + FP)
    
    return precision, recall, f1, accuracy


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
    # v3->main
    # main->v3
    # in models: changed variant with non-variant
    if variant == 0:
        model = BrenoPeterandSydneysSuperCoolConvolutionalNeuralNetworkVeryAccurateAndGood(kernel_size=7)
        model.load_state_dict(torch.load('models/model.pth'))

    if variant == 1:
        model = BrenoPeterandSydneysSuperCoolConvolutionalNeuralNetworkVeryAccurateAndGood(kernel_size=4)
        model.load_state_dict(torch.load('models/model_variant1.pth'))

    if variant == 2:
        model = BrenoPeterandSydneysSuperCoolConvolutionalNeuralNetworkVeryAccurateAndGood_Variant1(kernel_size=7)
        model.load_state_dict(torch.load('models/model_variant2.pth'))

    if variant == 3:
        model = BrenoPeterandSydneysSuperCoolConvolutionalNeuralNetworkVeryAccurateAndGood_Variant1(kernel_size=4)
        model.load_state_dict(torch.load('models/model_variant3.pth'))


    _, _, testing_data_arr = loadData("./randomized_data.json")
    #predict_emotion(model=model, 

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


def display_micro_stats(variant=0):
    print('\n')
    print(f"Variant {variant}")
    _, micro_matrices = generate_confusion_matrices(variant=variant)
    precision_micro_vector = []
    recall_micro_vector = []
    f1_micro_vector = []
    accuracy_micro_vector = []
    for emotion in micro_matrices.keys():
        micro_matrix = micro_matrices[emotion]
        precision, recall, f1, accuracy = generate_micro_metrics(micro_matrix) 
        precision_micro_vector.append(precision)
        recall_micro_vector.append(recall)
        f1_micro_vector.append(f1)
        accuracy_micro_vector.append(accuracy)
        # print("Emotion:", emotions[emotion])
        # for row in micro_matrices[emotion]:
        #     print(row)
        # print("Precision:", precision)
        # print("Recall:", recall)
        # print("F1:", f1)
        # print("Accuracy:", accuracy)
        # print()
    precision = sum(precision_micro_vector) / 4
    f1 = sum(f1_micro_vector) / 4
    recall = sum(recall_micro_vector) / 4
    accuracy = sum(accuracy_micro_vector) / 4
    print("Accuracy:", accuracy)
    print("Precision:", precision) 
    print("F1:", f1) 
    print("Recall:", accuracy)

def display_macro_stats(variant=0):
    print('\n')
    print(f"Variant {variant}")
    general_matrix, _ = generate_confusion_matrices(variant=variant)
    precision_vector, f1_vector, recall_vector, accuracy= generate_metrics_row(general_matrix)
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


if __name__ == "__main__":
    # emotions = {
    #     0: "happy",
    #     1: "angry",
    #     2: "engaged",
    #     3: "neutral"
    # }
    #
    # if len(sys.argv) == 3:
    #     try:
    #         variant_choice = int(sys.argv[2])
    #     except:
    #         print("Unknown argument for variant.")
    #     else:
    #         if sys.argv[1] == '-i':
    #             display_micro_stats(variant_choice)
    #         elif sys.argv[1] == '-a':
    #             display_macro_stats(variant_choice)
    # elif len(sys.argv) == 2:
    #     if sys.argv[1] == '-i':
    #         for i in range(4):
    #             display_micro_stats(i)
    #     elif sys.argv[1] == '-a':
    #         for i in range(4):
    #             display_macro_stats(i)
    # else:
    #     print("evaluation.py")
    #     print("Usage:\t\tevaluation.py <micro/macro> <variant_number>")
    #     print("micro/macro:\t-i: micro statistics, -a: macro statistics")
    #     print("Variant number (optional): 0 (default, main model), 1, 2, 3")

    model = BrenoPeterandSydneysSuperCoolConvolutionalNeuralNetworkVeryAccurateAndGood_Variant1(kernel_size=7)
    model.load_state_dict(torch.load('models/model_variant2.pth'))
    predict_emotion(model,"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") 
