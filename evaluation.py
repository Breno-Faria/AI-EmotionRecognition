#accuracy = (confusion_matrix[0][0] + confusion_matrix[1][1]) / 2

import numpy as np

def generate_confusion_matrix(model, labeled_data):

    confusion_matrix = [
        [0, 0],
        [0, 0]
    ]

    for data in labeled_data:
        actual_rating = data[1]
        predicted_rating = model.predict([data[0]])
        
        if actual_rating == predicted_rating:
            if actual_rating == 1:
                data[0][0] +=1 # true positive
            else:
                data[1][1] +=1 # true negative
        else:
            if actual_rating == 1:
                data[0][1] +=1 # false negative
            else:
                data [1][0] +=1 # false positive

    return confusion_matrix

def generate_micro_metrics_row(confusion_matrix_array):

    confusion_matrix_sum = np.sum(confusion_matrix_array, axis=0)
    return generate_metrics_row(confusion_matrix_sum)

def generate_macro_metrics_row(confusion_matrix_array):

    metrics_results = np.arary([0.0, 0.0, 0.0])

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