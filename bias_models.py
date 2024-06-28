from multiprocessing import freeze_support
from train import train_model
from evaluation import generate_confusion_matrices, generate_micro_metrics, generate_metrics_row
import sys
import os

def train_biased_models():

    for category in os.listdir('./bias_data/json/'):
        if category != ".DS_Store":
            name = f'biased_model_{category.lower()}.pth'
            source_file = f'./bias_data/json/{category}/randomized_data.json'
            print(f'training model: {name}')
            train_model(model_name=name, biased=True, biased_file=source_file, training_batch_size=10)


def generate_sectioned_stats():
    for i in range(5):
        _, conf_matrix, _= generate_confusion_matrices(data_dir='./randomized_data_2.json',sectioned=True, section=i)
        _, _, _, accuracy= generate_metrics_row(conf_matrix)
        print(f'Accuracy for section {i}: {accuracy}')


def generate_biased_stats(name="model.pth", old_dataset=False):

    overall_ma_p = 0
    overall_ma_r = 0
    overall_ma_f = 0
    overall_mi_p = 0
    overall_mi_r = 0
    overall_mi_f = 0
    overall_accuracy = 0

    overall_ma_p_age = 0
    overall_ma_r_age = 0
    overall_ma_f_age = 0
    overall_mi_p_age = 0
    overall_mi_r_age = 0
    overall_mi_f_age = 0
    overall_accuracy_age = 0

    overall_ma_p_gender = 0
    overall_ma_r_gender = 0
    overall_ma_f_gender = 0
    overall_mi_p_gender = 0
    overall_mi_r_gender = 0
    overall_mi_f_gender = 0
    overall_accuracy_gender = 0

    accuracy_map = {}
    categories_array = ["female", "male", "other", "middle_aged", "senior", "young"]
    for i, category in enumerate(categories_array):
        if old_dataset and category == "young":
            biased_dir = f'bias_data/json/{category}/old_encoded_images.json'
        else:
            biased_dir = f'bias_data/json/{category}/encoded_images.json'
        n, conf_matrix, micro_conf_matrix = generate_confusion_matrices(model_name=name, data_dir=biased_dir, biased=True)

        precision_micro_vector = []
        recall_micro_vector = []
        f1_micro_vector = []
        accuracy_micro_vector = []

        for emotion in micro_conf_matrix.keys():
            micro_matrix = micro_conf_matrix[emotion]
            precision, recall, f1, accuracy = generate_micro_metrics(micro_matrix) 
            precision_micro_vector.append(precision)
            recall_micro_vector.append(recall)
            f1_micro_vector.append(f1)
            accuracy_micro_vector.append(accuracy)

        mi_precision = sum(precision_micro_vector) / 4
        mi_f1 = sum(f1_micro_vector) / 4
        mi_recall = sum(recall_micro_vector) / 4
        accuracy = sum(accuracy_micro_vector) / 4

        precision_vector, f1_vector, recall_vector, accuracy= generate_metrics_row(conf_matrix)
        macro_averaged_precision = sum(precision_vector) / 4
        macro_averaged_f1_measure = sum(f1_vector) / 4
        macro_averaged_recall = sum(recall_vector) / 4

        overall_mi_p += mi_precision
        overall_mi_r += mi_recall
        overall_mi_f += mi_f1
        overall_ma_p += macro_averaged_precision
        overall_ma_r += macro_averaged_recall
        overall_ma_f += macro_averaged_f1_measure
        overall_accuracy += accuracy

        if i >= 3:
            overall_mi_p_age += mi_precision
            overall_mi_r_age += mi_recall
            overall_mi_f_age += mi_f1
            overall_ma_p_age += macro_averaged_precision
            overall_ma_r_age += macro_averaged_recall
            overall_ma_f_age += macro_averaged_f1_measure
            overall_accuracy_age += accuracy
        else:
            overall_mi_p_gender += mi_precision
            overall_mi_r_gender += mi_recall
            overall_mi_f_gender += mi_f1
            overall_ma_p_gender += macro_averaged_precision
            overall_ma_r_gender += macro_averaged_recall
            overall_ma_f_gender += macro_averaged_f1_measure
            overall_accuracy_gender += accuracy

        print(f'\n\n{category}:\nn={n}\nMicro Precision: {mi_precision}\nMicro Recall: {mi_recall}\nMicro F1: {mi_f1}\n\n' +
              f'Macro Precision: {macro_averaged_precision}\nMacro Recall: {macro_averaged_recall}\nMacro F1: {macro_averaged_f1_measure}\n\n' +
              f'Overall Accuracy: {accuracy}\n\n')
        accuracy_map[category] = (accuracy, n)

    
    overall_ma_p /= 6
    overall_ma_r /= 6
    overall_ma_f /= 6
    overall_mi_p /= 6
    overall_mi_r /= 6
    overall_mi_f /= 6
    overall_accuracy /= 6

    overall_ma_p_age /= 3
    overall_ma_r_age /= 3
    overall_ma_f_age /= 3
    overall_mi_p_age /= 3
    overall_mi_r_age /= 3
    overall_mi_f_age /= 3
    overall_accuracy_age /= 3

    overall_ma_p_gender /= 3
    overall_ma_r_gender /= 3
    overall_ma_f_gender /= 3
    overall_mi_p_gender /= 3
    overall_mi_r_gender /= 3
    overall_mi_f_gender /= 3
    overall_accuracy_gender /= 3

    print(f'\n\nOverall age-biased statistics (averaged):\nMacro Precision: {overall_ma_p_age}\nMacro Recall: {overall_ma_r_age}\nMacro F1: {overall_ma_f_age}\n'+
          f'Micro Precision: {overall_mi_p_age}\nMicro Recall: {overall_mi_r_age}\nMicro F1: {overall_mi_f_age}\nOverall Accuracy (averaged): {overall_accuracy_age}\n\n')
    
    print(f'\n\nOverall gender-biased statistics (averaged):\nMacro Precision: {overall_ma_p_gender}\nMacro Recall: {overall_ma_r_gender}\nMacro F1: {overall_ma_f_gender}\n'+
          f'Micro Precision: {overall_mi_p_gender}\nMicro Recall: {overall_mi_r_gender}\nMicro F1: {overall_mi_f_gender}\nOverall Accuracy (averaged): {overall_accuracy_gender}\n\n')

    print(f'\n\nOverall statistics (averaged):\nMacro Precision: {overall_ma_p}\nMacro Recall: {overall_ma_r}\nMacro F1: {overall_ma_f}\n'+
          f'Micro Precision: {overall_mi_p}\nMicro Recall: {overall_mi_r}\nMicro F1: {overall_mi_f}\nOverall Accuracy (averaged): {overall_accuracy}\n\n')

    for emotion in accuracy_map.keys():
        em_accuracy, n = accuracy_map[emotion]
        print(f'{emotion}: {round(100*em_accuracy, 3)}%, n={n}')



if __name__ == "__main__":

    freeze_support()
   
    if len(sys.argv) == 1:
        print("Please enter a model name (python3 bias_models.py [modelname.pth] [options]")
    else:
        if "-o" in sys.argv:
            generate_biased_stats(sys.argv[1], old_dataset=True)
        else:
            generate_biased_stats(sys.argv[1])
    # if len(sys.argv) == 1:
    #     print("Usage:\tpython3 k_fold.py [-t | -d]")
    #     print("\t-t\tTrain biased models, according to different categories")
    #     print("\t-d\tDisplay statistics for previously trained models")
    # elif len(sys.argv) == 3:
    #     if sys.argv[1] == "-t":
    #         train_biased_models()
    #     elif sys.argv[1] == "-d":
    #         generate_biased_stats()
    # elif len(sys.argv) == 2:
    #     if sys.argv[1] == "-t":
    #         train_biased_models()
    #     elif sys.argv[1] == "-d":
    #         generate_biased_stats()
    #     elif sys.argv[1] == "-s":
    #         generate_sectioned_stats()
    # else:
    #     print("Usage:\tpython3 k_fold.py [-t | -d]")
    #     print("\t-t\tTrain biased models, according to different categories")
    #     print("\t-d\tDisplay statistics for previously trained models")
