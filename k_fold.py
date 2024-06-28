from multiprocessing import freeze_support
from train import train_model
from dataset import k_loadData
from evaluation import generate_confusion_matrices, generate_macro_metrics_row, generate_micro_metrics, generate_metrics_row
import sys

def train_models_k_fold(k=10):

    for i in range(k):
        name = f'model_{i}_kfold.pth'
        train_model(k_fold = True, iteration=i, fold_data=k_loadData("randomized_data.json", k=k), model_name=name)

def generate_stats_for_k_fold(k=10):

    overall_ma_p = 0
    overall_ma_r = 0
    overall_ma_f = 0
    overall_mi_p = 0
    overall_mi_r = 0
    overall_mi_f = 0
    overall_accuracy = 0
    for i in range(k):
        name = f'model_{i}_kfold.pth'
        _, conf_matrix, micro_conf_matrix = generate_confusion_matrices(name, kfold=True, iteration=i)

        precision_micro_vector = []
        recall_micro_vector = []
        f1_micro_vector = []
        #accuracy_micro_vector = []

        for emotion in micro_conf_matrix.keys():
            micro_matrix = micro_conf_matrix[emotion]
            print(micro_matrix[0])
            print(micro_matrix[1])
            print()
            precision, recall, f1, _= generate_micro_metrics(micro_matrix) 
            precision_micro_vector.append(precision)
            recall_micro_vector.append(recall)
            f1_micro_vector.append(f1)
            #accuracy_micro_vector.append(accuracy)

        mi_precision = sum(precision_micro_vector) / 4
        mi_f1 = sum(f1_micro_vector) / 4
        mi_recall = sum(recall_micro_vector) / 4
        #accuracy = sum(accuracy_micro_vector) / 4
        for row in conf_matrix:
            print(row)
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

        print(f'\n\n{name}:\nMicro Precision: {mi_precision}\nMicro Recall: {mi_recall}\nMicro F1: {mi_f1}\n\n' +
              f'Macro Precision: {macro_averaged_precision}\nMacro Recall: {macro_averaged_recall}\nMacro F1: {macro_averaged_f1_measure}\n\n' +
              f'Overall Accuracy: {accuracy}\n\n')

    
    overall_ma_p /= k
    overall_ma_r /= k
    overall_ma_f /= k
    overall_mi_p /= k
    overall_mi_r /= k
    overall_mi_f /= k
    overall_accuracy /= k

    print(f'\n\nOverall statistics (averaged):\nMacro Precision: {overall_ma_p}\nMacro Recall: {overall_ma_r}\nMacro F1: {overall_ma_f}\n'+
          f'Micro Precision: {overall_mi_p}\nMicro Recall: {overall_mi_r}\nMicro F1: {overall_mi_f}\nOverall Accuracy (averaged): {overall_accuracy}\n\n')



if __name__ == "__main__":

    freeze_support()
    
    if len(sys.argv) == 1:
        print("Usage:\tpython3 k_fold.py [-t | -d] [-k <number>]")
        print("\t-t\tTrain models using k-fold cross-validation")
        print("\t-d\tDisplay statistics for previously trained models")
        print("\t-k <number>\t(Optional) Specify the number of folds (default is 10)")
    elif len(sys.argv) == 3:
        if sys.argv[1] == "-t":
            train_models_k_fold(int(sys.argv[2]))
        elif sys.argv[1] == "-d":
            generate_stats_for_k_fold(int(sys.argv[2]))
    elif len(sys.argv) == 2:
        if sys.argv[1] == "-t":
            train_models_k_fold()
        elif sys.argv[1] == "-d":
            generate_stats_for_k_fold()
    else:
        print("Usage:\tpython3 k_fold.py [-t | -d] [-k <number>]")
        print("\t-t\tTrain models using k-fold cross-validation")
        print("\t-d\tDisplay statistics for previously trained models")
        print("\t-k <number>\t(Optional) Specify the number of folds (default is 10)")


    

