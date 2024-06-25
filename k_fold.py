from multiprocessing import freeze_support
import evaluation as eval
import model
from train import train_model
import torch
from dataset import k_loadData
import sys
from randomize_dataset import read_and_randomize

if __name__ == "__main__":

    freeze_support()
    k = 10
    for i in range(k):
        name = f'model_{i}_kfold'
        train_model(k_fold = True, iteration=i, fold_data=k_loadData("data.json"), model_name=name)

