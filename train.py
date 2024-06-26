from model import BrenoPeterandSydneysSuperCoolConvolutionalNeuralNetworkVeryAccurateAndGood, BrenoPeterandSydneysSuperCoolConvolutionalNeuralNetworkVeryAccurateAndGood_Variant1
import os
import sys
import dataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from multiprocessing import freeze_support


model_dir = "models"
temp_model_dir = model_dir + os.sep + "tmp"


def train_model(training_batch_size=100, kernel_size=7, learning_rate=0.001, model_name="model.pth", variant=False, k_fold = False, iteration=0, fold_data=[], biased=False, biased_file=""):
    if variant:
        print('Generating variant model')
    for model_file in os.listdir(temp_model_dir + os.sep):
        os.remove(temp_model_dir + os.sep + model_file)

    print("HYPERPARAMETERS:")
    print('Training batch size:', training_batch_size)
    print('Kernel size:', kernel_size)
    print('Learning rate:', learning_rate)

    transform = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    if k_fold:
        idx = len(fold_data[iteration][0])//10
        train_data, validation_data, test_data = fold_data[iteration][0][idx:], fold_data[iteration][0][:idx], fold_data[iteration][1]
    elif biased:
        train_data, validation_data, test_data = dataset.loadData(biased_file, biased=True)
    else:
        train_data, validation_data, test_data = dataset.loadData("randomized_data.json")
        
    
    trainset = dataset.EmotionDataset(json_array = train_data, transform=transform)
    validationset = dataset.EmotionDataset(json_array = validation_data, transform=transform)
    testset = dataset.EmotionDataset(json_array = test_data, transform=transform)
    num_epocs = 10

    train_loader = DataLoader(trainset, batch_size=training_batch_size, shuffle=True, num_workers=2)
    validation_loader = DataLoader(validationset, batch_size=25, shuffle=False, num_workers=2)
    test_loader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    if variant == True:
        model = BrenoPeterandSydneysSuperCoolConvolutionalNeuralNetworkVeryAccurateAndGood_Variant1(kernel_size=kernel_size)
    else:
        model = BrenoPeterandSydneysSuperCoolConvolutionalNeuralNetworkVeryAccurateAndGood(kernel_size=kernel_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)



    accuracy_per_epoc = {}

    for epoc in range(num_epocs):
        for images, labels in train_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in validation_loader:
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            accuracy = (correct / total) * 100
            model_filename = temp_model_dir + os.sep + f'epoc_{epoc}.pth'
            torch.save(model.state_dict(), model_filename)
            accuracy_per_epoc[epoc] = accuracy
            print(f'Epoch [{epoc+1}/{num_epocs}], Validation Accuracy: {round((correct/total)*100, 2)}')

        model.train()
    
    max_epoc = 0
    for epoc in accuracy_per_epoc.keys():
        if accuracy_per_epoc[epoc] > accuracy_per_epoc[max_epoc]:
            max_epoc = epoc

    print(accuracy_per_epoc)
    if variant == False:
        model = BrenoPeterandSydneysSuperCoolConvolutionalNeuralNetworkVeryAccurateAndGood(kernel_size=kernel_size)
    else:
        print("getting variant")
        model = BrenoPeterandSydneysSuperCoolConvolutionalNeuralNetworkVeryAccurateAndGood_Variant1(kernel_size=kernel_size)

    print(f"Best model is from epoc {max_epoc+1} with accuracy {accuracy_per_epoc[max_epoc]}")
    model.load_state_dict(torch.load(temp_model_dir + os.sep + f'epoc_{max_epoc}.pth'))

    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in validation_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        accuracy = (correct / total) * 100
        print(f'Final Validation Accuracy: {round((correct/total)*100, 2)}')
        
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print(f"Final Training Accuracy: {(correct/total)*100}")

    model_filename = model_dir + os.sep + model_name
    torch.save(model.state_dict(), model_filename)
    for model_file in os.listdir(temp_model_dir + os.sep):
        os.remove(temp_model_dir + os.sep + model_file)
    return model

if __name__ == "__main__":
    freeze_support()
    filename = ""
    kernel_size = 7
    variant = False
    if len(sys.argv) == 1:
        print("train.py")
        print("Usage: train.py <model_filename> <variant?> <kernel_size>")
        print("Filename: must end in .pth, i.e. model.pth")
        print("Variant: -v for variant or nothing for original")
        print("Kernal size: 4 or 7, default 7")
        print("Example: python3 train.py model.pth -v 4")

    if sys.argv[1].endswith('.pth'):
        filename = sys.argv[1]
    else:
        raise ValueError("Invalid filename for model.")

    if len(sys.argv) == 3:
        if sys.argv[2] == '-v':
            variant = True
        else:
            if sys.argv[2] == '4':
                kernel_size = 4
            elif sys.argv[2] == '7':
                kernel_size = 7
            else:
                raise ValueError("Unknown argument:", sys.argv[2])

    if len(sys.argv) == 4:
        if sys.argv[2] == '-v':
            variant = True
        else:
            raise ValueError("Unknown argument, should be -v:", sys.argv[2])

        if sys.argv[3] == '4':
            kernel_size = 4
        elif sys.argv[3] == '7':
            kernel_size = 7
        else:
            raise ValueError("Unknown argument:", sys.argv[3])
                
    if len(sys.argv) > 4:
        raise ValueError("Too many arguments.")

    assert(filename != "")
    assert(kernel_size == 4 or kernel_size == 7)
    print("training model.")
    print("filename:", filename)
    print("kernel size:", kernel_size)
    print("variant:", variant)
    train_model(model_name=filename, kernel_size=kernel_size, variant=variant)

