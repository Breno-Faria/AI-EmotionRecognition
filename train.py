from model import BrenoPeterandSydneysSuperCoolConvolutionalNeuralNetworkVeryAccurateAndGood
import os
import sys
import dataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torchvision.transforms as transforms

model_dir = "models"
temp_model_dir = model_dir + os.sep + "tmp"


def train_model(training_batch_size=100, kernel_size=4, learning_rate=0.001, model_name="model.pth"):
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

    train_data, validation_data, test_data = dataset.loadData("data.json")
    trainset = dataset.EmotionDataset(json_array = train_data, transform=transform)
    validationset = dataset.EmotionDataset(json_array = validation_data, transform=transform)
    testset = dataset.EmotionDataset(json_array = test_data, transform=transform)
    print("Loaded Dataset")

    num_epocs = 10

    train_loader = DataLoader(trainset, batch_size=training_batch_size, shuffle=True, num_workers=2)
    validation_loader = DataLoader(validationset, batch_size=25, shuffle=False, num_workers=2)
    test_loader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)


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
            model_filename = temp_model_dir + os.sep + f'epoc_{epoc+1}.pth'
            torch.save(model.state_dict(), model_filename)
            accuracy_per_epoc[epoc] = accuracy
            print(f'Epoch [{epoc+1}/{num_epocs}], Validation Accuracy: {round((correct/total)*100, 2)}')

        model.train()
    
    max_epoc = 0
    for epoc in accuracy_per_epoc.keys():
        if accuracy_per_epoc[epoc] > accuracy_per_epoc[max_epoc]:
            max_epoc = epoc

    model = BrenoPeterandSydneysSuperCoolConvolutionalNeuralNetworkVeryAccurateAndGood()
    model.load_state_dict(torch.load(temp_model_dir + os.sep + f'epoc_{max_epoc}.pth'))
    print(f"Best model is from epoc {max_epoc+1} with accuracy {accuracy_per_epoc[max_epoc]}")

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
    return model

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1].endswith('.pth'):
            train_model(model_name=sys.argv[1])
        else:
            print("Invalid file type, must end with .pth.")
    else:
        print("No filename specified, saving to models/model.pth.")
        train_model()
    #train_model(kernel_size=3)
    #train_model(kernel_size=5)
    #train_model(kernel_size=6)
    #train_model(kernel_size=7)

    #train_model(training_batch_size=25)
    #train_model(training_batch_size=50)
    #train_model(training_batch_size=100)
    #train_model(training_batch_size=200)

    #train_model(learning_rate=0.0005)
    #train_model(learning_rate=0.0001)
    #train_model(learning_rate=0.0002)



