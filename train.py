from model import BrenoPeterandSydneysSuperCoolConvolutionalNeuralNetworkVeryAccurateAndGood
import dataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets



def train_model():

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
    train_fraction = 1400
    validation_fraction = 300
    test_fraction = 300
    batch_size = int(train_fraction / num_epocs)
    assert(train_fraction + validation_fraction + test_fraction == 2000)

    train_loader = DataLoader(trainset, batch_size=100, shuffle=True, num_workers=2)
    validation_loader = DataLoader(validationset, batch_size=25, shuffle=False, num_workers=2)
    test_loader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)


    learning_rate = 0.001
    model = BrenoPeterandSydneysSuperCoolConvolutionalNeuralNetworkVeryAccurateAndGood()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)





    for epoc in range(num_epocs):
        loss_list = []
        accuracy_list = []
        batch_loss = 0
        total_batches = 0
        for i, (images, labels) in enumerate(train_loader):
            # print(f"Batch {i}:")
            # print(f"  Images shape: {images.shape}")
            # print(f"  Labels shape: {labels.shape}")
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss_list.append(loss.item())
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
            print('accuracy on validation set: {} %'
              .format((correct / total) * 100))
        model.train()

        # for i, (images, labels) in enumerate(validation_loader):
        #     outputs = model(images)
        #     loss = criterion(outputs, labels)
        #     loss_list.append(loss.item())
        #     
        #     total = labels.size(0)
        #     _, predicted = torch.max(outputs.data, 1)
        #     correct = (predicted == labels).sum().item()
        #     accuracy_list.append(correct / total)
        #
        #print(f'Epoch: [{epoc+1}/{num_epocs}], Loss: {loss.item()}, Validation Accuracy: {correct / total}')

    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print('accuracy on training set: {} %'
          .format((correct / total) * 100))
    return model

if __name__ == "__main__":
    train_model()
