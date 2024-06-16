from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets

class BrenoPeterandSydneysSuperCoolConvolutionalNeuralNetworkVeryAccurateAndGood(nn.Module):
    def __init__(self, kernel_size=4):
        super(BrenoPeterandSydneysSuperCoolConvolutionalNeuralNetworkVeryAccurateAndGood, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, stride=1, kernel_size=kernel_size),
            nn.LeakyReLU(),
            nn.BatchNorm2d(32),

            nn.Conv2d(in_channels=32, out_channels=32, stride=1, kernel_size=kernel_size),
            nn.LeakyReLU(),
            nn.BatchNorm2d(32),

            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=32, out_channels=64, stride=1, kernel_size=kernel_size),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64),

            nn.Conv2d(in_channels=64, out_channels=64, stride=1, kernel_size=kernel_size),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64),
            
            nn.MaxPool2d(kernel_size=2, stride=2),

        )

        width_map = {
            3: 9,
            4: 7,
            5: 6,
            6: 4,
            7: 3,
        }
        if kernel_size in width_map.keys():
            width = width_map[kernel_size]
        else:
            width = 7
        
        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(64*width*width, 1000),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(1000, 512),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(512, 4),
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)
        return x



class BrenoPeterandSydneysSuperCoolConvolutionalNeuralNetworkVeryAccurateAndGood_Variant1(nn.Module):
    def __init__(self, kernel_size=4):
        super(BrenoPeterandSydneysSuperCoolConvolutionalNeuralNetworkVeryAccurateAndGood_Variant1, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, stride=1, kernel_size=kernel_size),
            nn.LeakyReLU(),
            nn.BatchNorm2d(32),

            nn.Conv2d(in_channels=32, out_channels=32, stride=1, kernel_size=kernel_size),
            nn.LeakyReLU(),
            nn.BatchNorm2d(32),

            nn.MaxPool2d(kernel_size=2, stride=2),

            
            nn.MaxPool2d(kernel_size=2, stride=2),

        )

        width_map = {
            3: 9,
            4: 7,
            5: 6,
            6: 4,
            7: 3,
        }
        if kernel_size in width_map.keys():
            width = width_map[kernel_size]
        else:
            width = 7
        
        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(3200, 1000),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(1000, 512),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(512, 4),
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)
        return x
