import torch
from model import BrenoPeterandSydneysSuperCoolConvolutionalNeuralNetworkVeryAccurateAndGood

# Initialize the model
model = BrenoPeterandSydneysSuperCoolConvolutionalNeuralNetworkVeryAccurateAndGood()

# Create a dummy input tensor with the same dimensions as the expected input (3 channels, 32x32 image)
dummy_input = torch.randn(1, 3, 32, 32)

# Forward pass through the convolutional layers to inspect the output shape
with torch.no_grad():
    conv_output = model.conv_layer(dummy_input)

print(conv_output.shape)