import torch
import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F
# Define the model with multiple convolutional layers
class SimpleConvNet(nn.Module):
    def __init__(self):
        super(SimpleConvNet, self).__init__()
        self.conv1 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        
        # Zero-initialize the weights
        nn.init.zeros_(self.conv1.weight)
        nn.init.zeros_(self.conv1.bias)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        return x

# Initialize the model, loss function, and optimizer
model = SimpleConvNet()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Example input and ground truth
input_tensor = torch.randn(1, 64, 128, 128)
ground_truth = torch.randn(1, 16, 128, 128)

# Set some channels of the ground truth to zero
ground_truth[:, [0, 3, 5], :, :] = 0

# Training loop
num_epochs = 200
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(input_tensor)
    loss = criterion(outputs, ground_truth)
    loss.backward()
    optimizer.step()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')

# Print final output for inspection
print(outputs)