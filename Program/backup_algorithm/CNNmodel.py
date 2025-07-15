import torch
import torch.nn as nn
from collections import OrderedDict

class BinaryCNN(nn.Module):
    def __init__(self):
        super(BinaryCNN, self).__init__()

        self.cnn = nn.Sequential(OrderedDict([
            ('conv_block_1', _ConvBlock(1, 4)),  
            ('max_pool_1', nn.MaxPool2d(2, 2)),

            ('conv_block_2', _ConvBlock(4, 8,  bn=True)),  
            ('max_pool_2', nn.MaxPool2d(2, 2)),
            ('conv_block_3', _ConvBlock(8, 16, stride = 2, bn=True)),  
            ('conv_block_4', _ConvBlock(16, 32, stride = 4, bn=True)),  
            
        ]))
        
        # Fully connected layers
        self.fc1 = nn.Linear(32 * 32 * 32, 256)  # Adjust input size based on image size
        self.fc2 = nn.Linear(256, 128) 
        self.fc3 = nn.Linear(128, 2)  # Output two parameters
        self.dropout = nn.Dropout(p = 0.1)  # Dropout with probability 0.1
        
    def forward(self, contact_image):
        # Process input images through CNN
        batch_size,  height, width = contact_image.size()
        #x = contact_image.view(-1, height, width)
        x = contact_image
        # Add channel dimension: (B, 1, H, W)
        x = x.unsqueeze(1)
        x = self.cnn(x)
        x = x.view(batch_size, -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
class _ConvBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bn=False):
        super(_ConvBlock, self).__init__()
        self.add_module('conv', nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))
        if bn:
            self.add_module('norm', nn.BatchNorm2d(out_channels))
        self.add_module('relu', nn.LeakyReLU(inplace=True))
        
if __name__ == "__main__":

    model = BinaryCNN()
    random_matrix = torch.rand(3,1,1024,1024)
    y = model.forward(random_matrix)


