# GRU_Net
import torch
torch.set_printoptions(profile="full")
torch.set_printoptions(profile="default") # reset
import torch.nn.functional as F
import torch.nn as nn
from collections import OrderedDict



class GRUNet(nn.Module):
    def __init__(self, **kwargs):
        super(GRUNet, self).__init__(**kwargs)
        self.cnn = nn.Sequential(OrderedDict([
            ('conv_block_1', _ConvBlock(4, 32)),  
            ('max_pool_1', nn.MaxPool2d(2, 2)),

            ('conv_block_2', _ConvBlock(32, 64, stride = 2, bn=True)),  
            ('max_pool_2', nn.MaxPool2d(2, 2)),
            
            ('conv_block_4_1', _ConvBlock(64, 64, stride = 4, bn=True)),  
            
        ]))

        # GRU layer to model temporal dynamics across image sequences
        self.gru = nn.GRU(input_size = 4096, hidden_size = 1024, 
                        batch_first=True)
        self.rnn = nn.RNN(input_size = 4096, hidden_size = 1024, num_layers = 2, batch_first=True)
        self.lstm = nn.LSTM(input_size = 4096, hidden_size = 1024, batch_first=True)

        # Fully connected layers for encoding solar geometry and component fractions (6D → 4096D)
        self.fc_layer1 = nn.Linear(6, 64)
        self.fc_layer2 = nn.Linear(64, 256)
        self.fc_layer3 = nn.Linear(256, 4096)
        # Activation function after each FC layer
        self.relu = nn.LeakyReLU() 
        self.transcript = nn.Linear(1024, 2) 
        # Final regression layer to predict LAI and MTA
        self.fc = nn.Linear(1024, 2)
    def forward(self, contact_image, percent_four):
        contact_image = contact_image.to(torch.float32)
        # CNN: Process multiple images in batch mode
        batch_size, num_images, channels, height, width = contact_image.size()
        x = contact_image.view(-1, channels, height, width)  
        x = self.cnn(x)
        # Restore time sequence shape for RNN
        x = x.view(batch_size, num_images, -1)
        # MLP: Process solar angles and four-component fractions
        y = self.relu(self.fc_layer1(percent_four))
        y = self.relu(self.fc_layer2(y))
        y = self.fc_layer3(y)
        # Combine CNN and MLP features
        x = x + y
        # h0 = torch.zeros(2, batch_size, 4096).to(x.device)
        out, _ = self.gru(x)
        # Final prediction from last time step
        out = self.fc(out[:, -1, :])
        return out

class _ConvBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bn=False):
        super(_ConvBlock, self).__init__()
        self.add_module('conv', nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))
        if bn:
            self.add_module('norm', nn.BatchNorm2d(out_channels))
        self.add_module('relu', nn.LeakyReLU(inplace=True))












