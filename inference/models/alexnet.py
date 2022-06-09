"""This file contains the architeture of a Gr-Convnet variant specified for
classification of 30 classes. 

The main backbone is unchanged and only a linear layer is concatenated to the
end to prevent any changes to the expressivity of the original architecture."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AlexNet(nn.Module):
    def __init__(self, input_channels=4, dropout=False, prob=0.0, channel_size=16):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, channel_size, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(channel_size+input_channels, 2*channel_size, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(3*channel_size+input_channels, 4*channel_size, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(7*channel_size+input_channels, 8*channel_size, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(15*channel_size+input_channels, 16*channel_size, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(16*channel_size, 5)

    def forward(self, x):
        identity=F.avg_pool2d(x,4,4)
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = torch.cat((x, identity), dim=1)

        identity=F.avg_pool2d(x,2,2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = torch.cat((x, identity), dim=1)

        identity=F.avg_pool2d(x,2,2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2, 2)
        x = torch.cat((x, identity), dim=1)
        
        identity=F.avg_pool2d(x,2,2)
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2, 2)
        x = torch.cat((x, identity), dim=1)

        x = F.relu(self.conv5(x))

        x=F.adaptive_avg_pool2d(x,1)
        x = x.reshape(x.size(0), -1)

        x = self.fc1(x)
        return x
