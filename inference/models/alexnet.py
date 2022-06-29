"""
This file contains three variants of our Alexnet class,
each with slightly modified architectures or weight initialization.
The current best model is trained using the class: 'PretrainedAlexnet'.

Alexnet:
    - Conventional Alexnet architecture with reduced no. of channels
      and fc-layers
myAlexNet:
    - Modified Alexnet architecture with added BatchNorm layers
PretrainedAlexnet:
    - Exact copy of Alexnet architecture with reduced fc layer
      (removed dropout layer proven to have better performance)
    - First two layers loaded with Imagenet pretraining weights


"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import alexnet


class AlexNet(nn.Module):
    def __init__(self, input_channels=3, dropout=False, channel_size=16, n_cls=5):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, channel_size, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(channel_size, 2*channel_size, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(2*channel_size, 4*channel_size, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(4*channel_size, 8*channel_size, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8*channel_size, 16*channel_size, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=.2),
            nn.Linear(16*channel_size * 6 * 6, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, n_cls),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class myAlexNet(nn.Module):
    def __init__(self, input_channels=4, dropout=False, prob=0.0, channel_size=16):
        super(myAlexNet, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, channel_size, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(channel_size)
        self.conv2 = nn.Conv2d(channel_size, 2*channel_size, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(2*channel_size)
        self.conv3 = nn.Conv2d(2*channel_size, 4*channel_size, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(4*channel_size)
        self.conv4 = nn.Conv2d(4*channel_size, 8*channel_size, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(8*channel_size)
        self.conv5 = nn.Conv2d(8*channel_size, 16*channel_size, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(16*channel_size)
        self.fc1 = nn.Linear(16*channel_size, 5)
        #self.fc2 = nn.Linear(64, 5)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight, gain=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)
        
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)

        x = self.conv5(x)
        x = self.bn5(x)
        x = F.relu(x)

        x=F.adaptive_avg_pool2d(x,1)
        x = x.reshape(x.size(0), -1)

        x = self.fc1(x)
        #x = F.relu(x)
        #x = self.fc2(x)
        return x


class PretrainedAlexnet(nn.Module):
    def __init__(self, n_cls=5):
        super(PretrainedAlexnet, self).__init__()
        pretrained_alexnet = alexnet(pretrained=True)
        self.features = pretrained_alexnet.features
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Linear(256 * 6 * 6, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, n_cls)
        )

        for i, param in enumerate(self.features.parameters()):
            if i < 4:
                param.requires_grad = False

        for i, m in enumerate(self.features.modules()):
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)) and i > 4:
                nn.init.xavier_uniform_(m.weight, gain=1)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    # Unfreeze pretrained layers (1st & 2nd CNN layer)
    def unfreeze_backbone(self):
        for param in self.features.parameters():
            param.requires_grad = True
