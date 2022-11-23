# -*- encoding: utf-8 -*-
import torch.nn as nn


class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        # self.cnn1 = nn.Sequential(
        # nn.Conv2d(3, 32, kernel_size=7),
        # nn.Conv2d(32, 32, kernel_size=7),
        # nn.BatchNorm2d(32),
        # nn.ReLU(inplace=True),
        # nn.MaxPool2d(2, stride=2),
        # nn.Conv2d(32, 64, kernel_size=3),
        # nn.Conv2d(64, 64, kernel_size=3),

        # nn.BatchNorm2d(64),
        # nn.ReLU(inplace=True),
        # nn.MaxPool2d(2, stride=2),
        # nn.Conv2d(64, 128, kernel_size=3),
        # nn.Conv2d(128, 128, kernel_size=3),

        # nn.BatchNorm2d(128),
        # nn.ReLU(inplace=True),
        # nn.MaxPool2d(2, stride=2))

        # self.fc1 = nn.Sequential(
        # nn.Linear(128 * 23 * 23, 4096),
        # nn.ReLU(inplace=True),
        # nn.Linear(4096, 16))

        self.cnn1 = nn.Sequential(
            # 112, 112, 64
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 56, 56, 128
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 28, 28, 256
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 14, 14, 512
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 7, 7, 512
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.fc1 = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 16))

    def forward_once(self, x):
        output = self.cnn1(x)
        # print(output.size())
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2
