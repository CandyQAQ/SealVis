import torch
import torch.nn as nn


class VggNet(nn.Module):
    def __init__(self, num_classes=1148):
        super(VggNet, self).__init__()
        self.Conv = torch.nn.Sequential(
            # 3*224*224  conv1
            torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            # 64*112*112   conv2
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            # 128*56*56    conv3
            torch.nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            # 256*28*28    conv4
            torch.nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))
        # 512*14*14   conv5
        # torch.nn.Conv2d(512, 512, kernel_size = 3, stride = 1, padding = 1),
        # torch.nn.ReLU(),
        # torch.nn.Conv2d(512, 512, kernel_size = 3, stride = 1, padding = 1),
        # torch.nn.ReLU(),
        # torch.nn.Conv2d(512, 512, kernel_size = 3, stride = 1, padding = 1),
        # torch.nn.ReLU(),
        # torch.nn.MaxPool2d(kernel_size = 2, stride = 2))
        # 512*7*7

        self.Classes = torch.nn.Sequential(
            torch.nn.Linear(14 * 14 * 512, 1060),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(1060, 1060),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(1060, num_classes))
        # torch.nn.Linear(1060, num_classes))

    def forward(self, inputs):
        x = self.Conv(inputs)
        x = x.view(-1, 14 * 14 * 512)
        x = self.Classes(x)
        return x


if __name__ == "__main__":
    model = VggNet(num_classes=1148)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # summary(model, (3, 224, 224))
