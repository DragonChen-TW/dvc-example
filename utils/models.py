from torch import nn
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.mp = nn.MaxPool2d(2)
        self.fc = nn.Linear(320, 10)

    def forward(self, x):
        in_size = x.size(0)

        out = self.mp(self.conv1(x))
        out = F.relu(out)
        out = self.mp(self.conv2(out))
        out = F.relu(out)
        out = out.view(in_size, -1)
        out = self.fc(out)

        return out

if __name__ == '__main__':
    import torch
    model = LeNet()
    x = torch.rand(64, 1, 28, 28)
    print(model(x).shape)