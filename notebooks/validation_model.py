import torch
import torch.nn as nn
import torch.nn.functional as F


class ValidationModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        # cnn
        self.conv1 = nn.Conv2d(3, 8, 2)     # 7x7
        self.conv2 = nn.Conv2d(8, 16, 3)    # 4x4
        self.mp = nn.MaxPool2d(2, 2)        # 2x2

        self.ln1 = nn.Linear(64, 32)
        self.bn1 = nn.BatchNorm1d(32)
        self.ln2 = nn.Linear(32, 8)

        # tabular
        self.ln3 = nn.Linear(3, 8)
        self.ln4 = nn.BatchNorm1d(8)

        # concat
        self.bn4 = nn.BatchNorm1d(16)
        self.ln5 = nn.Linear(16, 1)

    def forward(self, img, tab):
        # cnn
        img = F.relu(self.conv1(img))
        img = F.relu(self.conv2(img))
        img = self.mp(img)

        img = img.reshape(img.shape[0], -1)

        img = F.relu(self.ln1(img))
        img = self.bn1(img)
        img = F.tanh(self.ln2(img))

        # tabular
        tab = F.relu(self.ln3(tab))
        tab = F.tanh(tab)

        #   for training [more than 1 input]
        if tab.ndim > 2:
            tab = tab.squeeze()

        # concat
        result = torch.cat((img, tab), dim=1)
        result = self.bn4(result)
        result = F.sigmoid(self.ln5(result))

        return result
