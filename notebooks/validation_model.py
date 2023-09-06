import torch
import torch.nn as nn
import torch.nn.functional as F


class ValidationModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        # CNN
        self.conv1 = nn.Conv2d(3, 16, (4, 8))
        self.mp1 = nn.MaxPool2d((2, 1))

        self.conv2 = nn.Conv2d(3, 16, (8, 4))
        self.mp2 = nn.MaxPool2d((1, 2))

        self.conv3 = nn.Conv2d(3, 16, 4)
        self.mp3 = nn.MaxPool2d(2)


        # Linear
        self.bn = nn.BatchNorm1d(128)

        self.ln1 = nn.Linear(128, 256)
        self.do1 = nn.Dropout(0.5)
        
        self.ln2 = nn.Linear(256, 128)
        self.do2 = nn.Dropout(0.5)

        self.ln3 = nn.Linear(128, 64)
        self.do3 = nn.Dropout(0.5)

        self.ln4 = nn.Linear(64, 1)
        
   
        # Weight initialization
        #     CNN
        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity="relu")
        nn.init.constant_(self.conv1.bias, 0)
        nn.init.kaiming_normal_(self.conv2.weight, nonlinearity="relu")
        nn.init.constant_(self.conv2.bias, 0)
        nn.init.kaiming_normal_(self.conv3.weight, nonlinearity="relu")
        nn.init.constant_(self.conv3.bias, 0)

        #    Linear
        nn.init.xavier_normal_(self.ln1.weight)
        nn.init.constant_(self.ln1.bias, 0)
        nn.init.xavier_normal_(self.ln2.weight)
        nn.init.constant_(self.ln2.bias, 0)
        nn.init.xavier_normal_(self.ln3.weight)
        nn.init.constant_(self.ln3.bias, 0)
        nn.init.xavier_normal_(self.ln4.weight)
        nn.init.constant_(self.ln4.bias, 0)

    def forward(self, img):
        # CNN
        img1 = torch.relu(self.conv1(img))
        img1 = self.mp1(img1)
        img1 = img1.reshape(img1.shape[0], -1)

        img2 = torch.relu(self.conv2(img))
        img2 = self.mp2(img2)
        img2 = img2.reshape(img2.shape[0], -1)

        img3 = self.conv3(img)
        img3 = self.mp3(img3)
        img3 = img3.reshape(img3.shape[0], -1)

        # Concatenate
        img = torch.cat([img1, img2, img3], dim=1)

        # Batch norm
        img = self.bn(img)

        # Linear
        img = torch.tanh(self.ln1(img))
        img = self.do1(img)

        img = torch.sigmoid(self.ln2(img))
        img = self.do2(img)

        img = torch.tanh(self.ln3(img))
        img = self.do3(img)

        img = torch.sigmoid(self.ln4(img))
        
        return img
