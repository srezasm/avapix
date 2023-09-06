import torch
import torch.nn as nn
import torch.nn.functional as F


class AvapixModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        # CNN
        self.conv1 = nn.Conv2d(3, 16, 2)
        self.do1 = nn.Dropout(0.5)

        self.conv2 = nn.Conv2d(16, 32, 2)
        self.do2 = nn.Dropout(0.5)

        self.conv3 = nn.Conv2d(32, 64, 2)
        self.do3 = nn.Dropout(0.5)

        self.conv4 = nn.Conv2d(64, 128, 2)
        self.do4 = nn.Dropout(0.5)

        self.conv5 = nn.Conv2d(128, 256, 2)
        self.do5 = nn.Dropout(0.5)

        self.conv6 = nn.Conv2d(256, 512, 2)
        self.bn = nn.BatchNorm2d(512)
        self.do6 = nn.Dropout(0.5)

        self.conv7 = nn.ConvTranspose2d(512, 256, 2)
        self.do7 = nn.Dropout(0.5)

        self.conv8 = nn.ConvTranspose2d(256, 128, 2)
        self.do8 = nn.Dropout(0.5)

        self.conv9 = nn.ConvTranspose2d(128, 64, 2)
        self.do9 = nn.Dropout(0.5)

        self.conv10 = nn.ConvTranspose2d(64, 32, 2)
        self.do10 = nn.Dropout(0.5)

        self.conv11 = nn.ConvTranspose2d(32, 16, 2)
        self.do11 = nn.Dropout(0.5)

        self.conv12 = nn.ConvTranspose2d(16, 3, 2)

        # Weight initialization
        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity="relu")
        nn.init.constant_(self.conv1.bias, 0)
        nn.init.xavier_normal_(self.conv2.weight)
        nn.init.constant_(self.conv2.bias, 0)
        nn.init.xavier_normal_(self.conv3.weight)
        nn.init.constant_(self.conv3.bias, 0)
        nn.init.kaiming_normal_(self.conv4.weight, nonlinearity="relu")
        nn.init.constant_(self.conv4.bias, 0)
        nn.init.xavier_normal_(self.conv5.weight)
        nn.init.constant_(self.conv5.bias, 0)
        nn.init.xavier_normal_(self.conv6.weight)
        nn.init.constant_(self.conv6.bias, 0)
        nn.init.kaiming_normal_(self.conv7.weight, nonlinearity="relu")
        nn.init.constant_(self.conv7.bias, 0)
        nn.init.xavier_normal_(self.conv8.weight)
        nn.init.constant_(self.conv8.bias, 0)
        nn.init.xavier_normal_(self.conv9.weight)
        nn.init.constant_(self.conv9.bias, 0)
        nn.init.kaiming_normal_(self.conv10.weight, nonlinearity="relu")
        nn.init.constant_(self.conv10.bias, 0)
        nn.init.xavier_normal_(self.conv11.weight)
        nn.init.constant_(self.conv11.bias, 0)
        nn.init.xavier_normal_(self.conv12.weight)
        nn.init.constant_(self.conv12.bias, 0)

    def forward(self, img):
        img = torch.relu(self.conv1(img))
        img = self.do1(img)

        img = torch.tanh(self.conv2(img))
        img = self.do2(img)

        img = torch.sigmoid(self.conv3(img))
        img = self.do3(img)

        img = torch.relu(self.conv4(img))
        img = self.do4(img)

        img = torch.tanh(self.conv5(img))
        img = self.do5(img)

        img = torch.sigmoid(self.conv6(img))
        img = self.bn(img)
        img = self.do6(img)

        img = torch.relu(self.conv7(img))
        img = self.do7(img)

        img = torch.tanh(self.conv8(img))
        img = self.do8(img)

        img = torch.sigmoid(self.conv9(img))
        img = self.do9(img)

        img = torch.relu(self.conv10(img))
        img = self.do10(img)

        img = torch.tanh(self.conv11(img))
        img = self.do11(img)

        img = torch.sigmoid(self.conv12(img))

        return img
