import torch
import torch.nn as nn
import torch.nn.functional as F


class AvapixModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.do1 = nn.Dropout(0.5)
        self.do2 = nn.Dropout(0.25)

        self.enc1 = nn.Conv2d(3, 16, 2)
        self.enc2 = nn.Conv2d(16, 32, 2)
        self.enc3 = nn.Conv2d(32, 64, 2)
        self.enc4 = nn.Conv2d(64, 128, 2)
        self.enc5 = nn.Conv2d(128, 256, 2)
        self.enc6 = nn.Conv2d(256, 512, 2)
        self.enc7 = nn.Conv2d(512, 1024, 2)

        self.bn1 = nn.BatchNorm2d(1024)

        self.dec1 = nn.ConvTranspose2d(1024, 512, 2)
        self.dec2 = nn.ConvTranspose2d(512, 256, 2)
        self.dec3 = nn.ConvTranspose2d(256, 128, 2)
        self.dec4 = nn.ConvTranspose2d(128, 64, 2)
        self.dec5 = nn.ConvTranspose2d(64, 32, 2)
        self.dec6 = nn.ConvTranspose2d(32, 16, 2)
        self.dec7 = nn.ConvTranspose2d(16, 3, 2)

        nn.init.kaiming_normal_(self.enc1.weight, nonlinearity="relu")
        nn.init.kaiming_normal_(self.enc2.weight, nonlinearity="relu")
        nn.init.kaiming_normal_(self.enc3.weight, nonlinearity="relu")
        nn.init.kaiming_normal_(self.enc4.weight, nonlinearity="relu")
        nn.init.kaiming_normal_(self.enc5.weight, nonlinearity="relu")
        nn.init.kaiming_normal_(self.enc6.weight, nonlinearity="relu")
        nn.init.kaiming_normal_(self.enc7.weight, nonlinearity="relu")
        nn.init.xavier_normal_(self.dec1.weight)
        nn.init.xavier_normal_(self.dec2.weight)
        nn.init.xavier_normal_(self.dec3.weight)
        nn.init.xavier_normal_(self.dec4.weight)
        nn.init.xavier_normal_(self.dec5.weight)
        nn.init.xavier_normal_(self.dec6.weight)
        nn.init.xavier_normal_(self.dec7.weight)

    def forward(self, img):
        img = torch.relu(self.enc1(img))
        img = torch.relu(self.enc2(img))
        img = torch.relu(self.enc3(img))
        img = torch.relu(self.enc4(img))
        img = torch.relu(self.enc5(img))
        img = torch.relu(self.enc6(img))
        img = torch.relu(self.enc7(img))
        img = self.bn1(img)
        img = torch.sigmoid(self.dec1(img))
        img = torch.sigmoid(self.dec2(img))
        img = torch.sigmoid(self.dec3(img))
        img = torch.sigmoid(self.dec4(img))
        img = torch.sigmoid(self.dec5(img))
        img = torch.sigmoid(self.dec6(img))
        img = torch.sigmoid(self.dec7(img))

        return img
