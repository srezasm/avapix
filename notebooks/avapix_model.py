import torch
import torch.nn as nn
import torch.nn.functional as F


class AvapixModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.do = nn.Dropout(0.5)

        self.ln1 = nn.Linear(192, 256)
        self.ln2 = nn.Linear(256, 256)
        self.ln3 = nn.Linear(256, 512)
        self.ln4 = nn.Linear(512, 512)
        self.ln6 = nn.Linear(512, 1024)
        self.ln7 = nn.Linear(1024, 1024)
        self.ln13 = nn.Linear(1024, 2048)
        self.ln14 = nn.Linear(2048, 2048)
        self.ln15 = nn.Linear(2048, 1024)
        self.ln8 = nn.Linear(1024, 512)
        self.ln9 = nn.Linear(512, 512)
        self.ln10 = nn.Linear(512, 256)
        self.ln11 = nn.Linear(256, 256)
        self.ln12 = nn.Linear(256, 192)

    def forward(self, img):
        img = img.reshape(img.shape[0], -1)

        img = torch.relu(self.ln1(img))
        img = torch.tanh(self.ln2(img))
        img = torch.relu(self.ln3(img))
        img = torch.tanh(self.ln4(img))
        img = torch.relu(self.ln6(img))
        img = torch.tanh(self.ln7(img))
        img = torch.relu(self.ln13(img))
        img = torch.tanh(self.ln14(img))
        img = torch.relu(self.ln15(img))
        img = torch.tan(self.ln8(img))
        img = torch.relu(self.ln9(img))
        img = torch.tanh(self.ln10(img))
        img = torch.relu(self.ln11(img))
        img = torch.sigmoid(self.ln12(img))
        
        img = img.reshape(img.shape[0], 3, 8, 8)

        return img
