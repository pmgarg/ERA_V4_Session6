import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Drastically reduced channels to meet <25K constraint
        self.conv1 = nn.Conv2d(1, 7, kernel_size=3, bias=False)  
        self.bn1 = nn.BatchNorm2d(7)  # 28

        self.conv2 = nn.Conv2d(7, 7, kernel_size=3, bias=False)  
        self.bn2 = nn.BatchNorm2d(7)  # 28

        self.conv3 = nn.Conv2d(7, 10, kernel_size=3,  bias=False)  
        self.bn3 = nn.BatchNorm2d(10)  # 32

        self.pool1 = nn.MaxPool2d(2, 2) #  receptive = 8

        self.conv4 = nn.Conv2d(10, 10, kernel_size=3, bias=False)  
        self.bn4 = nn.BatchNorm2d(10)  # 36
        self.conv5 = nn.Conv2d(10, 12, kernel_size=3, bias=False)  
        self.bn5 = nn.BatchNorm2d(12) # 40
        self.pool2 = nn.MaxPool2d(2, 2)  # 14->7 receptive = 18

        self.conv6 = nn.Conv2d(12, 16, kernel_size=3, padding=1, bias=False)  
        self.bn6 = nn.BatchNorm2d(16) # 44
        self.conv7 = nn.Conv2d(16, 18, kernel_size=3, padding=1, bias=False)  
        self.bn7 = nn.BatchNorm2d(18) # 44


        self.gap  = nn.AdaptiveAvgPool2d(1)   # add this
        self.fc   = nn.Linear(18, 10)         #  22*10 + 10 =220

        # self.drop2d = nn.Dropout2d(p=0.1)  # use after conv blocks (feature maps)
        # self.dropfc = nn.Dropout(p=0.1)    # use before final Linear (vectors)



    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        # x = self.drop2d(x)
        x = F.relu(self.bn2(self.conv2(x)))
        # x = self.drop2d(x)
        x = F.relu(self.bn3(self.conv3(x)))
        # x = self.drop2d(x)
        x = self.pool1(x)

        x = F.relu(self.bn4(self.conv4(x)))
        # x = self.drop2d(x)
        x = F.relu(self.bn5(self.conv5(x)))
        # x = self.drop2d(x)
        x = self.pool2(x)

        x = F.relu(self.bn6(self.conv6(x)))  # 5x5 -> 3x3
        # x = self.drop2d(x)
        x = F.relu(self.bn7(self.conv7(x)))  # 5x5 -> 3x3
        # x = self.drop2d(x)

        x = self.gap(x)                             # (N, 26, 1, 1)
        x = torch.flatten(x, 1)                     # (N, 26) — SAFE (never drops batch)
        # x = self.dropfc(x)
        x = self.fc(x)                              # (N, 10)
        # x =  self.conv8(x)  # (N, 10) 
        # x = x.view(-1, 14)
        return x

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)