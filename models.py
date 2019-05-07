import torch
import torch.nn as nn

from torchvision import models

class Flatten(nn.Module):
    def forward(self, x):
        N, C, H, W = x.size() # read in N, C, H, W
        return x.view(N, -1)  # "flatten" the C * H *

class BasicCNN(nn.Module):

    def __init__(self):
        super().__init__()
        out_1 = 32
        out_2 = 64
        out_3 = 128
        out_4 = 256

        k_size_1 = 3
        padding_1 = 1

        self.cnn = nn.Sequential(
            nn.Conv2d(3, out_1, padding=padding_1, kernel_size=k_size_1, stride=1), # out_1-k_size_1+1 = 26
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_1),
            nn.Conv2d(out_1 , out_1, padding= padding_1, kernel_size=k_size_1, stride=1), #26 - 4 + 1 = 23
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_1),
            nn.Conv2d(out_1 , out_1, padding= padding_1, kernel_size=k_size_1, stride=1), # 23 -3 = 20
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_1),

            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(out_1 , out_2, padding= padding_1, kernel_size=k_size_1, stride=1), # 20 -3 = 17
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_2),
            nn.Conv2d(out_2 , out_2, padding= padding_1, kernel_size=k_size_1, stride=1), # 17 -3 = 14
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_2),
            nn.Conv2d(out_2 , out_2, padding= padding_1, kernel_size=k_size_1, stride=1), # 17 -3 = 14
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_2),

            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(out_2 , out_3, padding= padding_1, kernel_size=k_size_1, stride=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_3),
            nn.Conv2d(out_3 , out_3, padding= padding_1, kernel_size=k_size_1, stride=1), # 17 -3 = 14
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_3),
            nn.Conv2d(out_3 , out_3, padding= padding_1, kernel_size=k_size_1, stride=1), # 17 -3 = 14
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_3),

            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(out_3 , out_4, padding= padding_1, kernel_size=k_size_1, stride=1), # 17 -3 = 14
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_4),
            nn.Conv2d(out_4 , out_4, padding= padding_1, kernel_size=k_size_1, stride=1), # 17 -3 = 14
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_4),
            nn.Conv2d(out_4 , out_4, padding= padding_1, kernel_size=k_size_1, stride=1), # 17 -3 = 14
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_4),

            nn.MaxPool2d(2, stride=2), #17/2 = 7
            Flatten(),

            nn.Linear(9216,512 ), # affine layer
            nn.ReLU(inplace=True),
            nn.Linear(512,10), # affine layer
            nn.ReLU(inplace=True),
            nn.Linear(10,2), # affine layer
        )

    def forward(self, x):
        return self.cnn(x)


