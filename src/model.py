import torch.nn as nn
from src import config
import torch


class ResidualBlock(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()

        ## cnn path
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels)
        )

        # skip connection
        self.skip = nn.Conv2d(in_channels, out_channels, 1)

        # add activation
        self.activation = nn.LeakyReLU()

        # reduce size
        self.pool = nn.MaxPool2d(2)

    def forward(self,x):
        identity = self.skip(x)
        out = self.conv(x)
        out = out + identity
        out = self.activation(out)
        out = self.pool(out)

        return out


class BrainTumorMRICNN(nn.Module):
    def __init__(self,num_classes = config.num_class):
        super().__init__()
        self.block1 = ResidualBlock(3, 8)
        self.block2 = ResidualBlock(8, 16)
        self.block3 = ResidualBlock(16, 32)
        self.block4 = ResidualBlock(32, 64)
        self.block5 = ResidualBlock(64, 128)

        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(128, 256),
            nn.LeakyReLU(),
            nn.Dropout(0.5),

            nn.Linear(256, num_classes)
        )


    def forward(self,x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.global_pool(x)

        x = torch.flatten(x, 1)

        x = self.fc(x)
        return x