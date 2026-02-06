from warnings import warn

import torch.nn.functional as F
import torch.nn as nn

def conditional_layer(condition: bool, layer_factory: callable):
    if condition:
        return layer_factory()
    else:
        return nn.Identity()


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity
        out = F.relu(out)
        return out


class ResidualBlockWithDownsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride=2):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 1×1 conv to match shape for the skip connection
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1,
                      stride=stride, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity
        out = F.relu(out)
        
        return out


class CNN(nn.Module):
    def __init__(
        self,
        num_classes=1,
        input_height=28,     # pixel height
        input_width=28,      # pixel width
        input_channels=3,     # 1: grayscale, 3: RGB
        batch_norm: tuple[int, int] | bool | None = (32, 64),
        max_pool: tuple[int, int] | bool | None = (2, 2),
        dropout: float | bool | None = 0.5,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.input_height = input_height
        self.input_width = input_width
        self.input_channels = input_channels

        use_batch_norm = batch_norm is not None and batch_norm is not False
        use_max_pool = max_pool is not None and max_pool is not False
        use_dropout = dropout is not None and dropout is not False

        # Convolutional feature extractor
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            conditional_layer(use_batch_norm, lambda: nn.BatchNorm2d(batch_norm[0])),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            conditional_layer(use_batch_norm, lambda: nn.BatchNorm2d(batch_norm[0])),
            nn.ReLU(inplace=True),
            conditional_layer(use_max_pool, lambda: nn.MaxPool2d(max_pool[0])),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            conditional_layer(use_batch_norm, lambda: nn.BatchNorm2d(batch_norm[1])),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            conditional_layer(use_batch_norm, lambda: nn.BatchNorm2d(batch_norm[1])),
            nn.ReLU(inplace=True),
            conditional_layer(use_max_pool, lambda: nn.MaxPool2d(max_pool[1])),
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            conditional_layer(use_dropout, lambda: nn.Dropout(dropout)),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv1(x)  # [B, 32, 14, 14]
        x = self.conv2(x)  # [B, 64, 7, 7]
        x = self.classifier(x)  # [B, num_classes]

        return x

class ResidualCNN(CNN):
    def __init__(
        self,
        num_classes=1,
        input_height=28,     # pixel height
        input_width=28,      # pixel width
        input_channels=3,     # 1: grayscale, 3: RGB
        batch_norm: tuple[int, int] | bool | None = (32, 64),
        max_pool: tuple[int, int] | bool | None = (2, 2),
        dropout: float | bool | None = 0.5,
    ):
        super().__init__(
            num_classes=num_classes,
            input_height=input_height,
            input_width=input_width,
            input_channels=input_channels,
            batch_norm=batch_norm,
            max_pool=max_pool,
            dropout=dropout,
        )

        #self.residual_block1 = ResidualBlock(32, 32)
        #self.residual_block2 = ResidualBlock(64, 64)
        
        self.res1 = ResidualBlockWithDownsample(3, 32, stride=2)
        self.res2 = ResidualBlockWithDownsample(32, 64, stride=2)
        
        self.res3 = ResidualBlock(32, 32)

    def forward(self, x):
        identity = x

        x = self.res1(x)
        x = self.res2(x)
        print(x.shape)
        
        #x = self.conv1(x)
        #x = self.residual_block1(x)
        #x = self.conv2(x)
        #x = self.residual_block2(x)
        #x = self.classifier(x)

        return x

class CNNWithResidualConnections(nn.Module):
    def __init__(
        self,
        num_classes=1,
        input_height=28,     # pixel height
        input_width=28,      # pixel width
        input_channels=3,     # 1: grayscale, 3: RGB
        batch_norm: tuple[int, int] | bool | None = (32, 64),
        max_pool: tuple[int, int] | bool | None = (2, 2),
        dropout: float | bool | None = True,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.input_height = input_height
        self.input_width = input_width
        self.input_channels = input_channels

        use_batch_norm = batch_norm is not None and batch_norm is not False
        use_max_pool = max_pool is not None and max_pool is not False
        use_dropout = dropout is not None and dropout is not False

        if not use_batch_norm:
            warn("Batch normalization cannot be disabled when using residual connections.")

        if not use_max_pool:
            warn("Max pooling cannot be disabled when using residual connections.")

        self.residual_block1 = ResidualBlock(self.input_channels, 32)
        self.residual_block2 = ResidualBlock(32, 64)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            conditional_layer(use_dropout, lambda: nn.Dropout(dropout)),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        print(x.shape)
        x = self.residual_block1(x)  # [B, 32, 28, 28]
        print(x.shape)
        x = self.residual_block2(x)   # [B, 64, 14, 14]
        print(x.shape)
        x = self.classifier(x)        # [B, num_classes]

        return x


class ClassificationHead(nn.Module):
    def __init__(
        self,
        in_features: int,
        num_classes: int,
        dropout: float | bool | None = 0.5,
    ):
        super().__init__()

        use_dropout = dropout is not None and dropout is not False

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 128),
            nn.ReLU(inplace=True),
            conditional_layer(use_dropout, lambda: nn.Dropout(dropout)),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.classifier(x)
