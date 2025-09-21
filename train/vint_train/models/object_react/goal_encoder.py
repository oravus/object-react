from torch import nn


# https://blog.paperspace.com/writing-resnet-from-scratch-in-pytorch/
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size=3, stride=stride, padding=1
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
        )
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(
        self, block, layers, num_classes=10, in_channels=3, numLayers=4, head=True
    ):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.head = head
        out_channels = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layers = []
        strides = [1, 2, 2, 2]
        for i in range(numLayers):
            self.layers.append(
                self._make_layer(
                    block, out_channels * (i + 1), layers[i], stride=strides[i]
                )
            )
        self.layers = nn.Sequential(*self.layers)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:

            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        for layer in self.layers:
            x = layer(x)

        if self.head:
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
        return x


class GoalEncoder(nn.Module):
    def __init__(
        self, goal_encoding_size=1024, in_channels=16, numLayers=2, preProjDims=None
    ):
        super().__init__()
        self.preProjDims = preProjDims
        if self.preProjDims is not None:
            self.conv1x1 = nn.Conv2d(in_channels, self.preProjDims, kernel_size=1)
            in_channels = self.preProjDims
        self.enc = ResNet(
            ResidualBlock,
            [3] * numLayers,
            in_channels=in_channels,
            numLayers=numLayers,
            head=False,
        )
        self.fc = nn.Linear(64 * numLayers, goal_encoding_size)

    def forward(self, goal_img):
        if self.preProjDims is not None:
            goal_img = self.conv1x1(goal_img)
        x = self.enc(goal_img).mean(dim=(2, 3))
        x = self.fc(x)
        return x
