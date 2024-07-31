import torch.nn as nn
import torch.nn.functional as F

class WideBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, width_factor=1):
        super(WideBasicBlock, self).__init__()
        out_channels = int(out_channels * width_factor)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class WideResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, width_factor=1):
        super(WideResNet, self).__init__()
        self.in_channels = 16
        
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1, width_factor=width_factor)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2, width_factor=width_factor)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2, width_factor=width_factor)
        self.linear = nn.Linear(64 * width_factor, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride, width_factor):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride, width_factor))
            self.in_channels = int(out_channels * width_factor)
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

# Function to create a WideResNet with specified layers and width factor
def create_wideresnet(num_blocks, width_factor):
    depths = {
        16: [2, 2, 2],
        22: [3, 3, 3],
        28: [4, 4, 4],
        40: [6, 6, 6]
    }
    
    assert num_blocks in depths, f"num_blocks must be one of {list(depths.keys())}"
    return WideResNet(WideBasicBlock, depths[num_blocks], width_factor=width_factor)
