import torch.nn as nn
import torch.nn.functional as F

#ResidualBlock (giữ nguyên, nhận param channels)
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out)) + residual
        out = F.relu(out)
        return out

# PolicyNet dày hơn: 20 ResNet blocks, num_channels=256
class PolicyNet(nn.Module):
    def __init__(self, num_res_blocks=20, num_channels=256, action_size=4096):
        """
        - num_res_blocks: 20 (tăng rất sâu)
        - num_channels: 256 (tăng kênh)
        - action_size: 4096 (64×64 mapping)
        """
        super(PolicyNet, self).__init__()
        # Layer đầu vào: từ 12 plane → num_channels
        self.conv_in = nn.Conv2d(12, num_channels, kernel_size=3, padding=1, bias=False)
        self.bn_in = nn.BatchNorm2d(num_channels)

        # Tạo 20 Residual Blocks
        self.res_blocks = nn.ModuleList([ResidualBlock(num_channels) for _ in range(num_res_blocks)])

        # Policy head: 1×1 conv giảm channels xuống 32, rồi FC → action_size
        self.conv_policy = nn.Conv2d(num_channels, 32, kernel_size=1)
        self.bn_policy = nn.BatchNorm2d(32)
        self.fc_policy = nn.Linear(32 * 8 * 8, action_size)

    def forward(self, x):
        # x: (batch, 12, 8, 8)
        out = F.relu(self.bn_in(self.conv_in(x)))
        for block in self.res_blocks:
            out = block(out)
        p = F.relu(self.bn_policy(self.conv_policy(out)))  # (batch, 32, 8, 8)
        p = p.view(p.size(0), -1)                           # (batch, 32*8*8)
        p = self.fc_policy(p)                               # (batch, action_size)
        return p