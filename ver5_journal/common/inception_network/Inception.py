import torch
import torch.nn as nn

# Inception Module 정의
class InceptionModule(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(InceptionModule, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, n_filters, kernel_size=1, padding=0)
        self.conv3 = nn.Conv1d(in_channels, n_filters, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(in_channels, n_filters, kernel_size=5, padding=2)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.conv_pool = nn.Conv1d(in_channels, n_filters, kernel_size=1, padding=0)

    def forward(self, x):
        conv1_out = self.conv1(x)
        conv3_out = self.conv3(x)
        conv5_out = self.conv5(x)
        pool_out = self.conv_pool(self.maxpool(x))
        return torch.cat([conv1_out, conv3_out, conv5_out, pool_out], dim=1)

# InceptionTime 모델 정의
class InceptionTime(nn.Module):
    def __init__(self, in_channels=12, num_classes=5, n_filters=32):
        super(InceptionTime, self).__init__()
        # 첫 번째 Inception Module에 입력 채널 12 설정
        self.inception1 = InceptionModule(in_channels, n_filters)
        self.inception2 = InceptionModule(n_filters * 4, n_filters)
        self.inception3 = InceptionModule(n_filters * 4, n_filters)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(n_filters * 4, num_classes)

    def forward(self, x):
        x = self.inception1(x)
        x = self.inception2(x)
        x = self.inception3(x)
        x = self.global_pool(x)
        x = x.squeeze(-1)
        x = self.fc(x)
        return x