import torch.nn as nn
from ver4_journal.common.network.BCResBlock import BCResBlock


def BCBlockStage(num_layers, last_channel, cur_channel, idx, use_stride):
    stage = nn.ModuleList()
    channels = [last_channel] + [cur_channel] * num_layers
    for i in range(num_layers):
        stride = (2, 1) if use_stride and i == 0 else (1, 1)
        stage.append(BCResBlock(channels[i], channels[i + 1], idx, stride))
    return stage


class BCResNets_MultiLabel(nn.Module):  # 모델 클래스 이름 변경
    def __init__(self, base_c, num_classes=5, in_channel=12):  # num_classes를 5로 설정 (NORM, MI, STTC, CD, HYP)
        super().__init__()
        self.num_classes = num_classes
        self.in_channel = in_channel
        self.n = [2, 2, 4, 4]  # identical modules repeated n times
        self.c = [
            base_c * 2,
            base_c,
            int(base_c * 1.5),
            base_c * 2,
            int(base_c * 2.5),
            base_c * 4,
        ]  # num channels
        self.embed_channel = 20
        self.s = [1, 2]  # stage using stride
        self._build_network()

    def _build_network(self):
        # Head: (Conv-BN-ReLU)
        self.cnn_first = nn.Sequential(
            nn.Conv1d(self.in_channel, self.embed_channel, 5, stride=(2), padding=2, bias=False),
            nn.ReLU(True),
        )
        self.cnn_head = nn.Sequential(
            nn.Conv2d(1, self.c[0], 5, (1, 2), (2, 2), bias=False),
            nn.BatchNorm2d(self.c[0]),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
        )
        # Body: BC-ResBlocks
        self.BCBlocks = nn.ModuleList([])
        for idx, n in enumerate(self.n):
            use_stride = idx in self.s
            self.BCBlocks.append(BCBlockStage(n, self.c[idx], self.c[idx + 1], idx, use_stride))

        # Classifier (one-stage multi-label classification with sigmoid activation)
        self.classifier = nn.Sequential(
            nn.Conv2d(
                self.c[-2], self.c[-2], (5, 5), bias=False, groups=self.c[-2], padding=(0, 2)
            ),
            nn.Conv2d(self.c[-2], self.c[-1], 1, bias=False),
            nn.BatchNorm2d(self.c[-1]),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Conv2d(self.c[-1], self.num_classes, 1),  # num_classes는 레이블 개수
            nn.Sigmoid()  # Multi-label classification을 위한 Sigmoid 활성화 함수
        )

    def forward(self, x):
        x = self.cnn_first(x)
        x = self.cnn_head(x.view(-1, 1, self.embed_channel, 500))
        for i, num_modules in enumerate(self.n):
            for j in range(num_modules):
                x = self.BCBlocks[i][j](x)
        x = self.classifier(x)
        x = x.view(-1, x.shape[1])  # batch_size x num_classes로 출력
        return x
