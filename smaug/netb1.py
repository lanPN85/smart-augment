from torch.nn import Sequential, Conv2d, BatchNorm2d,\
    Dropout, Linear, Sigmoid, ReLU, MaxPool2d

from smaug.modules import Flatten


class NetworkB1(Sequential):
    def __init__(self, channels=3, flat_length=968, labels=2, dropout=0.25):
        super().__init__(
            Conv2d(in_channels=channels, out_channels=16, kernel_size=(3, 3), padding=1),
            ReLU(),
            BatchNorm2d(16),
            MaxPool2d(kernel_size=(3, 3), padding=1),
            BatchNorm2d(16),
            Conv2d(in_channels=16, out_channels=8, kernel_size=(3, 3), padding=1),
            ReLU(),
            BatchNorm2d(8),
            MaxPool2d(kernel_size=(3, 3), padding=1),
            Flatten(),
            Dropout(p=dropout),
            Linear(flat_length, flat_length // 2),
            Sigmoid(),
            Dropout(p=dropout),
            Linear(flat_length // 2, labels),
        )
