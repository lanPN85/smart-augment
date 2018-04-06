from torch.nn import Sequential, Conv2d, Sigmoid, ReLU


class NetworkA1(Sequential):
    def __init__(self, channels=6):
        super().__init__(
            Conv2d(channels, out_channels=16, kernel_size=(3, 3), padding=1),
            ReLU(),
            Conv2d(16, 16, kernel_size=(5, 5), padding=2),
            ReLU(),
            Conv2d(16, 32, kernel_size=(7, 7), padding=3),
            ReLU(),
            Conv2d(32, 32, kernel_size=(5, 5), padding=2),
            ReLU(),
            Conv2d(32, channels // 2, kernel_size=(1, 1)),
            Sigmoid()
        )
