import os
import time
import torch
import torch.autograd as autograd
import torch.nn as nn


class SmartAugmentSingle:
    def __init__(self, net_a, net_b, alpha=0.7, beta=0.3, cuda=False):
        super().__init__()

        self.net_a = net_a
        self.net_b = net_b
        self.alpha = alpha
        self.beta = beta
        self.__cuda = False

        if cuda:
            self.cuda()

    def cuda(self, device=0):
        self.net_a.cuda(device)
        self.net_b.cuda(device)
        self.__cuda = True

    def forward_a(self, img1, img2):
        pass

    def forward_b(self, images):
        return self.net_b(images)

    def train(self, epochs, lr=0.01, save_dir='models/default'):
        os.makedirs(save_dir, exist_ok=True)

        optimizer = torch.optim.SGD(self.net_a.parameters() + self.net_b.parameters(),
                                    lr=lr, momentum=0.9, nesterov=False)
        criterion = nn.CrossEntropyLoss()

        for ep in range(epochs):
            total_loss = 0.0
            t_start = time.time()
            self.net_a.train()
            self.net_b.train()

            # TODO Finish this
