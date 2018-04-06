import sys
sys.path.append('.')

import torch
import torch.autograd as autograd

from smaug.neta1 import NetworkA1


if __name__ == '__main__':
    net = NetworkA1()
    images = autograd.Variable(torch.randn(10, 6, 96, 96))
    out = net(images)
