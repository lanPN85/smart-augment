import sys
sys.path.append('.')

import torch
import torch.autograd as autograd

from smaug.netb1 import NetworkB1


if __name__ == '__main__':
    net = NetworkB1()
    images = autograd.Variable(torch.randn(10, 3, 96, 96))
    out = net(images)
