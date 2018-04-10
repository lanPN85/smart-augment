import sys
sys.path.append('.')

import cv2
import torch
import torch.autograd as autograd

from torch.utils.data import DataLoader

from smaug import SmartAugmentSingle
from smaug.dataset import SingleAugmentDataset
from smaug.data_bridge import feret
from smaug.utils import raw_collate


if __name__ == '__main__':
    model = SmartAugmentSingle.load('models/default/net_a.pth', 'models/default/net_b.pth', cuda=False)
    files, labels = feret.get_data('data/colorferet/test', cutoff=15)
    dataset = SingleAugmentDataset(files, labels, augment=False)
    loader = DataLoader(dataset, collate_fn=raw_collate)

    all = list(loader)
    images, lbl = all[0]
    im1, im2, im3 = images
    im1 = autograd.Variable(im1)
    im2 = autograd.Variable(im2)
    im3 = autograd.Variable(im3)
    img = model.get_net_a_image(im1, im2)
    cv2.imwrite('tests/inf.png', img)

    img3 = model.denormalize(im3[0])
    cv2.imwrite('tests/inp.png', img3)
