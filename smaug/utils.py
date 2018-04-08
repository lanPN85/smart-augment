import torch
import numpy as np


def raw_collate(batch):
    i1, i2, i3, labels = [], [], [], []

    for im, label in batch:
        img1, img2, img3 = im
        i1.append(torch.from_numpy(img1))
        i2.append(torch.from_numpy(img2))
        i3.append(torch.from_numpy(img3))
        labels.append(torch.from_numpy(label))

    ti1 = torch.stack(i1, dim=0).float()
    ti2 = torch.stack(i2, dim=0).float()
    ti3 = torch.stack(i3, dim=0).float()
    tl = torch.stack(labels, dim=0)

    return (ti1, ti2, ti3), tl


if __name__ == '__main__':
    _im = (np.random.randn(3, 96, 96), np.random.randn(3, 96, 96), np.random.randn(3, 96, 96))
    _label = np.random.randint(0, 5, size=(1,))
    o = raw_collate([(_im, _label)])
