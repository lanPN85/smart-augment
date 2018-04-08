import os
import time
import torch
import torch.autograd as autograd
import torch.nn as nn

from torch.utils.data import DataLoader

from smaug.utils import raw_collate


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
        inp = torch.cat([img1, img2], dim=1)
        return self.net_a(inp)

    def forward_b(self, images):
        return self.net_b(images)

    def train(self, dataset, test_dataset, epochs, lr=0.01, save_dir='models/default'):
        os.makedirs(save_dir, exist_ok=True)

        optimizer = torch.optim.SGD(list(self.net_a.parameters()) + list(self.net_b.parameters()),
                                    lr=lr, momentum=0.9, nesterov=False)
        criterion_a = nn.MSELoss()
        criterion_b = nn.CrossEntropyLoss()
        train_loader = DataLoader(dataset, batch_size=1, shuffle=True,
                                  collate_fn=raw_collate)
        test_loader = DataLoader(test_dataset, batch_size=1, collate_fn=raw_collate)

        for ep in range(epochs):
            total_loss = 0.
            t_start = time.time()
            self.net_a.train()
            self.net_b.train()

            for i, (images, labels) in enumerate(train_loader):
                im1, im2, im3 = images
                im1 = autograd.Variable(im1)
                im2 = autograd.Variable(im2)
                im3 = autograd.Variable(im3)
                labels = autograd.Variable(labels)
                labels = torch.cat([labels, labels], dim=0)
                labels = torch.squeeze(labels, dim=1)

                if self.__cuda:
                    im1 = im1.cuda()
                    im2 = im2.cuda()
                    im3 = im3.cuda()
                    labels = labels.cuda()

                optimizer.zero_grad()

                new_img = self.forward_a(im1, im2)
                inp_batch = torch.cat([new_img, im3], dim=0)
                out = self.forward_b(inp_batch)

                loss_a = criterion_a(new_img, im3)
                loss_b = criterion_b(out, labels)
                loss = self.alpha * loss_a + self.beta * loss_b
                total_loss += loss

                loss.backward()
                optimizer.step()
                print('Epoch %d/%d - Iter %d/%d - Loss: %6.4f\r' %
                      (ep+1, epochs, i+1, len(dataset), loss))

            t_elapsed = time.time() - t_start
            print()
            print('Epoch %d/%d - Avg. loss: %.4f - Time: %.2fs' %
                  (ep + 1, epochs, total_loss / len(dataset), t_elapsed))
            snap_path_a = os.path.join(save_dir, 'epoch_%d_a.pth' % (ep + 1))
            snap_path_b = os.path.join(save_dir, 'epoch_%d_b.pth' % (ep + 1))
            self.save(snap_path_a, snap_path_b)

            # TODO Evaluate on epoch end

    def save(self, path_a, path_b):
        torch.save(self.net_a, path_a)
        torch.save(self.net_b, path_b)
