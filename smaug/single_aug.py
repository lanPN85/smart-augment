import os
import random
import time
import cv2
import torch
import torch.autograd as autograd
import torch.nn as nn
import numpy as np

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
        else:
            self.cpu()

    def cuda(self, device=0):
        self.net_a.cuda(device)
        self.net_b.cuda(device)
        self.__cuda = True

    def cpu(self):
        self.net_a.cpu()
        self.net_b.cpu()
        self.__cuda = False

    def forward_a(self, img1, img2):
        inp = torch.cat([img1, img2], dim=1)
        return self.net_a(inp)

    def forward_b(self, images):
        return self.net_b(images)

    def train(self, dataset, test_dataset, epochs, lr=0.01, save_dir='models/default',
              snapshot_freq=5, gradient_norm=400):
        img_dir = os.path.join(save_dir, 'images')
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(img_dir, exist_ok=True)

        optimizer = torch.optim.SGD(list(self.net_a.parameters()) + list(self.net_b.parameters()),
                                    lr=lr, momentum=0.9, nesterov=False)
        criterion_a = nn.MSELoss(size_average=False)
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

                loss_a = criterion_a(new_img, im3) #/ (new_img.shape[-1] * new_img.shape[-2])
                loss_b = criterion_b(out, labels)
                loss = self.alpha * loss_a + self.beta * loss_b
                total_loss += loss

                loss.backward()
                nn.utils.clip_grad_norm(list(self.net_a.parameters()) + list(self.net_b.parameters()), gradient_norm)
                optimizer.step()
                print('Epoch %d/%d - Iter %d/%d - Loss@A: %6.4f - Loss@B: %6.4f - Loss: %6.4f' %
                      (ep+1, epochs, i+1, len(dataset), loss_a, loss_b, loss), end='\r')

            t_elapsed = time.time() - t_start
            print()
            print('Epoch %d/%d - Avg. loss: %.4f - Time: %.2fs' %
                  (ep + 1, epochs, total_loss / len(dataset), t_elapsed))
            snap_path_a = os.path.join(save_dir, 'epoch_%d_a.pth' % (ep + 1))
            snap_path_b = os.path.join(save_dir, 'epoch_%d_b.pth' % (ep + 1))
            self.save(snap_path_a, snap_path_b)

            # Evaluate accuracy
            print('Evaluating...')
            correct, total = 0., 0.
            for i, (images, labels) in enumerate(test_loader):
                _, _, im3 = images
                im3 = autograd.Variable(im3)
                if self.__cuda:
                    im3 = im3.cuda()

                out = self.get_net_b_pred(im3)[0]
                _, pred = torch.max(out, 0)

                if pred.data[0] == labels[0][0]:
                    correct += 1.
                total += 1.
            acc = correct / total
            print('Val accuracy: %.4f' % acc)

            if ep % snapshot_freq == 0:
                print('Testing smart augment...')
                epoch_img_dir = os.path.join(img_dir, '%d' % (ep + 1))
                os.makedirs(epoch_img_dir, exist_ok=True)
                print('Saving results in %s' % epoch_img_dir)
                # Get 5 images from net A
                all_ = list(test_loader)
                for i in range(5):
                    images, _ = random.sample(all_, k=1)[0]
                    im1, im2, _ = images
                    im1 = autograd.Variable(im1)
                    im2 = autograd.Variable(im2)
                    if self.__cuda:
                        im1 = im1.cuda()
                        im2 = im2.cuda()

                    out_img = self.get_net_a_image(im1, im2)
                    cv2.imwrite(os.path.join(epoch_img_dir, '%03d_in1.png' % (i+1)), self.denormalize(im1[0]))
                    cv2.imwrite(os.path.join(epoch_img_dir, '%03d_in2.png' % (i+1)), self.denormalize(im2[0]))
                    cv2.imwrite(os.path.join(epoch_img_dir, '%03d_out.png' % (i+1)), out_img)

    def save(self, path_a, path_b):
        torch.save(self.net_a, path_a)
        torch.save(self.net_b, path_b)

    @staticmethod
    def denormalize(img):
        out = img.data.cpu().numpy()
        out = np.transpose(out, [1, 2, 0])
        out *= 255.
        out = out.astype(np.int)
        return out

    def get_net_a_image(self, img1, img2):
        self.net_a.eval()

        inp = torch.cat([img1, img2], dim=1)
        out = self.net_a(inp)[0]

        return self.denormalize(out)

    def get_net_b_pred(self, images):
        self.net_b.eval()

        out = self.net_b(images)
        return nn.Softmax(dim=1)(out)

    @classmethod
    def load(cls, path_a, path_b, **kwargs):
        net_a = torch.load(path_a)
        net_b = torch.load(path_b)
        return cls(net_a, net_b, **kwargs)
