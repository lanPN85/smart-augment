from argparse import ArgumentParser

import os
import shutil

from smaug import NetworkA1, NetworkB1, SmartAugmentSingle
from smaug.dataset import SingleAugmentDataset
from smaug.data_bridge import feret

BRIDGES = {
    'feret': feret
}


def parse_arguments():
    parser = ArgumentParser()

    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--alpha', type=float, default=0.7)
    parser.add_argument('--beta', type=float, default=0.3)
    parser.add_argument('--train-dir', default='data/colorferet/train')
    parser.add_argument('--val-dir', default='data/colorferet/val')
    parser.add_argument('--train-cutoff', type=int, default=None)
    parser.add_argument('--val-cutoff', type=int, default=None)
    parser.add_argument('--no-augment', action='store_true')
    parser.add_argument('--data-type', default='feret', help='feret')
    parser.add_argument('--grayscale', action='store_true')
    parser.add_argument('--random-crop', default=0.8, type=float)
    parser.add_argument('--rotate', default=0.5, type=float)
    parser.add_argument('--dropout', default=0.25, type=float)
    parser.add_argument('--epochs', default=500, type=int)
    parser.add_argument('--save-dir', default='models/default')
    parser.add_argument('--snapshot-freq', default=5, type=int)
    parser.add_argument('--grad-norm', default=400., type=float)
    parser.add_argument('--flat-length', default=968, type=int, help='968 for color')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()

    print('Preparing data...')
    bridge = BRIDGES[args.data_type]
    train_files, train_labels = bridge.get_data(args.train_dir, cutoff=args.train_cutoff)
    val_files, val_labels = bridge.get_data(args.val_dir, cutoff=args.val_cutoff)
    train_dataset = SingleAugmentDataset(train_files, train_labels, img_size=bridge.get_img_size(),
                                         augment=(not args.no_augment), random_crop=args.random_crop,
                                         rotate=args.rotate, grayscale=args.grayscale)
    val_dataset = SingleAugmentDataset(val_files, val_labels, img_size=bridge.get_img_size(), augment=False,
                                       grayscale=args.grayscale)

    print('Creating model...')
    channels = 1 if args.grayscale else 3
    net_a = NetworkA1(channels=2*channels)
    net_b = NetworkB1(channels=channels, flat_length=968,
                      labels=bridge.get_num_labels(), dropout=args.dropout)
    model = SmartAugmentSingle(net_a, net_b, alpha=args.alpha, beta=args.beta, cuda=args.cuda)

    if os.path.exists(args.save_dir):
        if input('Model directory already exists. Overwrite ? (Y/n) ').lower() == 'y':
            shutil.rmtree(args.save_dir)
        else:
            exit(0)

    print('Starting training...')
    try:
        model.train(train_dataset, val_dataset, args.epochs, lr=args.lr, save_dir=args.save_dir,
                    snapshot_freq=args.snapshot_freq, gradient_norm=args.grad_norm)
        print('Training complete.')
    except KeyboardInterrupt:
        print('\nTraining interrupted.')

