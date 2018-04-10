from argparse import ArgumentParser

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
    parser.add_argument('--channels', default=3, type=int)
    parser.add_argument('--random-crop', default=0.8, type=float)
    parser.add_argument('--rotate', default=0.5, type=float)
    parser.add_argument('--dropout', default=0.25, type=float)
    parser.add_argument('--epochs', default=500, type=int)
    parser.add_argument('--save-dir', default='models/default')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()

    print('Preparing data...')
    bridge = BRIDGES[args.data_type]
    train_files, train_labels = bridge.get_data(args.train_dir, cutoff=args.train_cutoff)
    val_files, val_labels = bridge.get_data(args.val_dir, cutoff=args.val_cutoff)
    train_dataset = SingleAugmentDataset(train_files, train_labels, img_size=bridge.get_img_size(),
                                         augment=(not args.no_augment), random_crop=args.random_crop,
                                         rotate=args.rotate)
    val_dataset = SingleAugmentDataset(val_files, val_labels, img_size=bridge.get_img_size(), augment=False)

    print('Creating model...')
    net_a = NetworkA1(channels=2*args.channels)
    net_b = NetworkB1(channels=args.channels, flat_length=968,
                      labels=bridge.get_num_labels(), dropout=args.dropout)
    model = SmartAugmentSingle(net_a, net_b, alpha=args.alpha, beta=args.beta, cuda=args.cuda)

    print('Starting training...')
    try:
        model.train(train_dataset, val_dataset, args.epochs, lr=args.lr, save_dir=args.save_dir)
        print('Training complete.')
    except KeyboardInterrupt:
        print('\nTraining interrupted.')
