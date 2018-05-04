import scipy.io
import os
import json
import shutil
import random


SOURCE_DIR = 'imdb/'
TARGET_DIR = 'imdb_split/'
RATIO = 0.7


if __name__ == '__main__':
    print('Loading metadata...')
    meta = scipy.io.loadmat(os.path.join(SOURCE_DIR, 'imdb.mat'))

    paths = meta['imdb'][0]['full_path'][0][0]
    genders = meta['imdb'][0]['gender'][0][0]

    shutil.rmtree(TARGET_DIR, ignore_errors=True)
    os.makedirs(TARGET_DIR, exist_ok=True)

    train_dir = os.path.join(TARGET_DIR, 'train')
    val_dir = os.path.join(TARGET_DIR, 'val')

    os.makedirs(train_dir)
    os.makedirs(val_dir)

    pairs = []
    for pth, gd in zip(paths, genders):
        if gd != 0. and gd != 1.:
            continue
        _pth = pth[0]
        pairs.append((os.path.join(SOURCE_DIR, _pth), gd))
    random.shuffle(pairs)

    del meta
    cutoff = int(len(pairs) * RATIO)
    train_pairs = pairs[:cutoff]
    val_pairs = pairs[cutoff:]
    train_map = {}
    val_map = {}

    for i, p in enumerate(train_pairs):
        old_path, gender = p
        ext = old_path.split('.')[-1]
        fname = '%06d.%s' % (i+1, ext)
        new_path = os.path.join(train_dir, fname)
        train_map[fname] = gender

        print('%s -> %s' % (old_path, new_path))
        shutil.copy2(old_path, new_path)

    for i, p in enumerate(val_pairs):
        old_path, gender = p
        ext = old_path.split('.')[-1]
        fname = '%06d.%s' % (i + 1, ext)
        new_path = os.path.join(val_dir, fname)
        val_map[fname] = gender

        print('%s -> %s' % (old_path, new_path))
        shutil.copy2(old_path, new_path)

    print('Saving labels...')
    json.dump(train_map, open(os.path.join(train_dir, 'genders.json'), 'wt'))
    json.dump(val_map, open(os.path.join(val_dir, 'genders.json'), 'wt'))
    print('Done.')
