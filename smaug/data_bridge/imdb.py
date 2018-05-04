import os
import json


def get_data(data_dir, cutoff=None):
    file_map = json.load(open(os.path.join(data_dir, 'genders.json'), 'rt'))
    full_paths, labels = [], []

    pairs = file_map.items()
    if cutoff is not None:
        pairs = list(pairs)[:cutoff]
    for fn, gender in pairs:
        try:
            labels.append(int(gender))
            full_paths.append(os.path.join(data_dir, fn))
        except ValueError:
            continue

    return full_paths, labels


def get_num_labels():
    return 2


def get_img_size():
    return 96, 96
