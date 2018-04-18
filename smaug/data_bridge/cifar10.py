import os
import json


def get_data(data_dir, cutoff=None):
    files = os.listdir(data_dir)
    files = list(filter(lambda f: f.endswith('.png'), files))
    label_map = json.load(open(os.path.join(data_dir, 'map.json')))

    if cutoff is not None:
        files = files[:cutoff]
    labels = list(map(lambda f: label_map[f], files))
    full_paths = list(map(lambda f: os.path.join(data_dir, f), files))

    return full_paths, labels


def get_num_labels():
    return 10


def get_img_size():
    return 32, 32
