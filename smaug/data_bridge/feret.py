import os
import csv


def get_data(data_dir, cutoff=None):
    files, labels = [], []
    with open(os.path.join(data_dir, 'genders.csv'), 'rt', newline='') as cf:
        reader = csv.reader(cf)
        for row in reader:
            fid, label = row
            fpath = os.path.join(data_dir, fid + '.ppm')
            label = 0 if label == 'M' else 1

            files.append(fpath)
            labels.append(label)

    if cutoff is not None:
        files = files[:cutoff]
        labels = labels[:cutoff]

    return files, labels


def get_num_labels():
    return 2


def get_img_size():
    return 96, 96
