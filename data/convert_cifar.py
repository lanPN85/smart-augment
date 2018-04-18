import pickle
import os
import cv2
import json
import numpy as np


train_batches = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
test_batches = ['test_batch']

ORIGIN = 'cifar-10-batches-py'
TARGET_TRAIN = 'cifar-10/train'
TARGET_TEST = 'cifar-10/test'

os.makedirs(TARGET_TRAIN, exist_ok=True)
os.makedirs(TARGET_TEST, exist_ok=True)


def convert_image(img_vec):
    assert img_vec.shape[0] == 3072
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    img[:, :, 2] = np.reshape(img_vec[:1024], (32, 32))
    img[:, :, 1] = np.reshape(img_vec[1024: 2048], (32, 32))
    img[:, :, 0] = np.reshape(img_vec[2048:], (32, 32))

    return img


train_set = {
    'data': [],
    'labels': []
}
for batch in train_batches:
    with open(os.path.join(ORIGIN, batch), 'rb') as f:
        print(os.path.join(ORIGIN, batch))
        d = pickle.load(f, encoding='bytes')
        train_set['labels'].extend(d[b'labels'])
        data = d[b'data']
        for i in range(data.shape[0]):
            img_ = convert_image(data[i, :])  # (Rx32 Gx32 Bx32) * row
            train_set['data'].append(img_)

test_set = {
    'data': [],
    'labels': []
}
for batch in test_batches:
    with open(os.path.join(ORIGIN, batch), 'rb') as f:
        print(os.path.join(ORIGIN, batch))
        d = pickle.load(f, encoding='bytes')
        test_set['labels'].extend(d[b'labels'])
        data = d[b'data']
        for i in range(data.shape[0]):
            img_ = convert_image(data[i, :])  # (Rx32 Gx32 Bx32) * row
            test_set['data'].append(img_)

train_map = {}
for i, (img, label) in enumerate(zip(train_set['data'], train_set['labels'])):
    fname = '%06d.png' % (i+1)
    cv2.imwrite(os.path.join(TARGET_TRAIN, fname), img)
    train_map[fname] = label
# pickle.dump(train_map, open(os.path.join(TARGET_TRAIN, 'map.pkl'), 'wb'))
json.dump(train_map, open(os.path.join(TARGET_TRAIN, 'map.json'), 'wt'))

test_map = {}
for i, (img, label) in enumerate(zip(test_set['data'], test_set['labels'])):
    fname = '%06d.png' % (i+1)
    cv2.imwrite(os.path.join(TARGET_TEST, fname), img)
    test_map[fname] = label
# pickle.dump(test_map, open(os.path.join(TARGET_TEST, 'map.pkl'), 'wb'))
json.dump(test_map, open(os.path.join(TARGET_TEST, 'map.json'), 'wt'))
