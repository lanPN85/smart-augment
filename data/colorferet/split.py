import os
import shutil
import random
import csv

from xml.etree import ElementTree as ET

TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.2
ROOTS = ['dvd1', 'dvd2']


def get_images(root_dir):
    dirs = os.listdir(root_dir)
    dirs = list(map(lambda d: os.path.join(root_dir, d), dirs))
    dirs = list(filter(lambda d: os.path.isdir(d), dirs))

    images = []

    for d in dirs:
        files = os.listdir(d)
        image_f = list(filter(lambda f: f.endswith('.ppm'), files))
        image_f = list(map(lambda f: os.path.join(d, f), image_f))
        for im in image_f:
            fname = os.path.split(im)[-1]
            full_id = fname.split('.')[0]
            id_parts = full_id.split('_')
            root_id = id_parts[0]

            images.append({
                'path': im,
                'full_id': full_id,
                'root_id': root_id
            })

    return images


def get_label(root_dir, image):
    xml_file = os.path.join(root_dir, image['root_id'], image['root_id'] + '.xml')
    tree = ET.parse(xml_file)
    root = tree.getroot()
    try:
        gender = root[0].find('Gender').get('value')
        if gender == 'Female':
            return 'F'
        elif gender == 'Male':
            return 'M'
        else:
            return None
    except:
        return None


if __name__ == '__main__':
    tagged = []

    for root in ROOTS:
        print('Processing %s ...' % root)

        img_dir = os.path.join(root, 'data', 'images')
        labels_dir = os.path.join(root, 'data', 'ground_truths', 'xml')

        print('Listing images...')
        images_ = get_images(img_dir)

        for img in images_:
            gender = get_label(labels_dir, img)
            if gender is not None:
                print('Tagging image at %s [Gender: %s]' % (img['path'], gender))
                tagged.append({
                    'path': img['path'],
                    'full_id': img['full_id'],
                    'root_id': img['root_id'],
                    'gender': gender
                })
            else:
                print('Image at %s has no gender tag' % img['path'])

    random.shuffle(tagged)
    train_len = int(len(tagged) * TRAIN_SPLIT)
    val_len = int(len(tagged) * VAL_SPLIT)
    train_set = tagged[:train_len]
    val_set = tagged[train_len:train_len + val_len]
    test_set = tagged[train_len + val_len:]

    for imgset, target_dir in zip([train_set, val_set, test_set], ['train', 'val', 'test']):
        shutil.rmtree(target_dir, ignore_errors=True)
        os.makedirs(target_dir,  exist_ok=True)

        print('Writing labels...')
        with open(os.path.join(target_dir, 'genders.csv'), 'wt', newline='') as cf:
            writer = csv.writer(cf)
            for img in imgset:
                writer.writerow([img['full_id'], img['gender']])

        print('Copying images...')
        for img in imgset:
            new_path = os.path.join(target_dir, img['full_id'] + '.ppm')
            print('%s -> %s' % (img['path'], new_path))
            shutil.copy2(img['path'], new_path)

    print('Done.')
