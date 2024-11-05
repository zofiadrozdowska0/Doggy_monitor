import os
import shutil
from pathlib import Path


# reverses changes caused by splitting the dataset folder
def revert_split(dataset_path):
    train_images = Path(dataset_path) / 'images/train'
    val_images = Path(dataset_path) / 'images/val'
    test_images = Path(dataset_path) / 'images/test'

    train_labels = Path(dataset_path) / 'labels/train'
    val_labels = Path(dataset_path) / 'labels/val'
    test_labels = Path(dataset_path) / 'labels/test'

    def move_files_back(source_images, source_labels, target_images, target_labels):
        if source_images.exists():
            for img_file in source_images.glob('*.PNG'):
                shutil.move(img_file, target_images / img_file.name)
            os.rmdir(source_images)

        if source_labels.exists():
            for label_file in source_labels.glob('*.txt'):
                shutil.move(label_file, target_labels / label_file.name)
            os.rmdir(source_labels)

    move_files_back(val_images, val_labels, train_images, train_labels)
    move_files_back(test_images, test_labels, train_images, train_labels)

    val_txt = Path(dataset_path) / 'val.txt'
    test_txt = Path(dataset_path) / 'test.txt'

    if val_txt.exists():
        os.remove(val_txt)
    if test_txt.exists():
        os.remove(test_txt)

    train_txt = Path(dataset_path) / 'train.txt'
    with open(train_txt, 'w') as f:
        for img_file in sorted(train_images.glob('*.PNG')):
            f.write(f'data/images/train/{img_file.name}\n')

    print('All done')

revert_split('2B_1-150')
