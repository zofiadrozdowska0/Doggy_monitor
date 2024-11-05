import os
import random
import shutil
from pathlib import Path


# Splits dataset into 3 folders: train (70%), val (20%) and test (10%)
def split_dataset(dataset_path, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    images_path = Path(dataset_path) / 'images/train'
    labels_path = Path(dataset_path) / 'labels/train'

    train_images = Path(dataset_path) / 'images/train'
    val_images = Path(dataset_path) / 'images/val'
    test_images = Path(dataset_path) / 'images/test'

    train_labels = Path(dataset_path) / 'labels/train'
    val_labels = Path(dataset_path) / 'labels/val'
    test_labels = Path(dataset_path) / 'labels/test'

    os.makedirs(val_images, exist_ok=True)
    os.makedirs(test_images, exist_ok=True)
    os.makedirs(val_labels, exist_ok=True)
    os.makedirs(test_labels, exist_ok=True)

    all_images = list(images_path.glob('*.PNG'))
    random.shuffle(all_images)

    total_images = len(all_images)
    train_count = int(total_images * train_ratio)
    val_count = int(total_images * val_ratio)
    test_count = total_images - train_count - val_count

    train_files = all_images[:train_count]
    val_files = all_images[train_count:train_count + val_count]
    test_files = all_images[train_count + val_count:]

    def copy_files(file_list, target_images, target_labels):
        for img_path in file_list:
            label_path = labels_path / (img_path.stem + '.txt')
            shutil.move(img_path, target_images / img_path.name)
            shutil.move(label_path, target_labels / label_path.name)

    copy_files(train_files, train_images, train_labels)
    copy_files(val_files, val_images, val_labels)
    copy_files(test_files, test_images, test_labels)

    def create_txt_file(file_list, txt_path, folder_name):
        with open(txt_path, 'w') as f:
            for img_path in file_list:
                f.write(f'data/images/{folder_name}/{img_path.name}\n')

    train_txt = Path(dataset_path) / 'train.txt'
    val_txt = Path(dataset_path) / 'val.txt'
    test_txt = Path(dataset_path) / 'test.txt'

    create_txt_file(train_files, train_txt, 'train')
    create_txt_file(val_files, val_txt, 'val')
    create_txt_file(test_files, test_txt, 'test')

    print('All done :)')

split_dataset('2B_1-150')
