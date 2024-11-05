import os
import shutil
from pathlib import Path


# Merging two datasets folders into one
def merge_yolo_datasets(dataset1_path, dataset2_path, output_path):
    dataset1_images = Path(dataset1_path) / 'images/train'
    dataset1_labels = Path(dataset1_path) / 'labels/train'
    dataset1_train_file = Path(dataset1_path) / 'train.txt'
    dataset1_yaml = Path(dataset1_path) / 'data.yaml'

    dataset2_images = Path(dataset2_path) / 'images/train'
    dataset2_labels = Path(dataset2_path) / 'labels/train'
    dataset2_train_file = Path(dataset2_path) / 'train.txt'
    dataset2_yaml = Path(dataset2_path) / 'data.yaml'

    output_images = Path(output_path) / 'images/train'
    output_labels = Path(output_path) / 'labels/train'
    output_train_file = Path(output_path) / 'train.txt'
    output_yaml = Path(output_path) / 'data.yaml'

    os.makedirs(output_images, exist_ok=True)
    os.makedirs(output_labels, exist_ok=True)

    def copy_files(source_images, source_labels, source_train_file, start_index=0):
        new_train_entries = []
        with open(source_train_file, 'r') as f:
            for line in f:
                img_name = line.strip()
                img_path = source_images / os.path.basename(img_name)
                label_path = source_labels / (img_path.stem + '.txt')

                if not img_path.exists() or not label_path.exists():
                    print(f"Skipping missing file: {img_path} or {label_path}")
                    continue

                new_img_name = f'frame_{start_index:06d}.PNG'
                new_img_path = output_images / new_img_name
                new_label_path = output_labels / (new_img_path.stem + '.txt')
                shutil.copy(img_path, new_img_path)
                shutil.copy(label_path, new_label_path)

                new_train_entries.append(f'data/images/train/{new_img_name}')
                start_index += 1
        return new_train_entries, start_index

    merged_train_entries, next_index = copy_files(
        dataset1_images, dataset1_labels, dataset1_train_file, start_index=0
    )

    next_index = len(list(output_images.glob('*.PNG')))

    # Copy dataset2 with updated start_index
    train_entries_2, _ = copy_files(
        dataset2_images, dataset2_labels, dataset2_train_file, start_index=next_index
    )
    merged_train_entries.extend(train_entries_2)

    with open(output_train_file, 'w') as f:
        f.write('\n'.join(merged_train_entries))

    shutil.copy(dataset1_yaml, output_yaml)

    print('All done :)')


# Example usage
data1 = "2B_1-13_100-150"
data2 = "2B_14-99"
output_name = "2B_1-150"
merge_yolo_datasets(data1, data2, output_name)
