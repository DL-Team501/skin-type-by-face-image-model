import os
import zipfile
from pathlib import Path
import pandas as pd

from src.consts import ROOT_PATH


def get_formatted_data():
    dry_oily_normal_zip_file = os.path.join(ROOT_PATH, 'data', 'dry-oily-normal.zip')
    other_classes_zip_file = os.path.join(ROOT_PATH, 'data', 'other-classes.zip')

    output_path = os.path.join(ROOT_PATH, 'data', 'raw')
    unzip_data(dry_oily_normal_zip_file, output_path)
    unzip_data(other_classes_zip_file, output_path)
    data = get_image_path_and_label_pairs(output_path)
    classes = data["label"].drop_duplicates().tolist()

    return data, classes


def unzip_data(file_path, output_path):
    zip_data = zipfile.ZipFile(file_path)
    zip_data.extractall(output_path)
    zip_data.close()


def get_image_path_and_label_pairs(data_path):
    image_path_list = list(Path(data_path).glob("*/*.jpg"))
    print(f'Total Images = {len(image_path_list)}')

    classes = os.listdir(Path(data_path))
    classes = sorted(classes)
    print("classes: ", classes)
    for c in classes:
        total_images_class = list(Path(os.path.join(Path(data_path), c)).glob("*.jpg"))
        print(f"* {c}: {len(total_images_class)} images")

    images_path = [None] * len(image_path_list)
    labels = [None] * len(image_path_list)

    for i, image_path in enumerate(image_path_list):
        images_path[i] = image_path
        labels[i] = image_path.parent.stem

    df_path_and_label = pd.DataFrame({'path': images_path,
                                      'label': labels})
    return df_path_and_label.sample(frac=1, random_state=123)
