import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import gdown
from sklearn.model_selection import train_test_split
from PIL import Image
from torchvision.models import vit_b_16, ViT_B_16_Weights

from src.data.preprocess.clean_data import get_formatted_data
from src.consts import SEED


class FaceImagesDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transforms, label_map: dict):
        self.df = df
        self.transforms = transforms
        self.label_map = label_map

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        df_new = self.df.copy()
        df_new = df_new.reset_index(drop=True)
        df_new["label"] = df_new["label"].map(self.label_map)
        image_path = df_new.iloc[idx, 0]
        image = Image.open(image_path).convert("RGB")
        image = self.transforms(image)
        label = df_new.iloc[idx, 1]

        return image, label


def download_from_drive(file_id, output_file_name):
    url = 'https://drive.google.com/uc?id=' + file_id
    output = './data' + output_file_name
    gdown.download(url, output)


def train_test_valid_split(dataset):
    train_pairs, df_rest = train_test_split(dataset,
                                            test_size=0.3,
                                            random_state=SEED,
                                            stratify=dataset["label"])

    valid_pairs, test_pairs = train_test_split(df_rest,
                                               test_size=0.5,
                                               random_state=SEED,
                                               stratify=df_rest["label"])

    return train_pairs, test_pairs, valid_pairs


def create_dataloaders(batch_size=100):
    download_from_drive('1sQMgIOdYb9JzMgyUeJcgxBhf7rrQ1pep', 'other-classes.zip')
    download_from_drive('12zR2REcKswslFkxnSYES5hq-_bOCssjR', 'dry-oily-normal.zip')

    data, classes = get_formatted_data()

    classes_map = dict(zip(classes, range(0, len(classes))))

    auto_transforms = ViT_B_16_Weights.DEFAULT.transforms()

    train_pairs, test_pairs, valid_pairs = train_test_valid_split(data)

    train_dataset = FaceImagesDataset(train_pairs, auto_transforms, classes_map)
    valid_dataset = FaceImagesDataset(valid_pairs, auto_transforms, classes_map)
    test_dataset = FaceImagesDataset(test_pairs, auto_transforms, classes_map)

    num_workers = os.cpu_count()

    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers)
    valid_dataloader = DataLoader(dataset=valid_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers)
    test_dataloader = DataLoader(dataset=test_dataset,
                                 batch_size=batch_size,
                                 shuffle=True,
                                 num_workers=num_workers)

    return train_dataloader, valid_dataloader, test_dataloader


if __name__ == "__main__":
    dataloaders = create_dataloaders(100)

    for batch in dataloaders:
        batch_images, batch_labels = next(iter(batch))
        print(batch_images.shape, batch_labels.shape)
