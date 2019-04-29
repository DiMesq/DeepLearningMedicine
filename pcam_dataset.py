import pandas as pd
import numpy as np
from skimage import io
from torch.utils.data import Dataset


class PcamDataset(Dataset):

    def __init__(self, images_path, labels_path, transforms=None, negative_only=False):
        self.images_path = images_path
        self.transforms = transforms

        df_labels = pd.read_csv(labels_path)
        self.labels = df_labels['label'].values
        self.image_ids = df_labels['id'].values

        if negative_only:
            labels = []
            images = []
            for i in range(len(self.labels)):
                if self.labels[i] == 0:
                    labels.append(self.labels[i])
                    images.append(self.image_ids[i])
            self.labels = np.array(labels)
            self.image_ids = np.array(images)
    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image = io.imread(f'{self.images_path}/{self.image_ids[idx]}.tif')
        if self.transforms:
            image = self.transforms(image)
        return image, self.labels[idx]
