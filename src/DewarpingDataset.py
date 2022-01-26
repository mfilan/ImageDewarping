import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torch import is_tensor

class DewarpingDataset(Dataset):
    """Dewarping images dataset."""

    def __init__(self, csv_file, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.images_frame = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.images_frame)

    def __getitem__(self, idx):
        if is_tensor(idx):
            idx = idx.tolist()

        image = Image.open(self.images_frame.iloc[idx, 0])
        original_image = Image.open(self.images_frame.iloc[idx, 1])
        if self.transform:
            image = self.transform(image)
            original_image = self.transform(original_image)
        return image,original_image