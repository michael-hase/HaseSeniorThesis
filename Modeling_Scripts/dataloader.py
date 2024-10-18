import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torchio as tio


class Mri3DDataLoader(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame, classification_values: list[str, ...], augment: bool = False,
                 batch_size: int = 1, crop: bool = True, device: torch.device = torch.device("cpu")) -> None:
        """
        Initialize dataloader.

        :param df: Dataframe with columns "path" and "DX". "path" is the path to the MRI image, "DX" is the diagnosis.
        :param classification_values: List of values in the "DX" column of df that should be used for classification.
        :param augment: Augment images using TorchIO. Default is False.
        :param batch_size: Batch size.
        :param crop: Crop images 20 pixels from each side to reduce margins. Default is True.
        :param device: Device to use for tensors (CPU or GPU). Default is CPU.
        """
        self.df = df
        self.classification_values = classification_values
        self.crop = crop
        self.device = device

        self.df = self.df[self.df["DX"].isin(self.classification_values)]
        dummies = pd.get_dummies(self.df["DX"])
        self.df = pd.concat([self.df, dummies], axis=1)

        if augment:
            self.augment = tio.Compose([
                tio.RandomAffine(p=0.5),
                tio.RandomGamma(p=0.5),
                tio.RandomSwap(p=0.5),
                tio.RescaleIntensity((0, 1), p=1)
            ])
        else:
            self.augment = None

        self.batch_size = batch_size
        self.shuffle()

    def shuffle(self) -> None:
        """Shuffle dataframe."""
        self.df = self.df.sample(frac=1).reset_index(drop=True)

    def __len__(self) -> int:
        """Get number of batches."""
        return len(self.df) // self.batch_size

    def get_single_item(self, idx: int) -> tuple[torch.Tensor, np.ndarray]:
        """Get single image and label based on row number (iloc)."""
        path = self.df.iloc[idx]["path"]
        image = nib.load(path).get_fdata()
        image = image.astype(np.float32)
        if self.crop:
            image = image[10:-10, 10:-10, 10:-10]
        image = np.expand_dims(image, axis=0)
        if self.augment is not None:
            image = self.augment(image)
        y = self.df.iloc[idx][self.classification_values].values
        image = np.expand_dims(image, axis=0)
        image /= image.max()
        image = torch.from_numpy(image).to(self.device)
        return image, y.astype(np.float32)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get batch of images and labels. Returns tuple of (images, one hot encoded labels)."""
        images = []
        ys = []
        for i in range(self.batch_size):
            item_idx = idx * self.batch_size + i
            image, y = self.get_single_item(item_idx)
            images.append(image)
            ys.append(y)
        return torch.cat(images, dim=0).float().to(self.device), torch.from_numpy(np.array(ys)).to(self.device)