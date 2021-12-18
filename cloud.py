from typing import Any, Callable, Dict, Optional, Sequence
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from PIL import Image
from rasterio.crs import CRS

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Subset
from torchgeo.datasets import Sentinel2
import pytorch_lightning as pl

# https://github.com/pytorch/pytorch/issues/60979
# https://github.com/pytorch/pytorch/pull/61045
DataLoader.__module__ = "torch.utils.data"

class Sentinel2CloudCover(Sentinel2):
    """Sentinel2 Cloud Cover Detection Competition dataset.

    ** NOTE: check if this is the description to be used **

    Satellite imagery is critical for a wide variety of applications from disaster 
    management and recovery, to agriculture, to military intelligence. A major 
    obstacle for all of these use cases is the presence of clouds, which cover 
    over 66% of the Earth's surface (Xie et al, 2020). Clouds introduce noise and 
    inaccuracy in image-based models, and usually have to be identified and removed. 
    Improving methods of identifying clouds can unlock the potential of an unlimited 
    range of satellite imagery use cases, enabling faster, more efficient, 
    and more accurate image-based research.

    See https://www.drivendata.org/competitions/83/cloud-cover/ for mor information about the competition.

    ** FUTURE NOTE: add details about radiant-mlhub and API_KEY **

    """

    filename_glob = "B02.*"
    filename_regex = """^(?P<band>B\d{2})"""
    bands = ["B02","B03","B04","B08"]

    def __init__(
        self,
        x_paths: pd.DataFrame,
        root: str = "data",
        split: str = "train",
        bands: Sequence[str] = [],
        y_paths: Optional[pd.DataFrame] = None,
        crs: Optional[CRS] = None,
        res: Optional[float] = None,
        transforms: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        download: bool = False,
        api_key: Optional[str] = None,
        cache: bool = True,
        checksum: bool = False,
    ) -> None:

        """Initialize a new Cloud Cover Detection Competition Dataset

        Args:
            root: root directory where dataset can be found
            crs: :term:`coordinate reference system (CRS)` to warp to
                (defaults to the CRS of the first file found)
            res: resolution of the dataset in units of CRS
                (defaults to the resolution of the first file found)
            bands: bands to return (defaults to all bands)
            transforms: a function/transform that takes an input sample
                and returns a transformed version
            cache: if True, cache file handle to speed up repeated sampling

        Raises:

        """
        
        super().__init__(root, crs, res, bands, transforms, cache)

        self.split = split
        self.data = x_paths
        self.label = y_paths
        self.checksum = checksum
        self.chip_ids = []

        # placeholder for configuring download from Radiant MLHub
        # if download:    
            # self._download(api_key)

    def __len__(self):
        """Return the number of data points in the dataset.

        Returns:
            length of the dataset
        """
        return len(self.data)

    def __getitem__(self, idx: int):
        """Return an index within the dataset.

        Args:
            idx: index to return
        Returns:
            data, labels, field ids, and metadata at that index
        """

        # Loads an n-channel image from a chip-level dataframe
        img = self.data.loc[idx]
        band_arrs = []
        for band in self.bands:
            with rasterio.open(img[f"{band}_path"]) as b:
                band_arr = b.read(1).astype("float32")
            band_arrs.append(band_arr)
        x_arr = np.stack(band_arrs, axis=-1)

        # Apply data augmentations, if provided
        if self.transforms:
            x_arr = self.transforms(image=x_arr)["image"]
        x_arr = np.transpose(x_arr, [2, 0, 1])

        # Prepare dictionary for item
        item = {
            "chip_id": img.chip_id, 
            "chip": x_arr
            }
        chip_ids.append(item["chip_id"])

        # Load label if available
        if self.label is not None:
            label_path = self.label.loc[idx].label_path
            with rasterio.open(label_path) as lp:
                y_arr = lp.read(1).astype("float32")
            # Apply same data augmentations to the label
            if self.transforms:
                y_arr = self.transforms(image=y_arr)["image"]
            item["label"] = y_arr

        return item

class Sentinel2CloudCoverDataModule(pl.LightningDataModule):
    """LightningDataModule implementation for Sentinel2 Cloud Cover Dataset.

    Implements 80/20 train/val splits based on cloud cover chip ids.
    """

    def __init__(
        self,
        root_dir: str,
        seed: int,
        batch_size: int = 64,
        num_workers: int = 0,
        api_key: Optional[str] = None,
        **kwargs: Any,
    ) -> None:

        super().__init__()
        self.root_dir = root_dir
        self.seed = seed
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.api_key = api_key

    def custom_transform(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Transform a single sample from the Dataset.

        Args:
            sample: dictionary containing image and target

        Returns:
            preprocessed sample
        """

        # Add any code to preprocess the images here

        return sample

    def prepare_data(self) -> None:
        """Initialize the main ``Dataset`` objects for use in :func:`setup`.

        This includes optionally downloading the dataset. This is done once per node,
        while :func:`setup` is done once per GPU.
        """
        Sentinel2CloudCover(
            self.root_dir, 
            bands=
            split="train",
            # checksum=False
            )

    def setup(self, stage: Optional[str] = None) -> None:
        """Create the train/val/test splits based on the original Dataset objects.

        The splits should be done here vs. in :func:`__init__` per the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html#setup.

        We split samples between train/val by the ``chip_id`` property. I.e. all
        samples with the same ``chip_id`` value will be either in the train or the val
        split. This is important to test one type of generalizability -- given a new
        storm, can we predict its windspeed. The test set, however, contains *some*
        storms from the training set (specifically, the latter parts of the storms) as
        well as some novel storms.

        Args:
            stage: stage to set up
        """

        self.all_train_dataset = Sentinel2CloudCover(
            self.root_dir,
            split="train",
            transforms=self.custom_transform,
            download=False,
        )

        self.all_test_dataset = Sentinel2CloudCover(
            self.root_dir,
            split="test",
            transforms=self.custom_transform,
            download=False,
        )        

        train_indices, val_indices = next(
            GroupShuffleSplit(test_size=0.2, n_splits=2, random_state=self.seed).split(
                self.all_train_dataset.chip_ids, groups=self.all_train_dataset.chip_ids
            )
        )

        self.train_dataset = Subset(self.all_train_dataset, train_indices)
        self.val_dataset = Subset(self.all_train_dataset, val_indices)
        self.test_dataset = Subset(
            self.all_test_dataset, range(len(self.all_test_dataset))
        )

    def train_dataloader(self) -> DataLoader[Any]:
        """Return a DataLoader for training.

        Returns:
            training data loader
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Return a DataLoader for validation.

        Returns:
            validation data loader
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Return a DataLoader for testing.

        Returns:
            testing data loader
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )