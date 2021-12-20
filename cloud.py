import sys, traceback, os
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence, List
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from PIL import Image
import rasterio
from rasterio.crs import CRS

import torch
from torch import Tensor
from torch.nn.functional import one_hot
from torch.utils.data import DataLoader, Subset
from torchgeo.datasets.geo import VisionDataset
import pytorch_lightning as pl

# https://github.com/pytorch/pytorch/issues/60979
# https://github.com/pytorch/pytorch/pull/61045
DataLoader.__module__ = "torch.utils.data"
CLOUD_BANDS = ["B04","B03","B02","B08"]

class Sentinel2CloudCover(VisionDataset):
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

    # filename_glob = "B0*.*"
    # filename_regex = """^(?P<band>B\d{2})"""
    feature_dir = "train_features"
    label_dir = "train_labels"
    n_classes = 2

    def __init__(
        self,
        root: str = "data",
        split: str = "train",
        bands: List[str] = [],
        transforms: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        download: bool = False,
        api_key: Optional[str] = None,
        cache: bool = True,
        checksum: bool = False,
    ) -> None:

        """Initialize a new Cloud Cover Detection Competition Dataset

        Args:
            root: str root directory where dataset can be found
            bands: List[str] bands to return (defaults to all bands)
            transforms: a function/transform that takes an input sample
                and returns a transformed version
            cache: if True, cache file handle to speed up repeated sampling

        Raises:

        """

        self.root = root
        self.split = split
        self.transforms = transforms
        self.checksum = checksum

        self.bands = bands if bands else CLOUD_BANDS
        self.chip_ids = []

        self._set_chip_ids()

        # placeholder for configuring download from Radiant MLHub
        # if download:    
            # self._download(api_key)

    def __len__(self):
        """Return the number of data points in the dataset.

        Returns:
            length of the dataset
        """
        return len(self.chip_ids)

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        """Return an sample based on index within the dataset.

        Args:
            idx: index to return
        Returns:
            sample with feature image and label
        """

        try:
            
            chip = self.chip_ids[index]
            # print(f"Loading chip id: {chip}\n")

            sample = {
                "image": self._load_feature(chip),
                "mask": self._load_label(chip)
                }

            sample["mask"] = sample["mask"].astype("int32")
            sample["image"] = torch.from_numpy(sample["image"])
            sample["mask"] = torch.from_numpy(sample["mask"])

            # Apply data augmentations, if provided
            if self.transforms is not None:
                sample = self.transforms(sample)

            return sample

        except Exception:

            print("Exception in code to get items by index:")
            print("-"*60)
            traceback.print_exc(file=sys.stdout)
            print("-"*60)

    def _load_feature(self, chip: str):

        chip_path = self.full_feature_path / chip
        band_arrs = []
        for band in self.bands:
            with rasterio.open(os.path.join(chip_path, f"{band}.tif")) as bimg:
                band_arr = bimg.read(1).astype("float32")
            band_arrs.append(band_arr)
        x_arr = np.stack(band_arrs, axis=-1)
        x_arr = np.transpose(x_arr, [2, 0, 1])

        return x_arr

    def _load_label(self, chip: str):

        label_path = Path(self.root) / self.label_dir / chip
        with rasterio.open(f"{label_path}.tif") as limg:
            y_arr = limg.read(1)#.astype("int32")

        return y_arr

    def _set_chip_ids(self) -> None:

        self.full_feature_path = Path(self.root) / self.feature_dir
        self.chip_ids = [x for x in os.walk(self.full_feature_path) if x[0].__contains__(self.feature_dir)][0][1]

class Sentinel2CloudCoverDataModule(pl.LightningDataModule):
    """LightningDataModule implementation for Sentinel2 Cloud Cover Dataset.

    Implements 80/20 train/val splits based on cloud cover chip ids.
    """

    def __init__(
        self,
        root_dir: str,
        seed: int,
        batch_size: int = 64,
        n_classes: int = 2,
        num_workers: int = 0,
        api_key: Optional[str] = None,
        **kwargs: Any,
    ) -> None:

        super().__init__()
        self.root_dir = root_dir
        self.seed = seed
        self.batch_size = batch_size
        self.n_classes = n_classes
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
        
        # sample["mask"] = torch.as_tensor(sample["mask"])
        # sample["mask"] = one_hot(sample["mask"], num_classes=self.n_classes)

        # sample["image"] = torch.from_numpy(sample["image"])
        # sample["mask"] = torch.from_numpy(sample["mask"])
        # sample["mask"] = sample["mask"].astype("int32")

        sample["image"] = sample["image"].float()
        sample["mask"] = sample["mask"].long()

        return sample

    def prepare_data(self) -> None:
        """Initialize the main ``Dataset`` objects for use in :func:`setup`.

        This is done once per node, while :func:`setup` is done once per GPU.
        """
        Sentinel2CloudCover(
            self.root_dir,
            split="train",
            bands=CLOUD_BANDS,
            transforms=None,
            download=False,
            checksum=False,
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
            bands=CLOUD_BANDS,
            transforms=self.custom_transform,
            download=False,
        )

        # self.all_test_dataset = Sentinel2CloudCover(
        #     self.root_dir,
        #     split="test",
        #     bands=CLOUD_BANDS,
        #     transforms=self.custom_transform,
        #     download=False,
        # )        

        train_indices, val_indices = next(
            GroupShuffleSplit(test_size=0.2, n_splits=2, random_state=self.seed).split(
                self.all_train_dataset.chip_ids, groups=self.all_train_dataset.chip_ids
            )
        )

        self.train_dataset = Subset(self.all_train_dataset, train_indices)
        self.val_dataset = Subset(self.all_train_dataset, val_indices)

        # self.test_dataset = Subset(
        #     self.all_test_dataset, range(len(self.all_test_dataset))
        # )

    # def teardown(self, stage: Optional[str] = None):
    #     pass

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

    # def test_dataloader(self) -> DataLoader[Any]:
    #     """Return a DataLoader for testing.

    #     Returns:
    #         testing data loader
    #     """
    #     return DataLoader(
    #         self.test_dataset,
    #         batch_size=self.batch_size,
    #         num_workers=self.num_workers,
    #         shuffle=False,
    #     )