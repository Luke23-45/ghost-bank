from __future__ import annotations

from torch.utils.data import DataLoader

from src.data.base import BaseDataModule
from src.data.synthetic.dataset import GaussianDataset, generate_gaussian_data
from src.data.synthetic.defaults import SyntheticConfig


class SyntheticDataModule(BaseDataModule):
    def __init__(self, config: SyntheticConfig) -> None:
        super().__init__()
        self.config = config
        self.train_dataset: GaussianDataset | None = None
        self.test_dataset: GaussianDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        train_data, test_data = generate_gaussian_data(
            seed=self.config.seed,
            imbalance_ratio=self.config.imbalance_ratio,
            majority_train=self.config.majority_train,
            test_per_class=self.config.test_per_class,
        )
        self.train_dataset = GaussianDataset(train_data)
        self.test_dataset = GaussianDataset(test_data)

    def _dataloader_kwargs(self) -> dict:
        kwargs: dict = {
            "batch_size": self.config.batch_size,
            "num_workers": self.config.num_workers,
            "pin_memory": self.config.pin_memory,
        }
        if self.config.num_workers > 0:
            kwargs["persistent_workers"] = self.config.persistent_workers
            kwargs["prefetch_factor"] = self.config.prefetch_factor
        return kwargs

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            **self._dataloader_kwargs(),
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            **self._dataloader_kwargs(),
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            **self._dataloader_kwargs(),
        )
