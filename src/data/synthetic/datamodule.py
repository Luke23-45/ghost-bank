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

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.batch_size,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.batch_size,
        )
