from __future__ import annotations

from abc import ABC

from pytorch_lightning import LightningDataModule


class BaseDataModule(ABC, LightningDataModule):
    pass
