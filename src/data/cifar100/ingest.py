from __future__ import annotations

import io
import json
import pickle
import tarfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


CIFAR100_HF_REPO = "uoft-cs/cifar100"
CIFAR100_HF_REVISION = "refs/convert/parquet"
CIFAR100_URL = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
CACHE_VERSION = "1"
EXPECTED_TAR_BYTES = 169_000_000


class CIFAR100Ingestor:
    """Downloads, extracts, and processes CIFAR-100 into serialized tensors.

    Data layout::

        root/
            raw/
                cifar-100-python.tar.gz
                cifar-100-python/
                    train   (pickle)
                    test    (pickle)
                    meta    (pickle)
            processed/
                train_images.pt   [N, 32, 32, 3]  uint8  NHWC
                train_targets.pt  [N]              int64
                test_images.pt    [10000, 32, 32, 3]  uint8  NHWC
                test_targets.pt   [10000]          int64
                val_images.pt     [M, 32, 32, 3]  uint8  NHWC   (if val_split > 0)
                val_targets.pt    [M]              int64
                metadata.json
                cache_key
    """

    def __init__(
        self,
        root: str = "./data/cifar100",
        val_split: float = 0.0,
        seed: int = 13,
    ) -> None:
        self.root = Path(root).resolve()
        self.raw_dir = self.root / "raw"
        self.processed_dir = self.root / "processed"
        self.val_split = val_split
        self._generator = torch.Generator().manual_seed(seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_cached(self) -> bool:
        if not self.processed_dir.is_dir():
            return False

        key_file = self.processed_dir / "cache_key"
        if not key_file.is_file() or key_file.read_text().strip() != CACHE_VERSION:
            return False

        required = [
            self.processed_dir / "train_images.pt",
            self.processed_dir / "train_targets.pt",
            self.processed_dir / "test_images.pt",
            self.processed_dir / "test_targets.pt",
        ]
        if self.val_split > 0:
            required.extend([
                self.processed_dir / "val_images.pt",
                self.processed_dir / "val_targets.pt",
            ])

        return all(p.is_file() for p in required)

    def ingest(self, force: bool = False) -> None:
        if self.is_cached and not force:
            print(f"[CIFAR100Ingestor] Using cached data at {self.processed_dir}")
            return
        if self._try_hf_parquet():
            return
        self._download_tar()
        self._extract()
        self._process_from_pickles()

    # ------------------------------------------------------------------
    # Hugging Face parquet path (fast CDN — needs huggingface_hub + pyarrow)
    # ------------------------------------------------------------------

    def _try_hf_parquet(self) -> bool:
        try:
            from huggingface_hub import hf_hub_download, list_repo_files
        except ImportError:
            return False

        try:
            import pandas as pd
        except ImportError:
            return False

        # Check for a parquet engine without importing at module level
        parquet_engine: str | None = None
        for eng in ("pyarrow", "fastparquet"):
            try:
                __import__(eng)
                parquet_engine = eng
                break
            except ImportError:
                continue
        if parquet_engine is None:
            return False

        print(f"[CIFAR100Ingestor] Downloading parquet from Hugging Face ({CIFAR100_HF_REPO}) ...")

        train_rel = "cifar100/train/0000.parquet"
        test_rel = "cifar100/test/0000.parquet"

        train_path = hf_hub_download(
            repo_id=CIFAR100_HF_REPO,
            filename=train_rel,
            repo_type="dataset",
            revision=CIFAR100_HF_REVISION,
        )
        test_path = hf_hub_download(
            repo_id=CIFAR100_HF_REPO,
            filename=test_rel,
            repo_type="dataset",
            revision=CIFAR100_HF_REVISION,
        )

        df_train = pd.read_parquet(train_path, engine=parquet_engine)
        df_test = pd.read_parquet(test_path, engine=parquet_engine)

        self._process_from_hf_dataframe(df_train, df_test)
        return True

    def _hf_img_to_numpy(self, raw: object) -> np.ndarray:
        """Decode an image cell from a Hugging Face parquet file.

        The cell may be:
        * bytes (raw JPEG/PNG)
        * a dict with ``{"bytes": …, "path": …}``
        * a NumPy array (already decoded)
        """
        if isinstance(raw, np.ndarray):
            return raw
        if isinstance(raw, dict):
            raw = raw.get("bytes") or raw.get("path")
        if isinstance(raw, str):
            raw = Path(raw).read_bytes()
        return np.array(Image.open(io.BytesIO(raw)))

    def _process_from_hf_dataframe(self, df_train, df_test) -> None:
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        print("[CIFAR100Ingestor] Converting Hugging Face parquet to tensors ...")

        train_imgs = torch.from_numpy(
            np.stack(
                [self._hf_img_to_numpy(img) for img in tqdm(df_train["img"], desc="Train images")],
                axis=0,
            )
        )
        test_imgs = torch.from_numpy(
            np.stack(
                [self._hf_img_to_numpy(img) for img in tqdm(df_test["img"], desc="Test images")],
                axis=0,
            )
        )
        # Images from PIL are uint8 NHWC already.
        if train_imgs.ndim == 4 and train_imgs.shape[-1] not in (1, 3):
            train_imgs = train_imgs.permute(0, 2, 3, 1)

        train_targets = torch.tensor(df_train["fine_label"].values, dtype=torch.long)
        test_targets = torch.tensor(df_test["fine_label"].values, dtype=torch.long)

        val_imgs, val_targets = None, None
        if self.val_split > 0:
            train_imgs, train_targets, val_imgs, val_targets = (
                self._stratified_split(train_imgs, train_targets)
            )

        torch.save(train_imgs, self.processed_dir / "train_images.pt")
        torch.save(train_targets, self.processed_dir / "train_targets.pt")
        torch.save(test_imgs, self.processed_dir / "test_images.pt")
        torch.save(test_targets, self.processed_dir / "test_targets.pt")

        if self.val_split > 0 and val_imgs is not None:
            torch.save(val_imgs, self.processed_dir / "val_images.pt")
            torch.save(val_targets, self.processed_dir / "val_targets.pt")

        metadata: dict[str, Any] = {
            "num_train": len(train_targets),
            "num_test": len(test_targets),
            "num_classes": 100,
            "image_shape": list(train_imgs.shape[1:]),
            "val_split": self.val_split,
            "cache_version": CACHE_VERSION,
            "source": "huggingface_parquet",
        }
        if self.val_split > 0 and val_targets is not None:
            metadata["num_val"] = len(val_targets)

        with open(self.processed_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        self.processed_dir.joinpath("cache_key").write_text(CACHE_VERSION)

        parts = [f"train={metadata['num_train']}"]
        if self.val_split > 0:
            parts.append(f"val={metadata.get('num_val', 0)}")
        parts.append(f"test={metadata['num_test']}")
        print(f"[CIFAR100Ingestor] Done — {', '.join(parts)}")

    # ------------------------------------------------------------------
    # tar.gz path (fallback)
    # ------------------------------------------------------------------

    def _download_tar(self) -> None:
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        dest = self.raw_dir / "cifar-100-python.tar.gz"

        if dest.is_file() and dest.stat().st_size > EXPECTED_TAR_BYTES * 0.9:
            return

        print(f"[CIFAR100Ingestor] Downloading from {CIFAR100_URL}")
        self._download_stream(CIFAR100_URL, dest)

        actual = dest.stat().st_size
        if actual < EXPECTED_TAR_BYTES * 0.9:
            raise RuntimeError(
                f"Downloaded file too small: {actual} bytes "
                f"(expected ~{EXPECTED_TAR_BYTES})"
            )

    def _extract(self) -> None:
        extract_dir = self.raw_dir / "cifar-100-python"
        if extract_dir.is_dir():
            return

        tar_path = self.raw_dir / "cifar-100-python.tar.gz"
        if not tar_path.is_file():
            raise FileNotFoundError(f"Archive not found: {tar_path}")

        print("[CIFAR100Ingestor] Extracting archive ...")
        extract_kwargs = {"path": self.raw_dir}
        if hasattr(tarfile, "data_filter"):
            extract_kwargs["filter"] = "data"
        with tarfile.open(tar_path, "r:gz") as tar:
            members = tar.getmembers()
            for member in tqdm(members, desc="Extracting", unit="file"):
                tar.extract(member, **extract_kwargs)

    def _process_from_pickles(self) -> None:
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        extract_dir = self.raw_dir / "cifar-100-python"
        if not extract_dir.is_dir():
            raise FileNotFoundError(f"Extracted data not found: {extract_dir}")

        print("[CIFAR100Ingestor] Processing CIFAR-100 tensors from pickle ...")

        train_dict = self._load_pickle(extract_dir / "train")
        test_dict = self._load_pickle(extract_dir / "test")
        meta_dict = self._load_pickle(extract_dir / "meta")

        train_images, train_targets = self._parse_batch(train_dict)
        test_images, test_targets = self._parse_batch(test_dict)

        val_images, val_targets = None, None
        if self.val_split > 0:
            train_images, train_targets, val_images, val_targets = (
                self._stratified_split(train_images, train_targets)
            )

        torch.save(train_images, self.processed_dir / "train_images.pt")
        torch.save(train_targets, self.processed_dir / "train_targets.pt")
        torch.save(test_images, self.processed_dir / "test_images.pt")
        torch.save(test_targets, self.processed_dir / "test_targets.pt")

        if self.val_split > 0 and val_images is not None:
            torch.save(val_images, self.processed_dir / "val_images.pt")
            torch.save(val_targets, self.processed_dir / "val_targets.pt")

        metadata: dict[str, Any] = {
            "num_train": len(train_targets),
            "num_test": len(test_targets),
            "num_classes": 100,
            "image_shape": list(train_images.shape[1:]),
            "val_split": self.val_split,
            "cache_version": CACHE_VERSION,
            "fine_label_names": [n.decode() for n in meta_dict[b"fine_label_names"]],
            "coarse_label_names": [n.decode() for n in meta_dict[b"coarse_label_names"]],
            "source": "toronto_tar",
        }
        if self.val_split > 0 and val_targets is not None:
            metadata["num_val"] = len(val_targets)

        with open(self.processed_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        self.processed_dir.joinpath("cache_key").write_text(CACHE_VERSION)

        parts = [f"train={metadata['num_train']}"]
        if self.val_split > 0:
            parts.append(f"val={metadata.get('num_val', 0)}")
        parts.append(f"test={metadata['num_test']}")
        print(f"[CIFAR100Ingestor] Done — {', '.join(parts)}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _download_stream(url: str, dest: Path) -> None:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (compatible; Ghost-Bank/0.1.0)"},
        )
        last_err: Exception | None = None
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=120) as response:
                    total = int(response.headers.get("Content-Length", 0))
                    block = 8192
                    with open(dest, "wb") as f, tqdm(
                        total=total,
                        unit="B",
                        unit_scale=True,
                        desc="Downloading",
                        miniters=1,
                    ) as pbar:
                        while True:
                            chunk = response.read(block)
                            if not chunk:
                                break
                            f.write(chunk)
                            pbar.update(len(chunk))
                return
            except (urllib.error.URLError, OSError) as exc:
                last_err = exc
                print(f"  Attempt {attempt + 1}/3 failed: {exc}")
        raise RuntimeError(
            f"Download failed after 3 attempts: {last_err}"
        )

    @staticmethod
    def _load_pickle(path: Path) -> dict:
        with open(path, "rb") as f:
            return pickle.load(f, encoding="bytes")

    @staticmethod
    def _parse_batch(batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        data = batch[b"data"]
        labels = batch.get(b"fine_labels", batch.get(b"labels"))
        if labels is None:
            raise KeyError("No label key found in batch (expected 'fine_labels')")

        images = (
            torch.from_numpy(data)
            .reshape(-1, 3, 32, 32)
            .permute(0, 2, 3, 1)
            .contiguous()
        )
        targets = torch.tensor(labels, dtype=torch.long)
        return images, targets

    def _stratified_split(
        self,
        images: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        classes = torch.unique(targets)
        train_idx, val_idx = [], []
        for c in classes:
            mask = targets == c
            idx = mask.nonzero(as_tuple=False).squeeze(-1)
            perm = idx[torch.randperm(len(idx), generator=self._generator)]
            n_val = max(1, int(len(idx) * self.val_split))
            val_idx.append(perm[:n_val])
            train_idx.append(perm[n_val:])

        train_i = torch.cat(train_idx)
        val_i = torch.cat(val_idx)
        return images[train_i], targets[train_i], images[val_i], targets[val_i]


# ======================================================================
# CLI entry point
# ======================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Ingest CIFAR-100: download, extract, and cache as .pt tensors."
    )
    parser.add_argument(
        "--root", default="./data/cifar100",
        help="Data root (default: ./data/cifar100)",
    )
    parser.add_argument(
        "--val-split", type=float, default=0.0,
        help="Fraction of training data to use as validation (default: 0.0)",
    )
    parser.add_argument(
        "--seed", type=int, default=13,
        help="Random seed for val split (default: 13)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-download and re-process even if cached",
    )
    args = parser.parse_args()

    ingestor = CIFAR100Ingestor(
        root=args.root,
        val_split=args.val_split,
        seed=args.seed,
    )
    ingestor.ingest(force=args.force)
