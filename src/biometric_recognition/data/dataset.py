"""Dataset module for loading biometric data."""

import logging
import os
from typing import List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset

from biometric_recognition.utils.aws_utils import get_data_path
from biometric_recognition.utils.image_utils import preprocess_image


class BiometricDataset(Dataset):
    """Dataset class for multimodal biometric data (fingerprints + iris)."""

    def __init__(
        self,
        data_path: str,
        num_people: int = 45,
        fingerprint_size: Tuple[int, int] = (128, 128),
        iris_size: Tuple[int, int] = (64, 64),
        preload: bool = True,
        config: Optional[dict] = None,
    ):
        """Initialize the dataset.

        Args:
            data_path: Local path (e.g., "data/") or S3 URI (e.g., "s3://bucket/path/")
            num_people: Number of people in the dataset
            fingerprint_size: Target size for fingerprint images
            iris_size: Target size for iris images
            preload: Whether to preload all images into memory (faster training)
            config: Configuration dictionary for cache_dir and AWS region
        """
        self.num_people = num_people
        self.fingerprint_size = fingerprint_size
        self.iris_size = iris_size
        self.preload = preload

        # Get the appropriate data path (handles S3 download if needed)
        if config:
            cache_dir = config.get("data", {}).get("cache_dir")
            aws_region = config.get("aws", {}).get("region", "us-east-1")
            self.data_path = get_data_path(data_path, cache_dir, aws_region)
        else:
            # Fallback - assume local path
            self.data_path = data_path

        self.samples = self._load_samples()
        logging.info(f"Found {len(self.samples)} samples from {num_people} people")

        if self.preload:
            logging.info("Pre-loading all images into memory...")
            self._preload_images()
            logging.info("Pre-loading completed!")

    def _load_samples(self) -> List[dict]:
        """Load all available samples from the dataset."""
        samples = []

        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Dataset path does not exist: {self.data_path}")

        for person_id in range(1, self.num_people + 1):
            person_path = os.path.join(self.data_path, str(person_id))
            if not os.path.exists(person_path):
                logging.warning(f"Directory for person {person_id} does not exist")
                continue

            # Get ALL available images for each modality
            fingerprint_paths = self._find_all_images(person_path, "Fingerprint")
            left_iris_paths = self._find_all_images(person_path, "left")
            right_iris_paths = self._find_all_images(person_path, "right")

            if all([fingerprint_paths, left_iris_paths, right_iris_paths]):
                # Create all possible combinations of the available images
                for fp_path in fingerprint_paths:
                    for left_path in left_iris_paths:
                        for right_path in right_iris_paths:
                            samples.append(
                                {
                                    "person_id": person_id - 1,  # Zero-based indexing
                                    "fingerprint_path": fp_path,
                                    "left_iris_path": left_path,
                                    "right_iris_path": right_path,
                                }
                            )

                num_samples = (
                    len(fingerprint_paths)
                    * len(left_iris_paths)
                    * len(right_iris_paths)
                )
                logging.info(
                    f"Person {person_id}: {len(fingerprint_paths)} fingerprints × "
                    f"{len(left_iris_paths)} left iris × {len(right_iris_paths)} "
                    f"right iris = {num_samples} samples"
                )
            else:
                logging.warning(f"Missing modality files for person {person_id}")

        return samples

    def _preload_images(self) -> None:
        """Pre-load all images into memory for faster training."""
        for i, sample in enumerate(self.samples):
            # Load images using shared utility
            fingerprint = preprocess_image(
                Image.open(sample["fingerprint_path"]),
                self.fingerprint_size,
                grayscale=False,
                add_batch_dim=False,
            )
            left_iris = preprocess_image(
                Image.open(sample["left_iris_path"]),
                self.iris_size,
                grayscale=True,
                add_batch_dim=False,
            )
            right_iris = preprocess_image(
                Image.open(sample["right_iris_path"]),
                self.iris_size,
                grayscale=True,
                add_batch_dim=False,
            )

            # Store in sample
            sample["fingerprint_tensor"] = fingerprint
            sample["left_iris_tensor"] = left_iris
            sample["right_iris_tensor"] = right_iris

            if (i + 1) % 10 == 0:
                logging.info(f"Pre-loaded {i + 1}/{len(self.samples)} samples")

    def _find_all_images(self, person_path: str, modality: str) -> List[str]:
        """Find ALL available images for a given modality."""
        modality_path = os.path.join(person_path, modality)
        if not os.path.exists(modality_path):
            return []

        all_images = []
        # Look for all supported image formats
        extensions = [".bmp", ".BMP", ".jpg", ".jpeg", ".png", ".tiff"]

        for ext in extensions:
            files = [f for f in os.listdir(modality_path) if f.endswith(ext)]
            for file in sorted(files):  # Sort for consistent ordering
                full_path = os.path.join(modality_path, file)
                all_images.append(full_path)

        return all_images

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        """Get a sample from the dataset."""
        sample = self.samples[idx]

        if self.preload and "fingerprint_tensor" in sample:
            # Use pre-loaded images
            fingerprint = sample["fingerprint_tensor"]
            left_iris = sample["left_iris_tensor"]
            right_iris = sample["right_iris_tensor"]
        else:
            # Load images on-demand using shared utility
            fingerprint = preprocess_image(
                Image.open(sample["fingerprint_path"]),
                self.fingerprint_size,
                grayscale=False,
                add_batch_dim=False,
            )
            left_iris = preprocess_image(
                Image.open(sample["left_iris_path"]),
                self.iris_size,
                grayscale=True,
                add_batch_dim=False,
            )
            right_iris = preprocess_image(
                Image.open(sample["right_iris_path"]),
                self.iris_size,
                grayscale=True,
                add_batch_dim=False,
            )

        return {
            "fingerprint": fingerprint,
            "left_iris": left_iris,
            "right_iris": right_iris,
            "label": torch.tensor(sample["person_id"], dtype=torch.long),
            "person_id": sample["person_id"],
        }
