"""Ray-based dataset module for distributed biometric data loading."""

import logging
import os
from typing import List, Optional, Tuple

import ray
import torch
from PIL import Image
from torch.utils.data import Dataset

from biometric_recognition.utils.image_utils import preprocess_image


# Ray remote functions for distributed processing
@ray.remote
def load_image_remote(
    image_path: str, target_size: Tuple[int, int], grayscale: bool = False
) -> torch.Tensor:
    """Load and preprocess a single image on a Ray worker."""
    from PIL import Image

    return preprocess_image(
        Image.open(image_path), target_size, grayscale, add_batch_dim=False
    )


@ray.remote
def load_sample_images_remote(
    sample_data: dict, fingerprint_size: Tuple[int, int], iris_size: Tuple[int, int]
) -> dict:
    """Load all images for a single sample on a Ray worker."""
    # Load fingerprint
    fingerprint = load_image_remote.remote(
        sample_data["fingerprint_path"], fingerprint_size, grayscale=False
    )

    # Load iris images
    left_iris = load_image_remote.remote(
        sample_data["left_iris_path"], iris_size, grayscale=True
    )

    right_iris = load_image_remote.remote(
        sample_data["right_iris_path"], iris_size, grayscale=True
    )

    # Wait for all images to load
    fingerprint_tensor = ray.get(fingerprint)
    left_iris_tensor = ray.get(left_iris)
    right_iris_tensor = ray.get(right_iris)

    return {
        "person_id": sample_data["person_id"],
        "fingerprint_tensor": fingerprint_tensor,
        "left_iris_tensor": left_iris_tensor,
        "right_iris_tensor": right_iris_tensor,
        "fingerprint_path": sample_data["fingerprint_path"],
        "left_iris_path": sample_data["left_iris_path"],
        "right_iris_path": sample_data["right_iris_path"],
    }


@ray.remote
def find_images_for_person_remote(data_path: str, person_id: int) -> Optional[dict]:
    """Find all images for a person using Ray distributed processing."""
    person_path = os.path.join(data_path, str(person_id))
    if not os.path.exists(person_path):
        return None

    def find_image(person_path: str, modality: str) -> Optional[str]:
        modality_path = os.path.join(person_path, modality)
        if not os.path.exists(modality_path):
            return None

        extensions = [".bmp", ".BMP", ".jpg", ".jpeg", ".png", ".tiff"]
        for ext in extensions:
            files = [f for f in os.listdir(modality_path) if f.endswith(ext)]
            if files:
                return os.path.join(modality_path, files[0])
        return None

    # Find all required images
    fingerprint_path = find_image(person_path, "Fingerprint")
    left_iris_path = find_image(person_path, "left")
    right_iris_path = find_image(person_path, "right")

    if all([fingerprint_path, left_iris_path, right_iris_path]):
        return {
            "person_id": person_id - 1,  # Zero-based indexing
            "fingerprint_path": fingerprint_path,
            "left_iris_path": left_iris_path,
            "right_iris_path": right_iris_path,
        }
    return None


class RayBiometricDataset(Dataset):
    """Ray-enabled dataset for distributed multimodal biometric data processing."""

    def __init__(
        self,
        data_path: str,
        num_people: int = 45,
        fingerprint_size: Tuple[int, int] = (128, 128),
        iris_size: Tuple[int, int] = (64, 64),
        preload: bool = True,
        ray_workers: int = None,
    ):
        """Initialize the Ray-based dataset.

        Args:
            data_path: Path to the dataset directory
            num_people: Number of people in the dataset
            fingerprint_size: Target size for fingerprint images
            iris_size: Target size for iris images
            preload: Whether to preload all images into memory using Ray
            ray_workers: Number of Ray workers to use (None = auto-detect)
        """
        self.data_path = data_path
        self.num_people = num_people
        self.fingerprint_size = fingerprint_size
        self.iris_size = iris_size
        self.preload = preload

        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init(num_cpus=ray_workers, logging_level=logging.ERROR)
            cpus = ray.cluster_resources().get("CPU", "auto")
            logging.info(f"Ray initialized with {cpus} CPUs")

        # Load samples using Ray for distributed file discovery
        self.samples = self._load_samples_distributed()
        logging.info(f"Found {len(self.samples)} samples from {num_people} people")

        if self.preload:
            logging.info("Pre-loading all images using Ray distributed processing...")
            self._preload_images_distributed()
            logging.info("Ray pre-loading completed!")

    def _load_samples_distributed(self) -> List[dict]:
        """Load all available samples using Ray distributed processing."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Dataset path does not exist: {self.data_path}")

        # Launch distributed tasks to find images for each person
        logging.info("Launching distributed file discovery...")
        futures = []
        for person_id in range(1, self.num_people + 1):
            future = find_images_for_person_remote.remote(self.data_path, person_id)
            futures.append(future)

        # Collect results
        logging.info("Collecting file discovery results...")
        results = ray.get(futures)

        # Filter out None results
        samples = [sample for sample in results if sample is not None]

        # Log any missing samples
        missing_count = len([r for r in results if r is None])
        if missing_count > 0:
            logging.warning(f"Missing files for {missing_count} people")

        return samples

    def _preload_images_distributed(self) -> None:
        """Pre-load all images using Ray distributed processing."""
        if len(self.samples) == 0:
            return

        # Launch distributed image loading tasks
        logging.info("Launching distributed image loading...")
        futures = []
        for sample in self.samples:
            future = load_sample_images_remote.remote(
                sample, self.fingerprint_size, self.iris_size
            )
            futures.append(future)

        # Process results in batches to avoid overwhelming memory
        batch_size = max(1, len(futures) // 4)  # Process in 4 batches

        for i in range(0, len(futures), batch_size):
            batch_futures = futures[i : i + batch_size]
            batch_results = ray.get(batch_futures)

            # Update samples with loaded tensors
            for j, loaded_sample in enumerate(batch_results):
                original_idx = i + j
                self.samples[original_idx].update(
                    {
                        "fingerprint_tensor": loaded_sample["fingerprint_tensor"],
                        "left_iris_tensor": loaded_sample["left_iris_tensor"],
                        "right_iris_tensor": loaded_sample["right_iris_tensor"],
                    }
                )

            batch_num = i // batch_size + 1
            logging.info(
                f"Ray pre-loaded batch {batch_num}/4 ({len(batch_results)} samples)"
            )

    def _load_image(
        self, image_path: str, target_size: Tuple[int, int], grayscale: bool = False
    ) -> torch.Tensor:
        """Load and preprocess an image (fallback for non-Ray loading)."""
        return preprocess_image(
            Image.open(image_path), target_size, grayscale, add_batch_dim=False
        )

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        """Get a sample from the dataset."""
        sample = self.samples[idx]

        if self.preload and "fingerprint_tensor" in sample:
            # Use Ray pre-loaded images
            fingerprint = sample["fingerprint_tensor"]
            left_iris = sample["left_iris_tensor"]
            right_iris = sample["right_iris_tensor"]
        else:
            # Fallback to regular loading (could also use Ray here)
            fingerprint = self._load_image(
                sample["fingerprint_path"], self.fingerprint_size, grayscale=False
            )
            left_iris = self._load_image(
                sample["left_iris_path"], self.iris_size, grayscale=True
            )
            right_iris = self._load_image(
                sample["right_iris_path"], self.iris_size, grayscale=True
            )

        return {
            "fingerprint": fingerprint,
            "left_iris": left_iris,
            "right_iris": right_iris,
            "label": torch.LongTensor([sample["person_id"]]).squeeze(),
            "person_id": sample["person_id"],
        }

    def __del__(self):
        """Cleanup Ray resources when dataset is destroyed."""
        if ray.is_initialized():
            ray.shutdown()


# Alternative: Ray Data API approach (more modern)
class RayDataBiometricDataset:
    """Modern Ray Data API approach for biometric dataset processing."""

    def __init__(self, data_path: str, num_people: int = 45, **kwargs):
        import ray.data

        # Create Ray dataset from file paths
        self.ray_dataset = ray.data.read_images(data_path, mode="RGB")

        # Apply distributed preprocessing
        self.processed_dataset = self.ray_dataset.map_batches(
            self._preprocess_batch,
            batch_format="numpy",
            num_cpus=0.5,  # Use half CPU per task
        )

    def _preprocess_batch(self, batch):
        """Preprocess a batch of images using Ray Data."""
        # This would contain the image preprocessing logic
        # Applied distributedly across Ray workers
        pass

    def to_torch(self):
        """Convert to PyTorch dataset."""
        return self.processed_dataset.to_torch(
            label_column="person_id",
            feature_columns=["fingerprint", "left_iris", "right_iris"],
        )
