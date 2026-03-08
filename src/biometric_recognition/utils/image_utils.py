"""Image processing utilities."""

import base64
from typing import Tuple

import numpy as np
import torch
from PIL import Image


def image_file_to_base64(image_path: str) -> str:
    """Convert image file to base64 string.

    Consolidates duplicate functions from inference.py and examples/api_client.py.

    Args:
        image_path: Path to image file

    Returns:
        Base64 encoded image string
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def preprocess_image(
    image: Image.Image,
    target_size: Tuple[int, int],
    grayscale: bool = False,
    add_batch_dim: bool = True,
) -> torch.Tensor:
    """Preprocess image for model input.

    Args:
        image: PIL Image to preprocess
        target_size: Target size (height, width)
        grayscale: Whether to convert to grayscale
        add_batch_dim: Whether to add batch dimension

    Returns:
        Preprocessed tensor (with or without batch dimension)
    """
    # Convert to appropriate mode and resize
    mode = "L" if grayscale else "RGB"
    image = image.convert(mode).resize(target_size)

    # Convert to tensor and normalize
    image_array = np.array(image, dtype=np.float32) / 255.0

    if grayscale:
        tensor = torch.FloatTensor(image_array).unsqueeze(0)  # Add channel dimension
    else:
        tensor = torch.FloatTensor(image_array).permute(2, 0, 1)  # HWC to CHW

    if add_batch_dim:
        tensor = tensor.unsqueeze(0)  # Add batch dimension

    return tensor


def prepare_batch_from_images(
    fingerprint_img: Image.Image,
    left_iris_img: Image.Image,
    right_iris_img: Image.Image,
    fingerprint_size: Tuple[int, int] = (128, 128),
    iris_size: Tuple[int, int] = (64, 64),
    device: torch.device = None,
) -> dict:
    """Prepare a batch dictionary from PIL images.

    Args:
        fingerprint_img: Fingerprint PIL Image
        left_iris_img: Left iris PIL Image
        right_iris_img: Right iris PIL Image
        fingerprint_size: Target size for fingerprint images
        iris_size: Target size for iris images
        device: Device to move tensors to

    Returns:
        Dictionary with preprocessed tensors
    """
    # Preprocess images
    fingerprint_tensor = preprocess_image(
        fingerprint_img, fingerprint_size, grayscale=False
    )
    left_iris_tensor = preprocess_image(left_iris_img, iris_size, grayscale=True)
    right_iris_tensor = preprocess_image(right_iris_img, iris_size, grayscale=True)

    batch = {
        "fingerprint": fingerprint_tensor,
        "left_iris": left_iris_tensor,
        "right_iris": right_iris_tensor,
    }

    # Move to device if specified
    if device is not None:
        batch = {key: tensor.to(device) for key, tensor in batch.items()}

    return batch
