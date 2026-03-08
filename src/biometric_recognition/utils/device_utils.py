"""Device utilities for PyTorch."""

import logging

import torch


def get_device(device_preference: str = "auto") -> torch.device:
    """Get the appropriate device for PyTorch operations.

    Args:
        device_preference: Device preference ("auto", "cpu", "cuda", "mps")

    Returns:
        torch.device: The selected device
    """
    if device_preference == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            gpu_name = torch.cuda.get_device_name(0)
            logging.info(f"Using CUDA device: {gpu_name}")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
            logging.info("Using MPS (Metal Performance Shaders) device")
        else:
            device = torch.device("cpu")
            logging.info("Using CPU device")
    else:
        device = torch.device(device_preference)
        logging.info(f"Using specified device: {device}")

    return device


def print_device_info() -> None:
    """Print information about available devices."""
    logging.info("=== Device Information ===")

    # CPU info
    logging.info(f"CPU cores available: {torch.get_num_threads()}")

    # CUDA info
    if torch.cuda.is_available():
        logging.info("CUDA available: Yes")
        logging.info(f"CUDA version: {torch.version.cuda}")
        logging.info(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logging.info(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            logging.info(f"    Memory: {memory:.1f} GB")
    else:
        logging.info("CUDA available: No")

    # MPS info
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        logging.info("MPS available: Yes")
    else:
        logging.info("MPS available: No")

    logging.info("==========================")
