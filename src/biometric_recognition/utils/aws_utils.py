"""Simplified AWS S3 utilities for biometric recognition system."""

import logging
import os
import tempfile
from pathlib import Path
from typing import Dict, Optional, Tuple

import boto3


class S3Utils:
    """S3 utility class for uploading and downloading files (singleton pattern)."""

    _instances: Dict[str, "S3Utils"] = {}

    def __new__(cls, region: str = "us-east-1"):
        if region not in cls._instances:
            cls._instances[region] = super().__new__(cls)
            cls._instances[region].s3_client = boto3.client("s3", region_name=region)
            cls._instances[region].region = region
        return cls._instances[region]

    def __init__(self, region: str = "us-east-1"):
        # Initialization already handled in __new__
        pass

    def _parse_s3_uri(self, s3_uri: str) -> Tuple[str, str]:
        """Parse S3 URI into bucket and key."""
        if not s3_uri.startswith("s3://"):
            raise ValueError(f"Invalid S3 URI: {s3_uri}")
        return s3_uri[5:].split("/", 1)

    def upload_to_s3(self, local_path: str, s3_uri: str) -> str:
        """Upload file to S3."""
        if not os.path.exists(local_path):
            raise FileNotFoundError(f"Local file not found: {local_path}")

        bucket, key = self._parse_s3_uri(s3_uri)
        self.s3_client.upload_file(local_path, bucket, key)
        logging.info(f"Uploaded {local_path} to {s3_uri}")
        return s3_uri

    def download_from_s3(self, s3_uri: str, local_path: str) -> str:
        """Download file from S3."""
        bucket, key = self._parse_s3_uri(s3_uri)
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        self.s3_client.download_file(bucket, key, local_path)
        logging.info(f"Downloaded {s3_uri} to {local_path}")
        return local_path

    def download_dataset_from_s3(self, s3_uri: str, local_dir: str) -> str:
        """Download entire dataset directory from S3."""
        bucket, prefix = self._parse_s3_uri(s3_uri)
        local_path = Path(local_dir)
        local_path.mkdir(parents=True, exist_ok=True)

        logging.info(f"Listing objects in s3://{bucket}/{prefix}...")
        paginator = self.s3_client.get_paginator("list_objects_v2")
        downloaded_count = 0

        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                obj_key = obj["Key"]
                if obj_key.endswith("/") or not (
                    relative_path := obj_key[len(prefix) :].lstrip("/")
                ):
                    continue

                local_file_path = local_path / relative_path
                local_file_path.parent.mkdir(parents=True, exist_ok=True)
                self.s3_client.download_file(bucket, obj_key, str(local_file_path))
                downloaded_count += 1

                if downloaded_count % 100 == 0:
                    logging.info(f"Downloaded {downloaded_count} files...")

        if downloaded_count == 0:
            raise RuntimeError(f"No files found at S3 URI: {s3_uri}")

        logging.info(
            f"Downloaded {downloaded_count} files from {s3_uri} to {local_dir}"
        )
        return str(local_path)


def get_data_path(
    path: str, cache_dir: Optional[str] = None, aws_region: str = "us-east-1"
) -> str:
    """Get data path - download from S3 if needed, return local path otherwise."""
    if not path.startswith("s3://"):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Local data path not found: {path}")
        logging.info(f"Using local data path: {path}")
        return path

    logging.info(f"S3 data path detected: {path}")
    s3_utils = S3Utils(aws_region)
    bucket, key = s3_utils._parse_s3_uri(path)

    cache_path = (
        Path(cache_dir or tempfile.gettempdir()) / "biometric_cache" / bucket / key
    )

    if cache_path.exists() and any(cache_path.iterdir()):
        logging.info(f"Using cached S3 dataset: {cache_path}")
        return str(cache_path)

    logging.info(f"Downloading dataset from S3 to: {cache_path}")
    s3_utils.download_dataset_from_s3(path, str(cache_path))
    return str(cache_path)
