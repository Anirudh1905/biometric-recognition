"""Model utilities for creation, loading, saving and inference."""

import logging
from pathlib import Path
from typing import Any, Dict

import torch
from omegaconf import DictConfig

from biometric_recognition.models import MultimodalBiometricModel
from biometric_recognition.utils.device_utils import get_device


def create_model(cfg: DictConfig, device: torch.device) -> MultimodalBiometricModel:
    """Create model from config.

    Args:
        cfg: Hydra configuration
        device: Device to place model on

    Returns:
        Configured model on device
    """
    model = MultimodalBiometricModel(
        num_classes=cfg.data.num_people,
        fingerprint_backbone=cfg.model.backbone_name,
        fingerprint_feature_dim=cfg.model.fingerprint_feature_dim,
        iris_feature_dim=cfg.model.iris_feature_dim,
        fusion_hidden_dim=cfg.model.fusion_hidden_dim,
        dropout=cfg.model.dropout,
        freeze_fingerprint_backbone=True,
    ).to(device)

    logging.info(
        f"Model created with {sum(p.numel() for p in model.parameters())} parameters"
    )

    return model


def move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """Move batch tensors to the specified device.

    Args:
        batch: Dictionary containing tensors and non-tensor values
        device: Target device for tensors

    Returns:
        Dictionary with tensors moved to device
    """
    batch_device = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch_device[key] = value.to(device)
        else:
            batch_device[key] = value
    return batch_device


def load_model_from_checkpoint(
    checkpoint_path: str,
    num_classes: int,
    backbone_name: str = "mobilenetv2_100",
    fingerprint_feature_dim: int = 1280,
    iris_feature_dim: int = 32,
    fusion_hidden_dim: int = 128,
    dropout: float = 0.5,
    device: torch.device | None = None,
) -> MultimodalBiometricModel:
    """Load trained model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file (local or S3 URI)
        num_classes: Number of classes in the model
        backbone_name: Name of the timm model to use as backbone
        fingerprint_feature_dim: Feature dimension for fingerprint branch
        iris_feature_dim: Feature dimension for iris branches
        fusion_hidden_dim: Hidden dimension for fusion layer
        dropout: Dropout probability
        device: Target device for the model

    Returns:
        Loaded and configured model
    """
    if device is None:
        device = get_device()

    logging.info(f"Loading model from {checkpoint_path}")

    # Download model if it's an S3 URI
    local_model_path = checkpoint_path
    if checkpoint_path.startswith("s3://"):
        from biometric_recognition.utils.aws_utils import S3Utils

        s3_utils = S3Utils()
        local_model_path = "./tmp_model.pth"
        s3_utils.download_from_s3(checkpoint_path, local_model_path)

    # Load checkpoint (with weights_only=False for compatibility with Hydra config)
    checkpoint = torch.load(local_model_path, map_location=device, weights_only=False)

    # Use saved config if available, otherwise use provided parameters
    # Always use pretrained=False when loading from checkpoint since weights come from checkpoint
    saved_config = checkpoint.get("config", {})
    if saved_config and hasattr(saved_config, "model"):
        model = MultimodalBiometricModel(
            num_classes=num_classes,
            fingerprint_backbone=getattr(
                saved_config.model, "backbone_name", backbone_name
            ),
            fingerprint_feature_dim=saved_config.model.fingerprint_feature_dim,
            iris_feature_dim=saved_config.model.iris_feature_dim,
            fusion_hidden_dim=saved_config.model.fusion_hidden_dim,
            dropout=saved_config.model.dropout,
            freeze_fingerprint_backbone=True,
            pretrained=False,
        ).to(device)
    else:
        # Fallback to provided parameters
        model = MultimodalBiometricModel(
            num_classes=num_classes,
            fingerprint_backbone=backbone_name,
            fingerprint_feature_dim=fingerprint_feature_dim,
            iris_feature_dim=iris_feature_dim,
            fusion_hidden_dim=fusion_hidden_dim,
            dropout=dropout,
            freeze_fingerprint_backbone=True,
            pretrained=False,
        ).to(device)

    # Load weights
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    val_acc = checkpoint.get("val_accuracy", "N/A")
    logging.info(f"Model loaded successfully. Validation accuracy: {val_acc}")
    return model


def predict_batch(
    model: MultimodalBiometricModel,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
) -> Dict[str, Any]:
    """Make predictions on a batch of data.

    Args:
        model: The loaded model
        batch: Dictionary containing input tensors
        device: Device to run inference on

    Returns:
        Dictionary containing predictions and confidence scores
    """
    # Move batch to device using shared utility
    batch_device = move_batch_to_device(batch, device)

    with torch.no_grad():
        # Get logits
        logits = model(batch_device)

        # Get probabilities and predictions
        probabilities = torch.softmax(logits, dim=1)
        predicted_classes = torch.argmax(logits, dim=1)
        confidence_scores = torch.max(probabilities, dim=1)[0]

        # Get top-k predictions
        top_k_probs, top_k_indices = torch.topk(
            probabilities, k=min(5, probabilities.size(1)), dim=1
        )

    return {
        "predicted_classes": predicted_classes.cpu().numpy(),
        "confidence_scores": confidence_scores.cpu().numpy(),
        "probabilities": probabilities.cpu().numpy(),
        "top_k_indices": top_k_indices.cpu().numpy(),
        "top_k_probabilities": top_k_probs.cpu().numpy(),
        "logits": logits.cpu().numpy(),
    }


def get_model_info(
    model: MultimodalBiometricModel, model_path: str | None = None
) -> Dict[str, Any]:
    """Get information about the loaded model.

    Args:
        model: The loaded model
        model_path: Path to the model checkpoint

    Returns:
        Dictionary containing model information
    """
    param_count = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "model_path": model_path or "unknown",
        "total_parameters": param_count,
        "trainable_parameters": trainable_params,
        "model_size_mb": param_count
        * 4
        / (1024 * 1024),  # Approximate size in MB (float32)
        "architecture": {
            "fingerprint_feature_dim": model.fingerprint_branch.backbone.num_features,
            "iris_feature_dim": 32,  # Fixed value from iris branch architecture
            "fusion_hidden_dim": model.fusion_module.fusion_layer[0].out_features,
        },
    }


def save_checkpoint(
    path: Path,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    val_accuracy: float,
    cfg: Any = None,
    train_losses: list[float] | None = None,
    val_losses: list[float] | None = None,
    val_accuracies: list[float] | None = None,
) -> None:
    """Save model checkpoint.

    Args:
        path: Path to save checkpoint
        epoch: Current epoch number
        model: Model to save
        optimizer: Optimizer state to save
        val_accuracy: Validation accuracy at this checkpoint
        cfg: Optional config to save (DictConfig or dict)
        train_losses: Optional list of training losses
        val_losses: Optional list of validation losses
        val_accuracies: Optional list of validation accuracies
    """
    from omegaconf import DictConfig, OmegaConf

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_accuracy": val_accuracy,
    }

    if cfg is not None:
        if isinstance(cfg, DictConfig):
            checkpoint["config"] = OmegaConf.to_container(cfg)
        else:
            checkpoint["config"] = cfg
    if train_losses is not None:
        checkpoint["train_losses"] = train_losses
    if val_losses is not None:
        checkpoint["val_losses"] = val_losses
    if val_accuracies is not None:
        checkpoint["val_accuracies"] = val_accuracies

    torch.save(checkpoint, path)
