import torch
import os
from modules.logging import init_logger

LOGGER = init_logger(__name__)


def load_device(force_cpu=False):
    if force_cpu:
        LOGGER.info("Device forced to CPU.")
        return torch.device("cpu")

    env_device = os.environ.get("DEVICE")
    if env_device:
        LOGGER.info(f"Using device from env: {env_device}")
        return torch.device(env_device)

    if torch.cuda.is_available():
        LOGGER.info("CUDA detected. Using NVIDIA GPU.")
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        LOGGER.info("Apple Silicon (MPS) detected. Using Metal Performance Shaders.")
        return torch.device("mps")
    else:
        LOGGER.info("No GPU detected. Falling back to CPU.")
        return torch.device("cpu")
