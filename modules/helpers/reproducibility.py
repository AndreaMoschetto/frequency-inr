import torch
import os
import random
from modules.logging import init_logger

LOGGER = init_logger(__name__)


def setup_reproducibility(seed=None):
    if seed is None:
        seed = os.environ.get("RANDOM_SEED")

    if seed is not None:
        try:
            seed = int(seed)

            torch.manual_seed(seed)
            random.seed(seed)

            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            LOGGER.info(f"Random seed set to: {seed}")
        except ValueError:
            LOGGER.warning(f"Invalid seed format: {seed}. Expected an integer.")
    else:
        LOGGER.warning("Random seed not set, results may not be reproducible")
