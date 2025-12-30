import argparse
import torch
from modules.device import load_device
from modules.helpers.config import load_config
from modules.logging import init_logger, setup_logging
from modules.packing import pack_model

LOGGER = init_logger(__name__)


def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str)
    parser.add_argument("state_dict_path", type=str)
    parser.add_argument("output_path", type=str)
    parser.add_argument("--mode", type=str, choices=["full", "quantized"], default="full")
    return parser.parse_args()


def pack(config, state_dict, output_path, device, mode="full"):
    LOGGER.info("Building model structure...")
    model = config.model_builder()
    model.to(device)

    LOGGER.info(f"Loading trained weights (Mode: {mode})...")
    model.load_state_dict(state_dict, strict=False)

    if mode == "full":
        # --- Full Precision (Standard PyTorch Save) ---
        LOGGER.info("Saving Raw State Dict (High Quality, No Compression)...")
        torch.save(model.state_dict(), output_path)

    elif mode == "quantized":
        # --- Compression (Entropy Coding) ---
        LOGGER.info("Packing model to Bitstream (Compressed)...")
        try:
            stream = pack_model(model)
            stream.save(output_path)
        except AttributeError as e:
            LOGGER.error("CRITICAL ERROR: 'pack' failed. Did you use DummyQuantizer with mode='quantized'?")
            raise e

    LOGGER.info(f"Model saved successfully to {output_path}")


def main():
    setup_logging()
    args = load_args()
    device = load_device()
    config = load_config(args.config)

    state_dict = torch.load(args.state_dict_path, map_location=device, weights_only=True)
    pack(config, state_dict, args.output_path, device, mode=args.mode)


if __name__ == "__main__":
    main()
