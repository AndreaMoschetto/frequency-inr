import argparse
import torch
from modules.data import dump_reconstructed_tensor, dump_reconstructed_fourier
from modules.device import load_device
from modules.helpers.config import load_config
from modules.logging import init_logger, setup_logging
from modules.packing.bytestream import ByteStream

LOGGER = init_logger(__name__)


def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=str)
    parser.add_argument("resolution", type=str)
    parser.add_argument("output_path", type=str)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--mode", type=str, choices=["full", "quantized"], default="full")
    return parser.parse_args()


def parse_resolution(resolution_str):
    parts = resolution_str.split("x")
    return (int(parts[0]), int(parts[1]))


def main():
    setup_logging()
    args = load_args()
    device = load_device()
    config = load_config(args.config)

    LOGGER.info(f"Building model for decoding (Mode: {args.mode})...")
    model = config.model_builder()
    model.to(device)

    if args.mode == "full":
        # --- Load Standard PyTorch File ---
        LOGGER.info(f"Loading raw weights from {args.model_path}...")
        state_dict = torch.load(args.model_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict, strict=False)

    elif args.mode == "quantized":
        # --- Decode Compressed Bitstream ---
        LOGGER.info(f"Unpacking bitstream from {args.model_path}...")
        stream = ByteStream.load(args.model_path)
        model.unpack(stream)

    resolution = parse_resolution(args.resolution)

    LOGGER.info("Running inference...")
    input_coordinates = model.generate_input(resolution).to(device)

    with torch.no_grad():
        reconstructed = model(input_coordinates)

    # Save output based on domain (Spatial vs Fourier)
    if args.config == "fourier":
        dump_reconstructed_fourier(reconstructed, args.output_path)
    else:
        dump_reconstructed_tensor(reconstructed, args.output_path)

    LOGGER.info(f"Reconstruction saved to {args.output_path}")


if __name__ == "__main__":
    main()
