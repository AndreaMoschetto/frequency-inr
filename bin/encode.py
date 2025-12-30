import argparse

from modules.data import ImageData, FourierImageData
from modules.device import load_device
from modules.helpers.config import load_config
from modules.helpers.reproducibility import setup_reproducibility
from modules.logging import init_logger, setup_logging
from modules.nn.quantizer.dummy import DummyQuantizer
from modules.nn.quantizer.uniform import UniformQuantizer
from bin.pack import pack
from bin.fit import fit

LOGGER = init_logger(__name__)


def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", type=str)
    parser.add_argument("output_path", type=str)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--mode", type=str, choices=["full", "quantized"], default="full",
                        help="Mode: 'full' (Best Quality, Dummy Quantizer) or 'quantized' (Compressed, Uniform Quantizer)")
    return parser.parse_args()


def main():
    setup_logging()
    setup_reproducibility(42)

    args = load_args()
    device = load_device()
    config = load_config(args.config)
    # ----------- Override Quantizer Based on Mode -------------------
    if args.mode == "full":
        LOGGER.info("MODE: Full Precision. Forcing DummyQuantizer.")
        # Override quantizer builder for all phases to return Dummy
        for phase_name in config.phases:
            config.phases[phase_name].quantizer_builder = lambda _: DummyQuantizer()

    elif args.mode == "quantized":
        LOGGER.info("MODE: Quantized. Forcing UniformQuantizer(8bit).")
        # Override quantizer builder for all phases to return Uniform
        for phase_name in config.phases:
            config.phases[phase_name].quantizer_builder = lambda _: UniformQuantizer(8)
    # ------------------------------

    # Image Loading Logic (Standard vs Fourier)
    if args.config == "fourier":
        LOGGER.info("Loading Fourier Image (FFT)...")
        image_data = FourierImageData(args.image_path, device)
    else:
        LOGGER.info("Loading Standard Image (RGB)...")
        image_data = ImageData(args.image_path, device)

    fitted_state_dict = fit(config, image_data, device)
    pack(config, fitted_state_dict, args.output_path, device, mode=args.mode)


if __name__ == "__main__":
    main()
