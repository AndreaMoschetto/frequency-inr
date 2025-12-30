import argparse

from modules.data import ImageData, FourierImageData, DCTImageData
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
    parser.add_argument("--mode", type=str, choices=["full", "quantized"], default="full")
    return parser.parse_args()


def main():
    setup_logging()
    setup_reproducibility(42)

    args = load_args()
    device = load_device()
    config = load_config(args.config)

    # Runtime Quantizer Swap logic
    if args.mode == "full":
        LOGGER.info("MODE: Full Precision. Forcing DummyQuantizer.")
        for phase_name in config.phases:
            config.phases[phase_name].quantizer_builder = lambda _: DummyQuantizer()
    elif args.mode == "quantized":
        LOGGER.info("MODE: Quantized. Forcing UniformQuantizer(8bit).")
        for phase_name in config.phases:
            config.phases[phase_name].quantizer_builder = lambda _: UniformQuantizer(8)

    if args.config == "fourier":
        LOGGER.info("Loading Fourier Image (FFT)...")
        try:
            image_data = FourierImageData(args.image_path, device)
        except NameError:
            LOGGER.error("FourierImageData non trovata. Usa config 'dct'.")
            return

    elif args.config == "dct":
        LOGGER.info("Loading DCT Image (Discrete Cosine Transform)...")
        image_data = DCTImageData(args.image_path, device)

    else:
        LOGGER.info("Loading Standard Image (RGB)...")
        image_data = ImageData(args.image_path, device)

    fitted_state_dict = fit(config, image_data, device)
    pack(config, fitted_state_dict, args.output_path, device, mode=args.mode)


if __name__ == "__main__":
    main()
