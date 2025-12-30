import torch

from config import FittingPhaseConfiguration
from modules.nn.image_representation.coordinates_based import (
    CoordinatesBasedRepresentation,
)
from modules.nn.positional_encoder import PositionalEncoder
#  from modules.nn.quantizer.uniform import UniformQuantizer
from modules.nn.quantizer.dummy import DummyQuantizer
from modules.nn.siren import Siren
from modules.training import Trainer, TrainerConfiguration


def model_builder():
    # Positional encoder: mapping (u,v) coordinates into a vector space
    encoder = PositionalEncoder(num_frequencies=16, scale=1.4)

    # Output: 6 channels (Real/Imaginary parts for R, G, B)
    TARGET_CHANNELS = 6

    network = Siren(
        input_features=encoder.output_features_for(2),
        hidden_features=256,
        hidden_layers=3,
        output_features=TARGET_CHANNELS,
    )

    return CoordinatesBasedRepresentation(encoder, network)


def trainer_builder_for(iterations: int):
    def __builder(model, image, device):
        return Trainer(
            TrainerConfiguration(
                optimizer_builder=optimizer_builder,
                scheduler_builder=scheduler_builder,
                loss_fn_builder=loss_fn_builder,
                iterations=iterations,
                log_interval=50,
                shuffle_factor=1,  # Pixel shuffle is not needed in Fourier domain
            ),
            model,
            image,
            device,
        )

    return __builder


def quantizer_builder(_):
    # return UniformQuantizer(8)
    return DummyQuantizer()


def optimizer_builder(parameters):
    # LR slightly lower than standard (1e-3) because frequencies are delicate
    return torch.optim.Adam(parameters, lr=5e-4)


def scheduler_builder(optimizer):
    # Cosine Annealing: starts at 5e-4 and gently decays to 1e-5
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 500, 1.0e-5)


def loss_fn_builder():
    return torch.nn.MSELoss()


phases = {
    "full_precision": FittingPhaseConfiguration(
        model_builder=model_builder,
        # 500 iterations are the bare minimum to see sharp results in FFT
        trainer_builder=trainer_builder_for(500),
    )
}
