import torch

from config import FittingPhaseConfiguration
from modules.nn.image_representation.coordinates_based import (
    CoordinatesBasedRepresentation,
)
from modules.nn.positional_encoder import PositionalEncoder
from modules.nn.quantizer.uniform import UniformQuantizer
from modules.nn.siren import Siren
from modules.training import Trainer, TrainerConfiguration


def model_builder():
    # Setup identical to the Fourier version for a fair comparison
    encoder = PositionalEncoder(num_frequencies=16, scale=1.4)

    network = Siren(
        input_features=encoder.output_features_for(2),
        hidden_features=256,  # MATCHED: Same width as Fourier config
        hidden_layers=3,      # MATCHED: Same depth as Fourier config
        output_features=3,    # ONLY DIFFERENCE: Here we want standard RGB (3 channels)
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
                shuffle_factor=1,  # MATCHED: Same shuffling factor (low/none)
            ),
            model,
            image,
            device,
        )

    return __builder


def quantizer_builder(_):
    return UniformQuantizer(8)


def optimizer_builder(parameters):
    # MATCHED: Same Learning Rate as Fourier
    return torch.optim.Adam(parameters, lr=5e-4)


def scheduler_builder(optimizer):
    # MATCHED: Same scheduler decay
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 500, 1.0e-5)


def loss_fn_builder():
    return torch.nn.MSELoss()


phases = {
    "full_precision": FittingPhaseConfiguration(
        model_builder=model_builder,
        trainer_builder=trainer_builder_for(500),  # MATCHED: 500 iterations
    ),
    # We disable the QAT (Quantization) phase here as well
    # to compare only the pure reconstruction capability.

    # "8bits_qat": FittingPhaseConfiguration(
    #     model_builder=model_builder,
    #     trainer_builder=trainer_builder_for(200),
    #     quantizer_builder=quantizer_builder,
    #     recalibrate_quantizers=True,
    # ),
}
