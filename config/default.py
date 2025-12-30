import torch

from config import FittingPhaseConfiguration
from modules.nn.image_representation.coordinates_based import (
    CoordinatesBasedRepresentation,
)
from modules.nn.positional_encoder import PositionalEncoder
from modules.nn.quantizer.dummy import DummyQuantizer
from modules.nn.siren import Siren
from modules.training import Trainer, TrainerConfiguration


def model_builder():
    encoder = PositionalEncoder(num_frequencies=16, scale=1.4)

    network = Siren(
        input_features=encoder.output_features_for(2),
        hidden_features=256,
        hidden_layers=3,
        output_features=3,
        period=30.0,
        a=6.0
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
                log_interval=100,
                shuffle_factor=1,
            ),
            model,
            image,
            device,
        )

    return __builder


def quantizer_builder(_):
    return DummyQuantizer()


def optimizer_builder(parameters):
    return torch.optim.Adam(parameters, lr=1e-3)


def scheduler_builder(optimizer):
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 1000, 1.0e-5)


def loss_fn_builder():
    return torch.nn.MSELoss()


phases = {
    "feature_learning": FittingPhaseConfiguration(
        model_builder=model_builder,
        trainer_builder=trainer_builder_for(1000),
        quantizer_builder=quantizer_builder,
        recalibrate_quantizers=False
    )
}

# Export
model_builder = model_builder
