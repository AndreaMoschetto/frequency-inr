
from torch import Tensor, Size, nn

from modules.helpers.coordinates import generate_coordinates_grid
from modules.logging import init_logger
from modules.nn.image_representation.base import ImplicitImageRepresentation

LOGGER = init_logger(__name__)


class CoordinatesBasedRepresentation(ImplicitImageRepresentation):
    def __init__(self, encoder: nn.Module, network: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.network = network

    def generate_input(self, output_shape: Size) -> Tensor:
        (height, width) = (output_shape[0], output_shape[1])
        return generate_coordinates_grid(height, width)

    def forward(self, coordinates: Tensor) -> Tensor:
        encoded_coordinates = self.encoder(coordinates)

        # Eseguiamo la rete
        reconstructed = self.network(encoded_coordinates)

        # --- SPIA DI DEBUG (Ora nel file giusto!) ---
        if reconstructed.shape[-1] != 6:
            print("\n" + "X" * 50)
            print(f"[CRITICAL DEBUG] La rete SIREN interna ha restituito: {reconstructed.shape}")
            print("[CRITICAL DEBUG] Sto forzando l'uscita? NO, restituisco quello che vedo.")
            print("X" * 50 + "\n")
        # --------------------------------------------

        return reconstructed
