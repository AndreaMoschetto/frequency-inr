from typing import Tuple
import torchvision
import PIL
import torch
from skimage import io
from torch import fft

from modules.logging import init_logger

LOGGER = init_logger(__name__)


class ImageData:
    """Standard spatial image data (RGB)"""

    def __init__(self, path, device):
        pil_image = PIL.Image.open(path).convert("RGB")
        self.path = path
        # Tensor shape loaded by torchvision is (3, H, W)
        self.tensor = torchvision.transforms.functional.to_tensor(pil_image).to(device)
        self.height = self.tensor.shape[1]
        self.width = self.tensor.shape[2]

    def resolution(self) -> Tuple[int, int]:
        return (self.height, self.width)

    def num_pixels(self) -> int:
        return self.width * self.height


class FourierImageData(ImageData):
    """Spectral domain data (FFT of RGB)"""

    def __init__(self, path, device):
        super().__init__(path, device)

        # Compute FFT2 (using ortho norm)
        # Input (3, H, W) -> Output (3, H, W) Complex
        fft_tensor = fft.fft2(self.tensor, norm="ortho")

        # Shift
        fft_shifted = fft.fftshift(fft_tensor)

        # Stack Real and Imaginary
        # Concatenate on channel dim (0) -> Result (6, H, W)
        self.tensor = torch.cat([fft_shifted.real, fft_shifted.imag], dim=0).to(device)

        LOGGER.info(f"Fourier Data created. Shape: {self.tensor.shape} (expected 6 channels)")


def dump_reconstructed_tensor(reconstructed_tensor: torch.Tensor, path: str):
    """
    Standard spatial dump.
    Expects input tensor in shape (H, W, C) from the model.
    """
    reconstructed_image = (
        reconstructed_tensor.detach()
        .clamp(0.0, 1.0)
        .mul(255.0)
        .round()
        .to(torch.uint8)
        .cpu()
        .numpy()
    )

    # Check shape
    if reconstructed_image.ndim == 3 and reconstructed_image.shape[0] < 10:
        # If by mistake we got (C, H, W), fix it to (H, W, C)
        LOGGER.warning(f"Tensor shape {reconstructed_image.shape} looks channel-first. Transposing.")
        reconstructed_image = reconstructed_image.transpose(1, 2, 0)

    LOGGER.debug(f"Saving image with shape: {reconstructed_image.shape}")
    io.imsave(path, reconstructed_image)


def dump_reconstructed_fourier(reconstructed_tensor: torch.Tensor, path: str):
    """
    Inverse pipeline: 
    Model Output (H, W, 6) -> Transpose to (6, H, W) -> Complex -> IFFT -> Real Image (C, H, W) -> Save
    """
    with torch.no_grad():
        # The model outputs (H, W, 6), but for torch operations we want (6, H, W)
        # So we move the last dim (Channels) to the first dim.
        tensor_ch_first = reconstructed_tensor.movedim(2, 0)  # (6, H, W)

        # Split back into Real and Imaginary
        real_part, imag_part = torch.chunk(tensor_ch_first, 2, dim=0)

        # Recombine into Complex Tensor
        complex_tensor = torch.complex(real_part, imag_part)

        # Inverse Shift
        unshifted = fft.ifftshift(complex_tensor)

        # Inverse FFT
        spatial_tensor = fft.ifft2(unshifted, norm="ortho").real  # Result is (3, H, W)

        # Prepare for saving: convert (3, H, W) to (H, W, 3) just because dump_reconstructed_tensor expects HWC
        spatial_tensor_hwc = spatial_tensor.movedim(0, 2)

        dump_reconstructed_tensor(spatial_tensor_hwc, path)
