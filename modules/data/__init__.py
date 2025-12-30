import torch
from torch import Tensor
from torchvision.io import read_image, write_png
from torchvision.transforms import ConvertImageDtype, Normalize

LOG_SCALE_FACTOR = 6.0


def save_spectrum_debug(tensor_cw, filename):
    """
    Helper function to visualize the spectrum.
    It expects a tensor (Channels, H, W) already log-compressed or scaled.
    We take only the magnitude of the first channel (R or G) for visualization.
    """

    # We assume structure: R_real, R_imag, G_real, ...
    # We take index 0 (Real) and 1 (Imag) if the tensor is raw complex separated
    # But here the tensor is already "packed". We estimate the global energy.

    energy = tensor_cw.abs().mean(dim=0)  # (H, W)

    v_min, v_max = energy.min(), energy.max()
    if v_max - v_min > 1e-5:
        vis = (energy - v_min) / (v_max - v_min)
    else:
        vis = energy

    write_png(ConvertImageDtype(torch.uint8)(vis.unsqueeze(0).cpu()), filename)


class ImageData:
    def __init__(self, image_path: str, device: str):
        self.image = read_image(image_path).to(device)
        self.image = ConvertImageDtype(torch.float32)(self.image)
        self.transform = Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        self.tensor = self.transform(self.image)

    def get_image(self) -> Tensor:
        return self.tensor


class FourierImageData:
    def __init__(self, image_path: str, device: str):
        raw_image = read_image(image_path).to(device)
        raw_image = ConvertImageDtype(torch.float32)(raw_image)

        fft = torch.fft.fft2(raw_image, dim=(-2, -1), norm="ortho")
        fft_shifted = torch.fft.fftshift(fft, dim=(-2, -1))

        real = fft_shifted.real
        imag = fft_shifted.imag

        self.tensor = torch.cat([real, imag], dim=0)
        self.tensor = torch.sign(self.tensor) * torch.log1p(self.tensor.abs())

        print("[DEBUG] Saving Ground Truth Spectrum to 'debug_spectrum_TRUTH.png'...")
        save_spectrum_debug(self.tensor, "debug_spectrum_TRUTH.png")

        self.tensor = self.tensor / LOG_SCALE_FACTOR

    def get_image(self) -> Tensor:
        return self.tensor


def dump_reconstructed_tensor(tensor: Tensor, output_path: str):
    tensor = tensor.permute(2, 0, 1)
    tensor = (tensor * 0.5) + 0.5
    tensor = tensor.clamp(0, 1)
    tensor = ConvertImageDtype(torch.uint8)(tensor)
    write_png(tensor.cpu(), output_path)


def dump_reconstructed_fourier(tensor: Tensor, output_path: str):
    # input tensor shape: (Height, Width, 6)
    tensor = tensor.permute(2, 0, 1)  # -> (6, H, W)

    # 1. Denormalize (x * 6.0) -> Return to Logarithmic space
    tensor_log_scaled = tensor * LOG_SCALE_FACTOR

    # --- DEBUG: Save the prediction ---
    # Save NOW, while still in logarithmic space (readable)
    print(f"[DEBUG] Saving Predicted Spectrum to 'debug_spectrum_PRED.png'...")
    save_spectrum_debug(tensor_log_scaled, "debug_spectrum_PRED.png")
    # --------------------------------------------------

    # 2. INVERSE LOG COMPRESSION
    tensor = torch.sign(tensor_log_scaled) * torch.expm1(tensor_log_scaled.abs())

    channels = tensor.shape[0] // 2
    real = tensor[:channels, :, :]
    imag = tensor[channels:, :, :]

    complex_spectrum = torch.complex(real, imag)

    unshifted_spectrum = torch.fft.ifftshift(complex_spectrum, dim=(-2, -1))
    reconstructed_image = torch.fft.ifft2(unshifted_spectrum, dim=(-2, -1), norm="ortho")
    image_real = reconstructed_image.real

    # Auto-Exposure
    v_min, v_max = image_real.min(), image_real.max()
    if v_max - v_min > 1e-5:
        image_normalized = (image_real - v_min) / (v_max - v_min)
        out_uint8 = ConvertImageDtype(torch.uint8)(image_normalized)
    else:
        image_clamped = image_real.clamp(0, 1)
        out_uint8 = ConvertImageDtype(torch.uint8)(image_clamped)

    write_png(out_uint8.cpu(), output_path)
