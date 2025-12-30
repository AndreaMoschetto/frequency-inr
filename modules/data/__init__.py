import torch
import os
from torch import Tensor
from torchvision.io import read_image, write_png
from torchvision.transforms import ConvertImageDtype, Normalize

# ==========================================
#  MATH FUNCTIONS (DCT/IDCT)
# ==========================================


def dct1(x):
    """DCT 1D type II"""
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)
    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)
    Vc = torch.fft.fft(v, dim=1)
    k = - torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * (3.141592653589793 / (2 * N))
    W_r = torch.cos(k)
    W_i = torch.sin(k)
    V = Vc.real * W_r - Vc.imag * W_i
    return V.view(*x_shape)


def idct1(X):
    """IDCT 1D type II"""
    x_shape = X.shape
    N = x_shape[-1]
    X_v = X.contiguous().view(-1, N) / 2
    k = torch.arange(N, dtype=X.dtype, device=X.device)[None, :] * (3.141592653589793 / (2 * N))
    W_r = torch.cos(k)
    W_i = torch.sin(k)
    V_t_r = X_v
    V_t_i = X_v * 0
    V_r = V_t_r * W_r - V_t_i * W_i
    V_i = V_t_r * W_i + V_t_i * W_r
    V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)
    v = torch.fft.irfft(torch.view_as_complex(V), n=N, dim=1)
    x = v.new_zeros(v.shape)
    x[:, ::2] += v[:, :N - (N // 2)]
    x[:, 1::2] += v.flip([1])[:, :N // 2]
    return x.view(*x_shape)


def dct2(x):
    return dct1(dct1(x.transpose(-1, -2)).transpose(-1, -2))


def idct2(X):
    return idct1(idct1(X.transpose(-1, -2)).transpose(-1, -2))


# ==========================================
#  SCALING FACTORS
# ==========================================

DCT_SCALE_FACTOR = 15.0
LOG_SCALE_FACTOR = 6.0


# ==========================================
#  DATA MANAGEMENT CLASSES
# ==========================================

class ImageData:
    def __init__(self, image_path: str, device: str):
        self.image = read_image(image_path).to(device)
        self.image = ConvertImageDtype(torch.float32)(self.image)
        self.transform = Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        self.tensor = self.transform(self.image)
        self.path = image_path

    def get_image(self) -> Tensor:
        return self.tensor

    def resolution(self):
        return self.image.shape[1], self.image.shape[2]


class DCTImageData:
    def __init__(self, image_path: str, device: str):
        self.path = image_path

        raw_image = read_image(image_path).to(device)
        raw_image = ConvertImageDtype(torch.float32)(raw_image)

        self.h, self.w = raw_image.shape[1], raw_image.shape[2]

        # 1. DCT
        dct_tensor = dct2(raw_image)

        # 2. Log Compression
        self.tensor = torch.sign(dct_tensor) * torch.log1p(dct_tensor.abs())

        # 3. Scaling
        self.tensor = self.tensor / DCT_SCALE_FACTOR

        print(f"[DCT INFO] Prepared Image shape: {self.tensor.shape}. Range: {self.tensor.min():.3f} to {self.tensor.max():.3f}")

    def get_image(self) -> Tensor:
        return self.tensor

    def resolution(self):
        return self.h, self.w


class FourierImageData:
    def __init__(self, image_path: str, device: str):
        self.path = image_path
        raw_image = read_image(image_path).to(device)
        self.h, self.w = raw_image.shape[1], raw_image.shape[2]

        raw_image = ConvertImageDtype(torch.float32)(raw_image)

        fft = torch.fft.fft2(raw_image, dim=(-2, -1), norm="ortho")
        fft_shifted = torch.fft.fftshift(fft, dim=(-2, -1))

        real = fft_shifted.real
        imag = fft_shifted.imag

        self.tensor = torch.cat([real, imag], dim=0)
        self.tensor = torch.sign(self.tensor) * torch.log1p(self.tensor.abs())
        self.tensor = self.tensor / LOG_SCALE_FACTOR

    def get_image(self) -> Tensor:
        return self.tensor

    def resolution(self):
        return self.h, self.w


# ==========================================
#  DUMP FUNCTIONS
# ==========================================

def dump_reconstructed_tensor(tensor: Tensor, output_path: str):
    """
    Standard saving for Spatial domain (pixels).
    """
    tensor = tensor.permute(2, 0, 1)
    tensor = (tensor * 0.5) + 0.5
    tensor = tensor.clamp(0, 1)
    tensor = ConvertImageDtype(torch.uint8)(tensor)
    write_png(tensor.cpu(), output_path)


def dump_reconstructed_dct(tensor: Tensor, output_path: str, original_image_path: str = None):
    original_image_path = None
    """
    Saves the image reconstructed from DCT.

    Args:
        tensor: The network output tensor.
        output_path: Where to save the generated image.
        original_image_path: (Optional) If provided (not None), activates DC Transplant.
                             If None, decodes normally without access to the original.
    """
    #  Inverse Rescaling
    tensor = tensor.permute(2, 0, 1) * DCT_SCALE_FACTOR

    #  Inverse Log
    tensor = torch.sign(tensor) * torch.expm1(tensor.abs())

    # --- DC TRANSPLANT LOGIC ---
    used_original = False
    if original_image_path and os.path.exists(original_image_path):
        try:
            # Read the original image for "DC Transplant"
            true_img = read_image(original_image_path).to(tensor.device)
            true_img = ConvertImageDtype(torch.float32)(true_img)
            true_dct = dct2(true_img)

            # Replace ONLY the DC coefficient [0,0]
            for c in range(3):
                tensor[c, 0, 0] = true_dct[c, 0, 0].item()

            used_original = True
        except Exception as e:
            print(f"[DCT WARN] Unable to use the original for DC fix: {e}")

    #  Inverse DCT
    reconstructed_image = idct2(tensor)

    #  Gamma Correction
    # We always apply it because SIRENs tend to underexpose anyway.
    reconstructed_image = reconstructed_image.clamp(1e-6, 1.0)
    reconstructed_image = torch.pow(reconstructed_image, 0.6)

    out_uint8 = ConvertImageDtype(torch.uint8)(reconstructed_image)
    write_png(out_uint8.cpu(), output_path)

    mode_str = "Hybrid (DC Transplant)" if used_original else "Blind (Raw)"
    print(f"[DCT INFO] Image saved to {output_path} [{mode_str}]")


def dump_reconstructed_fourier(tensor: Tensor, output_path: str, original_image_path: str = None):
    """
    Saves the image reconstructed from FFT.

    Args:
        original_image_path: (Optional) If provided, activates DC Transplant.
    """
    #  Inverse Scaling and Inverse Log
    tensor = tensor.permute(2, 0, 1)
    tensor = tensor * LOG_SCALE_FACTOR
    tensor = torch.sign(tensor) * torch.expm1(tensor.abs())

    # --- DC TRANSPLANT LOGIC ---
    used_original = False
    if original_image_path and os.path.exists(original_image_path):
        try:
            true_img = read_image(original_image_path).to(tensor.device)
            true_img = ConvertImageDtype(torch.float32)(true_img)

            true_fft = torch.fft.fft2(true_img, dim=(-2, -1), norm="ortho")
            true_shifted = torch.fft.fftshift(true_fft, dim=(-2, -1))

            # Center = DC Component
            h, w = true_shifted.shape[-2], true_shifted.shape[-1]
            cy, cx = h // 2, w // 2

            num_channels = 3
            for c in range(num_channels):
                # Replace Real and Imaginary at the center
                tensor[c, cy, cx] = true_shifted[c, cy, cx].real
                tensor[c + num_channels, cy, cx] = true_shifted[c, cy, cx].imag

            used_original = True
        except Exception as e:
            print(f"[FFT WARN] Unable to use the original for DC fix: {e}")

    #  Complex Reconstruction
    channels = tensor.shape[0] // 2
    real = tensor[:channels, :, :]
    imag = tensor[channels:, :, :]
    complex_spectrum = torch.complex(real, imag)

    unshifted_spectrum = torch.fft.ifftshift(complex_spectrum, dim=(-2, -1))
    reconstructed_image = torch.fft.ifft2(unshifted_spectrum, dim=(-2, -1), norm="ortho")

    # Real Part
    reconstructed_image = reconstructed_image.real

    #  Gamma Correction
    reconstructed_image = reconstructed_image.clamp(1e-6, 1.0)
    reconstructed_image = torch.pow(reconstructed_image, 0.6)

    out_uint8 = ConvertImageDtype(torch.uint8)(reconstructed_image)
    write_png(out_uint8.cpu(), output_path)

    mode_str = "Hybrid (DC Transplant)" if used_original else "Blind (Raw)"
    print(f"[FFT INFO] Image saved to {output_path} [{mode_str}]")
