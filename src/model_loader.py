import os
import torch
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet


def load_realesrgan(
    scale: int = 4,
    model_path: str = "models/realesrgan/weights/RealESRGAN_x4plus.pth",
    use_gpu: bool = False
):
    """
    Loads the Real-ESRGAN model for image super-resolution.

    Args:
        scale (int): Upscaling factor (default = 4).
        model_path (str): Path to pretrained model weights.
        use_gpu (bool): Whether to use CUDA if available.

    Returns:
        RealESRGANer: Initialized upscaling model.
    """

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model weights not found at {model_path}. "
            "Run scripts/download_models.py first."
        )

    device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"

    # Define the RRDBNet architecture used by Real-ESRGAN
    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        scale=scale
    )

    upsampler = RealESRGANer(
        scale=scale,
        model_path=model_path,
        model=model,
        tile=0,          # no tiling for now
        tile_pad=10,
        pre_pad=0,
        half=False,      # FP32 for CPU stability
        device=device
    )

    print(f"✔ Real-ESRGAN loaded on {device.upper()}")
    return upsampler
