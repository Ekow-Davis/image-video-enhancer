import os
import cv2
from tqdm import tqdm

from model_loader import load_realesrgan


SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def enhance_images(
    input_dir: str = "input/images",
    output_dir: str = "output/images",
    scale: int = 4,
    use_gpu: bool = False
):
    """
    Enhances all images in the input directory using Real-ESRGAN.

    Args:
        input_dir (str): Folder containing input images.
        output_dir (str): Folder to save enhanced images.
        scale (int): Upscaling factor.
        use_gpu (bool): Whether to use GPU if available.
    """

    os.makedirs(output_dir, exist_ok=True)

    # Load model
    upsampler = load_realesrgan(scale=scale, use_gpu=use_gpu)

    image_files = [
        f for f in os.listdir(input_dir)
        if f.lower().endswith(SUPPORTED_EXTENSIONS)
    ]

    if not image_files:
        print(" No images found in input directory.")
        return

    print(f" Found {len(image_files)} image(s). Starting enhancement...\n")

    for filename in tqdm(image_files, desc="Enhancing images"):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        img = cv2.imread(input_path, cv2.IMREAD_COLOR)

        if img is None:
            print(f" Failed to read image: {filename}")
            continue

        try:
            enhanced_img, _ = upsampler.enhance(img, outscale=scale)
            cv2.imwrite(output_path, enhanced_img)
        except Exception as e:
            print(f" Error processing {filename}: {e}")

    print("\n✔ Image enhancement complete.")


if __name__ == "__main__":
    enhance_images(use_gpu=False)
