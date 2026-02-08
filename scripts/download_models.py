import os
import urllib.request

MODEL_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
MODEL_DIR = os.path.join("models", "realesrgan", "weights")
MODEL_PATH = os.path.join(MODEL_DIR, "RealESRGAN_x4plus.pth")


def download_model():
    os.makedirs(MODEL_DIR, exist_ok=True)

    if os.path.exists(MODEL_PATH):
        print("✔ Model already exists. Skipping download.")
        return

    print("⬇ Downloading RealESRGAN_x4plus model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("✔ Model downloaded successfully.")


if __name__ == "__main__":
    download_model()
