# image-video-enhancer

## Project Structure

```
image_video_enhancer/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── config/
│   └── settings.yaml
│
├── models/
│   └── realesrgan/
│       └── weights/
│           └── RealESRGAN_x4plus.pth
│
├── input/
│   ├── images/
│   └── videos/
│
├── output/
│   ├── images/
│   └── videos/
│
├── src/
│   ├── main.py
│   ├── image_enhancer.py
│   ├── model_loader.py
│   ├── utils.py
│   └── logger.py
│
└── scripts/
    └── download_models.py
```