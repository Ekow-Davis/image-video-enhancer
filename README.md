# Image & Video Enhancer

## Overview

**Image & Video Enhancer** is a small open-source Python project for enhancing low-resolution images (and later videos) using **pretrained deep learning super-resolution models**.

The project is designed primarily for **educational use in image processing**, with an emphasis on:
- Practical super-resolution pipelines
- Folder-based batch processing
- Clean, modular, and readable Python code
- Using pretrained models (no training required)

At the moment, the project supports **image enhancement only** using **Real-ESRGAN**.  
Video enhancement will be added later using a frame-by-frame approach.

---

## What This Project Does

- Takes low-resolution images (e.g. ~480p)
- Upscales them (×4 by default)
- Improves perceived sharpness, edges, and overall visual quality
- Saves enhanced images to a dedicated output folder

### Best suited for
- Images of people (faces and full-body shots)
- Natural scenes
- General photographic content

### Limitations
- The model does **not recover real missing details**; it hallucinates plausible detail
- Not suitable for forensic or factual image reconstruction
- Very blurry images or heavy motion blur may produce poor results

---

## Technologies Used

- **Python 3.11**
- **PyTorch**
- **Real-ESRGAN** (pretrained super-resolution model)
- **OpenCV**
- **NumPy**
- **tqdm**

All tools and models used are **free and open-source**.

---

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