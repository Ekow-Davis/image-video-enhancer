import os
import cv2
from tqdm import tqdm

from model_loader import load_realesrgan

SUPPORTED_VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv")


def enhance_videos_in_folder(
    input_dir: str = "input/videos",
    output_dir: str = "output/videos",
    scale: int = 4,
    use_gpu: bool = False
):
    """
    Enhances all videos in a folder using frame-by-frame super-resolution.
    """

    os.makedirs(output_dir, exist_ok=True)
    upsampler = load_realesrgan(scale=scale, use_gpu=use_gpu)

    video_files = [
        f for f in os.listdir(input_dir)
        if f.lower().endswith(SUPPORTED_VIDEO_EXTS)
    ]

    if not video_files:
        print("⚠ No video files found.")
        return

    for video_name in video_files:
        input_path = os.path.join(input_dir, video_name)
        output_path = os.path.join(
            output_dir,
            f"{os.path.splitext(video_name)[0]}_enhanced.mp4"
        )

        cap = cv2.VideoCapture(input_path)

        if not cap.isOpened():
            print(f"❌ Cannot open video: {video_name}")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        out_width = width * scale
        out_height = height * scale

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (out_width, out_height))

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"\n🎞 Processing {video_name} ({frame_count} frames)")

        for _ in tqdm(range(frame_count), desc="Enhancing frames"):
            ret, frame = cap.read()
            if not ret:
                break

            enhanced_frame, _ = upsampler.enhance(frame, outscale=scale)
            out.write(enhanced_frame)

        cap.release()
        out.release()

        print(f"✔ Saved: {output_path}")


if __name__ == "__main__":
    enhance_videos_in_folder(use_gpu=False)
z