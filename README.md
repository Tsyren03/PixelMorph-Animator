



# PixelMorph Animator

## Project Description
**PixelMorph Animator** is a Python-based computer vision project that creates a smooth, visually captivating morphing animation between two different images. Inspired by pixel-rearranging art, this program does not simply crossfade one image into another. Instead, it intelligently maps and visually moves the pixels from a source image to reconstruct the layout of a target image.

By utilizing **OpenCV** and **NumPy**, the script divides high-resolution (1024x1024) images into blocks, calculates the mean colors, and pairs them using Euclidean distance. The resulting pixel transition is rendered at 60 FPS using an ease-in-out algorithm, generating a premium, high-quality video output (`.mp4`).

## Key Features
* **Smart Block Matching:** Intelligently maps pixels by comparing the color similarity (mean RGB values) between source and target image blocks.
* **Smooth Animation:** Implements an `ease_in_out` mathematical function to ensure the pixel movement starts and ends naturally.
* **High-Resolution Processing:** Automatically crops and resizes input images without distortion, maintaining a 1024x1024 aspect ratio for crisp results.

## Requirements
* Python 3.x
* `opencv-python` (cv2)
* `numpy`

## Usage
1. Place your target image (e.g., `Trump.jpg`) and source image (e.g., `random.jpg`) in the root directory.
2. Run the script:
   ```bash
   python main.py

<img width="1638" height="2048" alt="Trump" src="https://github.com/user-attachments/assets/e95a2624-de66-4260-8783-140bd2bac349" />
<img width="828" height="795" alt="스크린샷 2026-05-16 17 25 26" src="https://github.com/user-attachments/assets/c3cd4643-746d-4bc4-a554-3e54a76cf0ec" />
<img width="828" height="795" alt="스크린샷 2026-05-16 17 25 13" src="https://github.com/user-attachments/assets/4fdd45c0-9566-496a-9109-885571cd2a0e" />
