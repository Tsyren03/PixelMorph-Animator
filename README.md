# PixelMorph Animator

## Project Description
**PixelMorph Animator** is a Python-based computer vision project that creates a smooth, visually captivating morphing animation between two different images. Inspired by pixel-rearranging art, this program does not simply crossfade one image into another. Instead, it intelligently maps and visually moves the pixels from a source image to reconstruct the layout of a target image.

By utilizing **OpenCV** and **NumPy**, the script divides high-resolution (1024x1024) images into blocks, calculates the mean colors, and pairs them using Euclidean distance. The resulting pixel transition is rendered at 60 FPS using an ease-in-out algorithm, generating a premium, high-quality video output (`.mp4`).

Inspiration video:
https://www.youtube.com/shorts/MeFi68a2pP8

## Key Features
* **Smart Block Matching:** Intelligently maps pixels by comparing the color similarity (mean RGB values) between source and target image blocks.
* **Smooth Animation:** Implements an `ease_in_out` mathematical function to ensure the pixel movement starts and ends naturally.
* **High-Resolution Processing:** Automatically crops and resizes input images without distortion, maintaining a 1024x1024 aspect ratio for crisp results.

<img width="828" height="795" alt="Trump" src="https://github.com/user-attachments/assets/e95a2624-de66-4260-8783-140bd2bac349" />
<img width="828" height="795" alt="스크린샷 2026-05-16 17 25 13" src="https://github.com/user-attachments/assets/4fdd45c0-9566-496a-9109-885571cd2a0e" />
<img width="828" height="795" alt="스크린샷 2026-05-16 17 25 26" src="https://github.com/user-attachments/assets/c3cd4643-746d-4bc4-a554-3e54a76cf0ec" />
# Mathematical Model: The "Perfect Morph" Algorithm

The core logic of the `animate_perfect_morph` function can be defined as a bijective spatial mapping driven by luminance sorting and interpolated via a smooth kinematic function.

---

## 1. Definitions and Setup

Let \( I_{\text{src}} \) be the source color image and \( I_{\text{tgt}} \) be the target grayscale image.

Both images share the exact same dimensions, containing a total of:

$$
N = H \times W
$$

pixels.

Let \( Y_{\text{src}} \) and \( Y_{\text{tgt}} \) represent one-dimensional sets containing the luminance values of every pixel:

$$
Y_{\text{src}} = \{ y_1, y_2, \dots, y_N \}
$$

$$
Y_{\text{tgt}} = \{ y_1, y_2, \dots, y_N \}
$$

---

## 2. Permutation (Brightness Sorting)

The algorithm computes two sorting permutations, \( \sigma_s \) and \( \sigma_t \), that arrange luminance values into strict ascending order:

$$
Y_{\text{src}}[\sigma_s(k)] \leq Y_{\text{src}}[\sigma_s(k+1)]
$$

$$
Y_{\text{tgt}}[\sigma_t(k)] \leq Y_{\text{tgt}}[\sigma_t(k+1)]
$$

where:

- \( k = 1 \) corresponds to the darkest pixel
- \( k = N \) corresponds to the brightest pixel

---

## 3. Bijective Spatial Mapping

Once both luminance arrays are ordered by rank \( k \), a strict one-to-one correspondence is established.

The pixel located at the coordinate of the \( k \)-th darkest source pixel:

$$
P_{\text{src}}(k)
$$

is mapped to the coordinate of the \( k \)-th darkest target pixel:

$$
P_{\text{tgt}}(k)
$$

---

## 4. Animation Kinematics

Let \( \tau \in [0,1] \) represent normalized animation time.

To produce smooth motion, the algorithm applies the Smoothstep easing function:

$$
E(\tau) = \tau^2 (3 - 2\tau)
$$

The position of pixel \( k \) at time \( \tau \) is then computed using linear interpolation between source and target coordinates:

$$
P(k,\tau)=
\left[
P_{\text{src}}(k)\,(1-E(\tau))
\right]
+
\left[
P_{\text{tgt}}(k)\,E(\tau)
\right]
$$
