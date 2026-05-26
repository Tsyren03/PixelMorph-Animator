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

Mathematical Model: The "Perfect Morph" Algorithm
The core logic of the animate_perfect_morph function can be defined as a bijective spatial mapping driven by luminance sorting and interpolated via a smooth kinematic function.
1. Definitions and Setup
Let I_src be the source color image and I_tgt be the target grayscale image. Both images share the exact same dimensions, containing a total of N = H × W pixels.
Let Y_src and Y_tgt represent one-dimensional sets containing the luminance (brightness) values of every pixel:
Y_src = { y_1, y_2, ..., y_N }
Y_tgt = { y_1, y_2, ..., y_N }
2. Permutation (Brightness Sorting)
The algorithm calculates two sorting functions (permutations), σ_s and σ_t, which arrange the luminance arrays into strict ascending order:
Y_src[σ_s(k)] ≤ Y_src[σ_s(k+1)]
Y_tgt[σ_t(k)] ≤ Y_tgt[σ_t(k+1)]
Here, k represents the "rank" of a pixel based on its brightness, where k=1 is the absolute darkest pixel and k=N is the absolute brightest.
3. Bijective Spatial Mapping
Because both arrays are now ordered identically by rank k, a strict one-to-one mapping is established. The pixel originally located at the 2D coordinate of the k-th darkest spot in the source image, P_src(k), is mathematically assigned to travel to the 2D coordinate of the k-th darkest spot in the target image, P_tgt(k).
4. Animation Kinematics
To animate the transition over a given number of frames, let τ (tau) represent the normalized time of the animation, where τ ∈ [0, 1]. To ensure natural movement instead of rigid linear speed, a Smoothstep function E(τ) is applied as an easing curve:
E(τ) = τ² × (3 - 2τ)
Finally, the exact 2D position P(k, τ) of any pixel k at time τ is calculated using linear interpolation between its starting and ending coordinates:
P(k, τ) = [ P_src(k) × (1 - E(τ)) ] + [ P_tgt(k) × E(τ) ]
