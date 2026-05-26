import cv2
import numpy as np


def crop_and_resize(image, target_size=(1024, 1024)):
    h, w = image.shape[:2]
    size = min(h, w)
    y1, y2 = (h - size) // 2, (h + size) // 2
    x1, x2 = (w - size) // 2, (w + size) // 2
    cropped = image[y1:y2, x1:x2]
    return cv2.resize(cropped, target_size, interpolation=cv2.INTER_AREA)


def prepare_image(image_path, target_size=(1024, 1024), to_grayscale=False):
    image = cv2.imread(image_path)
    if image is None: raise ValueError(f"Error: Could not load image at '{image_path}'.")
    resized_image = crop_and_resize(image, target_size)
    if to_grayscale: return cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    return resized_image


def ease_in_out(t):
    return t * t * (3.0 - 2.0 * t)


# ==========================================
# ALGORITHM 1: 1D Brightness Sort (Standard)
# ANIMATION: "The Grid Collapse" (Standard Linear read)
# ==========================================
def rearrange_pixels(source_color_img, target_gray_img):
    source_gray = cv2.cvtColor(source_color_img, cv2.COLOR_BGR2GRAY)
    source_color_flat = source_color_img.reshape(-1, 3)
    source_gray_flat = source_gray.flatten()
    target_gray_flat = target_gray_img.flatten()

    source_sort_indices = np.argsort(source_gray_flat)
    target_sort_indices = np.argsort(target_gray_flat)

    sorted_source_colors = source_color_flat[source_sort_indices]
    output_flat = np.zeros_like(source_color_flat)
    output_flat[target_sort_indices] = sorted_source_colors
    return output_flat.reshape(source_color_img.shape)


def animate_perfect_morph(source_color_img, target_color_img, output_filename='perfect_morph.mp4', num_frames=180,
                          fps=60):
    h, w, _ = source_color_img.shape
    source_gray = cv2.cvtColor(source_color_img, cv2.COLOR_BGR2GRAY)
    target_gray = cv2.cvtColor(target_color_img, cv2.COLOR_BGR2GRAY)

    source_color_flat = source_color_img.reshape(-1, 3)
    source_sort_indices = np.argsort(source_gray.flatten())
    target_sort_indices = np.argsort(target_gray.flatten())

    # Animation 1: Pixels fly from their exact original grid locations
    start_y = source_sort_indices // w
    start_x = source_sort_indices % w

    end_y = target_sort_indices // w
    end_x = target_sort_indices % w
    colors = source_color_flat[source_sort_indices]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_filename, fourcc, fps, (w, h))

    for frame_idx in range(num_frames):
        t = frame_idx / (num_frames - 1)
        smooth_t = ease_in_out(t)
        cur_y = (start_y * (1 - smooth_t) + end_y * smooth_t).astype(np.int32)
        cur_x = (start_x * (1 - smooth_t) + end_x * smooth_t).astype(np.int32)
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[cur_y, cur_x] = colors
        out.write(frame)

    final_frame = np.zeros((h, w, 3), dtype=np.uint8)
    final_frame[end_y, end_x] = colors
    for _ in range(fps * 2): out.write(final_frame)
    out.release()


# ==========================================
# ALGORITHM 2: 3D Lexicographical Sort (Pro)
# ANIMATION: "The RGB Separation" (Pixels fly in from R, G, B corners)
# ==========================================
def rearrange_pixels_lexsort(source_color_img, target_color_img):
    src_flat = source_color_img.reshape(-1, 3)
    tgt_flat = target_color_img.reshape(-1, 3)
    src_sort_idx = np.lexsort(src_flat.T)
    tgt_sort_idx = np.lexsort(tgt_flat.T)
    output_flat = np.zeros_like(src_flat)
    output_flat[tgt_sort_idx] = src_flat[src_sort_idx]
    return output_flat.reshape(source_color_img.shape)


def animate_lexsort_morph(source_color_img, target_color_img, output_filename='lexsort_morph.mp4', num_frames=180,
                          fps=60):
    h, w, _ = source_color_img.shape
    src_flat = source_color_img.reshape(-1, 3)
    tgt_flat = target_color_img.reshape(-1, 3)

    src_sort_idx = np.lexsort(src_flat.T)
    tgt_sort_idx = np.lexsort(tgt_flat.T)
    colors = src_flat[src_sort_idx]

    # Animation 2: Pixels start separated based on their dominant color channel
    start_y = np.zeros_like(src_sort_idx)
    start_x = np.zeros_like(src_sort_idx)

    # Red-heavy pixels start Top-Left, Green Top-Right, Blue Bottom-Center
    dominant_channel = np.argmax(colors, axis=1)
    start_y[dominant_channel == 0] = h - 1  # Blue bottom
    start_x[dominant_channel == 0] = w // 2
    start_y[dominant_channel == 1] = 0  # Green top right
    start_x[dominant_channel == 1] = w - 1
    start_y[dominant_channel == 2] = 0  # Red top left
    start_x[dominant_channel == 2] = 0

    end_y = tgt_sort_idx // w
    end_x = tgt_sort_idx % w

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_filename, fourcc, fps, (w, h))

    for frame_idx in range(num_frames):
        t = frame_idx / (num_frames - 1)
        smooth_t = ease_in_out(t)
        cur_y = (start_y * (1 - smooth_t) + end_y * smooth_t).astype(np.int32)
        cur_x = (start_x * (1 - smooth_t) + end_x * smooth_t).astype(np.int32)
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[cur_y, cur_x] = colors
        out.write(frame)

    final_frame = np.zeros((h, w, 3), dtype=np.uint8)
    final_frame[end_y, end_x] = colors
    for _ in range(fps * 2): out.write(final_frame)
    out.release()


# ==========================================
# ALGORITHM 3: Perceptual Color Sort (HSV Space)
# ANIMATION: "The Hue Tornado" (Pixels spin in based on their Hue)
# ==========================================
def rearrange_pixels_hsv(source_color_img, target_color_img):
    src_hsv = cv2.cvtColor(source_color_img, cv2.COLOR_BGR2HSV)
    tgt_hsv = cv2.cvtColor(target_color_img, cv2.COLOR_BGR2HSV)
    src_flat = source_color_img.reshape(-1, 3)
    src_hsv_flat = src_hsv.reshape(-1, 3)
    tgt_hsv_flat = tgt_hsv.reshape(-1, 3)

    src_sort_idx = np.lexsort((src_hsv_flat[:, 0], src_hsv_flat[:, 1], src_hsv_flat[:, 2]))
    tgt_sort_idx = np.lexsort((tgt_hsv_flat[:, 0], tgt_hsv_flat[:, 1], tgt_hsv_flat[:, 2]))
    output_flat = np.zeros_like(src_flat)
    output_flat[tgt_sort_idx] = src_flat[src_sort_idx]
    return output_flat.reshape(source_color_img.shape)


def animate_hsv_morph(source_color_img, target_color_img, output_filename='hsv_morph.mp4', num_frames=180, fps=60):
    h, w, _ = source_color_img.shape
    src_hsv = cv2.cvtColor(source_color_img, cv2.COLOR_BGR2HSV)
    tgt_hsv = cv2.cvtColor(target_color_img, cv2.COLOR_BGR2HSV)
    src_flat = source_color_img.reshape(-1, 3)
    src_hsv_flat = src_hsv.reshape(-1, 3)
    tgt_hsv_flat = tgt_hsv.reshape(-1, 3)

    src_sort_idx = np.lexsort((src_hsv_flat[:, 0], src_hsv_flat[:, 1], src_hsv_flat[:, 2]))
    tgt_sort_idx = np.lexsort((tgt_hsv_flat[:, 0], tgt_hsv_flat[:, 1], tgt_hsv_flat[:, 2]))
    colors = src_flat[src_sort_idx]
    hues = src_hsv_flat[src_sort_idx, 0]  # Extract sorted Hues

    # Animation 3: Pixels start in a circle, positioned by their Hue angle
    # Hue in OpenCV is 0-179. We map it to 0-360 degrees (in radians)
    angles = (hues / 179.0) * 2 * np.pi
    radius = min(h, w) // 2
    center_y, center_x = h // 2, w // 2

    start_y = (center_y + radius * np.sin(angles)).astype(np.int32)
    start_x = (center_x + radius * np.cos(angles)).astype(np.int32)

    end_y = tgt_sort_idx // w
    end_x = tgt_sort_idx % w

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_filename, fourcc, fps, (w, h))

    for frame_idx in range(num_frames):
        t = frame_idx / (num_frames - 1)
        smooth_t = ease_in_out(t)
        cur_y = (start_y * (1 - smooth_t) + end_y * smooth_t).astype(np.int32)
        cur_x = (start_x * (1 - smooth_t) + end_x * smooth_t).astype(np.int32)
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[cur_y, cur_x] = colors
        out.write(frame)

    final_frame = np.zeros((h, w, 3), dtype=np.uint8)
    final_frame[end_y, end_x] = colors
    for _ in range(fps * 2): out.write(final_frame)
    out.release()


# ==========================================
# ALGORITHM 4: PCA Projection Sort (Dynamic)
# ANIMATION: "The Data Stream" (Pixels flow down from a single line)
# ==========================================
def get_pca_sort_indices(color_img):
    flat_img = color_img.reshape(-1, 3).astype(np.float32)
    mean_color = np.mean(flat_img, axis=0)
    centered = flat_img - mean_color
    cov = np.dot(centered.T, centered) / (centered.shape[0] - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    pc1 = eigenvectors[:, np.argmax(eigenvalues)]
    projection = np.dot(centered, pc1)
    return np.argsort(projection)


def rearrange_pixels_pca(source_color_img, target_color_img):
    src_flat = source_color_img.reshape(-1, 3)
    src_sort_idx = get_pca_sort_indices(source_color_img)
    tgt_sort_idx = get_pca_sort_indices(target_color_img)
    output_flat = np.zeros_like(src_flat)
    output_flat[tgt_sort_idx] = src_flat[src_sort_idx]
    return output_flat.reshape(source_color_img.shape)


def animate_pca_morph(source_color_img, target_color_img, output_filename='pca_morph.mp4', num_frames=180, fps=60):
    h, w, _ = source_color_img.shape
    src_flat = source_color_img.reshape(-1, 3)
    src_sort_idx = get_pca_sort_indices(source_color_img)
    tgt_sort_idx = get_pca_sort_indices(target_color_img)
    colors = src_flat[src_sort_idx]

    # Animation 4: Pixels start arranged linearly across the top to represent the 1D PCA projection
    start_y = np.zeros_like(src_sort_idx)
    start_x = np.linspace(0, w - 1, len(src_sort_idx)).astype(np.int32)

    end_y = tgt_sort_idx // w
    end_x = tgt_sort_idx % w

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_filename, fourcc, fps, (w, h))

    for frame_idx in range(num_frames):
        t = frame_idx / (num_frames - 1)
        smooth_t = ease_in_out(t)
        cur_y = (start_y * (1 - smooth_t) + end_y * smooth_t).astype(np.int32)
        cur_x = (start_x * (1 - smooth_t) + end_x * smooth_t).astype(np.int32)
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[cur_y, cur_x] = colors
        out.write(frame)

    final_frame = np.zeros((h, w, 3), dtype=np.uint8)
    final_frame[end_y, end_x] = colors
    for _ in range(fps * 2): out.write(final_frame)
    out.release()


# ==========================================
# ALGORITHM 5: Sobel Edge-Mapping Sort
# ANIMATION: "The Edge Implosion" (Pixels explode from the center out to the edges)
# ==========================================
def get_sobel_target_indices(target_color_img):
    target_gray = cv2.cvtColor(target_color_img, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(target_gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(target_gray, cv2.CV_64F, 0, 1, ksize=3)
    target_edges = cv2.magnitude(sobelx, sobely).flatten()
    return np.argsort(target_edges)


def rearrange_pixels_sobel(source_color_img, target_color_img):
    src_flat = source_color_img.reshape(-1, 3)
    source_gray = cv2.cvtColor(source_color_img, cv2.COLOR_BGR2GRAY).flatten()
    source_sort_idx = np.argsort(source_gray)
    target_sort_idx = get_sobel_target_indices(target_color_img)

    output_flat = np.zeros_like(src_flat)
    output_flat[target_sort_idx] = src_flat[source_sort_idx]
    return output_flat.reshape(source_color_img.shape)


def animate_sobel_morph(source_color_img, target_color_img, output_filename='sobel_morph.mp4', num_frames=180, fps=60):
    h, w, _ = source_color_img.shape
    src_flat = source_color_img.reshape(-1, 3)
    source_gray = cv2.cvtColor(source_color_img, cv2.COLOR_BGR2GRAY).flatten()
    source_sort_idx = np.argsort(source_gray)
    target_sort_idx = get_sobel_target_indices(target_color_img)
    colors = src_flat[source_sort_idx]

    # Animation 5: Pixels explode outward from the exact center point
    start_y = np.full_like(source_sort_idx, h // 2)
    start_x = np.full_like(source_sort_idx, w // 2)

    end_y = target_sort_idx // w
    end_x = target_sort_idx % w

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_filename, fourcc, fps, (w, h))

    for frame_idx in range(num_frames):
        t = frame_idx / (num_frames - 1)
        smooth_t = ease_in_out(t)
        cur_y = (start_y * (1 - smooth_t) + end_y * smooth_t).astype(np.int32)
        cur_x = (start_x * (1 - smooth_t) + end_x * smooth_t).astype(np.int32)
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[cur_y, cur_x] = colors
        out.write(frame)

    final_frame = np.zeros((h, w, 3), dtype=np.uint8)
    final_frame[end_y, end_x] = colors
    for _ in range(fps * 2): out.write(final_frame)
    out.release()