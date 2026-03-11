import cv2
import numpy as np


def crop_and_resize(image, target_size=(1024, 1024)):
    """이미지 비율 왜곡 없이 중앙을 자르고 크기를 맞춘다."""
    h, w = image.shape[:2]
    size = min(h, w)
    y1, y2 = (h - size) // 2, (h + size) // 2
    x1, x2 = (w - size) // 2, (w + size) // 2

    cropped = image[y1:y2, x1:x2]
    return cv2.resize(cropped, target_size, interpolation=cv2.INTER_AREA)


def prepare_image(image_path, target_size=(1024, 1024)):
    """이미지를 불러와 중앙을 자르고 해상도를 맞춘다."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Error: 파일 확인 필요 - '{image_path}'")

    return crop_and_resize(image, target_size)


def ease_in_out(t):
    """애니메이션 속도를 부드럽게 조절한다."""
    return t * t * (3.0 - 2.0 * t)


def animate_premium(source_color_img, target_color_img, output_filename='premium_morph.mp4', num_frames=180, fps=60,
                    block_size=32):
    """고해상도 및 부드러운 애니메이션을 생성한다."""
    h, w, _ = source_color_img.shape
    num_blocks_y, num_blocks_x = h // block_size, w // block_size

    source_gray = cv2.cvtColor(source_color_img, cv2.COLOR_BGR2GRAY)
    target_gray = cv2.cvtColor(target_color_img, cv2.COLOR_BGR2GRAY)

    s_blocks = []
    t_blocks = []
    for by in range(num_blocks_y):
        for bx in range(num_blocks_x):
            sy1, sy2 = by * block_size, (by + 1) * block_size
            sx1, sx2 = bx * block_size, (bx + 1) * block_size

            s_mean = source_color_img[sy1:sy2, sx1:sx2].mean(axis=(0, 1))
            t_mean = target_color_img[sy1:sy2, sx1:sx2].mean(axis=(0, 1))

            s_blocks.append({'idx': (by, bx), 'mean': s_mean})
            t_blocks.append({'idx': (by, bx), 'mean': t_mean})

    matched_s = set()
    pairs = []
    for t in t_blocks:
        best_dist = float('inf')
        best_s = None
        for s in s_blocks:
            if s['idx'] in matched_s:
                continue
            dist = np.linalg.norm(t['mean'] - s['mean'])
            if dist < best_dist:
                best_dist = dist
                best_s = s
        matched_s.add(best_s['idx'])
        pairs.append((best_s['idx'], t['idx']))

    grid_indices = np.arange(h * w).reshape(h, w)
    s_indices = []
    t_indices = []

    for s_idx, t_idx in pairs:
        sby, sbx = s_idx
        tby, tbx = t_idx

        sy1, sy2 = sby * block_size, (sby + 1) * block_size
        sx1, sx2 = sbx * block_size, (sbx + 1) * block_size
        ty1, ty2 = tby * block_size, (tby + 1) * block_size
        tx1, tx2 = tbx * block_size, (tbx + 1) * block_size

        s_flat = grid_indices[sy1:sy2, sx1:sx2].flatten()
        t_flat = grid_indices[ty1:ty2, tx1:tx2].flatten()

        s_g = source_gray[sy1:sy2, sx1:sx2].flatten()
        t_g = target_gray[ty1:ty2, tx1:tx2].flatten()

        s_indices.extend(s_flat[np.argsort(s_g)])
        t_indices.extend(t_flat[np.argsort(t_g)])

    source_sort_indices = np.array(s_indices)
    target_sort_indices = np.array(t_indices)
    source_color_flat = source_color_img.reshape(-1, 3)

    start_y = source_sort_indices // w
    start_x = source_sort_indices % w
    end_y = target_sort_indices // w
    end_x = target_sort_indices % w

    colors = source_color_flat[source_sort_indices]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_filename, fourcc, fps, (w, h))

    print(f"Generating {num_frames} frames at {fps} FPS...")

    for frame_idx in range(num_frames):
        t = frame_idx / (num_frames - 1)
        smooth_t = ease_in_out(t)

        cur_y = (start_y * (1 - smooth_t) + end_y * smooth_t).astype(np.int32)
        cur_x = (start_x * (1 - smooth_t) + end_x * smooth_t).astype(np.int32)

        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[cur_y, cur_x] = colors
        out.write(frame)

        if frame_idx % 20 == 0:
            print(f"Rendered frame {frame_idx}/{num_frames}")

    print("Holding final image for 2 seconds...")
    final_frame = np.zeros((h, w, 3), dtype=np.uint8)
    final_frame[end_y, end_x] = colors
    for _ in range(fps * 2):
        out.write(final_frame)

    out.release()
    print(f"Animation saved to {output_filename}!")


if __name__ == "__main__":
    TARGET_FILE = 'Trump.jpg'
    SOURCE_FILE = 'random.jpg'
    OUTPUT_VIDEO = 'premium_morph.mp4'

    print("고해상도 이미지 전처리 중...")
    target_img = prepare_image(TARGET_FILE, target_size=(1024, 1024))
    source_img = prepare_image(SOURCE_FILE, target_size=(1024, 1024))

    animate_premium(source_img, target_img, output_filename=OUTPUT_VIDEO, num_frames=180, fps=60, block_size=32)