import cv2
import gradio as gr
from morph_engine import (
    crop_and_resize,
    rearrange_pixels,
    animate_perfect_morph,
    rearrange_pixels_lexsort,
    animate_lexsort_morph,
    rearrange_pixels_hsv,
    animate_hsv_morph,
    rearrange_pixels_pca,
    animate_pca_morph,
    rearrange_pixels_sobel,  # <-- Algorithm 5 Import
    animate_sobel_morph  # <-- Algorithm 5 Import
)


def process_morph(source_img, target_img, resolution_size, num_frames, algorithm_choice):
    fps = 60

    if source_img is None or target_img is None:
        raise gr.Error("Please upload both a Source and Target image.")

    target_size = (resolution_size, resolution_size)

    # 1. Color Conversion
    if source_img.shape[-1] == 4:
        source_bgr = cv2.cvtColor(source_img, cv2.COLOR_RGBA2BGR)
    else:
        source_bgr = cv2.cvtColor(source_img, cv2.COLOR_RGB2BGR)

    if target_img.shape[-1] == 4:
        target_bgr = cv2.cvtColor(target_img, cv2.COLOR_RGBA2BGR)
    else:
        target_bgr = cv2.cvtColor(target_img, cv2.COLOR_RGB2BGR)

    src_bgr = crop_and_resize(source_bgr, target_size)
    tgt_bgr = crop_and_resize(target_bgr, target_size)
    output_video_path = "web_morph_output.mp4"

    # 2. Routing Logic & Dynamic Explanations
    if algorithm_choice == "Algorithm 5: Sobel Edge-Mapping Sort (Computer Vision)":
        static_bgr = rearrange_pixels_sobel(src_bgr, tgt_bgr)
        animate_sobel_morph(src_bgr, tgt_bgr, output_video_path, num_frames, fps)
        explanation = """
        ### 🔪 How it Works: Sobel Edge-Mapping Sort
        This algorithm introduces **Convolutional Spatial Filtering**. It uses a Sobel Operator (a core computer vision edge-detection filter) to calculate the spatial gradient and find all high-contrast edges in the Target image.
        It then sorts the Target image by edge magnitude and sorts the Source image by brightness. 

        **The Result:** The absolute brightest pixels of the source map directly onto the strongest edges of the target, while dark pixels fill the flat backgrounds. It creates a spectacular glowing outline effect.
        """

    elif algorithm_choice == "Algorithm 4: PCA Projection Sort (Dynamic)":
        static_bgr = rearrange_pixels_pca(src_bgr, tgt_bgr)
        animate_pca_morph(src_bgr, tgt_bgr, output_video_path, num_frames, fps)
        explanation = """
        ### 📊 How it Works: Principal Component Analysis (PCA)
        This algorithm analyzes the specific data of the images uploaded. 
        It calculates the **Covariance Matrix** of all RGB pixels to find an **Eigenvector** that represents the axis of maximum color variance. It then uses a dot product to project every 3D pixel onto this custom 1D line for sorting.

        **The Result:** A dynamically adapted color sort that shifts completely depending on the specific palettes of the uploaded images.
        """

    elif algorithm_choice == "Algorithm 3: Perceptual Color Sort (HSV Space)":
        static_bgr = rearrange_pixels_hsv(src_bgr, tgt_bgr)
        animate_hsv_morph(src_bgr, tgt_bgr, output_video_path, num_frames, fps)
        explanation = """
        ### 🧠 How it Works: Perceptual Color Sort (HSV Space)
        This algorithm converts the images into **HSV (Hue, Saturation, Value)**. 
        It sorts the pixels exactly how the human eye perceives light: matching absolute brightness, breaking ties with color saturation, and breaking final ties with exact color hue.

        **The Result:** A perfectly smooth 1-to-1 mapping that maintains stunning color distribution and highly realistic lighting.
        """

    elif algorithm_choice == "Algorithm 2: 3D Color Match (RGB Lexicographical)":
        static_bgr = rearrange_pixels_lexsort(src_bgr, tgt_bgr)
        animate_lexsort_morph(src_bgr, tgt_bgr, output_video_path, num_frames, fps)
        explanation = """
        ### 🟥🟩🟦 How it Works: 3D Color Match (RGB Lexicographical)
        This algorithm uses a **Lexicographical Sort** (like alphabetical order) on the raw pixel data. It ranks every pixel from lowest to highest based on Blue, then breaks ties with Green, and finally Red.

        **The Result:** Full color retention, but because computers process RGB linearly while humans do not, the final image features a highly artistic, "glitchy" mosaic texture.
        """

    else:
        tgt_gray = cv2.cvtColor(tgt_bgr, cv2.COLOR_BGR2GRAY)
        static_bgr = rearrange_pixels(src_bgr, tgt_gray)
        animate_perfect_morph(src_bgr, tgt_bgr, output_video_path, num_frames, fps)
        explanation = """
        ### 🌗 How it Works: 1D Brightness Sort (Standard)
        This is the foundational algorithm. It flattens the target image into 1D grayscale to create a strict "Luminance Map." 
        It then finds the absolute darkest pixel in the source image and assigns it to the absolute darkest spot in the target layout, proceeding up to the brightest white.

        **The Result:** A perfectly smooth shape reconstruction that acts like a monochromatic "tint" or "color filter."
        """

    # 3. Output Translation
    src_rgb = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2RGB)
    tgt_rgb = cv2.cvtColor(tgt_bgr, cv2.COLOR_BGR2RGB)
    static_rgb = cv2.cvtColor(static_bgr, cv2.COLOR_BGR2RGB)

    gallery_images = [
        (src_rgb, "1. Source Image"),
        (tgt_rgb, "2. Target Image"),
        (static_rgb, "3. Final Static Result")
    ]

    return gallery_images, output_video_path, explanation


# ==========================================
# Gradio UI Layout Definition
# ==========================================
with gr.Blocks(theme=gr.themes.Soft()) as ui:
    gr.Markdown("# 🎨 PixelMorph Animator (Master Edition)")
    gr.Markdown(
        "Upload two images and select an algorithm. The pixels from the **Source Image** will be reorganized to form the **Target Image**.")

    with gr.Row():
        source_input = gr.Image(label="Source Image (Pixels)", type="numpy")
        target_input = gr.Image(label="Target Image (Shape)", type="numpy")

    with gr.Row():
        res_slider = gr.Slider(minimum=256, maximum=1024, value=512, step=256, label="Resolution Size (Pixels)")
        frames_slider = gr.Slider(minimum=60, maximum=300, value=120, step=30, label="Animation Frames")

        algo_dropdown = gr.Dropdown(
            choices=[
                "Algorithm 5: Sobel Edge-Mapping Sort (Computer Vision)",
                "Algorithm 4: PCA Projection Sort (Dynamic)",
                "Algorithm 3: Perceptual Color Sort (HSV Space)",
                "Algorithm 2: 3D Color Match (RGB Lexicographical)",
                "Algorithm 1: 1D Brightness Sort (Standard)"
            ],
            value="Algorithm 5: Sobel Edge-Mapping Sort (Computer Vision)",
            label="Mapping Algorithm"
        )

    run_btn = gr.Button("Morph Images!", variant="primary")

    explanation_box = gr.Markdown(
        "### Algorithm Explanation\n*Run the morph to see how your selected algorithm works!*")

    gr.Markdown("### Results")
    image_gallery = gr.Gallery(label="Image Processing Pipeline", columns=3, rows=1, height="auto")
    video_output = gr.Video(label="Morphing Animation Video")

    run_btn.click(
        fn=process_morph,
        inputs=[source_input, target_input, res_slider, frames_slider, algo_dropdown],
        outputs=[image_gallery, video_output, explanation_box],
        api_name=False
    )

if __name__ == "__main__":
    ui.launch(share=True)