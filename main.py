import cv2
import numpy as np


def prepare_image(image_path, target_size=(512, 512), to_grayscale=False):
    """Loads, resizes, and optionally converts an image to grayscale."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Error: Could not load '{image_path}'. Check the file name.")

    resized_image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

    if to_grayscale:
        return cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    return resized_image


def rearrange_pixels(source_color_img, target_gray_img):
    """Sorts source pixels by brightness and maps them to the target."""
    source_gray = cv2.cvtColor(source_color_img, cv2.COLOR_BGR2GRAY)

    # Flatten arrays
    source_color_flat = source_color_img.reshape(-1, 3)
    source_gray_flat = source_gray.flatten()
    target_gray_flat = target_gray_img.flatten()

    # Get sorting indices (remembers original X/Y coordinates)
    source_sort_indices = np.argsort(source_gray_flat)
    target_sort_indices = np.argsort(target_gray_flat)

    # Sort source colors and map to target addresses
    sorted_source_colors = source_color_flat[source_sort_indices]
    output_flat = np.zeros_like(source_color_flat)
    output_flat[target_sort_indices] = sorted_source_colors

    # Reshape back to 2D image
    return output_flat.reshape(source_color_img.shape)


if __name__ == "__main__":
    # --- CONFIGURATION ---
    # Change these strings to match your exact downloaded file names
    TARGET_FILE = 'brad_pitt.jpg'
    SOURCE_FILE = 'my_random_picture.jpg'
    OUTPUT_FILE = 'final_result.jpg'

    print("Loading and preparing images...")
    target_img = prepare_image(TARGET_FILE, target_size=(512, 512), to_grayscale=True)
    source_img = prepare_image(SOURCE_FILE, target_size=(512, 512), to_grayscale=False)

    print("Rearranging pixels... (this takes a fraction of a second)")
    result_img = rearrange_pixels(source_img, target_img)

    print(f"Saving result to {OUTPUT_FILE}...")
    cv2.imwrite(OUTPUT_FILE, result_img)

    # Display the result on screen
    cv2.imshow("Pixel Sorted Image", result_img)
    print("Press any key on the image window to close it.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()