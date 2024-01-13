import cv2
import numpy as np

def fit_image_to_box(image_path, box_size):
    # Read the image using cv2
    img = cv2.imread(image_path)

    # Get the original dimensions
    original_height, original_width, _ = img.shape

    # Calculate the scaling factors for width and height
    width_scale = box_size / original_width
    height_scale = box_size / original_height

    # Choose the minimum scaling factor to fit the image into the box
    min_scale = min(width_scale, height_scale)

    # Calculate the new dimensions
    new_width = int(original_width * min_scale)
    new_height = int(original_height * min_scale)

    # Resize the image using cv2
    resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # Create a new blank image with the target size
    result_img = 255 * np.ones((box_size, box_size, 3), dtype=np.uint8)

    # Calculate the position to paste the resized image
    paste_position = ((box_size - new_width) // 2, (box_size - new_height) // 2)

    # Paste the resized image onto the blank image
    result_img[paste_position[1]:paste_position[1] + new_height,
               paste_position[0]:paste_position[0] + new_width] = resized_img

    # Save or display the result
    cv2.imwrite("../temp_img/test1.png", result_img)