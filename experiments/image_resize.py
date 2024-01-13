from PIL import Image 

def fit_image_to_box(image_path, box_size):
    # Open the image
    img = Image.open(image_path)

    # Get the original dimensions
    original_width, original_height = img.size

    # Calculate the scaling factors for width and height
    width_scale = box_size / original_width
    height_scale = box_size / original_height

    # Choose the minimum scaling factor to fit the image into the box
    min_scale = min(width_scale, height_scale)

    # Calculate the new dimensions
    new_width = int(original_width * min_scale)
    new_height = int(original_height * min_scale)

    # Resize the image
    resized_img = img.resize((new_width, new_height), Image.ANTIALIAS)

    # Create a new blank image with the target size
    result_img = Image.new('RGB', (box_size, box_size), (0, 0, 0))

    # Calculate the position to paste the resized image
    paste_position = ((box_size - new_width) // 2, (box_size - new_height) // 2)

    # Paste the resized image onto the blank image
    result_img.paste(resized_img, paste_position)

    # Save or display the result
    result_img.show()