import requests
import os
from dotenv import load_dotenv
import base64
import cv2
import numpy as np
import time
from tqdm import tqdm

load_dotenv()


def diffuse_from_text(weather_prompt, image_name, output_dir, counter):
    engine_id = "stable-diffusion-xl-1024-v1-0"
    api_host = os.getenv("API_HOST", "https://api.stability.ai")
    api_key = os.getenv("STABILITY_API_KEY")

    if api_key is None:
        raise Exception("Missing Stability API key.")

    response = requests.post(
        f"{api_host}/v1/generation/{engine_id}/image-to-image",
        headers={
            "Accept": "application/json",
            "Authorization": f"Bearer {api_key}"
        },
        files={
            "init_image": open(image_name, "rb")
        },
        data={
            "image_strength": 0.45,
            "init_image_mode": "IMAGE_STRENGTH",
            "text_prompts[0][text]": f"{weather_prompt}",
            "cfg_scale": 7,
            "samples": 1,
            "steps": 30,
        }
    )

    if response.status_code != 200:
        raise Exception("Non-200 response: " + str(response.text))

    data = response.json()

    for i, image in enumerate(data["artifacts"]):
        output_filename = f"img2img_{counter}_{i}.png"
        with open(os.path.join(output_dir, output_filename), "wb") as f:
            f.write(base64.b64decode(image["base64"]))

def fit_image_to_box(image_path, box_size, output_path):
    img = cv2.imread(image_path)
    original_height, original_width, _ = img.shape
    width_scale = box_size / original_width
    height_scale = box_size / original_height
    min_scale = min(width_scale, height_scale)
    new_width = int(original_width * min_scale)
    new_height = int(original_height * min_scale)
    resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
    result_img = 255 * np.ones((box_size, box_size, 3), dtype=np.uint8)
    paste_position = ((box_size - new_width) // 2, (box_size - new_height) // 2)
    result_img[paste_position[1]:paste_position[1] + new_height,
               paste_position[0]:paste_position[0] + new_width] = resized_img
    cv2.imwrite(output_path, result_img)

def process_images(input_dir, output_dir, weather_prompt):
    counter = 0
    for filename in tqdm(os.listdir(input_dir)):
        if filename.endswith(".png"):
            image_path = os.path.join(input_dir, filename)
            resized_image_path = os.path.join(input_dir, "resized_" + filename)
            fit_image_to_box(image_path, 1024, resized_image_path)
            diffuse_from_text(weather_prompt, resized_image_path, output_dir, counter)
            counter += 1

input_dir = "data"  # Source directory containing PNG images
output_dir = "diffused_data"  # Directory where processed images will be saved
weather_prompt = "Create new version of the given image, in a heavily snowy condition. Do not change anything else, just change the weather condition, and make it snowy. Do not remove any of the people or objects form the image, keep it's location exactly where it is."

process_images(input_dir, output_dir, weather_prompt)


