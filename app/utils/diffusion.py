import requests
import os
from dotenv import load_dotenv
import base64

load_dotenv()

"""
img_data object contains
{
    "weather_prompt": description of adverse weather,
    "image_name": file name of png to diffuse
}
"""

def diffuse_from_text(weather_prompt, base64_image):

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
            "init_image": base64_image,
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
        with open(f"temp_img_out/img2img_{i}.png", "wb") as f:
            f.write(base64.b64decode(image["base64"]))

