import requests
import os
from dotenv import load_dotenv

load_dotenv()

"""
vid_data object contains
{
    "image_name": file name of png to diffuse
}
"""

def diffuse_video(image_name):
    response = requests.post(
        "https://api.stability.ai/v2alpha/generation/image-to-video",
        headers={
            "authorization": f"Bearer {os.getenv('STABILITY_AI_KEY')}",
        },
        data={
            "seed": 0,
            "cfg_scale": 2.5,
            "motion_bucket_id": 40
        },
        files={
            "image": ("file", open(f"../temp_img_out/{image_name}.png", "rb"), "image/png")
        },
    )

    if response.status_code != 200:
        raise Exception("Non-200 response: " + str(response.text))

    data = response.json()
    generation_id = data["id"]
    print(generation_id)

    # # //////////////////////////////////////////////////////////////////////////////////////////////////
    # generation_id = "d51f96f4ae9653e7433b921b607acf906fe5d3bbce5fdb83f7ff2a74799beb47"

    response = requests.request(
        "GET",
        f"https://api.stability.ai/v2alpha/generation/image-to-video/result/{generation_id}",
        headers={
            'Accept': None, # Use 'application/json' to receive base64 encoded JSON
            'authorization': f"{os.getenv('STABILITY_AI_KEY')}"
        },
    )

    if response.status_code == 202:
        print("Generation in-progress, try again in 10 seconds.")
    elif response.status_code == 200:
        print("Generation complete!")
        with open('../temp_vid_out/output.mp4', 'wb') as file:
            file.write(response.content)
    else:
        raise Exception("Non-200 response: " + str(response.json()))
    
vid_data = {
            "image_name": "img2img_0"
            }

diffuse_video(vid_data["image_name"])