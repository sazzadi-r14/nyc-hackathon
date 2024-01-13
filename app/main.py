import streamlit as st
from PIL import Image
import base64
import requests
import os
from utils.llm import *
from utils.diffusion import diffuse_from_text
from image_resize import fit_image_to_box

test1 = test1()

def encode_image(image):
  img_data = image.convert("RGB").save("temp.jpg", "JPEG")
  with open("temp.jpg", "rb") as img_file:
    return base64.b64encode(img_file.read()).decode('utf-8')


def main():
  st.title("Weather Condition Transformer")
  api_key = os.environ['OPENAI_API_KEY']

  uploaded_file = st.file_uploader("Choose an Image...", type="jpg")
  if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")

    # Save the image as a PNG file in the 'temp_img' folder
    temp_img_folder = "../temp_img"
    os.makedirs(temp_img_folder, exist_ok=True)
    temp_img_path = os.path.join(temp_img_folder, 'test1.png')
    image.save(temp_img_path, format='PNG')

    # Resize to 1024 by 1024
    fit_image_to_box("../temp_img/test1.png", 1024)

    

    # base64_image = encode_image(image)

    weather_condition = st.text_input(
        "What weather condition do you want to see in the image?")
    if st.button("Transform"):
      result = test1.generate_prompt(weather_condition)
      diffused_image = diffuse_from_text(result, "test1")
      st.write(result)
      st.image("../temp_img_out/img2img_0.png", caption='diffused_image', use_column_width=True)


if __name__ == '__main__':
  main()
