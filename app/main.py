import streamlit as st
from PIL import Image
import base64
import requests
import os
from utils.llm import *

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

    base64_image = encode_image(image)

    weather_condition = st.text_input(
        "What weather condition do you want to see in the image?")
    if st.button("Transform"):
      result = test1.generate_prompt(weather_condition)
      st.write(result)


if __name__ == '__main__':
  main()
