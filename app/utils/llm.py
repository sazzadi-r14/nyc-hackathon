import logging
import copy
import numpy as np
import requests
import json
import openai
import backoff
import os
from dotenv import load_dotenv
import time

from ratelimit import limits, sleep_and_retry
import base64
import json

load_dotenv()

logs_dir = "logs"
if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)

logging.basicConfig(filename=os.path.join(logs_dir, 'lm_class.log'), 
                    level=logging.INFO, 
                    format='%(asctime)s:%(levelname)s:%(message)s')

def backoff_hdlr(details):
    logging.warning("Backing off {wait:0.1f} seconds after {tries} tries calling function {target} with args {args} and kwargs {kwargs}".format(**details))


logger = logging.getLogger(__name__)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY
GPT_MODEL = "gpt-4-0613"

# Define rate limit
ONE_MINUTE = 60

class LLMPromptGeneration(object):
    def __init__(self, name):
        self.name = name
        
    @sleep_and_retry
    @limits(calls=40, period=ONE_MINUTE)
    @backoff.on_exception(backoff.expo, (requests.exceptions.RequestException, Exception), max_tries=10, on_backoff=backoff_hdlr)
    def post_correction(self, document_metadata):
        raise NotImplementedError()



class test1(LLMPromptGeneration):
    def __init__(self):
        super().__init__("test-v1")
        
    @sleep_and_retry
    @limits(calls=40, period=ONE_MINUTE)
    @backoff.on_exception(backoff.expo, (requests.exceptions.RequestException, Exception), max_tries=10, on_backoff=backoff_hdlr)
    def generate_prompt(self, user_prompt):
        
        messages = [
            {
                "role": "system",
                "content": "You will be given a weather condition. Based on this condition, generate 1 prompt describing extreme edge cases of this type of weather. These weather condition descriptions will be used for stable diffusion to create images of adverse weather conditions on top of autonomous vehicle driving images to simulate adverse driving conditions. Do not focus on how this affects the car, focus on how this impacts/obscures the surroundings.",
            },
            {
                "role": "user",
                "content": f"""user's prompt: {user_prompt}
return each of the prompts generated as a string value in a python dictionary 
return code in python list format.

example: [same_in_all + (user prompted weather condition)]

where "same_in_all" quote is:
You have to create an image based off of this image where you change the weather condition of the originla image. Under no condition should you eliminate any objects, people, cars, from the scene. Photorealistic, not at all like a cartoon, should look like real world. It is absolutely essential that the genereated image has the same point of view as the original one. The weather condition is:
""",
            },
        ]
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {OPENAI_API_KEY}",
        }

        payload = {
            "model": GPT_MODEL,
            "messages": messages,
            "max_tokens": 2042,
            "temperature": 0.0,
        }

        try:
            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, data=json.dumps(payload))
            response.raise_for_status()  # Raises a HTTPError if the response was an HTTP 4xx or 5xx
            response_json = response.json()
            result = response_json["choices"][0]["message"]["content"].strip()
            return result
        except requests.exceptions.HTTPError as errh:
            logging.error(f"HTTP Error: {errh}")
            return "failed"
        except requests.exceptions.ConnectionError as errc:
            logging.error(f"Error Connecting: {errc}")
            return "failed"
        except requests.exceptions.Timeout as errt:
            logging.error(f"Timeout Error: {errt}")
            return "failed"
        except requests.exceptions.RequestException as err:
            logging.error(f"Something went wrong with the request: {err}")
            return "failed"
        except Exception as e:
            logging.error(f"Unknown error: {e}")
            return "failed"
