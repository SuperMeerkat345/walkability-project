import os

import torch
import torchvision
import transformers

# FOR API
from dotenv import load_dotenv
import requests

load_dotenv() # Get the env variables into the environment

API_KEY = os.getenv('HUGGINGFACE_API_KEY')
API_URL = "https://api-inference.huggingface.co/models/openai/clip-vit-base-patch32"

test_img = "./ValidationSet/39035117400/(41.56136772139518, -81.56501323510685)_270.png"
headers = {"Authorization": f"Bearer {API_KEY}"}  


prompts = [
    "No visible sidewalks, paths, or pedestrian infrastructure. No safe crossing points or pedestrian-friendly zones. Unsafe or obstructed pathways (e.g., heavy traffic, no buffer from cars). Wide roads with fast-moving traffic, making pedestrians feel exposed. Unpleasant, unattractive environment (e.g., industrial feel, trash, graffiti, no greenery).",
    "Sidewalks may be present but are narrow, poorly maintained, or interrupted. Inconsistent or minimal pedestrian infrastructure (e.g., sidewalk gaps). Unsafe crossings or poorly lit areas. Roads are still wide with limited traffic-calming features. Sparse landscaping or aesthetic elements; few trees or visual appeal.",
    "Sidewalks are visible but may be narrow, uneven, or partially obstructed. Some pedestrian infrastructure, but with maintenance or connectivity issues. A few crosswalks, but crossing can still be difficult or intimidating. Moderately sized roads; some pedestrian protection like curb extensions or medians. Some greenery or pleasant features (trees, shopfronts), but not consistent.",
    "Wide, well-maintained sidewalks that allow comfortable walking. Clear, safe crosswalks with pedestrian-friendly traffic signals. Visible amenities like benches, trees, or small parks. Streets are narrower or include pedestrian buffers, making crossing safer. Clean and visually appealing environment, with consistent shade and landscaping.",
    "Wide, smooth, continuous sidewalks in excellent condition. Numerous, well-designed crosswalks and pedestrian-only areas. Abundant amenities like benches, parks, cafés, and shaded spots. Narrow or traffic-calmed streets that prioritize pedestrians over cars. Highly attractive surroundings: clean, landscaped, well-lit, and inviting.",
]

def query(image_path, prompt):
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    files = {
        "image": image_bytes,
    }
    data = {
        "inputs": prompt  # The CLIP model expects a text input as the 'inputs' field
    }
    
    # Make the POST request
    response = requests.post(API_URL, headers=headers, data=data, files=files)

    # Check the response
    if response.status_code == 200:
        result = response.json()
        return result
    else:
        print("Error:", response.status_code, response.text)
        return None


response = query(test_img, prompts[0])
print(response)