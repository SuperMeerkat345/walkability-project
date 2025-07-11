#https://huggingface.co/docs/transformers/en/model_doc/clip?usage=AutoModel

import requests
import torch
from PIL import Image # download with pip install pillow
from transformers import AutoProcessor, AutoModel
from tqdm import tqdm
import numpy as np

# VARIABLES ===============================================================================
# Establish prompts/labels
labels = [
    "No visible sidewalks, paths, or pedestrian infrastructure. No safe crossing points or pedestrian-friendly zones. Unsafe or obstructed pathways (e.g., heavy traffic, no buffer from cars). Wide roads with fast-moving traffic, making pedestrians feel exposed. Unpleasant, unattractive environment (e.g., industrial feel, trash, graffiti, no greenery).",
    "Sidewalks may be present but are narrow, poorly maintained, or interrupted. Inconsistent or minimal pedestrian infrastructure (e.g., sidewalk gaps). Unsafe crossings or poorly lit areas. Roads are still wide with limited traffic-calming features. Sparse landscaping or aesthetic elements; few trees or visual appeal.",
    "Sidewalks are visible but may be narrow, uneven, or partially obstructed. Some pedestrian infrastructure, but with maintenance or connectivity issues. A few crosswalks, but crossing can still be difficult or intimidating. Moderately sized roads; some pedestrian protection like curb extensions or medians. Some greenery or pleasant features (trees, shopfronts), but not consistent.",
    "Wide, well-maintained sidewalks that allow comfortable walking. Clear, safe crosswalks with pedestrian-friendly traffic signals. Visible amenities like benches, trees, or small parks. Streets are narrower or include pedestrian buffers, making crossing safer. Clean and visually appealing environment, with consistent shade and landscaping.",
    "Wide, smooth, continuous sidewalks in excellent condition. Numerous, well-designed crosswalks and pedestrian-only areas. Abundant amenities like benches, parks, cafés, and shaded spots. Narrow or traffic-calmed streets that prioritize pedestrians over cars. Highly attractive surroundings: clean, landscaped, well-lit, and inviting.",
]

correlation_dic = {
    "No visible sidewalks, paths, or pedestrian infrastructure. No safe crossing points or pedestrian-friendly zones. Unsafe or obstructed pathways (e.g., heavy traffic, no buffer from cars). Wide roads with fast-moving traffic, making pedestrians feel exposed. Unpleasant, unattractive environment (e.g., industrial feel, trash, graffiti, no greenery).": "very_low",
    "Sidewalks may be present but are narrow, poorly maintained, or interrupted. Inconsistent or minimal pedestrian infrastructure (e.g., sidewalk gaps). Unsafe crossings or poorly lit areas. Roads are still wide with limited traffic-calming features. Sparse landscaping or aesthetic elements; few trees or visual appeal.": "low",
    "Sidewalks are visible but may be narrow, uneven, or partially obstructed. Some pedestrian infrastructure, but with maintenance or connectivity issues. A few crosswalks, but crossing can still be difficult or intimidating. Moderately sized roads; some pedestrian protection like curb extensions or medians. Some greenery or pleasant features (trees, shopfronts), but not consistent.": "medium",
    "Wide, well-maintained sidewalks that allow comfortable walking. Clear, safe crosswalks with pedestrian-friendly traffic signals. Visible amenities like benches, trees, or small parks. Streets are narrower or include pedestrian buffers, making crossing safer. Clean and visually appealing environment, with consistent shade and landscaping.": "high",
    "Wide, smooth, continuous sidewalks in excellent condition. Numerous, well-designed crosswalks and pedestrian-only areas. Abundant amenities like benches, parks, cafés, and shaded spots. Narrow or traffic-calmed streets that prioritize pedestrians over cars. Highly attractive surroundings: clean, landscaped, well-lit, and inviting.": "very_high"
}

traitsv1 = [
    "wide sidewalk",
    "visible crosswalk",
    "green trees",
    "benches",
    "cafe or shopfront",
    "trash or graffiti",
    "fast-moving traffic"
]

traitsv2 = [
    "a wide sidewalk where people can walk",
    "a clearly marked pedestrian crosswalk",
    "green trees or landscaping along the street",
    "public benches for people to sit",
    "a cafe or shop with outdoor seating",
    "trash, litter, or graffiti on the street",
    "a road with lots of fast-moving cars",
    "a sidewalk that is flush with the street and hard to distinguish"
]

# NORMALIZATION SCORES
#sidewalk_scores 6.63279427573101
#crosswalk 33.3272013172479
#trees 1.8898278321906994
#benches 19.6483719278534
#cafe 8.340942509817296
#trash 69.92117795427296
#cars 9.542470992665601

traits = traitsv2
# END VARIABLES ==========================================================================

# Where imgs_path is an array of images
def labelImgs(imgs_path):
    imgs = [Image.open(img_path) for img_path in tqdm(imgs_path, desc="Loading images")]

    # Load the model
    model = AutoModel.from_pretrained(
        "openai/clip-vit-base-patch32", # Model name to load from huggingface
        torch_dtype=torch.bfloat16, # Data type precision
        attn_implementation="sdpa") # Attention mechanism

    processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

    trait_results = []
    scores = []

    crack_scores = []

    for img in tqdm(imgs, desc="Running images through CLIP"):
        inputs = processor( # convert PARAMETERS/INPUTS into readable format for model
            text=traits, # parameters
            images=[img], # image
            return_tensors="pt",
            padding=True
        )

        with torch.no_grad():
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1).squeeze()  # Shape: (num_traits,)

        # Build a dictionary of {trait: score}
        image_trait_scores = {trait: float(probs[i]) for i, trait in enumerate(traits)}

        trait_results.append(image_trait_scores)
        crack_scores.append(image_trait_scores["a sidewalk that is flush with the street and hard to distinguish"])

    print(crack_scores)
    return [[classify_walkability(post_processing(trait_result)) for trait_result in trait_results], [post_processing(trait_result) for trait_result in trait_results]]
    
def post_processing(traits_dict):
    score = 0
    score += (3) * (6.63279427573101) * traits_dict["a wide sidewalk where people can walk"] # 1/<mean_of_test_set> = normalization factor
    score += 2.0 * traits_dict["a clearly marked pedestrian crosswalk"]
    score += (2) * (1.89) * traits_dict["green trees or landscaping along the street"]
    score += 1.5 * traits_dict["public benches for people to sit"]
    score += 1.5 * traits_dict["a cafe or shop with outdoor seating"]
    score -= 2.0 * traits_dict["trash, litter, or graffiti on the street"]
    score -= 2.0 * traits_dict["a road with lots of fast-moving cars"]
    score -= 2.45 * traits_dict["a sidewalk that is flush with the street and hard to distinguish"]
    return score

def classify_walkability(score):
    if score >= 7:
        return "very_high"
    elif score >= 5:
        return "high"
    elif score >= 3:
        return "medium"
    elif score >= 1:
        return "low"
    else:
        return "very_low"


#print(labelImgs(["./ValidationSet/39035117400/(41.56136772139518, -81.56501323510685)_270.png",
#"./ValidationSet/39035113801/(41.49346267071857, -81.64873941215296)_0.png"]))

#print(labelImgs(["./ValidationSet/39035116300/(41.54400470175995, -81.60667936607943)_180.png"]))

#print(labelImgs(["./ValidationSet/39035117500/(41.5513681169817, -81.57537870166561)_90.png"]))