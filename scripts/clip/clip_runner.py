import requests
import torch
from PIL import Image # download with pip install pillow
from transformers import AutoProcessor, AutoModel
from tqdm import tqdm

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

    print("Pre-processing inputs...")
    # Preprocess inputs (return prompts and images in a format the model expects)
    inputs = processor(
        text=labels, # Prompts
        images=imgs, # image array
        return_tensors="pt", # Return result as pytorch tensors
        padding=True)

    print("Running model...")
    # Make fast faster
    with torch.no_grad():
        # Run the model
        outputs = model(**inputs) # unpack dictionary
        logits_per_image = outputs.logits_per_image # Similarity between each image and each prompt (raw scores, not probability)

        # Compute probabilities
        probs = logits_per_image.softmax(dim=1) 

    # Find highest probability label
    for i in tqdm(range(len(most_likely_idxs)), desc="Labeling images"):
        idx = most_likely_idxs[i].item()
        correlation_dic[labels[idx]]  


    return readable_labels