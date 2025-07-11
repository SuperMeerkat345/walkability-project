import requests
import torch
import pandas as pd

from PIL import Image
from transformers import AutoProcessor, AutoModelForVisualQuestionAnswering
from tqdm import tqdm

# VARIABLES =======================================================
data = pd.read_csv("./full_validation.csv")
correlation_dic = {
    "1": "very_low",
    "2": "low",
    "3": "medium",
    "4": "high",
    "5": "very_high"
}
# END OF VARIABLES ================================================
def analyze_labels(img_paths):
    print("==> Loading model")
    processor = AutoProcessor.from_pretrained("Salesforce/blip-vqa-base")
    model = AutoModelForVisualQuestionAnswering.from_pretrained(
        "Salesforce/blip-vqa-base", 
        torch_dtype=torch.float16,
        device_map="auto"
    )

    print("==> Loading images")
    img_paths = pd.read_csv("./full_validation.csv")["path"]
    images = [Image.open(img_path).convert("RGB") for img_path in tqdm(img_paths, desc="Loading images")]

    print("==> Running inferences")
    results = []
    question = "What is the walkability in this image on a scale of 1 to 5?"
    for image in tqdm(images, desc="Running inferences"):
        inputs = processor(images=image, text=question, return_tensors="pt").to("cuda", torch.float16)

        output = model.generate(**inputs)
        results.append(processor.batch_decode(output, skip_special_tokens=True)[0])


    results = [correlation_dic[res] if res in correlation_dic else "unknown" for res in results]
    data["label_model"] = results
    data.to_csv("./full_validation.csv", index=False)

    return results
