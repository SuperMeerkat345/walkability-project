import requests
import torch
import pandas as pd

from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
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
    print("==> Loading BLIP-2 model")
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
    model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-flan-t5-xl", 
        torch_dtype=torch.float16,
        device_map="auto"
    )

    print("==> Loading images")
    images = [Image.open(img_path).convert("RGB") for img_path in tqdm(img_paths, desc="Loading images")]

    print("==> Running inferences")
    results = []
    question = "What is the walkability in this image on a scale of 1 to 5?"

    for image in tqdm(images, desc="Running inferences"):
        inputs = processor(images=image, text=question, return_tensors="pt").to("cuda", torch.float16)
        with torch.no_grad():
            output = model.generate(**inputs)
        answer = processor.batch_decode(output, skip_special_tokens=True)[0].strip()
        results.append(answer)

    # Convert BLIP-2 answers to labels
    labeled_results = [correlation_dic.get(res, "unknown") for res in results]
    data["label_model"] = labeled_results
    data.to_csv("./full_validation.csv", index=False)

    print("==> Done. Results saved to full_validation.csv")
    return labeled_results

print(analyze_labels(data["path"].head(2).tolist()))