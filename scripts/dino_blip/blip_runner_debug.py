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
def analyze_labels(img_paths, dino_results):
    print("==> Loading BLIP-2 model")
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
    model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-flan-t5-xl", 
        torch_dtype=torch.float16,
        device_map="auto"
    )

    print("==> Preparing input")
    print("-> Loading images")
    images = [Image.open(img_path).convert("RGB") for img_path in tqdm(img_paths, desc="Loading images")]

    print("-> Loading DINO labels")
    dino_labels = [dino_result["labels"] for dino_result in dino_results]
    #print("dino_labels:", dino_labels)

    print("==> Running inferences")
    results = []
    
    for image, labels in tqdm(zip(images, dino_labels), total=len(images), desc="Running inferences"):
        context = ", ".join(set(labels))  # remove duplicates
        question = (
            f"Image contains: {context}.\n"
            "Rate walkability 1-5:\n"
            "1=No sidewalks, dangerous\n"
            "2=Poor pedestrian access\n" 
            "3=Basic sidewalks\n"
            "4=Good walkability\n"
            "5=Excellent for walking\n"
            "Answer with one number only:"
        )

        inputs = processor(images=image, text=question, return_tensors="pt").to("cuda", torch.float16)
        with torch.no_grad():
            output = model.generate(**inputs)
        answer = processor.batch_decode(output, skip_special_tokens=True)[0].strip()
        results.append(correlation_dic[answer] if answer in correlation_dic else "unkown")

    # # Convert BLIP-2 answers to labels
    # labeled_results = [correlation_dic.get(res, "unknown") for res in results]
    # data["label_model"] = results # labeled_results
    # data.to_csv("./full_validation.csv", index=False)

    #print("==> Done. Results saved to full_validation.csv")
    return results

#print(analyze_labels(data["path"]))