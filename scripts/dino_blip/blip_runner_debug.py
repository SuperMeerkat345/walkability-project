import torch
import pandas as pd

from PIL import Image
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
from tqdm import tqdm
from collections import Counter

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
    print("==> Loading BLIP-1 model")
    processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-7b")
    model = InstructBlipForConditionalGeneration.from_pretrained(
        "Salesforce/instructblip-vicuna-7b",
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
        label_counts = Counter(labels)
        context = ", ".join([f"[{count}] {label}" if count > 1 else label for label, count in label_counts.items()])
        
        question = (
            f"Rate the walkability of the image on a scale of 1-5"
        )
        inputs = processor(images=image, text=question, return_tensors="pt").to("cuda", torch.float16)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=1,
                do_sample=False
            )
        
        answer = processor.batch_decode(output, skip_special_tokens=True)[0].strip()
        print(context, answer)
        results.append(correlation_dic[answer] if answer in correlation_dic else answer)
        #results.append(answer)
    torch.cuda.empty_cache()

    # # # Convert BLIP-2 answers to labels
    # labeled_results = [correlation_dic.get(res, "unknown") for res in results]
    # data["label_model"] = results # labeled_results
    # data.to_csv("./full_validation.csv", index=False)

    #print("==> Done. Results saved to full_validation.csv")
    return results

#print(analyze_labels(data["path"]))