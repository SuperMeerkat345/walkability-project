import torch
import timm
import os

from PIL import ImageDraw, ImageFont, Image
from tqdm import tqdm

# result = {
#     'scores': [0.7743, 0.4954, 0.42610], 
#     'boxes': [[249.0021, 211.7118, 287.4543, 233.2357],
#                     [ 42.7825, 200.8615, 399.8435, 399.9040],
#                     [233.3115,  85.4666, 362.6517, 226.3072]], 
#     'text_labels': ['a car', 'a sidewalk', 'a tree'], 
#     'labels': ['a car', 'a sidewalk', 'a tree']
# }
# img_path="./ValidationSet/39035117600/(41.5767606, -81.5547632)_270.png"

def show_result(img_path, result, show_scores=False):
    image = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    print("==> Drawing boxes")
    # Go through all boxes
    for i, box in enumerate(result["boxes"]):
        # Convert box from tensor to list of floats
        if isinstance(box, torch.Tensor):
            box = box.tolist()
        box = [float(x) for x in box]

        label = result["labels"][i]
        score = result["scores"][i]
        if isinstance(score, torch.Tensor):
            score = score.item()

        label_text = f"{label}"
        if show_scores:
            label_text += f" ({score:.2f})"

        draw.rectangle(box, outline="red", width=3)
        draw.text((box[0], box[1] - 10), label_text, fill="red")

    print("-> Saving image to temp_output.png for viewing")
    image.save("temp_output.png")  # Save the annotated image
    print("==> Opening image in feh image viewer")
    os.system(f"feh --zoom 300 --geometry 1200x1200+3000+0 \"./temp_output.png\" &") # zoom 300%, some geometry parameters specified as well

    prompt = input("Press Enter to continue or type 'exit' to quit: ")
    if prompt.lower() == 'exit':
        print("Exiting...")
        exit(0) 

#show_result(img_path, result, show_scores=True)