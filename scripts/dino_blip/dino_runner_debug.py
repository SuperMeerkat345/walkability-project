import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from tqdm import tqdm

# VARIABLES =======================================================
# LABELS = [
#     "a pedestrian sidewalk which is seperated from the road", 
#     "an asphalt road", 
#     "a crosswalk",
#     "a tree"
# ]

# END OF VARIABLES ================================================

# method for getting all the possible labels
#label_names = set()

def label_imgs(img_paths, LABELS, box_threshold=0.4, text_threshold=0.3, batch_size=1):
    # Load model
    print("==> Loading model")
    model_id = "IDEA-Research/grounding-dino-tiny"
    device = "cuda"

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

    # Begin inputs
    print("==> Preparing inputs")

    # Load images (needs to be in double array or not?)
    print("-> Loading images")
    images = [Image.open(img_path).convert("RGB") for img_path in tqdm(img_paths, desc="Loading images")]

    # Preparing labels for each image
    print("-> Preparing labels")
    text_labels = [LABELS] * len(images)

    # Where all results will be stored
    results_all = []

    # Processing images by batches
    print(f"==> Processing inputs and running model inferece")
    for i in tqdm(range(0, len(images), batch_size), desc="Processing images"):
        batch_imgs = images[i:i + batch_size]
        batch_paths = img_paths[i:i + batch_size]
        text_labels = [LABELS] * len(batch_imgs)  # repeat labels per image

        # Set inputs
        inputs = processor(images=batch_imgs, text=text_labels, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)
        torch.cuda.empty_cache() # Reset the memory to avoid the CUDA out of memory error

        # Get the actual results
        target_sizes = [img.size[::-1] for img in batch_imgs]
        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=box_threshold, # 0.4
            text_threshold=text_threshold, # 0.3
            target_sizes=target_sizes
        )

        # # getting label names 
        # print(results)
        # for label in results[0]["labels"]:
        #     if label not in label_names:
        #         label_names.add(label)


        # Append the batch to the results
        results_all.extend(results)

    return results_all


# print(label_imgs(["./ValidationSet/39035117600/(41.5767606, -81.5547632)_270.png", 
# "./ValidationSet/39035117400/(41.56136772139518, -81.56501323510685)_270.png"], LABELS))