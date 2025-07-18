print("===> Loading modules...")
print("-> pandas")
import pandas as pd
print("-> os")
import os
print("-> tqdm")
from tqdm import tqdm

print("-> dino")
from dino_runner_debug import label_imgs
# print("-> dino debugger")
# from show_dino_results import show_result
print("-> dino post-processing")
from dino_post_processing import post_process
# print("-> score categorization")
# from categorize_score import categorize_score


# VARIABLES ===================================================================
data = pd.read_csv("./full_walkability_data.csv")

labelsv11 = [
    # "a narrow sidewalk not connected to the road", # Get rid of dino detection for sidewalks
    # "a sidewalk that looks like a road",
    #"a very narrow sidewalk beside houses",
    #"a wide road for vehicles",
    "a car or vehicle",
    "a fence or barrier",
    "a green leafy tree",
    "a house", 
    "a green lawn with green grassy grass",
    "a green bush or shrub",
    "a black trashcan or garbage bin",
    #"a fire hydrant", # make sure it doesn't get confused with a garbage bin
    #"a street light or lamp post",
    "trash or litter on the ground",
    #"a traffic sign", # DINO thinks that the google logo is a traffic sign LMAO
    #"cracks in the ground or pavement",
]


LABELS = labelsv11

img_paths = data["path"] # Use all paths for images
# END OF VARIABLES ============================================================

def run_model(img_paths, LABELS):
    print("===> Labeling images with DINO...")
    dino_results = label_imgs(img_paths, LABELS, box_threshold=0.3, text_threshold=0.25, batch_size=1)
    #print(dino_results)

    print("===> Postprocessing DINO results...")
    post_processing_results = post_process(dino_results)
    #print(post_processing_results)

    print("===> Writing results to validation CSV...")
    data["label_score"] = post_processing_results
    data.to_csv("./full_walkability_data.csv", index=False)

    print("===> Exited without issues")
    return 1

run_model(img_paths, LABELS)