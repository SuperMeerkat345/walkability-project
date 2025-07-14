print("===> Loading modules...")
print("-> pandas")
import pandas as pd
print("-> os")
import os
print("-> tqdm")
from tqdm import tqdm

print("-> dino")
from dino_runner_debug import label_imgs
print("-> dino debugger")
from show_dino_results import show_result
print("-> blip")
from blip_runner_debug import analyze_labels

# VARIABLES ===================================================================
data = pd.read_csv("./full_validation.csv")
labelsv1 = [
    # Driveway
    "a driveway leading away from the road",

    # Sidewalks vs roads
    "a concrete pedestrian sidewalk",
    "an asphalt road for vehicles",
    "a painted pedestrian crosswalk",
    "a white concrete sidewalk branching off the main road",
    "a curving pedestrian sidewalk"

    # Vegetation
    "a green leafy tree",
    "a small bush or shrub",
    "a patch of grass or lawn",
    "a flower bed",

    # Cars
    "a parked car",
    "a moving car",
    "a white car",
    "a red car"
]

labelsv2 = [
    "a flat concrete surface",
    "a paved area",
    "an asphalt road",
    "the ground",
    "a pedestrian crosswalk",
    "a green tree",
    "a car"
]

labelsv3 = [
    # Sidewalks
    "a pedestrian sidewalk",
    "a residential sidewalk",
    "a sidewalk with a curb",
    "a sidewalk crossing a driveway",
    "a concrete sidewalk",
    "a sidewalk with a curb",
    "a sidewalk crossing a driveway",
    "a diagonal sidewalk",
    "a sloped sidewalk",
    "a curved sidewalk",
    "a sidewalk with a curb and grass",
    "a sidewalk with grass on the side",
    "a sidewalk which is going across the image",
    "a sidewalk which is going along the image",
    "a sidewalk which is going across the image and has a curb on the side",
    "a sidewalk which is going along the image and has a curb on the side",
    "a sidewalk which is going across the image and has grass on the side",
    "a sidewalk which is going along the image and has grass on the side",
    "a sidewalk which is going across the image and has a curb and grass on the side",
    "a sidewalk which is going along the image and has a curb and grass on the side",
    "",

    # Driveways
    "a residential driveway",
    "a concrete driveway",
    "a driveway leading to a garage",
    "a driveway connecting to the road",
    "a driveway opening up to the street",
    "a driveway entrance",

    # Roads
    "an asphalt road",
    "a paved road for vehicles",
    "a road intersection",
    "a road with lane markings",

    # Trees
    "a leafy green tree",
    "a tree beside the sidewalk",
    "a tree along the road",

    # Cars
    "a parked car on the driveway",
    "a moving car on the road",
    "a car beside the sidewalk",
    "a white car",
    "a red car"
]

labelsv4 = [
    "a brown or white concrete sidewalk",
    "a concrete sidewalk next to a road",
    "a concrete sidewalk parallel to the road",
    "a paved road",
    "a residential driveway",
    "a green grassy lawn",
    "a green leafy tree with brown branches",
    "a crosswalk crossing the road to another side"
]

labelsv5 = [
    "a white concrete sidewalk",
    "a brown concrete sidewalk",
    "a concrete sidewalk next to a road",
    "a sidewalk parallel to the road branchoff",
    "a paved road",
    "a residential driveway",
    "a green grassy lawn",
    "a green leafy tree with brown branches",
    #"a crosswalk",
    #"a crosswalk across the road",
]

labelsv6 = [
    "a concrete sidewalk for pedestrians",
    "a paved road",
    "a residential driveway",
    "a green grassy lawn",
    "a green leafy tree with brown branches",
    "a car in the road"
]

labelsv7 = [
    "a concrete sidewalk for pedestrians",
    "a crosswalk or pedestrian crossing",
    "a traffic light or stop sign",
    "a paved road with light traffic",
    "a paved road with heavy traffic",
    "a residential driveway",
    "a parking lot or parked cars",
    "a fence or barrier blocking pedestrians",
    "a green grassy lawn",
    "a green leafy tree providing shade",
    "street lighting or lamp posts",
    "a bus stop or public transit",
    "construction or road work",
    "a bike lane or cycling path"
]

LABELS = labelsv7

img_paths = data["path"]
# END OF VARIABLES ============================================================


def test_model(img_paths, LABELS):
    print("===> Labeling images with DINO...")
    dino_results = label_imgs(img_paths, LABELS, box_threshold=0.29, text_threshold=0.24, batch_size=1)
    #print(dino_results)

    print("===> Showing DINO results...")
    for i in tqdm(range(len(img_paths)), desc="Showing results"):
        show_result(img_paths[i], dino_results[i], show_scores=True)

    print("===> Analyzing labels with BLIP-2...")
    blip_results = analyze_labels(img_paths, dino_results)
    #print(blip_results)

    print("===> Writing results to validation CSV...")
    data["label_model"] = blip_results
    data.to_csv("./full_validation.csv", index=False)
    
    print("===> Exited without issues")
    return 1

test_model(img_paths, LABELS)
    



# # For the agree column in the validation csv
#     agree_column = []

#     #print(results)
#     print("===> Analyzing labels with BLIP...")
#     for i in range(len(results)):
#         result = results[i]
#         image = img_paths[i]

#         show_result(image, result, show_scores=True)

#         agree = input(f"Do you agree with the labeling (y/n) [i: {i}/{len(img_paths)}]:\n\n")

#         if agree == "y":
#             agree_column.append("yes")
#         else:
#             agree_column.append("no")

#         os.system("kill $(pidof feh)")