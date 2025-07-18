import numpy as np
from tqdm import tqdm

label_scores = {
    'a green tree a green bush': 1, 
    'a house': 0.5, 
    'litter': -2, 
    'vehicle': -1, 
    'a fence': -1,  # Fences often block pedestrian access or indicate private property
    'a green leafy tree a green bush': 1, 
    'a car': -1, 
    'a green lawn': 1, 
    'ay tree a green bush': 1,  # This looks like a typo for "a green tree"
    'a green bush': 1, 
    'trash': -2, 
    'a black trash trash': -2, 
    'a green leafy tree': 1, 
    'a black trashcan': 0.5  # Trash cans are actually good infrastructure, not litter
}

def post_process(dino_results):
    scores = []
    for dino_result in tqdm(dino_results, desc="Scoring"):
        score = 0 
        for label in dino_result["text_labels"]:
            score += 0 if label not in label_scores else label_scores[label]
        scores.append(score)
    return scores

# def post_process_avg(dino_results):
#     scores = []
#     for dino_result in dino_results:
#         score = 0 
#         for label in dino_result["text_labels"]:
#             score += 0 if label not in label_scores else label_scores[label]
#         scores.append(score)
#     return np.mean(scores)

