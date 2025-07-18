import os
import pandas as pd

from tqdm import tqdm

# Gets the paths and sets them up
def start_data_csv(data_path, output_path):
    data = {
        "index": [],
        "path": [],
        "model_label": [],
        "model_score": [],
        "model_score_normalized": [],
        "census_tract": []
    }
    
    # index for the index array in data
    curr_index = 0

    for tract in tqdm(os.listdir(data_path), desc="Looping through Census tracts"):
        tract_path = data_path + "/" + tract
        for img in os.listdir(tract_path):
            img_path = tract_path + "/" + img

            # Add data to the row
            data["index"].append(curr_index)
            data["path"].append(img_path)

            data["model_label"].append("NA")
            data["model_score"].append("NA")
            data["model_score_normalized"].append("NA")

            data["census_tract"].append(tract)

            # Next index
            curr_index += 1

    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)

    return df
    
start_data_csv("./CensusTracts", "./full_walkability_data.csv")