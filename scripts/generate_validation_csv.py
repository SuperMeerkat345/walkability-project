import pandas as pd
import os

# datapath should be to the ValidationSet
def generate_validation_csv(dataPath, outputPath):
    img_list = {
        "label": [],
        "path": []
    }
    
    for tract in os.listdir(dataPath):
        img_path = os.path.join(dataPath, tract)

        for img in os.listdir(img_path):
            # Full path to the image file
            img_full_path = os.path.join(img_path, img)

            img_list["path"].append(img_full_path)
            img_list["label"].append("Null")
    
    csv = pd.DataFrame(img_list)
    print(csv)
    csv.to_csv(outputPath, index=False)
            
    
generate_validation_csv("./ValidationSet", "./validation.csv")
