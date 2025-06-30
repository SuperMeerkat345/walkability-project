import os
import subprocess
import pandas as pd

with open('linenumber', 'r') as file: # get current linenumber
    linenumber = int(file.read().strip())

data = pd.read_csv("validation.csv")

# PATH
path = data["path"].iloc[linenumber - 2] # -1 for index; -1 for column names line 
label = data["label"].iloc[linenumber - 2]

# Open this bad boy in the feh image viewer (you may need to install)
print(f"Loading {path} on line [{linenumber}]")
os.system(f"feh --zoom 300 --geometry 1200x1200+3000+0 \"{path}\" &") # zoom 300%, some geometry parameters specified as well

os.system("clear")
rating = input(f"MODIFYING LINE {linenumber} (label = {label})\nRate walkability on a scale of (1-5)\n[1] Very Low | [2] Low | [3] Medium | [4] High | [5] Very High | [0] Null | [] No Change:\n")

# kill feh once done with it
os.system("kill $(pidof feh)")

# Generate rating mappings:
rating_mappings = {
    "0": "Null",
    "1": "very_low",
    "2": "low",
    "3": "medium",
    "4": "high",
    "5": "very_high",
    "": label # No change
}


# PROCESS RATING INTO STRING
if rating in rating_mappings:
    text_rating = rating_mappings[rating] # Convert based on dictionary
    data.loc[linenumber-2, "label"] = text_rating

    # Write the updated content back to the CSV file
    data.to_csv('validation.csv', index=False)

    # Update to the next line
    with open('linenumber', 'w') as file:
        file.writelines([str(linenumber+1)])

    if label == text_rating:
        print(f"Label has not been changed from '{text_rating}' for line {linenumber}")
    else:
        print(f"Updated rating to '{text_rating}' for line {linenumber}")
    
    print(f"linenumber has been changed from {linenumber-1} to {linenumber}")
else:
    print(f"Invalid input '{rating}' Please input a digit 0-5 to assign to label or no input for no change\nlinenumber and validation.csv has not been modified")