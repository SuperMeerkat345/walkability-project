rating_mappings = {
    "1": "very_low",
    "2": "low",
    "3": "mid",
    "4": "high",
    "5": "very_high"
}

import os
import subprocess

with open('linenumber', 'r') as file: # get current linenumber
    linenumber = int(file.read().strip())

with open('validation.csv', 'r') as file: # get data from linenumber
    lines = file.readlines()

data = lines[linenumber - 1].strip()

# ISOLATE PATH
path = data.split(",")[1:]
path = ",".join(path)

# Open this bad boy in the feh image viewer (you may need to install)
print(f"Loading {path} on line [{linenumber}]")
os.system(f"feh --zoom 300 --geometry 1200x1200+3000+0 {path} &") # zoom 300%, some geometry parameters specified as well

os.system("clear")
rating = input(f"MODIFYING LINE {linenumber}\nRate walkability on a scale of (1-5)\n[1] Very Low | [2] Low | [3] Medium | [4] High | [5] Very High:\n")

# kill feh once done with it
os.system("kill $(pidof feh)")

# PROCESS RATING INTO STRING
if rating in rating_mappings:
    text_rating = rating_mappings[rating]

    new_data = data.split(",")
    new_data[0] = text_rating
    new_data = ",".join(new_data)

    lines[linenumber-1] = new_data + "\n"

    # Write the updated content back to the CSV file
    with open('validation.csv', 'w') as file:
        file.writelines(lines)

    # Update to the next line
    with open('linenumber', 'w') as file:
        file.writelines([str(linenumber+1)])

    print(f"Updated rating to '{text_rating}' for line {linenumber}")
else:
    pring(f"Invalid input '{rating}' Please input a digit 1-5\nlinenumber and validation.csv has not been modified")