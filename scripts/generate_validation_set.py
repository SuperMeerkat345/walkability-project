# Data frame and data manipulation
import pandas as pd
import numpy as np

# File/Folder manipulation
import shutil
import os

#File format
#Census tract -> block group -> block

def generate_validation_set(dataPath, outputPath, validationSetName = "validationSet", imgNum = 2):
    print("Starting generation in:", os.getcwd())
    # Check if output directory exists, if not create it
    if not os.path.exists(outputPath):
        os.makedirs(outputPath)
        
    # region Warning
    # WARNING: THIS REPLACES ANY DIRECTORIES IN THIS FOLDER THAT ALREADY EXIST
    print("\033[91mWARNING: THIS REPLACES ANY DIRECTORIES IN THIS FOLDER THAT ALREADY EXIST IN \033[0m" + outputPath)
    warning = input("WARNING: This may replace files that were already in the folder, please confirm you want to run generate_validation_set (y/n):\n")
    
    if warning == "n" or warning == "no":
        print("Terminating execution")
        return
    elif warning == "y" or warning == "yes":
        print("Continuing execution of generate_valiation_set")
    else:
        raise Exception("Unrecognized value, please input 'yes'/'y' or 'no'/'n'")
    # endregion

    # Loop through census tracts
    for tract in os.listdir(dataPath):
        currTractPath = outputPath + "/" + tract

        # If the directory already exists, delete it and recreate it
        if os.path.exists(currTractPath):
            print(f"Directory {currTractPath} already exists, removing it.")
            shutil.rmtree(currTractPath)  # Delete the existing directory and its contents

        # Recreate the directory
        os.makedirs(currTractPath)  # Make the census tract dirs
        print(f"Created directory: {currTractPath}")

        # Add the photos
        tractPath = dataPath + "/" + tract
        groupPath = tractPath + "/" + os.listdir(tractPath)[0]
        blockPath = groupPath + "/" + os.listdir(groupPath)[0] 

        # Deal with case where it gets the skyview
        if os.path.isfile(blockPath) and len(os.listdir(groupPath)) <= 1:
            continue # no more images so just move on from this tract
        elif os.path.isfile(blockPath) and len(os.listdir(groupPath)) > 1:
            blockPath = groupPath + "/" + os.listdir(groupPath)[1]
        #blockPath = groupPath + "/" + os.listdir(groupPath)[1] if os.path.isfile(blockPath) else blockPath
        
        
        for imgI in range(imgNum):
            if imgI >= len(os.listdir(blockPath)):
                break # ran out of images in the directory
            
            imgPart = os.listdir(blockPath)[imgI]
            source_file = os.path.join(blockPath, imgPart)
            destination_file = os.path.join(outputPath, tract, imgPart)

            # else: copy the img to the validation set
            shutil.copy(source_file, destination_file)
            print(f"File copied from {source_file} to {destination_file}")

generate_validation_set("./ClevelandData", "./ValidationSet")