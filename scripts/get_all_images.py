# Data frame and data manipulation
import pandas as pd
import numpy as np

# File/Folder manipulation
import shutil
import os

#File format
#Census tract -> block group -> block

def get_all_images(dataPath, outputPath, validationSetName = "validationSet", imgNum = 2):
    print("Starting generation in:", os.getcwd())
    # Check if output directory exists, if not create it
    #if not os.path.exists(outputPath):
        #os.makedirs(outputPath)
        
    # Loop through census tracts
    for tract in os.listdir(dataPath):
        # Now loop through the general coordinates
        blockGroupPath = dataPath + "/" + tract

        for blockGroup in os.listdir(blockGroupPath):
            # Loop through sub coordinates
            blockPath = blockGroupPath + "/" + blockGroup

            for block in os.listdir(blockPath):
                # Skip any files (There are some skyviews which aren't needed for the project)
                if os.path.isfile(blockPath + "/" + block):
                    continue

                # Loop through the images
                imgPath = blockPath + "/" + block

                for img in os.listdir(imgPath):
                    print(img) # Do an action

    
    
generate_validation_set("./ClevelandData/", "./")