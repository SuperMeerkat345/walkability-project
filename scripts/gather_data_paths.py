# File/Folder manipulation
import shutil
import os


def gather_data_paths(dataPath, outputPath):
    # Tracks amount of files copied
    copy_count = 0

    # Loop through census tracts
    for tract in os.listdir(dataPath):
        tract_input_path = os.path.join(dataPath, tract)
        tract_output_path = os.path.join(outputPath, tract)

        # Remove existing directory if it exists
        if os.path.exists(tract_output_path):
            print(f"Directory {tract_output_path} already exists, removing it.")
            shutil.rmtree(tract_output_path)

        os.makedirs(tract_output_path)
        print(f"Created directory: {tract_output_path}")

        # Iterate over groups
        for group in os.listdir(tract_input_path):
            group_path = os.path.join(tract_input_path, group)

            # Iterate over blocks
            for block in os.listdir(group_path):
                block_path = os.path.join(group_path, block)

                # Skip files (like the skyview image)
                if not os.path.isdir(block_path):
                    continue

                # Copy all image files to the output directory
                for img_file in os.listdir(block_path):
                    src = os.path.join(block_path, img_file)
                    dst = os.path.join(tract_output_path, img_file)

                    shutil.copy(src, dst)
                    copy_count += 1
                    print(f"Copied: {src} -> {dst}")
     
    print("Successfully completed data transfer")
    return copy_count
                



print(gather_data_paths("./ClevelandData", "CensusTracts"))