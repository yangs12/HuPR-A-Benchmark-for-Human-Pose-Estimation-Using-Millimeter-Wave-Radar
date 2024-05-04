import os
import numpy as np

folder_path = "/home/jupyter/hupr/radar_processed/"

# Get the list of files in the folder
file_list = os.listdir(folder_path)

# Iterate over each file
file_exists = []
for file_name in file_list:
    file_path = os.path.join(folder_path, file_name)

    # Check if the file is a subfolder
    if os.path.isdir(file_path+'/hori/'):
        pass
        # print(f"{file_name} is a subfolder")
    else:
        file_exists.append(file_name)

print(file_exists)
