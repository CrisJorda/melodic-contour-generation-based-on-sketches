import os
import shutil
import random

# Source and destination folders
src_folder = "/home/cristina/melodic-contour-generation-based-on-sketches/irishtunes_dataset/"
dst_folder = "/home/cristina/melodic-contour-generation-based-on-sketches/"

# Get a list of all .mid files in the source folder
mid_files = [f for f in os.listdir(src_folder)]

# Choose one random file
chosen_file = random.choice(mid_files)

# Full paths
src_path = os.path.join(src_folder, chosen_file)
dst_path = os.path.join(dst_folder, chosen_file)

# Copy the file
shutil.copy2(src_path, dst_path)
print(f"Copied {chosen_file} to {dst_folder}")
