import os
import subprocess
from pathlib import Path
from tqdm import tqdm

# ---------------------------------------------------------------
# Confgiguration
# ---------------------------------------------------------------

# Activate your conda environment paths (if needed)
conda_python_path = "/Users/charlesbeaulieu/anaconda3/envs/GIF-7005/bin/python"
dataset_path = Path("data/stuctured_dataset").expanduser()

stl_folder = "data/structured_dataset/STL"
output_folder = "generated"
# Blender executable path
blender_exec = "/Applications/Blender.app/Contents/MacOS/Blender"
blender_script = "blend_stl_to_sythetic_scan1.py"


# ---------------------------------------------------------------
# Processing
# ---------------------------------------------------------------
# This part of the script just iterates over the STL files in the folder
# and runs the Blender script for each file to avoid overloading the memory
# with blender
for stl_file in tqdm(os.listdir(stl_folder), desc="Processing STL files", unit="file"):
    if stl_file.endswith(".stl"):
        stl_path = os.path.join(stl_folder, stl_file)

        subprocess.run(
            [
                blender_exec,
                "--background",
                "--python",
                blender_script,
                "--",
                stl_path,
                output_folder,
            ],
            check=True,
        )
