import subprocess
from pathlib import Path
from dataset import Dataset
import os

# Activate your conda environment paths (if needed)
conda_python_path = "/Users/charlesbeaulieu/anaconda3/envs/GIF-7005/bin/python"
dataset_path = Path("data/stuctured_dataset").expanduser()  # Change this to your dataset path

# Load dataset
ds = Dataset(dataset_path)
ds.load_metadata()
stl_folder = "data/structured_dataset/STL"

parts = [str(pn) for pn, _ in ds.iterate_parts()]
total_parts = len(parts)

# Blender executable path
blender_exec = "/Applications/Blender.app/Contents/MacOS/Blender"
blender_script = (
    "blend_stl_to_sythetic_scan.py"  # Make sure to provide full path if not in current directory
)

# for idx, part in enumerate(parts, start=1):
#     print(f"Processing part {part} ({idx} of {total_parts})...")
for stl_file in os.listdir(stl_folder):

    # Run Blender command
    subprocess.run(
        [blender_exec, "--background", "--python", blender_script, "--", part], check=True
    )
