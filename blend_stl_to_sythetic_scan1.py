import sys
import os

# Add the current script's directory to sys.path.
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

# Now import your custom modules.
from blender_utils import generate_point_cloud
from pathlib import Path
from dataset import Dataset


def process_single_part(stl, output):
    """
    Process a single part (by its identifier) from the dataset.
    Looks up the STL file corresponding to the part and then generates
    the point cloud and depth maps.
    """
    # stl_path = str(part_dataset.get_stl_path(part_identifier))

    generate_point_cloud(
        stl_path=str(stl),
        model_location=(0, 0, 0),  # Position the model at the origin.
        model_scale=(0.001, 0.001, 0.001),  # Scale the model appropriately.
        num_cameras=10,  # Number of cameras to generate views.
        distance_to_camera=6000,  # Distance from the camera to the model.
        output_folder=output,  # Output folder for generated data.
    )


if __name__ == "__main__":
    # Parse command-line arguments passed after "--"
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []

    if len(argv) < 2:
        print(
            "Usage: blender --background --python generate_data_from_views.py -- <stl_file_path> <output_folder>"
        )
        sys.exit(1)

    # Get the arguments from the command-line.
    stl_file_path = Path(argv[0]).expanduser()
    output_folder = Path(argv[1]).expanduser()

    # Ensure the output folder exists
    output_folder.mkdir(parents=True, exist_ok=True)

    # Process the single part
    process_single_part(stl_file_path, output_folder)
