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


def process_single_part(part_dataset, part_identifier):
    """
    Process a single part (by its identifier) from the dataset.
    Looks up the STL file corresponding to the part and then generates
    the point cloud and depth maps.
    """
    stl_path = str(part_dataset.get_stl_path(part_identifier))

    print(f"Processing part {part_identifier} using STL file: {stl_path}")
    generate_point_cloud(
        stl_path=stl_path,
        model_location=(0, 0, 0),  # Position the model at the origin.
        model_scale=(0.001, 0.001, 0.001),  # Scale the model appropriately.
        num_cameras=10,  # Number of cameras to generate views.
        part_number=part_identifier,  # Use the part identifier.
    )


if __name__ == "__main__":
    # Parse command-line arguments passed after "--"
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []

    if len(argv) < 1:
        print("Usage: blender --background --python generate_data_from_views.py -- <part_id>")
        sys.exit(1)

    # Get the part identifier from the command-line.
    part_id = argv[0]

    # Define and load the dataset.
    dataset_path = Path(
        "~/Documents/LavalUniversity/Maitrise/data/DataSBI/structured_dataset"
    ).expanduser()
    dataset = Dataset(dataset_path)
    dataset.load_metadata()

    # Process the single part.
    process_single_part(dataset, part_id)
