import pathlib
import shutil

import trimesh
from tqdm import tqdm
import open3d as o3d
import json
import numpy as np
import gmsh


class Dataset:
    def __init__(self, step_folder_path: pathlib.Path):
        self.step_folder = step_folder_path.expanduser()

        # Define the dataset_final folder
        self.dataset_path = self.step_folder.parent / "structured_dataset"
        self.dataset_path.mkdir(parents=True, exist_ok=True)

        # Define the STP folder inside dataset_final
        self.stp_folder = self.dataset_path / "STP"
        self.stp_folder.mkdir(parents=True, exist_ok=True)

        # Copy all STEP files (.stp or .step) to the STP folder
        for step_file in self.step_folder.glob("*.stp") or self.step_folder.glob("*.step"):
            step_final_path = self.stp_folder / step_file.name
            shutil.copy(step_file, step_final_path)

        # Define STL and PLY folders inside dataset_final
        self.stl_folder = self.dataset_path / "STL"
        self.stl_folder.mkdir(parents=True, exist_ok=True)
        self.ply_folder = self.dataset_path / "PLY"
        self.ply_folder.mkdir(parents=True, exist_ok=True)

        self.metadata_path = self.dataset_path / "metadata.json"
        self.metadata = {}

    def compute_stl(self):
        """Convert STEP files to STL format and save them"""
        print(f"Converting files in {self.stp_folder} to {self.stl_folder}")

        # Set GMSH verbosity to a lower level to suppress messages
        gmsh.initialize()
        gmsh.option.setNumber("General.Verbosity", 0)

        # Collect all STEP files (.stp and .step) from the STP folder
        files = [
            file
            for file in self.stp_folder.iterdir()
            if file.is_file()
            and file.suffix.lower() in [".stp", ".step"]
            and not file.name.startswith("._")
        ]

        for file in tqdm(files, desc="Converting STEP to STL"):
            try:
                # Load the STEP file and convert to STL
                mesh = trimesh.Trimesh(
                    **trimesh.interfaces.gmsh.load_gmsh(
                        file_name=str(file), gmsh_args=[("Mesh.Algorithm", 5)]
                    )
                )
                # Save the STL file in the STL folder
                stl_output_path = self.stl_folder / (file.stem + ".stl")
                mesh.export(str(stl_output_path))
            except Exception as e:
                print(f"Error processing file {file}: {e}")

    def compute_ply(self, number_of_points: int):
        """Convert STL files to PLY format and save them with sampled points."""
        print(f"Converting STL files in {self.stl_folder} to PLY in {self.ply_folder}")

        # Collect all STL files from the STL folder
        stl_files = [
            file
            for file in self.stl_folder.iterdir()
            if file.is_file() and file.suffix.lower() == ".stl"
        ]

        for file in tqdm(stl_files, desc=f"Sampling {number_of_points} points"):
            try:
                # Read the STL file
                pcd = o3d.io.read_triangle_mesh(str(file))
                # Sample points from the mesh
                sampled_pcd = pcd.sample_points_poisson_disk(number_of_points)

                # Save the sampled point cloud as a PLY file
                ply_output_file = self.ply_folder / (file.stem + ".ply")
                o3d.io.write_point_cloud(str(ply_output_file), sampled_pcd)
            except Exception as e:
                print(f"Error processing file {file}: {e}")

    def compute_eig(self):
        """Compute eigenvalues and eigenvectors for each PLY file."""
        results = {}

        # Collect all PLY files from the PLY folder
        ply_files = [
            file
            for file in self.ply_folder.iterdir()
            if file.is_file() and file.suffix.lower() == ".ply"
        ]

        for ply_file in tqdm(ply_files, desc="Computing eigenvalues and eigenvectors"):
            try:
                # Load the point cloud
                pcd = o3d.io.read_point_cloud(str(ply_file))
                points = np.asarray(pcd.points)

                # Compute the covariance matrix
                covariance_matrix = np.cov(points.T)

                # Compute eigenvalues and eigenvectors
                eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

                # Store the results
                part_number = ply_file.stem
                results[part_number] = {
                    "eigenvalues": eigenvalues.tolist(),
                    "eigenvectors": eigenvectors.tolist(),
                }

            except Exception as e:
                print(f"Error processing file {ply_file}: {e}")

        return results

    def generate_metadata(self):
        """Generate metadata for the dataset and save it as a JSON file"""
        metadata = {}

        # Compute eigenvalues and eigenvectors
        eig_results = self.compute_eig()

        # Collect STEP files and create metadata entries
        for step_file in self.stp_folder.glob("*.stp") or self.stp_folder.glob("*.step"):
            part_number = step_file.stem  # Get the part number (name without extension)
            metadata[part_number] = {
                "step": str(step_file.relative_to(self.dataset_path)),
                "stl": str(
                    (self.stl_folder / (part_number + ".stl")).relative_to(self.dataset_path)
                ),
                "ply": str(
                    (self.ply_folder / (part_number + ".ply")).relative_to(self.dataset_path)
                ),
                "eigenvalues": eig_results.get(part_number, {}).get("eigenvalues", []),
                "eigenvectors": eig_results.get(part_number, {}).get("eigenvectors", []),
            }

        # Save the metadata to a JSON file
        with open(self.metadata_path, "w") as metadata_file:
            json.dump(metadata, metadata_file, indent=4)

        print(f"Metadata saved to {self.metadata_path}")

    def load_metadata(self):
        """Load metadata from the JSON file"""
        with open(self.metadata_path, "r") as metadata_file:
            self.metadata = json.load(metadata_file)
        return self.metadata

    def get_item_by_part_number(self, part_number):
        """Get item by part number"""
        return self.metadata.get(part_number)

    def get_ply_path(self, part_number):
        """Get the PLY path for a specific part number"""
        try:
            item = self.get_item_by_part_number(part_number)
            if item:
                return self.dataset_path / item["ply"]
        except Exception as e:
            print(f"Error getting PLY path for part number {part_number}: {e}")
        return None

    def get_stl_path(self, part_number):
        """Get the STL path for a specific part number"""
        item = self.get_item_by_part_number(part_number)
        if item:
            return self.dataset_path / item["stl"]
        return None

    def get_eigenvalues(self, part_number):
        """Get the eigenvalues for a specific part number"""
        item = self.get_item_by_part_number(part_number)
        if item:
            return item.get("eigenvalues", [])
        return []

    def iterate_parts(self):
        """Iterate through all parts in the dataset and perform an action"""
        for part_number, data in self.metadata.items():
            yield part_number, data

    def __getitem__(self, index):
        """Get item by index"""
        keys = list(self.metadata.keys())
        if index < 0 or index >= len(keys):
            raise IndexError("Index out of range")
        part_number = keys[index]
        return self.metadata[part_number]

    def __iter__(self):
        """Return an iterator over the dataset"""
        return iter(self.metadata.values())

    def __len__(self):
        """Return the length of the dataset"""
        return len(self.metadata)


if __name__ == "__main__":
    step_path = pathlib.Path(
        "~/Documents/LavalUniversity/Maitrise/data/DataSBI/structured_dataset"
    ).expanduser()
    dataset = Dataset(step_path)

    print("STEP Path:", dataset.stp_folder)
    print("STL Path:", dataset.stl_folder)
    print("PLY Path:", dataset.ply_folder)
    print("Metadata Path:", dataset.metadata_path)

    # Specify the number of points for sampling
    # number_of_points = 10000
    # dataset.compute_stl()
    # dataset.compute_ply(number_of_points)
    # dataset.generate_metadata()  # Generate metadata

    # Load metadata
    dataset.load_metadata()
    print(dataset.metadata)

    # Get metadata for a specific part number
    part_number = "763620"
    part_metadata = dataset.get_item_by_part_number(part_number)
    print(part_metadata)

    # Get PLY path for a specific part number
    ply_path = dataset.get_ply_path(part_number)
    print(f"PLY path for part number {part_number}: {ply_path}")

    # Get eigenvalues for a specific part number
    eigenvalues = dataset.get_eigenvalues(part_number)
    print(f"Eigenvalues for part number {part_number}: {eigenvalues}")

    # Iterate through all parts in the dataset
    for part_number, data in dataset.iterate_parts():
        print(f"Part number: {part_number}")

    # Access item by index
    index = 0
    item_by_index = dataset[index]
    print(f"Item at index {index}: {item_by_index}")
