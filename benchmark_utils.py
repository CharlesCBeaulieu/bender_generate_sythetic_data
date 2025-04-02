import numpy as np
import open3d as o3d

def compute_eig(pcd: o3d.geometry.PointCloud):
    """
    Compute the eigenvalues and eigenvectors of the covariance matrix of a point cloud.
    """
    # Convert point cloud to numpy array
    points = np.asarray(pcd.points)

    # Center the point cloud
    centroid = np.mean(points, axis=0)
    centered_points = points - centroid

    # Compute the covariance matrix
    cov_matrix = np.cov(centered_points.T)

    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    # Sort eigenvalues and eigenvectors in descending order
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    return eigenvalues, eigenvectors


def compute_eigen_ratios(eigenvalues):
    """
    Compute all possible ratios between sorted eigenvalues.
    """
    eigenvalues = np.sort(eigenvalues)[::-1]  # Sort eigenvalues in descending order
    ratio_xy = eigenvalues[0] / eigenvalues[1] if eigenvalues[1] != 0 else float('inf')  # x / y
    ratio_xz = eigenvalues[0] / eigenvalues[2] if eigenvalues[2] != 0 else float('inf')  # x / z
    ratio_yz = eigenvalues[1] / eigenvalues[2] if eigenvalues[2] != 0 else float('inf')  # y / z
    return ratio_xy, ratio_xz, ratio_yz


def evaluate_point_cloud(pcd: o3d.geometry.PointCloud):
    """
    Evaluate a point cloud by computing eigenvalues and eigenvectors.
    """
    # Compute ratios between eigenvalues
    eigenvalues, _ = compute_eig(pcd)
    eig_ratio_xy, eig_ratio_xz, eig_ratio_yz = compute_eigen_ratios(eigenvalues)
    
    # Compute anisotropy
    eigenvalues = np.sort(eigenvalues)[::-1]  # Sort eigenvalues in descending order
    anisotropy = (eigenvalues[0] - eigenvalues[2]) / eigenvalues[0]
    
    metrics = {
        "eig_ratio_xy": eig_ratio_xy,
        "eig_ratio_xz": eig_ratio_xz,
        "eig_ratio_yz": eig_ratio_yz,
        "anisotropy": anisotropy,
        "eigenvalues": eigenvalues
    }
    
    return metrics


def compute_similarity_ratio(metrics1, metrics2):
    """
    Compute the similarity between two point clouds based on their eigen ratios.
    """
    xy_diff = np.abs(metrics1["eig_ratio_xy"] - metrics2["eig_ratio_xy"])
    xz_diff = np.abs(metrics1["eig_ratio_xz"] - metrics2["eig_ratio_xz"])
    yz_diff = np.abs(metrics1["eig_ratio_yz"] - metrics2["eig_ratio_yz"])
    
    # Aggregate differences into a similarity score
    similarity = xy_diff + xz_diff + yz_diff

    return similarity