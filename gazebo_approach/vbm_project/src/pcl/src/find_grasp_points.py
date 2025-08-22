import open3d as o3d
import numpy as np

def remove_invalid_points(pcd):
    """
    Remove invalid points (NaN or Inf) from the point cloud.
    
    Parameters:
        pcd (o3d.geometry.PointCloud): Input point cloud.

    Returns:
        o3d.geometry.PointCloud: Point cloud with only valid points.
    """
    # Get the coordinates of the points
    points = np.asarray(pcd.points)

    # Filter out points that contain NaN or Inf values
    valid_points = np.isfinite(points).all(axis=1)

    # Create a new point cloud object with only the valid points
    pcd_valid = o3d.geometry.PointCloud()
    pcd_valid.points = o3d.utility.Vector3dVector(points[valid_points])

    # If the point cloud has color information, also retain the color data
    if pcd.has_colors():
        colors = np.asarray(pcd.colors)
        pcd_valid.colors = o3d.utility.Vector3dVector(colors[valid_points])

    return pcd_valid

def compute_center_of_mass(pcd):
    """
    Compute the center of mass of the point cloud.
    
    Parameters:
        pcd (o3d.geometry.PointCloud): Input point cloud.

    Returns:
        np.ndarray: Center of mass coordinates.
    """
    # Compute the center of mass of the point cloud
    points = np.asarray(pcd.points)
    center_of_mass = np.mean(points, axis=0)
    return center_of_mass

def compute_angle_between_vectors(v1, v2):
    """
    Compute the angle between two vectors (in radians).
    
    Parameters:
        v1 (np.ndarray): First vector.
        v2 (np.ndarray): Second vector.

    Returns:
        float: Angle in radians.
    """
    # Calculate the cosine of the angle using the dot product
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    cos_theta = np.clip(cos_theta, -1.0, 1.0)  # Ensure the value is within the range [-1, 1]
    angle = np.arccos(cos_theta)  # Calculate the angle in radians
    return angle

def find_grasp_near_center_of_mass(pcd, center_of_mass, min_distance_threshold=0.01, 
                                     max_distance_threshold=0.05, angle_threshold=0.9, 
                                     y_threshold=0.01, z_threshold=0.01, 
                                     normal_angle_threshold=np.pi/4):
    """
    Find a pair of grasp points near the center of mass of the point cloud.
    
    Parameters:
        pcd (o3d.geometry.PointCloud): Input point cloud.
        center_of_mass (np.ndarray): Center of mass coordinates.
        min_distance_threshold (float): Minimum distance between grasp points.
        max_distance_threshold (float): Maximum distance between grasp points.
        angle_threshold (float): Threshold for angle similarity of normals.
        y_threshold (float): Threshold for Y distance from center of mass.
        z_threshold (float): Threshold for Z distance from center of mass.
        normal_angle_threshold (float): Threshold for normal angle.

    Returns:
        tuple: Best pair of grasp points or (None, None) if no pair found.
    """
    points = np.asarray(pcd.points)  # Extract point coordinates
    normals = np.asarray(pcd.normals)  # Extract normals
    best_pair = None
    best_score = float('inf')  # Initialize the best score

    # Iterate through all pairs of points
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            # Calculate the Euclidean distance between the two points
            dist = np.linalg.norm(points[i] - points[j])

            # Check if the distance is within the specified min and max thresholds
            if min_distance_threshold < dist < max_distance_threshold:
                # Calculate the distance of the points from the center of mass
                xyz_distance_to_center_of_mass = (
                    abs(points[i][0] - center_of_mass[0]) +
                    abs(points[j][0] - center_of_mass[0]) +
                    abs(points[i][1] - center_of_mass[1]) +
                    abs(points[j][1] - center_of_mass[1]) +
                    abs(points[i][2] - center_of_mass[2]) +
                    abs(points[j][2] - center_of_mass[2])
                )

                # Compute the dot product of the normals to evaluate symmetry
                cos_similarity = np.dot(normals[i], normals[j])

                # Compute the vector between the two grasp points
                grasp_line = points[j] - points[i]

                # Calculate the angle between the normal and the grasp line for both points
                angle_i = compute_angle_between_vectors(normals[i], -grasp_line)
                angle_j = compute_angle_between_vectors(normals[j], grasp_line)

                # Check if the sum of the angles is less than the threshold
                if angle_i + angle_j < normal_angle_threshold and abs(cos_similarity + 1) < angle_threshold:
                    # Combine distance, normal symmetry, and center of mass proximity into a score
                    score = dist + abs(cos_similarity + 1) + xyz_distance_to_center_of_mass * 10
                    if score < best_score:  # Check if this score is better than the best
                        best_score = score
                        best_pair = (i, j)  # Update best grasp points

    if best_pair:
        print(f"Best grasp points found between point {best_pair[0]} and point {best_pair[1]}")
        print(f"Point 1: {points[best_pair[0]]}, Normal 1: {normals[best_pair[0]]}")
        print(f"Point 2: {points[best_pair[1]]}, Normal 2: {normals[best_pair[1]]}")
        print(f"Score: {best_score}")
        return points[best_pair[0]], points[best_pair[1]]  # Return the best grasp points
    else:
        print("No suitable grasp points found.")
        return None, None

def visualize_grasp(pcd, point1, point2):
    """
    Visualize the point cloud and the grasping points.
    
    Parameters:
        pcd (o3d.geometry.PointCloud): Input point cloud.
        point1 (np.ndarray): Coordinates of the first grasp point.
        point2 (np.ndarray): Coordinates of the second grasp point.
    """
    # Set the color of the entire point cloud to yellow
    pcd.paint_uniform_color([1, 1, 0])  # RGB: [1, 1, 0] corresponds to yellow

    if point1 is not None and point2 is not None:
        # Create grasp points as small spheres to mimic point contacts
        grasp_point1 = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)  # Small sphere for point contact
        grasp_point1.paint_uniform_color([1, 0, 0])  # Color grasp point red
        grasp_point1.translate(point1)  # Place at the first grasp point

        grasp_point2 = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)  # Same for second grasp point
        grasp_point2.paint_uniform_color([1, 0, 0])  # Color grasp point red
        grasp_point2.translate(point2)  # Place at the second grasp point

        # Create a grasp line by adding small spheres between the points
        grasp_line_spheres = []
        num_segments = 5  # Number of segments for the grasp line
        for i in range(num_segments + 1):
            t = i / num_segments
            point_on_line = (1 - t) * np.array(point1) + t * np.array(point2)  # Linear interpolation
            small_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.00001)  # Very small radius for line points
            small_sphere.paint_uniform_color([0, 1, 0])  # Color grasp line green
            small_sphere.translate(point_on_line)  # Place along the line
            grasp_line_spheres.append(small_sphere)

        # Visualize the point cloud and grasping elements
        o3d.visualization.draw_geometries([pcd, grasp_point1, grasp_point2] + grasp_line_spheres, point_show_normal=True)

def adaptive_voxel_downsampling(pcd, target_points=5000, initial_voxel_size=0.001, max_iterations=50):
    """
    Perform adaptive voxel downsampling on the point cloud.
    
    Parameters:
        pcd (o3d.geometry.PointCloud): Input point cloud.
        target_points (int): Desired number of points after downsampling.
        initial_voxel_size (float): Starting voxel size for downsampling.
        max_iterations (int): Maximum number of downsampling iterations.

    Returns:
        tuple: Final voxel size used and downsampled point cloud.
    """
    voxel_size = initial_voxel_size  # Initialize voxel size
    downsampled_pcd = pcd  # Start with the original point cloud
    for _ in range(max_iterations):
        downsampled_pcd = pcd.voxel_down_sample(voxel_size)  # Downsample the point cloud
        num_points = len(downsampled_pcd.points)  # Get the number of points

        # Check if we are close to the target number of points
        if num_points <= target_points:
            break  # Stop if the target is reached
        
        # Increase the voxel size to reduce the number of points in the next iteration
        voxel_size *= 1.3  # Increase by 30%

    print(f"Voxel size: {voxel_size}, Points left: {num_points}")

    return voxel_size, downsampled_pcd

if __name__ == "__main__":
    # Load the point cloud file
    pcd = o3d.io.read_point_cloud("/home/sumanth/vbm_project/src/coco_can.pcd")

    # Preprocess the point cloud
    pcd = remove_invalid_points(pcd)  # Remove invalid points
    optimal_voxel_size, downsampled_pcd = adaptive_voxel_downsampling(pcd, target_points=2000)  # Downsample the point cloud
    downsampled_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=100, max_nn=30))  # Estimate normals

    # Normalize and orient normals consistently
    downsampled_pcd.normalize_normals()
    downsampled_pcd.orient_normals_consistent_tangent_plane(100)

    # Print point cloud bounds for reference
    print("Point cloud bounds:")
    print("Min bound:", pcd.get_min_bound())
    print("Max bound:", pcd.get_max_bound())

    # Calculate a threshold based on the point cloud bounds
    min_distance_threshold = min(pcd.get_max_bound() - pcd.get_min_bound()) * 0.8
    print("min_distance_threshold", min_distance_threshold)

    # Compute the center of mass of the point cloud
    center_of_mass = compute_center_of_mass(downsampled_pcd)
    print(f"Center of mass of the point cloud: {center_of_mass}")

    # Find the best grasp points near the center of mass
    point1, point2 = find_grasp_near_center_of_mass(downsampled_pcd, center_of_mass, 
                                                      min_distance_threshold=min_distance_threshold, 
                                                      angle_threshold=3.14)

    # Visualize the point cloud and display the grasp points and grasp line
    visualize_grasp(downsampled_pcd, point1, point2)
