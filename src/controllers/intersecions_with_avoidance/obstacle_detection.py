import numpy as np
import cv2
from distances import compute_distances

def calculate_predicted_distances(fov_vertical, num_segments, camera_mount_height, margin_factor=1):
    """
    Calculate the predicted distances for each segment based on the field of view and camera mount height.

    Args:
        fov_vertical (float): The vertical field of view (in radians).
        num_segments (int): The number of segments to divide the image into.
        margin_factor (float): The margin factor to adjust the predicted distance.
        camera_mount_height (float): The height at which the camera is mounted.

    Returns:
        np.ndarray: An array of predicted distances for each segment.
    """
    predicted_distances = np.zeros(num_segments)
    
    for i in range(num_segments):
        # Compute the vertical angle for the segment
        alpha = (num_segments - i - 0.5) * fov_vertical / num_segments + 0.01
        theta = np.radians(90) - fov_vertical / 2 + alpha
        
        # If the angle exceeds 90 degrees, set distance to infinity
        if theta > np.radians(90):
            predicted_distances[i] = float('inf')
        else:
            # Calculate the predicted distance based on the camera's height and angle
            predicted_distances[i] = (camera_mount_height / np.cos(theta)) * margin_factor
            
    return predicted_distances

def calculate_centers_of_segments(num_segments, camera_width, camera_height):
    segment_width = camera_width // num_segments
    segment_height = camera_height // num_segments
    
    segment_coordinates = []
    for i in range(num_segments):
        for j in range(num_segments):
            # Calculate the center of each segment
            segment_x_center = (j * segment_width + (j + 1) * segment_width) // 2
            segment_y_center = (i * segment_height + (i + 1) * segment_height) // 2
            segment_coordinates.append((segment_x_center, segment_y_center))
            
    return segment_coordinates

def detect_obstacles(image, camera_height, camera_width,
                     lidar_width, lidar_height, lidar_data,
                     segment_coordinates, predicted_distances,
                     distance_treshhold, fov_horizontal, car_width,
                     num_segments=20, display_results = True):
    """
    Detect obstacles in the image based on the lidar data and predicted distances.

    Args:
        image (np.ndarray): The image to process.
        camera_height (int): The height of the camera image.
        camera_width (int): The width of the camera image.
        lidar_width (int): The width of the lidar data.
        lidar_height (int): The height of the lidar data.
        lidar_data (list): The lidar data.
        segment_coordinates (list): List of (x, y) coordinates for the centers of each segment.
        predicted_distances (np.ndarray): List of predicted distances for each segment.
        distance_treshhold (float): distance to obstacle to be detected
        num_segments (int): Number of segments in each dimension (default is 20).
    """
    # Calculate segment dimensions
    segment_width = camera_width // num_segments
    segment_height = camera_height // num_segments

    # Compute the distances for each segment based on lidar data
    segment_distances = compute_distances(segment_coordinates, camera_height, camera_width, lidar_width, lidar_height, lidar_data, kernel_size=1)
    
    x = distance_treshhold * np.tan(fov_horizontal/2)
    j_offset = int((((x - car_width/2) / x ) * num_segments) / 2)

    isObstacle = False
    distance_index = 0
    
    if display_results:
        visual_image = image.copy()
    
    for i in range(num_segments):
        for j in range(num_segments):
            # Define the corners of each segment
            x1 = j * segment_width
            y1 = i * segment_height
            x2 = (j + 1) * segment_width
            y2 = (i + 1) * segment_height

            # Get the distance for the current segment and its predicted distance
            distance = segment_distances[distance_index]
            predicted_distance = predicted_distances[i]
            distance_index += 1

            # Check if the detected distance is within the predicted range
            if np.isfinite(distance) and distance < predicted_distance and distance < distance_treshhold and j > j_offset and j < num_segments - j_offset:
                color = (0, 0, 255)  # Red for obstacle
                alpha = 0.5  # Red with transparency
                isObstacle = True
            else:
                color = (0, 255, 0)  # Green for no obstacle
                alpha = 0.5  # Green with transparency

            if display_results:
                # Create a copy of the original image to apply the alpha blending
                overlay = visual_image.copy()
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)  # -1 to fill the rectangle

                # Blend the overlay with the original image using alpha transparency
                cv2.addWeighted(overlay, alpha, visual_image, 1 - alpha, 0, visual_image)

    if display_results:
        # Display the image with obstacle detection annotations
        cv2.imshow("Obstacle Detection in Segments", visual_image)
        cv2.waitKey(1)
    
    return isObstacle