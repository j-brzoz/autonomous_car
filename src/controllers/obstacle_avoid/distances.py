import numpy as np
import cv2

def prepare_image(image, camera_width, camera_height):
    # Convert raw byte data to a 4-channel BGRA image and then to a 3-channel BGR image.
    image = np.frombuffer(image, np.uint8).reshape((camera_height, camera_width, 4))
    image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    
    return image

def compute_distances(camera_coordinates, camera_height, camera_width, lidar_width, lidar_height, lidar_data, kernel_size=3):
    """
    Compute the distance from the camera to the lidar data at the given coordinates.

    Args:
        camera_coordinates (list): List of (x, y) coordinates from the camera.
        camera_height (int): The height of the camera image.
        camera_width (int): The width of the camera image.
        lidar_width (int): The width of the lidar data.
        lidar_height (int): The height of the lidar data.
        lidar_data (list): Lidar data in a flat list format.
        kernel_size (int): The size of the kernel to average over.

    Returns:
        list: List of computed distances at the given camera coordinates.
    """
    # Reshape lidar data to 2D for easier manipulation
    lidar_data_2d = np.array(lidar_data).reshape((lidar_height, lidar_width))
    distances = []
    half_kernel = kernel_size // 2

    for u, v in camera_coordinates:
        # Map camera coordinates to lidar coordinates
        lidar_x = int(u * lidar_width / camera_width)
        lidar_y = int(v * lidar_height / camera_height)
        
        # Check if the lidar coordinates are within bounds
        if 0 <= lidar_x < lidar_width and 0 <= lidar_y < lidar_height:
            # Define the region to sample from lidar data (kernel)
            x_start = max(0, lidar_x - half_kernel)
            x_end = min(lidar_width, lidar_x + half_kernel + 1)
            y_start = max(0, lidar_y - half_kernel)
            y_end = min(lidar_height, lidar_y + half_kernel + 1)
            
            # Calculate average distance in the kernel region
            window = lidar_data_2d[y_start:y_end, x_start:x_end]
            average_distance = np.mean(window)
            distances.append(average_distance)
        else:
            # Return NaN for points outside the lidar data bounds
            distances.append(float('nan'))
    
    return distances

def show_distances_from_yolo(results, camera_height, camera_width, lidar_width, lidar_height, lidar_data):
    """
    Annotate YOLO detection results with distances calculated from lidar data.

    Args:
        results (list): List of YOLO detection results.
        camera_height (int): Height of the camera image.
        camera_width (int): Width of the camera image.
        lidar_width (int): Width of the lidar data.
        lidar_height (int): Height of the lidar data.
        lidar_data (list): Lidar data in a flat list format.
    """
    # Get coordinates for bounding boxes from YOLO detection results
    camera_coordinates = []
    for result in results:
        for box in result.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)
            x_center = (x1 + x2) // 2
            y_center = (y1 + y2) // 2
            camera_coordinates.append((x_center, y_center))

    # Compute distances for each detected coordinate
    distances = compute_distances(camera_coordinates, camera_height, camera_width, lidar_width, lidar_height, lidar_data)
    
    # Plot the image with YOLO boxes and distance annotations
    base_image = results[0].plot()

    for result in results:
        for box, distance in zip(result.boxes.xyxy, distances):
            x1, y1, x2, y2 = map(int, box)

            # Set text color and display distance (or 'N/A' if no valid distance)
            color = (255, 255, 255)
            distance_text = f"{distance:.2f} m" if np.isfinite(distance) else "N/A"
            cv2.putText(base_image, distance_text, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    cv2.imshow("YOLO Detection with Distances", base_image)
    cv2.waitKey(1)
    
def show_segments(image, camera_height, camera_width, lidar_width, lidar_height, lidar_data, num_segments_x=10, num_segments_y=10):
    """
    Divide the image into segments and display distances for each segment.

    Args:
        image (np.ndarray): The image to process.
        camera_height (int): Height of the camera image.
        camera_width (int): Width of the camera image.
        lidar_width (int): Width of the lidar data.
        lidar_height (int): Height of the lidar data.
        lidar_data (list): Lidar data in a flat list format.
        num_segments_x (int): Number of segments along the x-axis.
        num_segments_y (int): Number of segments along the y-axis.
    """
    # Initialize segment dimensions
    segment_coordinates = []
    segment_width = image.shape[1] // num_segments_x
    segment_height = image.shape[0] // num_segments_y

    # Calculate segment centers (coordinates)
    for i in range(num_segments_y):
        for j in range(num_segments_x):
            segment_x_center = (j * segment_width + (j + 1) * segment_width) // 2
            segment_y_center = (i * segment_height + (i + 1) * segment_height) // 2
            segment_coordinates.append((segment_x_center, segment_y_center))

    # Compute distances for each segment center
    segment_distances = compute_distances(segment_coordinates, camera_height, camera_width, lidar_width, lidar_height, lidar_data)

    # Annotate the image with segment information
    distance_index = 0
    for i in range(num_segments_y):
        for j in range(num_segments_x):
            x1 = j * segment_width
            y1 = i * segment_height
            x2 = (j + 1) * segment_width
            y2 = (i + 1) * segment_height

            # Draw the rectangle for each segment
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 1)

            # Get the distance for the current segment
            distance = segment_distances[distance_index]
            distance_index += 1
            
            # Display distance text on the image
            distance_text = f"{distance:.1f}" if np.isfinite(distance) else "N/A"
            text_x = x1 + 2
            text_y = y1 + segment_height // 2
            cv2.putText(image, distance_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow("Segmented Obstacle Detection", image)
    cv2.waitKey(1)