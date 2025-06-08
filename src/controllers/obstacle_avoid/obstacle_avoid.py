from vehicle import Driver # type: ignore
import torch

from distances import prepare_image, show_segments, compute_distances
from obstacle_detection import detect_obstacles, calculate_predicted_distances, calculate_centers_of_segments
from vehicle_control import VehicleController, PidController, PurePursuitController
from line_segmentation import compute_line_mask, compute_center_mask

# Constans
DETECTION_INTERVAL = 10

# For all lidars and cameras
HORIZONTAL_FOV = 1.57
VERTICAL_FOV = 0.92
MOUNT_HEIGHT = 0.9

# Device selection
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load YOLOPv2 model
yolopv2 = torch.jit.load("../../../models/yolopv2.pt", map_location=device)
yolopv2.to(device)
yolopv2.eval()
if device.type == 'cpu':
    torch.set_num_threads(12)
    print(f"Num threads: {torch.get_num_threads()}")
print("Got the model!")

# Initialization of the driver and timestep
driver = Driver()
TIME_STEP = int(driver.getBasicTimeStep())

# Initialize devices
camera_front = driver.getDevice("camera_front_obstacle")
camera_front_height = camera_front.getHeight()
camera_front_width = camera_front.getWidth()

camera_left = driver.getDevice("camera_left_obstacle")
camera_left_height = camera_left.getHeight()
camera_left_width = camera_left.getWidth()

camera_right = driver.getDevice("camera_right_obstacle")
camera_right_height = camera_right.getHeight()
camera_right_width = camera_right.getWidth()

lidar_front = driver.getDevice("lidar_front_obstacle")
lidar_front_width = lidar_front.getHorizontalResolution()
lidar_front_height = lidar_front.getNumberOfLayers()

lidar_left = driver.getDevice("lidar_left_obstacle")
lidar_left_width = lidar_left.getHorizontalResolution()
lidar_left_height = lidar_left.getNumberOfLayers()

lidar_right = driver.getDevice("lidar_right_obstacle")
lidar_right_width = lidar_right.getHorizontalResolution()
lidar_right_height = lidar_right.getNumberOfLayers()

# Enable the devices
DEVICE_INTERVAL = TIME_STEP * DETECTION_INTERVAL
camera_front.enable(DEVICE_INTERVAL)
lidar_front.enable(DEVICE_INTERVAL)
camera_left.enable(DEVICE_INTERVAL)
lidar_left.enable(DEVICE_INTERVAL)
camera_right.enable(DEVICE_INTERVAL)
lidar_right.enable(DEVICE_INTERVAL)

# Controllers for steering the car
pid = PidController(Kp=0.3, Ki=0.055, Kd=0.001)
pure_pursuit = PurePursuitController(L=3, min_ld=8, max_ld=15, Kdd=0.5)
vehicle_controller = VehicleController(pure_pursuit, pid, driver, max_speed=50.0, min_speed=0.0)

# Data for obstacles detection
num_segments = 20
segments_coordinates = calculate_centers_of_segments(num_segments, camera_front_width, camera_front_height)
predicted_distances = calculate_predicted_distances(VERTICAL_FOV, num_segments, MOUNT_HEIGHT, margin_factor=1)

# Frame counter 
frame_counter = 0

# Main simulation loop
with torch.no_grad():
    while driver.step() != -1:
        if frame_counter % DETECTION_INTERVAL == 0:
            
            # Get camera images
            image_front = prepare_image(camera_front.getImage(), camera_front_width, camera_front_height)
            image_left = prepare_image(camera_left.getImage(), camera_left_width, camera_left_height)
            image_right = prepare_image(camera_right.getImage(), camera_right_width, camera_right_height)
            
            # Get LiDAR data
            lidar_data_front = lidar_front.getRangeImage()
            lidar_data_left = lidar_left.getRangeImage()
            lidar_data_right = lidar_right.getRangeImage()
            
            #show_segments(image_front.copy(), camera_front_height, camera_front_width, lidar_front_width, lidar_front_height, lidar_data_front, num_segments, num_segments)
            
            # Detect obstacles
            obstacle_front = detect_obstacles(
                image_front, camera_front_height, camera_front_width, 
                lidar_front_width, lidar_front_height, lidar_data_front,
                segments_coordinates, predicted_distances, 15, 1.57, 6, num_segments, display_results = False)
            
            obstacle_left = detect_obstacles(
                image_left, camera_left_height, camera_left_width, 
                lidar_left_width, lidar_left_height, lidar_data_left,
                segments_coordinates, predicted_distances, 7, 1.57, 10, num_segments, display_results = False)
            
            obstacle_right = detect_obstacles(
                image_right, camera_right_height, camera_right_width, 
                lidar_right_width, lidar_right_height, lidar_data_right,
                segments_coordinates, predicted_distances, 7, 1.57, 10, num_segments, display_results = False)
            
            obstacles = {"front": obstacle_front, "left": obstacle_left, "right": obstacle_right}
            
            # Make line segmentation with yolop and compute centers of the lines
            line_mask = compute_line_mask(image_front, camera_front_height, camera_front_width, yolopv2, visualize=False)
            center_mask = compute_center_mask(line_mask)
            distances_front = compute_distances(segments_coordinates, camera_front_height, camera_front_width, lidar_front_width, lidar_front_height, lidar_data_front)
            
            # Update vehicle controller
            vehicle_controller.update(distances_front, center_mask, camera_front_width, camera_front_height, num_segments, HORIZONTAL_FOV, obstacles, image_front, visualize=True)
            
        frame_counter += 1